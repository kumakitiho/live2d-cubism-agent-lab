# 自動セグメンテーション

## 境界

この機能は、1枚のsource画像とcanonical `asset_generation_queue.yaml`から、Live2Dパーツごとのマスク候補を生成します。候補は確定マスクではありません。confidence、stability、semantic prompt、bbox、provenance、レビュー理由を持つ`segmentation_result.yaml`として出力し、rankingとassignment reviewを経て初めて別のqueue候補へ反映できます。

実装済み:

- deterministicな`MockSegmentationBackend`
- optionalかつlazy-loadの`Sam2SegmentationBackend`
- 差し替え可能なgrounding backendとSAM 2を合成する`GroundedSam2SegmentationBackend`
- local-onlyのTransformers Grounding DINO adapter
- point、box、semantic、existing-mask refinement request契約
- automatic mask generation request契約
- soft、binary、overlay previewの同一canvas出力
- confidence、左右、expected region、source alpha、競合、対称性、taxonomyを用いるranking
- human review必須のassignment planと、承認済みpartだけのqueue候補適用
- `asset_generation_orchestrator`からsource抽出・生成inpainting・qualityへ接続するrun単位のhandoff

未実装・対象外:

- model checkpointのdownloadやmodel registryへのnetwork access
- segmentation CLI単体での生成inpainting、redraw、隠れ領域生成（生成inpaintingはorchestratorが別backendへhandoff）
- Cubism操作、PSD生成、rigging
- ranking結果の自動確定
- `protect_mask`、`edge_extension_mask`、`inpaint_mask`をsegmentation成功だけで生成済みにする処理

## backend契約

backend-neutral contractは`tools/backends/segmentation/contracts.py`にあります。`SegmentationRequest`はsourceと同じcanvasのpoint、box、existing mask、fixture maskを保持します。`SegmentationResult.status`は次のいずれかです。

- `completed`: 実際に処理し、0件以上の候補を返した
- `not_run`: dry-runで、モデル・推論を実行していない
- `unavailable`: optional dependency、local checkpoint、local cache、設定が不足している
- `failed`: 導入済みruntimeが推論中に失敗した

CLIは`--execute`時の0候補を成功扱いしません。`not_run`や`unavailable`も`completed`へ読み替えません。

### Mock

CPUだけで動き、downloadは行いません。fixture maskがあればcoverage値を変更せず返します。fixtureがなければrequest内容とsource bytesから決定的なsoft maskを作ります。同一requestは同一candidate IDと画素を返します。

### SAM 2 / SAM 2.1

SAM関連moduleは`--execute`後にだけimportします。次のいずれかを明示してください。

- `--checkpoint PATH --model-config HYDRA_CONFIG_NAME`
- `--model-id REPOSITORY_ID --model-revision REVISION`（公式SAM 2対応IDの既存Hugging Face cacheだけを`local_files_only`で解決）

`--model-id`がcacheにない場合もdownloadしません。cache済みcheckpointと公式のmodel ID→Hydra config対応を解決し、local checkpoint経路と同じ`build_sam2`へ渡します。device、model ID、revision、checkpoint、configはprovenanceへ記録します。point/box/existing maskのいずれもないrequestはautomatic mask generationへ、いずれかがあるrequestはpredictorへ渡します。

### Grounded SAM 2

semantic promptをlocal Grounding DINOへ渡し、返された各bbox、score、labelを保持したままSAM 2のbox promptへ変換します。検出が複数なら候補も複数のまま返します。標準adapterはAPI tokenを使いません。`--grounding-model`は既存local directoryだけを受け付け、Transformersにも`local_files_only=True`を渡します。

## 出力

候補ごとに次をatomic replaceで公開します。source、canonical masks、part画像、queue、他のderivativeと衝突するpathは`--force`があっても拒否します。

```text
<layer>.<candidate>.soft.png
<layer>.<candidate>.binary.png
<layer>.<candidate>.preview.png
```

- `soft.png`: backendが返した8-bit grayscale coverageを保持
- `binary.png`: `--binary-threshold`（既定128）で別途二値化
- `preview.png`: sourceへsoft maskを45% opacityでoverlay

全画像はsourceと同じwidth、height、origin `(0, 0)`です。入力は上書きしません。

## ranking

`tools.segmentation_candidate_ranker`は次をscoreとレビュー理由へ反映します。

- candidate confidenceとstability
- `side: L / R / C`とbbox中心
- normalized `expected_region`
- mask面積とsource alpha overlap
- 他layer候補との重なり
- L/R pairの鏡像位置と面積比
- semantic prompt、role、layer IDのtaxonomy整合
- draw order metadataの妥当性と競合解釈（単独で候補を確定しない）

低confidence、side不明、位置不整合、面積異常、source alpha外への漏れ、候補競合、左右非対称は`requires_review`と`rejection_reasons`へ残ります。rankingは`rank`を付けるだけでselected stateを作りません。

## assignment reviewとqueue適用

assignment plannerは各layerのrank 1を「提案」しますが、自動承認しません。`target_mask`だけが選択したsoft candidateを参照します。既存`protect_mask`、`edge_extension_mask`、`inpaint_mask`は保持し、それぞれを別工程で導出すべきことを`derivation`へ記録します。

resultからassignmentまではcanonical queueのfile SHA-256、正規化した論理内容のSHA-256、source画像bytesのSHA-256を伝播します。`apply`はqueue path、論理内容、queue bytes、source bytes、segmentation run IDを再検証し、レビュー後にqueueまたはsourceが変わっていれば停止します。

適用前に人間が次を確認します。

1. 候補のpreview、confidence、レビュー理由を確認する。
2. 採用しないlayerは`needs_review`のまま残す。
3. 採用layerだけ`status: approved`、`requires_review: false`にする。
4. rootの`review_status`を`approved`にする。
5. `apply`で新しいqueue pathへ書く。

`apply`が変更できるfieldは次だけです。

```text
target_mask
protect_mask
edge_extension_mask
inpaint_mask
segmentation_backend
segmentation_model_id
segmentation_model_revision
segmentation_run_id
segmentation_confidence
```

未選択partのmappingとqueueのjobs/merge gateは変更しません。`apply`のファイル出力では、block-style `assets:`内の承認part blockだけを置換し、未選択partとその他のYAML bytes（コメント、quote、flow style、改行を含む）をそのままコピーします。安全に限定patchできないflow-styleのroot `assets`等は、全体を再serializeせず明示的に停止します。入力queueと同じpathへの出力は拒否します。

## テスト

通常CIはmockとinjected runtimeだけを使い、checkpointを取得しません。

```powershell
python -m pytest -q
python -m ruff check .
python -m mypy tools tests
python -m pip check
git diff --check
```

実SAM/GPUテストを追加する場合は必ず`@pytest.mark.gpu`を付け、通常CIから分離します。

```powershell
python -m pytest -m gpu
```
