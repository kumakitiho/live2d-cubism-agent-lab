# 自動マスク候補導出

`automatic_mask_deriver` は、segmentationレビュー後のcanonical queueにあるsoft
`target_mask`、source alpha、draw order、他レイヤーのmaskから、次のreview-only候補を
決定的に作ります。

- `protect_mask`: source alpha内のtargetを保守的にerosionした領域
- `edge_extension_mask`: targetの外側をdilationしたsoft ring
- `inpaint_mask`: 初期実装では前髪に隠れた顔・額だけ

候補は自動確定されません。canonical queueと入力maskは上書きせず、結果、候補PNG、
review plan、新しいqueue candidateを別々の成果物として扱います。

## 実行順

既定はdry-runです。dry-runでもqueue構造、source/全target maskの存在とcanvas、hash、
base-dir、出力衝突、候補導出を検証しますが、ファイルは書きません。ファイルを書き出す
処理には明示的な `--execute` が必要です。

```powershell
python -m tools.automatic_mask_deriver `
  generated/asset_generation_queue.segmented.yaml `
  --output generated/mask_derivation_result.yaml

python -m tools.automatic_mask_deriver `
  generated/asset_generation_queue.segmented.yaml `
  --output-dir generated/mask-derivation `
  --output generated/mask_derivation_result.yaml `
  --execute

python -m tools.mask_derivation_ranker `
  generated/mask_derivation_result.yaml `
  --output generated/mask_derivation_review.yaml `
  --execute
```

review planは常にrootが `review_status: pending`、各layerが
`status: needs_review` と `requires_review: true` で始まります。人間が画像、conflict、
derivation reasonを確認した後だけ、rootを `approved`、採用layerを `approved`、
`requires_review: false` に変更します。未承認layerはapply時に無視されます。

```powershell
python -m tools.mask_derivation_assignment apply `
  generated/asset_generation_queue.segmented.yaml `
  generated/mask_derivation_review.yaml `
  --output generated/asset_generation_queue.masks.yaml `
  --execute
```

applyが変更できるのは、承認済みlayerの次のfieldだけです。

```text
protect_mask
edge_extension_mask
inpaint_mask
mask_derivation_run_id
mask_derivation_confidence
mask_derivation_status
```

`target_mask`、`source_file`、未承認layer、jobs、merge gateなどは変更しません。

## 導出ポリシー

protect候補は `erode(target_mask, radius)` を基本とし、source alphaと交差させます。
soft grayscaleをbinaryへ丸めず、Pillowのgrayscale morphologyでcoverageを保持します。
目、まつ毛、細い髪束などはradiusを1pxへ制限し、指定radiusで消滅する場合は少なくとも
1pxが残るまでradiusを下げます。これは曖昧な境界を絶対保護領域へ含めないための
保守的な候補です。

edge-extension候補は `dilate(target_mask, radius) - target_mask` です。source alpha外、
backgroundを含む他の独立part内部を候補から除外します。近接する前面layerをdraw orderから
探し、その近傍ringを高いsoft coverageで優先し、その他のmotion余白は低いcoverageで残します。
`adjacent_layers` と `overlap_purpose` には目的の前面layerを記録します。前面でない近接layer、
canvas端でのclip、source alpha外へ達したraw ringはreview conflictとして残します。初期版では
色連続性やmotion preview自体は直接解析しません。微小島検出は大きなmaskを座標setへ
展開せず、byte buffer上のscanline flood fillで行います。

inpaint候補は、roleが `face`、`face_base`、`face_hidden_fill`、`head`、`skin_face` の
いずれかで、draw order上前面に `front_hair`、`hair_front`、`bangs`、`fringe` がある場合
だけ作ります。`expected_region` があればその楕円を、なければ可視顔の左右対称形状を
使い、visible targetとprotectを引いたうえで前髪occluder内へ限定します。十分な完全形状を
推定できない場合は空maskを捏造せず、`status: unavailable` と
`reason: complete_shape_not_estimable` を返します。

## conflictとreview

最低限、同一layerのprotect/inpaint、protect/edge、edge/inpaint重複、layer間inpaint重複、
source alpha外へのedge拡張、面積上限、微小島、targetから離れた島、左右part非対称、
draw order矛盾、canvas clipを検出します。conflict領域は自動修正せず、review reasonまたは
rejection reasonにします。previewはsource、target（赤）、候補（mask type別）、conflict
（黄）、隣接layer（シアン）を同じcanvasで表示します。

出力PNGは候補ごとに次の3種類です。

```text
<layer-token>.<type>.<artifact-scope>.soft.png
<layer-token>.<type>.<artifact-scope>.binary.png
<layer-token>.<type>.<artifact-scope>.preview.png
```

## 整合性と再実行

結果にはcanonical queue bytes、canonical内容、source画像、選択外contextを含む全入力
target mask、soft/binary/preview候補PNGのSHA-256、segmentation run ID、mask derivation run IDを
記録します。run IDを省略した場合、これらのhashと設定からUUID v5を作るため、同じ入力と
設定では結果が決定的です。candidate pathにはrun IDと入力内容から作るartifact scopeを含め、
部分実行の選択layer集合も `execution_scope` とrun/artifact scopeへ含めます。別runやfull/partial
runの `--force` が過去の承認済みPNGを上書きしないようにします。rankingとapplyはhashを
再検証し、queue/source/target/context/candidate maskの変更、resultの変更、run ID混在を拒否します。
PNG/YAMLは同じディレクトリの一時ファイルからatomic replaceします。既存出力は
`--force` なしでは上書きしません。全パスは `--base-dir` 内に限定します。

dry-runは候補画像を保持しません。execute時のPNG payloadは1 MiBを超えるとspooled temporary
fileへ退避し、全layer分のRGBA previewをRAMへ同時保持しません。最終出力はpayloadを1件ずつ
同一出力ディレクトリの一時ファイルへ書き、atomic replaceします。

特定layerだけ再実行する場合は `--layer face_base` を使います。部分失敗resultから失敗
layerだけを選ぶ場合は次を使います。

```powershell
python -m tools.automatic_mask_deriver QUEUE `
  --retry-failed-from generated/mask_derivation_result.yaml `
  --output generated/mask_derivation_retry.yaml `
  --execute
```

## 初期版の未実装role

Priority Cの目・虹彩・口腔・髪の連続形状・首/胴体・衣装別の完全形状推定は未実装です。
これらは `complete_shape_not_estimable` として人間へ戻します。また、motion previewからの
露出予測とsource RGBの境界色類似度によるedge方向制御も未実装です。外部AIや実画像model
には依存せず、通常CIはPillow fixtureだけで完結します。
