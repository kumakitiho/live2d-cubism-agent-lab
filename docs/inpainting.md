# 生成inpainting

この層は、canonicalな `asset_generation_queue.yaml` から準備された1部品分のrequestを受け、
`inpaint_mask` 内だけを生成して候補・品質情報・selection planを作ります。segmentation、queue全体の
orchestration、redraw、PSD、Cubism操作は担当しません。

## 安全境界

- 既定はdry-runで、request YAML以外の画像・modelを読み込まず、成果物を生成／更新しません。
- `--execute` 時もsource、current part、mask、canonical queueを上書きしません。
- model出力はcropに限定し、元canvasへ直接採用しません。
- cropを元サイズへ戻した後、soft `inpaint_mask` で1回だけ局所合成します。
- `protect_mask` はcurrent partから復元し、`inpaint_mask` 外もcurrent partと一致させます。
- `edge_extension_mask` はprovenanceとして記録しますが、生成許可maskには加えません。
- candidateは常に `inferred: true` 相当かつ `requires_review: true` です。
- selectionは `review.status: approved`、reviewer、notesが揃うまでqueue候補へ適用できません。
- queue適用も別ファイルを作るだけで、canonical queueを直接変更しません。
- selection適用時は元resultのSHA-256、candidate metadataの完全一致、PNG/previewのSHA-256と
  base-dir内パス、candidate canvasを再検証します。review欄以外を編集したselectionは適用できません。

## Backend

`mock` はCPUだけで動くdeterministic fixtureです。candidate countとseedを再現でき、modelや
optional dependencyを必要としません。

`diffusers` は `AutoPipelineForInpainting` を実行時だけimport/loadします。model ID、revision、
seed、inference steps、guidance scale、strength、device、dtype、scheduler、CPU offload、attention
slicingを `backend_config` で指定できます。`local_files_only` は既定でtrueで、`offline: true` も
強制的にlocal-onlyにします。リモート取得を許可する場合は、利用者が `local_files_only: false` を
requestへ明記し、`--execute` を指定する必要があります。

`flux_fill` はDiffusersの `FluxFillPipeline` 用optional adapterです。FLUX Fillは通常のinpaintingと
異なり `strength` を受けないためadapter内で渡しません。model IDはhard-codeしていません。
使用前に、選択したcheckpointの配布ページでライセンス、商用利用条件、出力利用条件、アクセス条件を
利用者自身が確認してください。adapterの実装や `inpainting` extraの導入は、model利用許諾を意味しません。

Optional runtimeだけを導入する場合:

```powershell
python -m pip install -e ".[inpainting]"
```

この操作ではDiffusers等のPython packageを導入しますが、checkpointは取得しません。

## Requestとprompt

request schemaは `schemas/inpainting_request.schema.yaml`、例は
`examples/inpainting_request.sample.yaml` です。`tools.inpainting_prompt_builder` はcharacter specの
identity、line style、palette、lighting、queueのrole、side、周辺geometry、隠れ領域の目的をまとめ、
透過背景・同一canvas/origin・mask内限定をpositive promptへ、全身再生成やpose・identity・色・線幅・
光源・無関係なアクセサリ・不透明背景・resize・mask外変更をnegative promptへ入れます。

```powershell
python -m tools.inpainting_prompt_builder `
  examples/character_spec.sample.yaml `
  examples/asset_generation_queue.sample.yaml `
  --layer-id face_hidden_fill `
  --output generated/inpainting_prompt.yaml
```

上はdry-runです。YAMLを書き出す場合だけ `--execute` を付けます。

## Candidate生成

```powershell
python -m tools.generative_inpainter `
  examples/inpainting_request.sample.yaml `
  --backend mock `
  --output generated/inpainting_result.yaml
```

上はmodel loadもファイル更新もしないplanです。実画像を生成する場合:

```powershell
python -m tools.generative_inpainter `
  generated/inpainting_request.yaml `
  --backend diffusers `
  --model-id <MODEL_ID> `
  --candidate-count 4 `
  --output generated/inpainting_result.yaml `
  --execute
```

処理順は `bbox -> padding -> model sizeへresize -> inference -> crop sizeへ復元 -> soft mask合成 ->
protect復元 -> mask外復元 -> 同一canvas/originへ配置` です。候補PNG、4 panel preview、seed、backend、
model provenance、crop/resize情報、quality metrics、PNG/previewのSHA-256を記録します。run全体を
一時ディレクトリへstageし、全候補とresultの検証後にresultを最後に公開します。公開中の失敗は既存
成果物へrollbackし、backendが途中で失敗したrunは候補を1つも公開しません。

protect差分、inpaint mask外差分、canvas/origin、alpha hole、mask外漏れ、required coverage、edge
continuity、boundary color、white halo、visual reconstruction contractをgateに使います。candidate層の
visual reconstructionは、生成candidateをsource画像の背面へ合成し、target/inpaint領域でsourceから
見える差分を測ります。全partのdraw order再合成は既存のglobal quality evaluatorで別途実行します。
`inpaint_region_source_difference_score` はprovenanceとranking参考情報に限り、FAIL条件にはしません。
edge continuityはseam alpha差の平均、alpha continuityは最大差、surrounding palette consistencyは
生成領域と隣接supportの平均色差として独立に記録します。

## Ranking、レビュー、queue候補

```powershell
python -m tools.inpainting_candidate_ranker `
  generated/inpainting_result.yaml `
  --output generated/inpainting_selection.yaml `
  --execute
```

全候補failならselectionは作りません。合格候補だけをedge、boundary、halo、alpha、palette、visual、
seed、candidate IDの決定的な順でrankします。bestを選んでも `review_required: true` で、出力直後の
`review.status` は `pending` です。previewを人間が確認後、selection YAMLのreviewを次のように更新します。

```yaml
review:
  status: approved
  reviewer: <REVIEWER>
  notes: <VISUAL_REVIEW_NOTES>
```

承認済みselectionから、canonical queueとは別の更新候補を作ります。

```powershell
python -m tools.inpainting_candidate_ranker apply `
  generated/asset_generation_queue.yaml `
  generated/inpainting_selection.yaml `
  --output generated/asset_generation_queue.inpainted.yaml `
  --execute
```

更新対象はselectionの1 partだけです。`source_file`、generation/provenance、quality/readinessだけを更新し、
他partは同じmappingのまま保持します。readinessは `generated` のままで、review済みcandidateを自動で
`approved` やCubism handoff可能にはしません。

## テスト境界

通常の `python -m pytest -q` は `not benchmark and not gpu` を既定にし、mockとadapter contractだけを
実行してmodelを取得しません。modelをローカルに用意したGPU統合テストは `python -m pytest -m gpu`
に隔離します。現時点では実checkpointを使うGPU fixtureは同梱していません。
