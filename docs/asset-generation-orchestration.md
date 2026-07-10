# AI素材生成orchestration

`tools.asset_generation_orchestrator` は、canonicalな `asset_generation_queue.yaml` を読み取り専用入力として、segmentation、assignment、source抽出、inpainting、selection、再合成、global quality、refinementをrun単位で接続します。

## 安全境界

- 既定はdry-runです。出力を作らず、backend adapterを解決してもmodelはloadしません。
- 実行には `--execute` が必要です。model IDはローカルcacheだけを使い、暗黙downloadは各backendが拒否します。
- canonical queueは上書きしません。assignment、extraction、inpainting selection、refinementごとに `queue-candidates/` へ候補を作ります。
- assignmentとselectionは既定で `waiting_for_review` です。`--auto-approve-mock` は有効なbackendがすべてmockの場合だけ利用できます。
- run stateはcanonical queueとsource画像のSHA-256を保持します。`--resume` 時に違いがあればstale runとして拒否します。
- base-dir外への出力、別runとのrun ID混在、既存runへの未指定上書き、token・credential・ローカル絶対パスのprovenance保存を拒否または除去します。

## 実行モード

完全mock CI:

```powershell
python -m tools.asset_generation_orchestrator generated/asset_generation_queue.yaml `
  --segmentation-backend mock --inpainting-backend mock `
  --run-id run-001 --auto-approve-mock --execute
```

segmentationだけを実行してassignment reviewで停止:

```powershell
python -m tools.asset_generation_orchestrator generated/asset_generation_queue.yaml `
  --segmentation-backend sam2 --inpainting-backend disabled `
  --run-id segment-001 --execute
```

承認済みmaskを持つqueueからinpaintingを再開:

```powershell
python -m tools.asset_generation_orchestrator generated/asset_generation_queue.segmented.yaml `
  --segmentation-backend disabled --inpainting-backend diffusers `
  --run-id inpaint-001 --execute
```

review後はrun内のassignmentまたはselection YAMLだけを承認済みにし、同じ入力、run ID、output directoryへ `--resume --execute` を指定します。completed stageは再実行しません。

## 成果物

```text
generated/runs/<run_id>/
  run.yaml
  segmentation/       result、ranking、candidate mask、provenance
  assignments/        assignment plan
  masks/              run-local mask manifest
  extracted-parts/    source画素から抽出した同一canvasのpart
  inpainting/         layer別request、candidate、selection、provenance
  quality/            global quality入力、report、difference
  previews/           再合成画像とdifference
  refinement/         failed partだけのrefinement plan
  queue-candidates/   各gate後のqueue候補とdiff summary
```

`run.yaml` のstage statusは `planned`、`running`、`completed`、`waiting_for_review`、`blocked`、`failed`、`skipped` のいずれかです。stage例外は後続を`blocked`にし、部分成果物が存在しても`completed`にはしません。quality評価自体が完了してpart FAILが見つかった場合は、refinement planにそのpartだけを登録し、run outcomeを`refinement_required`にします。

## Backend registryとresource制御

`tools.backend_registry.registry` は次をlazyに解決します。

- segmentation: `mock`、`sam2`、`grounded_sam2`
- inpainting: `mock`、`diffusers`、`flux_fill`

availabilityは理由付きで取得でき、registry取得だけではmodelをloadしません。`ResourceScheduler` はdependencyのないCPU taskを並列化し、GPU worker上限とglobal model exclusive lockを適用します。GPU stageは既定で1 workerのため、segmentation modelとinpainting modelを同時常駐させません。
