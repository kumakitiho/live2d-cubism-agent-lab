# Orchestration contract

## Stage 1: character spec

入力はsource画像とユーザー希望。出力はvalidator済み `character_spec.yaml`。`open_questions` が残る場合は停止する。

## Stage 2: asset generation

入力はcharacter spec。状態の単一ソースは `asset_generation_queue.yaml`。`asset_manifest.yaml` と `layer_map.yaml` はqueueから生成し、`model_import.psd` はそのmanifestを入力にbuildする。queue merge gateとmanifest handoff gateを両方要求する。

## Stage 3: Cubism

入力は承認済みPSD、layer map、action plan。出力はoperation report、evidence、`cubism_evaluation.yaml`、必要に応じて変換した `asset_feedback.yaml`。

## Feedback transition

`asset_feedback.yaml` の `target_layer_id` をlayer mapで検証する。`requested_action` を対応するqueue jobへ追加し、そのjobとmerge gateを再実行する。変更したlayerを使うCubism stepだけを再確認する。

## State labels

- `waiting_for_user`: 人間の意図または権利確認待ち
- `planning_assets`: 素材計画・queue作成中
- `waiting_for_assets`: 外部画像処理またはレビュー待ち
- `ready_for_cubism`: merge/handoff gate通過済み
- `cubism_review`: import・mesh・parameter確認中
- `asset_rework`: feedbackを素材工程へ返した状態
- `complete_with_manual_qa`: 自動gate通過後、明記済みの目視QAを残す状態
