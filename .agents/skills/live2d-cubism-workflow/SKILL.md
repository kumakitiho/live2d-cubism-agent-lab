---
name: live2d-cubism-workflow
description: Live2DまたはVTuber制作で、承認済みmodel_import.psd、layer_map.yaml、action_plan.yamlを受け取り、Cubismの高レベルUIマクロ、External API、実行レポート、素材へのasset_feedback.yamlを安全に扱うときに使用する。キャラ仕様収集、素材生成、任意座標操作、完全自動リギングには使用しない。
---

# Live2D Cubism Workflow

承認済み素材をCubismへ取り込み、決められたaction planをdry-run、実行、評価する。キャラクター仕様や素材を上流工程の代わりに作らない。

## 必須入力

- 実体のある承認済み `model_import.psd`
- validator済み `layer_map.yaml`
- validator済み `action_plan.yaml`
- handoff readyな `asset_manifest.yaml`

入力が不足する場合は `$image-to-live2d-assets` へ戻し、このSkill内で補完しない。

## ワークフロー

1. `AGENTS.md`、README、上流handoff、直近ログを読む。
2. `references/workflow.md` を読み、入力gateを確認する。
3. action plan作成時は `references/action_plan_schema.md` を読み、`assets/action_plan.template.yaml` から始める。
4. `python -m scripts.validate_layer_map <path>` と `python -m scripts.validate_action_plan <path>` を実行する。
5. 実機操作前にBridgeまたは個別CLIをdry-runする。
6. ユーザーが実行を求めた場合だけ `--execute` を付ける。
7. operation report、スクリーンショット、パラメータ結果を確認する。
8. `assets/cubism_evaluation.template.yaml` にeye、mouth、mesh、textureの評価結果とevidenceを記録する。
9. `python -m tools.cubism_evaluation validate <evaluation> --layer-map <layer-map>` を実行する。
10. 素材由来のwarn/failがあれば `python -m tools.cubism_evaluation to-feedback <evaluation> --layer-map <layer-map> --output <feedback>` で `asset_feedback.yaml` に変換する。
11. validator済みfeedbackを `$image-to-live2d-assets` へ返す。

## 操作境界

- PSDインポート、ショートカット、既知ダイアログ入力、自動メッシュ生成、保存、Undo、スクリーンショットは名前付きUIマクロで扱う。
- パラメータ、ドキュメント、モデルUIDはCubism External APIで扱う。
- 任意座標クリック、汎用click、自由ドラッグ、中間点配置を追加しない。
- 想定外ダイアログ、未知のアクセシビリティ名、未承認API、失敗した検証では停止する。
- `manual_checkpoint` は見た目判断とfeedback記録にだけ使う。
- character spec、画像prompt、素材queue、mask、inpainting、PSD分離計画を作らない。
- CubismとVTube Studioで目視確認するまで完成と呼ばない。

## 所有するテンプレート

- `assets/action_plan.template.yaml`
- `assets/asset_feedback.template.yaml`
- `assets/cubism_evaluation.template.yaml`
- `assets/rigging_plan.template.md`

## 主要コマンド

```powershell
python -m scripts.validate_layer_map generated/layer_map.yaml
python -m scripts.validate_action_plan examples/action_plan.real_assets.sample.yaml
python -m tools.cubism_bridge run-action-plan examples/action_plan.real_assets.sample.yaml
python -m tools.cubism_evaluation validate examples/cubism_evaluation.sample.yaml --layer-map examples/layer_map.sample.yaml
python -m tools.asset_feedback_validator examples/asset_feedback.sample.yaml --layer-map examples/layer_map.sample.yaml
```

`--execute` なしはdry-runである。実行後は `outputs/action_plan_report.md` を読む。
