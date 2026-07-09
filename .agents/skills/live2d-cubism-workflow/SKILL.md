---
name: live2d-cubism-workflow
description: Live2DまたはVTuber制作で、キャラ仕様・layer map・Cubism action planを作成し、高レベルUIマクロ、Cubism External API、実行レポートを安全に使うときに使用する。PSDインポート、自動メッシュ生成、パラメータ確認、dry-run、Cubism操作失敗の記録を扱う。任意座標クリック、自由ドラッグ、完成済みモデルや完全自動リギングの主張には使用しない。
---

# Live2D Cubism Workflow

## ワークフロー

1. `AGENTS.md`、README、既存の仕様・素材・直近ログを読む。
2. 現在のフェーズを `spec`、`asset/layer preparation`、`Cubism operation`、`rigging/QA` から選ぶ。
3. 計画前に `references/workflow.md` を読む。UI操作時は `references/ui_macro_policy.md`、API時は `references/cubism_api_policy.md`、action plan作成時は `references/action_plan_schema.md` も読む。
4. 仕様がなければ `assets/character_spec.template.yaml`、レイヤー計画がなければ `assets/layer_map.template.yaml` から作る。不足素材、画像prompt、PSD分離、リギングは対応するasset templateから始める。
5. Cubism操作は `action_plan.yaml` にし、`python -m scripts.validate_action_plan <path>` で検証する。
6. 仕様とlayer mapも対応validatorで検証する。
7. 実機操作前に必ずBridgeまたは個別CLIをdry-runする。実行はユーザーが求めた場合だけ `--execute` を付ける。
8. `outputs/` のレポート、スクリーンショット、失敗理由、次の行動を確認・記録する。

## 操作境界

- PSDインポート、ショートカット、既知ダイアログ入力、自動メッシュ生成、保存、Undo、スクリーンショットは名前付きUIマクロで扱う。
- パラメータ、ドキュメント、モデルUIDはCubism External APIで扱う。
- 任意座標クリック、汎用click、自由ドラッグ、中間点配置を追加しない。
- 想定外ダイアログ、未知のアクセシビリティ名、未承認API、失敗した検証では停止する。
- `manual_checkpoint` は見た目の良否や想定外状態の判断にだけ使う。定型ボタン操作を人間へ戻すために使わない。
- CubismプロジェクトとVTube Studioで目視確認するまで完成と呼ばない。

## 成果物テンプレート

- `assets/character_spec.template.yaml`
- `assets/asset_generation_queue.template.yaml`
- `assets/layer_map.template.yaml`
- `assets/image_prompt.template.md`
- `assets/psd_separation_instructions.template.md`
- `assets/rigging_plan.template.md`

## 主要コマンド

```powershell
python -m scripts.validate_character_spec examples/character_spec.sample.yaml
python -m scripts.validate_layer_map examples/layer_map.sample.yaml
python -m scripts.validate_action_plan examples/action_plan.sample.yaml
python -m tools.cubism_bridge run-action-plan examples/action_plan.sample.yaml
```

`--execute` なしはdry-runである。実行後は `outputs/action_plan_report.md` を読む。
