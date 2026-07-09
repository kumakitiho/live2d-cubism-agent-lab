# Workflow

## フェーズ

1. **spec**: `character_spec.yaml` に目的、範囲、外見、表情、動き、制約、納品物を記録する。
2. **asset/layer preparation**: source素材を評価し、`layer_map.yaml` とPSD分離指示を作る。素材分けmasterとCubism import PSDは分ける。
3. **Cubism operation**: validator済みの `action_plan.yaml` をdry-runし、名前付きUIマクロとExternal APIを実行する。
4. **rigging/QA**: パラメータ、デフォーマ、物理演算、表情、VTube Studio動作を確認する。

## 成果物

- `character_spec.yaml`
- `layer_map.yaml`
- `action_plan.yaml`
- `rigging_plan.md`
- `outputs/action_plan_report.md`
- `outputs/*.png`

## 完了条件

- 構造ファイルがvalidatorを通る。
- UI/API操作はdry-runとの差分が説明できる。
- PSD importでは実行前後のModelingDocument数と、実行後current ModelUIDの所属を検証する。
- 失敗時のレポートとスクリーンショットが残る。
- CubismとVTube Studioで目視確認するまで、完成または本番投入可能と扱わない。
