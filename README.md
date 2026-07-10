# Live2D Cubism Agent Lab

Live2Dモデル制作を題材に、Codexの **Skill → Script Tool → 高レベルUIマクロ → External API Bridge** を学びながら使うためのWindows向けMVPです。`live2d-one-image-vtuber` が、仕様、素材、Cubismの3工程を構造化成果物でつなぎます。完成モデルを一発生成する仕組みではありません。

## できること

- 1枚絵からmodel scope・motion level別のLive2Dパーツ計画を生成
- source画像の観測情報とユーザー意図を分けたcharacter spec生成
- 目、口、髪、体、隠れ補完を並列jobとして管理するasset queue
- 隠れ部分を `inferred`、要確認素材を `review_required` として追跡
- Cubismの問題を `asset_feedback.yaml` で対象layerへ戻すfeedback loop
- `asset_manifest.yaml` の構造、必須部品、実ファイル、layer map整合、Cubism handoff条件を検証
- manifestからPSDレイヤー順を作るplan-only builder stub
- `character_spec.yaml` と `layer_map.yaml` のテンプレート・軽量検証
- allowlist方式の `action_plan.yaml` 検証
- Cubismのフォーカス、PSDインポート、自動メッシュ生成、保存、Undo、スクリーンショットを名前付きマクロとして計画・実行
- Cubism External APIの登録、許可状態、ドキュメント、モデルUID、パラメータ取得・設定
- UIマクロとAPI確認を1つの実行レポートへまとめる
- PSD取り込み前後を同じAPI sessionで比較し、新規DocumentUIDと取り込み後current ModelUIDを照合する
- 実機なしでのdry-run、API payloadテスト、UI操作列テスト

任意座標クリック、頂点の自由ドラッグ、中間点の手作業配置は公開コマンドにしません。PSDインポートやボタン操作は、人間へ戻すのではなく高レベルマクロで自動化します。人間確認は見た目の良否や想定外ケースの判断に限定します。

## 1枚絵からVTuber化する流れ

```text
権利確認済みsource画像 + ユーザー希望
  ↓ character-spec-generator
character_spec.yaml
  ↓ image-to-live2d-assets
parallel queue → mask/inpainting計画 → manifest → merge gate
  ↓ 外部生成・segmentation・inpainting・描画ツール（将来接続）
generated/parts/*.png
  ↓ PSD backend（現在はbuild planのみ）
model_import.psd + layer_map.yaml
  ↓ live2d-cubism-workflow
Cubism import → auto mesh → parameter/API確認 → 目視QA
  ↓ 素材起因の問題
asset_feedback.yaml → image-to-live2d-assetsの対象jobへ戻す
```

責務は次のように分けます。

| Skill | 担当 | 担当しないこと |
|---|---|---|
| `character-spec-generator` | 画像観測、ユーザー希望、model scope、motion、runtimeの仕様化 | 素材生成、PSD、Cubism操作 |
| `image-to-live2d-assets` | 画像解析計画、パーツ設計、mask、inpainting、prompt、manifest、PSD出力計画 | Cubism操作、リギング、完成モデル判定 |
| `live2d-cubism-workflow` | 承認済みPSDのimport、高レベルUI macro、External API、操作レポート | 隠れ素材の生成、権利判断、自由座標操作 |
| `live2d-one-image-vtuber` | 3 Skillのhandoff、gate、feedback loop | 各工程の実処理、Agents SDK state machine |

MVPでは画像やPSDの実生成より、計画・manifest・validationを優先します。`tools.psd_asset_builder` は空のPSDを作らず、実backend接続前は必ず `status: plan_only` を返します。

著作権、二次創作ガイドライン、肖像・商標、画像生成サービスの利用条件を含め、source画像をモデル化する権利が不明な場合は先へ進めません。`source_image.rights_status: confirmed` はユーザーが権利を確認した後だけ設定してください。

## セットアップ

Python 3.11以上を使います。Windowsで実機GUI操作も行う場合は `windows` extraを入れてください。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[windows,dev]"
python -m pytest -q
```

## no-assets dry-run

source画像やPSDがなくても、次のaction planはvalidatorとBridge dry-runを通せます。`--execute` は付けません。

```powershell
python -m tools.cubism_ui save
python -m tools.cubism_ui apply-auto-mesh --preset Standard --alpha 10
python -m tools.cubism_api get-documents
python -m scripts.validate_action_plan examples/action_plan.no_assets.sample.yaml
python -m tools.cubism_bridge run-action-plan examples/action_plan.no_assets.sample.yaml
```

## real-assets workflow

以下は実在する `assets/source/character.png` が必要です。source画像はGit管理されません。

```powershell
python -m tools.material_planner assets/source/character.png --model-scope bust_up --motion-level standard
python -m tools.asset_feedback_validator examples/asset_feedback.sample.yaml --layer-map examples/layer_map.sample.yaml
python -m tools.asset_generation_queue_validator examples/asset_generation_queue.sample.yaml
python -m tools.asset_manifest_validator examples/asset_manifest.sample.yaml --base-dir .
python -m tools.psd_asset_builder examples/asset_manifest.sample.yaml --base-dir .
```

上のqueueコマンドはサンプルの構造確認用で、未解決feedbackがあるため `merge_ready: false` が正常です。実素材をCubismへhandoffする直前は、実成果物のqueueに対して終了コードも含めた必須gateを実行します。

```powershell
python -m tools.asset_generation_queue_validator generated/asset_generation_queue.yaml --base-dir . --manifest generated/asset_manifest.yaml --require-merge-ready
python -m tools.asset_manifest_validator generated/asset_manifest.yaml --base-dir . --require-handoff-ready
```

1つ目のコマンドはqueueの `output_manifest` と指定manifestの実パス・project・source画像を照合し、2つ目はmanifest本体と実ファイルのCubism handoff readinessを検証します。どちらか片方だけではhandoff gate完了と扱いません。

計画YAMLだけを書き出す場合は `--execute`、PSD build planを書き出す場合は `--write-plan` を使います。後者もPSD自体は生成しません。

```powershell
python -m tools.material_planner assets/source/character.png --model-scope half_body --motion-level expressive --execute --output outputs/asset_generation_plan.yaml
python -m tools.psd_asset_builder examples/asset_manifest.sample.yaml --base-dir . --write-plan outputs/psd_build_plan.yaml
```

次のコマンドは実在する `generated/model_import.psd` と `generated/layer_map.yaml` を要求します。サンプルYAMLだけでは実行できません。

```powershell
python -m scripts.validate_action_plan examples/action_plan.real_assets.sample.yaml
python -m tools.cubism_ui import-psd generated/model_import.psd
python -m tools.cubism_bridge run-action-plan examples/action_plan.real_assets.sample.yaml
```

## 実機Cubismで実行する

1. Live2D Cubism Editorを起動し、Modeling Workspaceを表示します。
2. External Application Integrationを有効化します。初回接続ではCubism側の「Allow」が必要です。
3. 先にdry-run出力を確認します。
4. 問題なければ `--execute` を付けます。

```powershell
python -m tools.cubism_api --execute register
python -m tools.cubism_api --execute get-approval
python -m tools.cubism_ui import-psd generated/model_import.psd --execute
python -m tools.cubism_bridge apply-auto-mesh-and-capture --preset Standard --alpha 10 --execute
```

Cubism APIのトークンは `.live2d-agent/cubism-token.json` に保存され、Git管理から除外されます。`SetParameterValues` はEditorの一時バッファへ値を送ります。解除には `clear-parameter-values` を使います。

## action plan

```powershell
python -m scripts.validate_action_plan examples/action_plan.no_assets.sample.yaml
python -m scripts.validate_action_plan examples/action_plan.real_assets.sample.yaml
python -m tools.cubism_bridge run-action-plan examples/action_plan.no_assets.sample.yaml
python -m tools.cubism_bridge run-action-plan examples/action_plan.real_assets.sample.yaml --execute
```

`action_plan.no_assets.sample.yaml` はAPI/UI payloadのdry-run境界確認用です。`action_plan.real_assets.sample.yaml` は承認済みPSDとlayer mapを使う実機用で、素材なしでは実行しません。

queue validatorは `feedback_inputs` の実YAMLとlayer mapも読み、queue・feedback・layer mapのproject、feedback IDのjob割当、severityに関係なく未解決のissueをmerge gateへ反映します。

`mode` は次の4種類です。

- `file`: 仕様・レイヤーマップなどの決定的検証
- `ui_macro`: 許可済みのCubism高レベルUIマクロ
- `api`: 許可済みのCubism External API操作
- `manual_checkpoint`: 見た目判断や想定外状態の確認。操作の肩代わりには使わない

未知のコマンド、座標指定、`click` / `drag` 系コマンドはvalidatorが拒否します。実行は最初の失敗で停止し、`outputs/action_plan_report.md` に途中結果を残します。
コマンドごとの必須・許可引数も検証し、action planからのスクリーンショット出力先は `outputs/*.png` に限定します。

## ディレクトリ

```text
.agents/skills/character-spec-generator/  source画像からcharacter specを作るSkill
.agents/skills/image-to-live2d-assets/   1枚絵からの素材生成Skillとテンプレート
.agents/skills/live2d-cubism-workflow/  承認済み素材をCubismで操作するSkill
.agents/skills/live2d-one-image-vtuber/  全体handoffとfeedback loop
tools/material_planner.py                scope/level別の素材計画
tools/asset_generation_queue_validator.py 並列queueとmerge gateの検証
tools/asset_feedback_validator.py        Cubism feedbackとlayer IDの検証
tools/asset_manifest_validator.py        manifestとhandoff gateの検証
tools/psd_asset_builder.py               PSD build plan stub
tools/cubism_ui.py                      Windows UIマクロ
tools/cubism_api.py                     WebSocket External API
tools/cubism_bridge.py                  action plan実行とレポート
scripts/                                構造成果物validator
examples/                               学習用の仕様・レイヤー・計画
assets/models/                          ユーザー所有PSD置き場（Git対象外）
assets/source/                          ユーザー所有source画像（Git対象外）
generated/                              生成part・manifest・PSD（Git対象外）
outputs/                                スクリーンショット・レポート（Git対象外）
```

## 安全性と既知の制約

- `image-to-live2d-assets` はsource画像の利用権を自動判定しません。未確認ならhandoffを停止します。
- inferred素材は、目視承認とmanifest更新が終わるまでCubism import可能と扱いません。
- handoff gateはsource画像・import PNG・PSD・layer mapの実在と基本signature、project/canvas/layer ID/nameの一致も確認します。画像の完全decodeやPSD内部構造検証は今後のbackend検証対象です。
- `required: true` の部品がimport対象外、またはredraw素材が未レビューならhandoffしません。
- segmentation、inpainting、画像生成、Photoshop Plugin、実PSD writerは未接続です。
- 実機操作はWindows UI Automationのラベルを使います。Cubismの言語・版によってアクセシビリティ名が異なる場合、推測して続行せず停止します。
- 対象ウィンドウはタイトルだけでなく、実行ファイル名が既定の `CubismEditor*.exe` に一致することも確認します。ViewerやUpdaterは対象外で、ダイアログは特定済みEditor process IDに限定します。
- 現在のMVPは日本語/英語の既知ラベルを扱います。実機ごとのUIA調整は、失敗スクリーンショットとcontrol情報を見て名前付きlocatorを追加してください。
- 自動メッシュ生成はパラメータ設定後に行うと変形を戻す可能性があります。取り込み直後に行う前提です。
- auto-meshのUndo recoveryは、マクロがauto-mesh適用を完了したと記録した場合だけ実行します。結果は常に目視確認が必要です。
- `save` はSave Asダイアログが開いた場合に完了扱いせず停止します。新規モデルは保存先を明示した別の名前付きマクロを追加してから自動保存対象にしてください。
- 実装時のローカル環境ではCubismの起動・標準インストール先を確認できなかったため、実機UIの成功までは保証していません。dry-runとモック境界は検証対象です。
- Cubism/VTube Studioで目視確認するまで、モデル完成・本番投入可能とは扱いません。

## 次の拡張

CLIの入出力が実機で安定した後に進めます。

1. segmentation / inpainting / 外部画像生成tool adapter
2. Photoshop PluginまたはPSD writer backend
3. 成熟したvalidatorとBridgeコマンドをMCP toolとして公開
4. UIA control treeの診断コマンドと版別profile
5. PSDレイヤー解析とparameter CSV生成
6. Director / Asset Evaluator / Layer Designer / Cubism PlannerへのAgent分割
7. Cubism書き出し通知とVTube Studio QA

## 参照した公式仕様

- [Live2D Cubism Editor External API](https://docs.live2d.com/en/cubism-editor-manual/external-application-integration-api/)
- [External API integration functions](https://docs.live2d.com/en/cubism-editor-manual/external-application-integration-api-list/)
- [PSD import](https://docs.live2d.com/en/cubism-editor-manual/psd-import/)
- [Automatic Mesh generator](https://docs.live2d.com/cubism-editor-manual/mesh-edit/)
