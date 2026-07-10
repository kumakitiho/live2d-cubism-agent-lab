# Cubism環境診断とsmoke test

この診断系は、固定座標や画像推測クリックを使わず、次の順でCubism Editorを確認します。

1. External API
2. 公式キーボードショートカット
3. Windows UI Automationのcontrol type・label・process ID

UIAで対象を一意に特定できない場合や、日本語・英語profileを一意に選べない場合は停止します。任意座標クリック、自由ドラッグ、`pyautogui.click(x, y)`は実装していません。

## dry-run

`--execute`を付けない限り、Cubism、UIA、External APIへ接続せず、ファイルも作成しません。実行予定と必要な依存だけを標準出力へ表示します。

```powershell
python -m tools.cubism_environment_probe `
  --output outputs/cubism-environment-report.yaml

python -m tools.cubism_smoke_test generated/model_import.psd `
  --preset Standard `
  --alpha 10 `
  --output outputs/cubism-smoke-test.yaml
```

## 環境診断を実行する

Cubism Editorを起動し、External Application IntegrationのAllowを確認した後で実行します。

```powershell
python -m tools.cubism_environment_probe `
  --control-tree-output outputs/cubism-control-tree.json `
  --screenshot outputs/cubism-diagnostics.png `
  --output outputs/cubism-environment-report.yaml `
  --execute
```

診断は実行ファイル名、window title、process IDを組み合わせてEditorを特定します。Viewer、Updater、タイトルだけが一致する別プロセスは拒否します。ダイアログとcontrol treeも特定済みprocess IDに限定されます。

External API接続先はloopback（`localhost`、`127.0.0.0/8`、`::1`）だけを許可します。保存済みtokenを任意ホストへ送信する指定は拒否されます。

control treeには、各controlの`name`、`automation_id`、`control_type`、`class_name`、`enabled`、`visible`、`process_id`、`parent_path`、`supported_patterns`を記録します。Edit controlの内容、credential/token/secret形式の値、ユーザーのローカルファイルパスは保存前に除去します。

profileは`tools/cubism_ui_profiles/`の日本語・英語定義から、window patternと実際に露出したlanguage markerを使って選択します。候補が0件または同点の場合は`unsupported_language`として停止し、候補と理由を診断reportへ残します。

現在のprofileはCubism 5用です。window titleと実行ファイル名から版を照合できない場合や、Cubism 4/6など非対応版では`unsupported_version`として停止します。

## smoke testを実行する

承認済みPSDだけを入力にし、環境診断reportを確認してから実行します。

```powershell
python -m tools.cubism_smoke_test generated/model_import.psd `
  --preset Standard `
  --alpha 10 `
  --output outputs/cubism-smoke-test.yaml `
  --execute
```

stageは次の順で進みます。

```text
preflight -> api_connection -> initial_snapshot -> psd_import
          -> import_verification -> auto_mesh -> visual_capture
          -> undo -> final_snapshot
```

前段が失敗した場合、後段は`blocked`になります。実行していないstageを`completed`にはしません。APIによるimport差分検証や保存可能状態の検証ができない場合は`waiting_for_user`となり、成功扱いしません。

各global shortcut送信直前に、foreground windowのprocess IDと実行ファイル名を再確認します。途中で別アプリへfocusが移った場合は`shortcut_failed`で停止します。

PSD importは、import前後の`ModelingDocuments`、新規`DocumentUID`、current `ModelUID`、current edit modeを比較します。自動メッシュは、profileで定義されたComboBox、alpha Edit、confirm ButtonをUIAで一意に特定します。ComboBoxが0件・複数件、alpha Editが0件、未知labelの場合は停止します。

Undoは自動メッシュ適用が記録された場合だけ実行します。適用後とUndo後のscreenshotを保存します。見た目の品質は自動PASSにせず、reportの`visual_review_required: true`を維持します。

自動メッシュのconfirm後にdialog-close待ちやscreenshotが失敗した場合も、適用済みmutationが記録されているときだけUndo recoveryを1回試行します。confirm前の失敗ではUndoしません。

## reportと失敗分類

- 環境report schema: `schemas/cubism_environment_report.schema.yaml`
- smoke test report schema: `schemas/cubism_smoke_test_report.schema.yaml`
- sanitized control tree: `outputs/cubism-control-tree.json`

主要な失敗分類は次のとおりです。

```text
cubism_not_found
wrong_process
window_not_found
unsupported_version
unsupported_language
external_api_unreachable
external_api_not_approved
wrong_edit_mode
dialog_not_found
control_not_exposed_by_uia
ambiguous_control
shortcut_failed
import_verification_failed
auto_mesh_verification_failed
undo_failed
save_as_dialog_opened
timeout
```

失敗時は可能な範囲で`cubism-smoke-failure.png`または指定された診断screenshotを保存します。出力は`--base-dir`内だけに制限されます。

screenshotは検証済みCubismウィンドウだけを撮影し、デスクトップ全体は撮影しません。smoke reportはaction名・件数・mutationとDocumentUID / ModelUID / edit modeだけを記録し、絶対PSDパスや生のdocument metadataは保存しません。

## 実機未検証事項

通常pytestはRecordingBackend、fake UIA tree、fake API clientだけを使用し、Cubism Editorを起動しません。次はWindows実機での目視確認が必要です。

- インストールされているCubism版での実行ファイル名・UIA label・control type
- External APIの承認、DocumentUID / ModelUID / edit mode応答
- PSD importダイアログと新規document差分
- 自動メッシュdialogのpreset、alpha、confirm control
- 自動メッシュ適用後とUndo後の見た目
- Save controlの有効状態
- Cubism projectとVTube Studioでの最終表示

これらを確認するまで、モデルやCubism操作フローを「完成」とは扱いません。
