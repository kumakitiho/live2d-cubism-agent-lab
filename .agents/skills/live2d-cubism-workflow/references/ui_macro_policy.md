# UI Macro Policy

## 許可する操作

- Cubismウィンドウのフォーカス
- `Ctrl+O` によるPSD選択
- 既知のModel Settings選択肢
- `Ctrl+A` と `Ctrl+Shift+A` による自動メッシュ生成
- 既知のプリセット・alpha入力
- `Ctrl+S`、`Ctrl+Z`
- スクリーンショット

## 禁止する操作

- 公開 `click(x, y)` / `drag(x1, y1, x2, y2)`
- モデルが自由に決める絶対座標
- 頂点の自由ドラッグや中間点配置
- ダイアログ未検出時のEnter連打

UI Automationの表示名で対象を特定できない場合は停止する。実機操作には `--execute` が必要で、失敗時はfailure screenshotを残す。
Cubismのmain windowはタイトルと実行ファイル名の両方で特定し、以後のダイアログは同じprocess IDに限定する。Undo recoveryはauto-mesh適用完了が記録された場合だけ許可する。
