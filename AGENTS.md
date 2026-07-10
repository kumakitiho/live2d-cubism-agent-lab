# Project guidance

- CubismのGUI操作は `tools.cubism_ui` の名前付き高レベルマクロ経由に限定する。
- 任意座標クリック、自由なドラッグ、汎用 `click(x, y)` APIを追加しない。
- 実機操作の前に必ずdry-runを実行し、`--execute` はユーザーが実機操作を求めた場合だけ使う。
- 想定外のウィンドウやダイアログでは停止し、ログとスクリーンショットを残す。
- CubismプロジェクトとVTube Studioで目視確認するまで「完成」と表現しない。
- `rtk` などの任意wrapperは存在確認後だけ使い、未導入環境では標準のshell commandへフォールバックする。
