# Session log

## 2026-07-10 — Initial environment

### 判断

- Live2D完成モデルの全自動生成ではなく、制作成果物とCubism操作を積み上げる学習・制作環境とする。
- Skill、deterministic validators、名前付きUI macro、External API、Bridgeの順で構成する。
- MCP化と複数Agent化は、実機でCLI入出力が安定した後に行う。
- UIの定型操作は自動化し、`manual_checkpoint` は見た目判断に限定する。
- 任意座標クリックと自由ドラッグは公開インターフェースにしない。

### Harness

- dry-runを既定にし、実機は `--execute` を必須にする。
- action planをallowlist検証する。
- API payloadとUI操作列を実機なしでテストする。
- 想定外状態では停止し、スクリーンショットとMarkdown reportを残す。

### 環境観察

- Python 3.13.5、Codex CLI 0.101.0を確認。
- 実装時点でCubismプロセスは起動しておらず、一般的なLive2Dインストールディレクトリも確認できなかった。
- 実機Cubism UI検証は未実施。dry-run・モック・静的検証を先に行う。

### 公式仕様メモ

- Cubism External APIはWebSocket/JSON、既定port 22033。
- 接続ごとにRegisterPluginし、初回はEditor側Allowが必要。
- PSDはFile > Openでも取り込み可能で、Model Settingsから新規モデルを選ぶ。
- 自動メッシュ生成はCtrl+A、Ctrl+Shift+Aで開ける。alpha調整とUndo回復を考慮する。

### Subagent review対応

- title regexだけでなくCubism実行ファイル名とprocess IDを照合するよう変更。
- auto-mesh適用が記録された後の失敗だけUndo recoveryを許可。
- import前のGetDocumentsと、import後のDocuments/current ModelUID snapshotを比較するverification stepを追加。
- action planにcommand別引数schemaと `outputs/*.png` 出力制約を追加。
- save時のSave Asダイアログ検出、auto-mesh dialog postcondition、visual reviewフラグを追加。
- 再レビューを受け、対象process名を `CubismEditor*.exe` に限定。
- Bridgeの複数API stepで同一WebSocket sessionを維持し、新規DocumentUIDのViewsにcurrent ModelUIDが属することを確認。
- import後auto-meshの前にverification sequenceを必須化する横断validatorを追加。

### Final verification / review

- Skill quick validation: pass
- Ruff: pass
- Mypy: pass（10 source files）
- Pytest: 31 passed
- pip check: pass
- character spec / layer map / action plan validators: pass
- CLI / installed entry point dry-run: pass
- 最終サブエージェント再レビュー: 新規所見なし、静的レビュー上のブロッカーなし

### 残リスクと次のハーネス候補

- Cubism未起動・標準インストール先未検出のため、実機UIA locatorは未検証。
- OSレベルで別ウィンドウが割り込む競合と、大容量PSDの取り込み完了タイミングは実機確認が必要。
- 次は、Cubismの版と言語を記録するUIA control tree診断、既知PSD fixtureでのimport smoke test、失敗スクリーンショット比較を追加する。

## 2026-07-10 — Image to Live2D assets MVP

### 判断

- 1枚絵からの素材生成工程を、Cubism操作Skillとは別の `image-to-live2d-assets` に分離した。
- MVPは画像生成ではなく、scope/level別planning、manifest、validation、PSD build planを対象にした。
- 隠れ部分は `inferred: true`、人間確認対象は `review_required: true` として追跡する。
- 実PSD backend未接続時は `status: plan_only` とし、空PSDや完成主張を禁止した。

### Harness

- source pathと拡張子をplannerで検証する。
- manifestの構造的validityとCubism handoff readinessを分ける。
- layer ID/name重複、inferred review、import制約、権利状態をvalidatorで確認する。
- scopeとmotion levelのtaxonomyをfixture不要の単体テストで固定する。

### 残リスクと次の候補

- source画像の内容解析、segmentation、inpainting、生成prompt実行、PSDバイナリ生成は未実装。
- 次は同一canvasのmask/PNG fixture検証、画像tool adapter contract、実PSDの再読込検証を追加する。

### Subagent review対応

- manifestフラグだけではhandoffできないようにし、source、import PNG、PSD、layer mapの実在・非空・基本signatureをgateへ追加した。
- layer mapのproject、canvas、layer ID/nameをmanifestと照合するようにした。
- `required: true` のimport除外をhandoff blockerにした。
- redraw素材へ `review_required: true` を必須化した。
- PSD backendは `ready_to_build: true` とmissing source解消前に呼べないようにした。
- sample manifestとlayer mapのproject、canvas、部品集合を揃え、未生成・未レビュー状態だけがhandoff blockerとして残る形にした。
- 再レビューを受け、layer mapの行数、重複、欠落、`(layer_id, name)`対応を厳密比較するようにした。
- builderもvalidatorと同じsource/PNG signature関数を使い、偽画像で `ready_to_build` にならないようにした。
- builderの相対PSD出力を `base_dir` 基準で絶対化し、将来backendが別cwdへ書かないようにした。
- 最終再レビューを受け、backend戻り値のpath一致、非空、`8BPS` signatureを成功条件にした。
