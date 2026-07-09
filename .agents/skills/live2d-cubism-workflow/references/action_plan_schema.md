# Action Plan Schema

```yaml
schema_version: 1
project: sample
steps:
  - id: validate_spec
    mode: file
    command: file.validate_character_spec
    args:
      path: examples/character_spec.sample.yaml
```

## mode

- `file`: 構造ファイルを検証する。
- `ui_macro`: 名前付きCubism UIマクロを実行する。
- `api`: Cubism External APIを呼ぶ。
- `manual_checkpoint`: `instruction` を必須とし、見た目判断を待つ。

各stepの`id`は一意にする。`args`はmappingにする。`ui_macro`と`api`はallowlist外のcommandを使えない。`x`、`y`、`coordinates`などの座標引数はUI stepで禁止する。
各commandは必須・許可引数を持ち、未知の引数を受け付けない。action planが指定するスクリーンショットとfailure screenshotは `outputs/` 配下の `.png` に限定する。
同じplanでPSD import後にauto-meshを行う場合、import前`get_documents`、import後`get_document_snapshot`、`verify_imported_document`をこの順で挟む。Bridgeは同一API session内で新規DocumentUIDとcurrent ModelUIDを照合する。
