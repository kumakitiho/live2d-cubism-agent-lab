# Action Plan Schema

```yaml
schema_version: 1
project: sample
steps:
  - id: validate_layer_map
    mode: file
    command: file.validate_layer_map
    args:
      path: generated/layer_map.yaml
```

## mode

- `file`: layer mapまたはimport結果を決定的に検証する。
- `ui_macro`: 名前付きCubism UIマクロを実行する。
- `api`: Cubism External APIを呼ぶ。
- `manual_checkpoint`: `instruction` を必須とし、見た目判断とfeedback記録を待つ。

各stepの`id`は一意にする。`args`はmappingにする。`ui_macro`と`api`はallowlist外のcommandを使えない。`x`、`y`、`coordinates`などの座標引数はUI stepで禁止する。
スクリーンショットとfailure screenshotは `outputs/` 配下の `.png` に限定する。
同じplanでPSD import後にauto-meshを行う場合、import前`get_documents`、import後`get_document_snapshot`、`verify_imported_document`をこの順で挟む。

character specの収集・検証はCubism action planへ含めず、上流の `character-spec-generator` で完了させる。
