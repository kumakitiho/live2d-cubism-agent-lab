---
name: character-spec-generator
description: 1枚のキャラクターsource画像とユーザーの希望からLive2D用character_spec.yamlを作るときに使用する。画像から外見、画角、見えるアクセサリ、表情、揺れ候補を推測し、model_scope、motion_level、expressions、physics_targets、target_runtimeなど人間の意図が必要な項目だけを確認する。素材生成やCubism操作には使用せず、仕様確定後はimage-to-live2d-assetsへhandoffする。
---

# Character Spec Generator

source画像から観測できる情報を先に埋め、ユーザーにしか決められない制作意図だけを質問して `character_spec.yaml` を完成させる。

## ワークフロー

1. source画像のパス、権利状態、ユーザーの希望を確認する。
2. `references/field_inference_policy.md` を読み、画像から観測できる外見、画角、衣装、アクセサリ、現在の表情、物理演算候補を記録する。
3. `assets/character_spec.template.yaml` を基に仕様draftを作る。
4. 推測値を `spec_provenance.image_inferred_fields`、明示希望を `user_confirmed_fields`、不確実な補完を `assumptions` へ分ける。
5. 画像から決められない項目だけを短く質問する。優先する項目は `model_scope`、`motion_level`、`target_runtime`、追加表情、必要な動き、権利確認である。
6. 既にユーザーが指定した項目を聞き直さない。画像で明らかな髪色や衣装を質問しない。
7. 回答を反映し、`spec_provenance.open_questions` を空にする。
8. `python -m scripts.validate_character_spec <path>` を実行する。
9. validator通過後だけ `$image-to-live2d-assets` へhandoffする。

## 必須出力

- `character_spec.yaml`
- 画像から観測した項目
- ユーザー確認済み項目
- 仮定と未解決質問
- 次のhandoff情報

## 境界

- segmentation、mask、inpainting、部品PNG、PSD、layer mapを生成しない。
- Cubism Editor、UI macro、External API、リギングを操作しない。
- source画像から見えない設定を事実として記録しない。
- `motion_level` や `target_runtime` を外見だけで決めない。
- 権利状態が不明なまま素材生成を承認しない。

## Handoff contract

次を満たしてから `$image-to-live2d-assets` を使う。

- `character_spec.yaml` がvalidatorを通る。
- `spec_provenance.open_questions` が空である。
- `model_scope`、`motion_level`、`target_runtime` がユーザー意図と一致する。
- source画像のパスと権利状態が記録されている。

```yaml
handoff:
  target_skill: image-to-live2d-assets
  character_spec: generated/character_spec.yaml
  source_image: assets/source/character.png
  validation:
    character_spec_valid: true
    open_questions_resolved: true
```
