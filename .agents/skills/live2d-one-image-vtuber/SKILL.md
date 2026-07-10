---
name: live2d-one-image-vtuber
description: 1枚のキャラクター画像からLive2D/VTuber化する全体工程をオーケストレーションするときに使用する。character-spec-generator、image-to-live2d-assets、live2d-cubism-workflowを順にhandoffし、Cubism評価の問題をasset_feedback.yamlとして素材生成へ戻す。各工程の実処理やAgents SDK実装には使用しない。
---

# Live2D One Image VTuber

3つの専門Skillを順に進め、構造化成果物とvalidation gateで状態を受け渡す。各Skillの作業を自分で代行しない。

## オーケストレーション

1. `$character-spec-generator` を使い、source画像とユーザー希望から `character_spec.yaml` を確定する。
2. character spec validatorが通り、未解決質問が空になるまで次へ進まない。
3. `$image-to-live2d-assets` を使い、素材計画とcanonical queueを作り、queueからmanifest、layer map、import PSD build planを派生する。
4. queueのmerge gateとmanifest handoff gateが通るまでCubismへ進まない。
5. `$live2d-cubism-workflow` を使い、承認済みPSD、layer map、action planをdry-runしてから実行する。
6. `cubism_evaluation.yaml` にeye、mouth、mesh、textureの結果を記録する。
7. 素材由来のwarn/failがあればevaluationを `asset_feedback.yaml` へ変換し、validatorを通す。
8. feedbackを `$image-to-live2d-assets` へ渡し、対象layerを含むqueue jobだけを再開する。
9. merge gate、PSD build、Cubismの影響stepを再実行する。

詳細な成果物と遷移条件は `references/orchestration_contract.md` を読む。

## Feedback loop

```text
character-spec-generator
  → image-to-live2d-assets
    → live2d-cubism-workflow
      → asset_feedback.yaml
        → image-to-live2d-assets（対象jobを再実行）
          → live2d-cubism-workflow（影響範囲を再確認）
```

同じ `target_layer_id` と `issue_type` で3回失敗したら4回目を自動で続けず、人間またはレビュー用サブエージェントへ相談し、採用判断を記録する。

## 境界

- character spec、画像素材、PSD、Cubism操作を直接生成・実行しない。
- gateを省略して次工程へ進めない。
- feedbackを自然文だけで渡さない。
- Agents SDK、MCP server、永続state machineはまだ実装しない。
- Cubism/VTube Studioの目視確認前に完成と判定しない。

## 完了条件

- 各handoff成果物とvalidator結果が記録されている。
- openまたはblockingのasset feedbackがない。
- Cubism operation reportに未解決の素材起因エラーがない。
- 人間の目視QA項目と未自動化工程が明記されている。
