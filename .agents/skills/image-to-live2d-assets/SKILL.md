---
name: image-to-live2d-assets
description: validator済みcharacter_spec.yamlと1枚のsource画像からLive2D/Cubism用素材パックを設計するときに使用する。単一ソースのasset generation queue、隠れ部分補完、mask/inpainting、prompt、queueから派生するmanifest/layer map、PSD build plan、Cubismから戻るasset_feedbackを扱う。character spec収集やCubism操作には使用しない。
---

# Image to Live2D Assets

確定済みcharacter specを、追跡可能なLive2D素材生成計画へ変換する。`asset_generation_queue.yaml` を状態の単一ソースとし、manifestとlayer mapを直接編集しない。

## 必須入力

- validator済み `character_spec.yaml`
- character specが参照するsource画像
- 任意のvalidator済み `asset_feedback.yaml`

`model_scope`、`motion_level`、表情、物理演算対象、runtimeをこのSkillで聞き直さずcharacter specから読む。

## ワークフロー

1. `references/part_taxonomy.md` を読み、scopeとmotion levelに対応する部品を選ぶ。
2. `python -m tools.material_planner <source> --model-scope ... --motion-level ...` をdry-runする。
3. `assets/asset_generation_queue.template.yaml` から目、口、髪、体、隠れ補完のcanonical assetと並列jobを作る。
4. feedback入力があれば `target_layer_id` に対応するjobの `feedback_refs` へIDを追加し、requested actionとevidenceを反映してそのjobを再開する。
5. `python -m tools.asset_generation_queue_validator <queue>` で構造を検証する。通常は `strict`、未完成素材を扱う開発中だけ `validation_mode: dev` を使う。
6. 各jobで抽出、mask、inpainting、再描画を行い、隠れ部分を `inferred: true`、`review_required: true` とする。
7. `python -m tools.asset_queue_builder <queue>` をdry-runし、`--execute` でqueueから `asset_manifest.yaml` と `layer_map.yaml` を再生成する。
8. `python -m tools.asset_manifest_validator <manifest> --base-dir .` を実行する。
9. `python -m tools.psd_asset_builder <manifest>` でPSD build planを確認する。
10. 実backendがなければ `model_import.psd` を生成済みと主張しない。
11. handoff直前に `python -m tools.asset_generation_queue_validator <queue> --base-dir . --manifest <manifest> --require-merge-ready` でqueueとmanifestのpath・project・sourceを結合検証し、続けて `python -m tools.asset_manifest_validator <manifest> --base-dir . --require-handoff-ready` を実行する。両方が終了コード0のときだけ `$live2d-cubism-workflow` へ引き継ぐ。

## 所有する成果物

- `asset_generation_plan.yaml`
- `asset_generation_queue.yaml`（唯一の編集対象）
- `asset_manifest.yaml`（queueからの派生物）
- `layer_map.yaml`（queueからの派生物）
- `image_prompt.md`
- `psd_separation_instructions.md`
- 部品mask・inpainting計画
- 将来の `generated/parts/*.png` と `model_import.psd`

素材状態、layer metadata、出力先、import制約はqueue templateにだけ保持する。

## Feedback処理

- feedbackのlayer IDをlayer mapで検証する。
- issueを対象jobへだけ割り当てる。
- requested actionをprompt/mask/segmentation計画へ反映する。
- 再生成結果を再レビューし、queueのasset readiness、job status、validationを更新する。
- すべてのrequired jobを再mergeし、queueからmanifest/layer map/PSD build planを再構築する。
- 同じlayer/issueで3回失敗したら4回目を止める。

## 境界

- character specを作成・変更しない。仕様変更が必要なら `$character-spec-generator` へ戻す。
- Cubism Editor、UI macro、External API、リギングを操作しない。
- source画像だけから見えない形状を事実として扱わない。
- 未レビューのinferred素材をimport可能と判定しない。
- 外部画像生成、segmentation、inpainting、Photoshop Pluginは後付け可能なadapterとして扱う。

## 将来分割予定

MVPでは1 Skillにまとめるが、責務が成熟したら次へ分割する。

- planner: character specからtaxonomy、job、prompt、mask計画を作る。
- generator: segmentation、inpainting、redraw adapterを実行し、queueのasset状態を更新する。
- manifest builder: queueからmanifest、layer map、PSD build planを決定的に生成する。

分割後もqueueを唯一の状態ソースとし、派生物からqueueへ逆同期しない。

## Handoff contract

次を満たしてから `$live2d-cubism-workflow` を使う。

- queueの全required jobとtarget assetがapprovedでmerge validationがtrueである。
- source画像の利用権状態がconfirmedである。
- manifest validatorがhandoff readyを返す。
- 実体のあるsource、全import PNG、`model_import.psd`、`layer_map.yaml` がある。
- unresolvedなasset feedbackが解消または明示的にrejectedである。
