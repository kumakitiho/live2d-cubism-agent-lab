---
name: image-to-live2d-assets
description: 1枚のキャラクター画像からLive2D/Cubism用の素材パックを設計するときに使用する。パーツ分類、隠れ部分補完、マスク・inpainting計画、生成prompt、asset_manifest、layer_map、PSD build plan、権利・目視レビュー状態を扱う。Cubism Editorの操作、リギング、完成モデル生成には使用せず、準備完了後はlive2d-cubism-workflowへhandoffする。
---

# Image to Live2D Assets

1枚のsource画像を、追跡可能なLive2D素材生成計画へ変換する。画像生成やPSD生成が未接続でも、計画、manifest、prompt、validationを先に完成させる。

## 入力を確認する

- source画像のパスと利用権確認状態を記録する。
- `model_scope` を `bust_up`、`half_body`、`full_body` から選ぶ。
- `motion_level` を `minimal`、`standard`、`expressive` から選ぶ。
- 目標キャンバス、必要表情、揺れ物、アクセサリを記録する。
- 権利状態が不明なら生成・handoffを承認済みとして扱わない。

## ワークフロー

1. `references/part_taxonomy.md` を読み、scopeとmotion levelに対応する部品を選ぶ。
2. `python -m tools.material_planner <source> --model-scope ... --motion-level ...` をdry-runし、`asset_generation_plan.yaml` を確認する。
3. `references/layer_generation_policy.md` を読み、各部品へ抽出、マスク、inpainting、再描画のいずれかを割り当てる。
4. 隠れ部分を生成する部品へ必ず `inferred: true` と `review_required: true` を設定する。詳細は `references/inpainting_policy.md` を読む。
5. `assets/asset_manifest.template.yaml` を基に `asset_manifest.yaml` を作る。
6. workspace rootから `python -m tools.asset_manifest_validator <manifest> --base-dir .` を実行し、構造エラーと実ファイルを含むhandoff未達条件を分けて直す。
7. `assets/layer_map.template.yaml` を基に `layer_map.yaml` を作る。
8. `references/psd_export_policy.md` を読み、`python -m tools.psd_asset_builder <manifest>` でPSD build planを確認する。
9. 実際の画像生成・mask合成・PSD backendが接続されていなければ、`model_import.psd` を生成済みと主張しない。
10. handoff条件を満たしたときだけ `live2d-cubism-workflow` へ引き継ぐ。

## 必須出力

- `asset_generation_plan.yaml`
- `asset_manifest.yaml`
- `layer_map.yaml`
- 部品ごとのmask・inpainting・prompt計画
- 将来の `generated/parts/*.png` と `model_import.psd` の出力計画
- validation結果と未レビュー項目

`assets/asset_generation_plan.template.yaml`、`assets/asset_manifest.template.yaml`、`assets/layer_map.template.yaml` を出力の起点に使う。

## 境界

- Cubism Editor、UIマクロ、External APIを操作しない。
- リギング、メッシュ、デフォーマ、パラメータ作成を行わない。
- source画像だけから見えない形状を事実として扱わない。
- 未レビューのinferred素材をCubism import可能と判定しない。
- 実PSD backendがない状態で空ファイルや偽の `model_import.psd` を作らない。
- 外部画像生成、segmentation、inpainting、Photoshop Pluginは後付け可能な生成手段として扱い、manifest契約を先に保つ。

## Handoff contract

次の条件をすべて満たしてから `$live2d-cubism-workflow` を使う。

- source画像の利用権状態が `confirmed` である。
- `asset_manifest.yaml` に構造エラーがない。
- source画像と全import PNGが実在し、空でなく、宣言した画像signatureを持つ。
- import対象の全レイヤー名とIDが一意である。
- `required: true` の全素材がimport対象に含まれる。
- `inferred: true` の全素材が `review_required: true` かつ `readiness: approved` である。
- `generation_method: redraw` の全素材が `review_required: true` である。
- 全import制約がtrueで、参照画像・ガイド・未解決maskを含まない。
- signatureを確認できる `model_import.psd` と、project/canvas/layer ID/nameがmanifestと一致する `layer_map.yaml` が存在する。

handoff時は次を渡す。

```yaml
handoff:
  target_skill: live2d-cubism-workflow
  model_import_psd: generated/model_import.psd
  layer_map: generated/layer_map.yaml
  asset_manifest: generated/asset_manifest.yaml
  validation:
    manifest_valid: true
    handoff_ready: true
    inferred_assets_reviewed: true
    rights_confirmed: true
```

条件未達ならhandoffせず、validatorの未達項目を次の作業として返す。
