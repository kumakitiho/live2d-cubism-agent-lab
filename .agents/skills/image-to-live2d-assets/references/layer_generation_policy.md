# Layer generation policy

## Generation methods

- `extract`: sourceで見えている画素を切り出す。
- `mask_extract`: segmentation maskで見えている領域を切り出す。
- `inpaint`: 隠れ領域を補完する。必ずinferredかつreview requiredにする。
- `redraw`: 人間または外部描画ツールが部品を描き直す。レビューを要求する。

## Asset loop

1. 1回に1つのpart familyだけを扱う。
2. sourceと同じキャンバス・同じ原点で透過PNGを生成する。
3. mask、prompt、生成tool、入力、出力をcanonical queueで追跡する。
4. 結果をqueue上で `approved`、`generated`、`rejected` のいずれかへ更新し、manifest/layer mapを再生成する。
5. 同じ失敗を3回繰り返したら手段を変え、無制限に再生成しない。

## Prompt contract

promptは部品の役割、sourceとの一致条件、透過背景、同一キャンバス、禁止変更を含める。キャラクター全体を毎回再生成せず、対象part familyだけを生成する。

## Import separation

制作中のsource、mask、guide、候補差分はmaterial workspaceへ保持する。Cubism import用PSDには採用済みdrawableだけを1部品1レイヤーで入れる。
