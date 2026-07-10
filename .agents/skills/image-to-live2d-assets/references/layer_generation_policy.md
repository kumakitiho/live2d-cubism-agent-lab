# Layer generation policy

## Generation methods

- `extract`: soft target maskをalphaへ乗算し、sourceで見えている画素をAA値ごと切り出す。
- `extract_and_edge_repair`: target edge近傍だけを処理し、より不透明な内側画素からRGB fringeを補修する。透明穴は埋めない。
- `transparency_fill`: binary inpaint mask内の完全透明穴だけを近傍色で補完する。AA edgeの既存画素は変更しない。
- `inpaint`: 隠れ領域を補完する。必ずinferredかつreview requiredにする。
- `redraw`: 人間または外部描画ツールが部品を描き直す。レビューを要求する。

source画素を保持する優先順位は `extract > extract_and_edge_repair > transparency_fill > inpaint > redraw` とする。品質gateで失敗したpartだけを次の方式へ進め、合格partをまとめて再生成しない。

protect領域の差分は通常の前進transitionで修復せず、sourceから `extract` をやり直す。生成inpaintingやredrawでpreserve違反を隠さない。

non-zero `overlap_margin_px` はtarget maskの自動膨張を意味しない。desired coverageはbinary target maskと明示的な `edge_extension_mask` の和集合とする。`inpaint_mask` は生成inpaintingが変更できる隠れ領域であり、overlap coverageへ流用しない。extensionが空なら初回extractをoverlap不足だけでfailにしない。plannerはnon-zero overlapの可視partを `extract_and_edge_repair` から開始する。

`draw_order` は背面から前面への順序で、小さい値を先に、大きい値を上へ合成する。hidden fillは対応するvisible baseより小さく、back hairはface・eyes・front hairより小さくする。

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
