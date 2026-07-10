# Field inference policy

## 画像から埋める項目

- 画角と写っている範囲
- 髪色、髪型、目の色、衣装、見えるアクセサリ
- source上の表情
- 前髪、横髪、後ろ髪、リボン、イヤリングなどの物理演算候補
- sourceに見えている左右差と隠れ領域候補

観測結果は断定しすぎず、画像から読めない内側、背面、素材、設定を `assumptions` へ入れる。

## 人間へ確認する項目

- 最終scopeが画像の画角と同じか、描き足して拡張するか
- `minimal` / `standard` / `expressive` のmotion level
- VTube Studioなどのtarget runtime
- sourceにない追加表情と配信用途
- 物理演算を有効にしたい部品と強さの好み
- source画像をモデル化・加工する権利状態

## 質問方針

- 1回に重要な未解決項目だけをまとめる。
- 既に明示された値を質問しない。
- 画像観測で十分な項目は質問せずdraftへ入れる。
- 回答がなくても仮定できる項目は `assumptions` へ置き、handoffを変える重要項目だけを `open_questions` に残す。
