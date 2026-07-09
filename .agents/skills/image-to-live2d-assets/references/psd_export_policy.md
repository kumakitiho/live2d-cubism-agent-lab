# PSD export policy

## Import PSD contract

- RGB、8 bit/channel、sRGBで出力する。
- 1 drawable partを1レイヤーへ統合する。
- 全レイヤー名を一意にする。
- 全part PNGを同じcanvas size・originで配置する。
- source参照、guide、mask、reject候補を除外する。
- inferred素材は承認済みだけを含める。
- `layer_map.yaml` とレイヤー順・名前を一致させる。

## Builder status

現MVPの `tools.psd_asset_builder` はmanifestから順序付きbuild planを作るstubであり、PSDバイナリを生成しない。build planの `status: plan_only` を保持し、実backendを接続するまで `model_import.psd` を生成済みと記録しない。将来backendを呼ぶ場合も、`ready_to_build: true`、import PNG実在、全part承認、全import制約、権利確認を先に要求する。

将来のbackendは `generated/parts/*.png` を読み、manifestの `order` と `layer_name` を使ってPSDを作る。出力後にレイヤー数、名前、canvas、color modeを再検証する。
