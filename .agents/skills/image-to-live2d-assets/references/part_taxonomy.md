# Part taxonomy

## Scope

`bust_up` は胸上、`half_body` は腰・腕・手、`full_body` は脚・足・靴までを対象にする。広いscopeは狭いscopeの部品を包含する。

| scope | 追加する主な部品 |
|---|---|
| `bust_up` | 顔、髪、首、耳、上半身、上衣 |
| `half_body` | 腰、左右の腕・手、下衣 |
| `full_body` | 腰下、左右の脚・足・靴 |

## Motion level

| level | 分割方針 |
|---|---|
| `minimal` | 目・口・髪を大きな単位で分ける |
| `standard` | 白目、虹彩、瞳孔、ハイライト、上下まぶた、口内、髪束を分ける |
| `expressive` | 閉じ目線、目影、追加口形状、前髪先端、横髪先端、ahogeを加える |

同一レベル内で粗い目レイヤーと詳細な目レイヤーを重複させない。上位levelでは下位の粗い部品を詳細部品へ置き換える。

## Naming

- `layer_id` と `layer_name` は一意にする。
- 左右はキャラクター本人基準で `_L` / `_R`、中央は必要に応じて `_C` を使う。
- 一時名、コピー名、連番だけの名前を使わない。
- source参照、mask、guideはimportレイヤーと別namespaceにする。

## Hidden regions

sourceで隠れている顔、口内、髪の根元、衣服の重なり、関節裏は観測事実ではない。補完するときは `generation_method: inpaint`、`inferred: true`、`review_required: true` とする。
