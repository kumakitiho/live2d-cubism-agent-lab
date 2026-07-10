# Inpainting policy

## Mandatory provenance

source画像で見えない画素を補完した部品は、すべて次を満たす。

```yaml
inferred: true
review_required: true
```

`generation_method` は実際に使った `extract_and_edge_repair`、`transparency_fill`、`inpaint`、`redraw` のいずれかを記録する。sourceで見えない画素を含む限り、局所fillであってもinferred/review requiredを解除しない。

prompt ID、mask path、使用tool、生成日時またはrun IDをmanifestへ追記できる形にする。

## Review

- 左右対称を盲信しない。
- 顔輪郭、口内、関節、衣服の重なり、髪の根元を優先確認する。
- sourceと矛盾する輪郭、塗り、光源、線幅をrejectする。
- 目視未確認のinferred素材を `approved` にしない。
- 採用しない候補をimport PSDへ入れない。

## Masks

maskはsourceと同じcanvas size・originを保つ。対象領域、保護領域、補完余白を分離できる形式にし、最終import PSDへ生maskを残さない。
