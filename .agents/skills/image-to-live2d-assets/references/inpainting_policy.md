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

maskはsourceと同じcanvas size・originを保つ。target maskはAA grayscale値を保持し、protect、edge extension、inpaint maskは用途ごとのalpha thresholdでbinary判定する。protectは変更禁止のsource領域、edge extensionはsource-preserving edge repairとoverlap、inpaintは生成inpaintingが変更可能な隠れ領域に限定する。最終import PSDへ生maskを残さない。
