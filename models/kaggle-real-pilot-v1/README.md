# Kaggle real-data pilot model

SegFormer-B0, four-class ONNX exported by Kaggle kernel
`guzzbr/draw-segformer-v1-4class` on 2026-10-30.

## Labels

| ID | Class |
|---:|---|
| 0 | background |
| 1 | visible paper |
| 2 | visible hand |
| 3 | user-added visible ink |

The model was trained with 2,400 synthetic images and 719 real pseudo-labeled
frames. The real masks were reviewed as contact-sheet proposals and had clear
mistakes corrected, but were not verified pixel by pixel. Treat this as an
experimental pilot, not a validated production model.

## Reported validation

Kaggle reported mean IoU `0.913673` on the synthetic validation split. Per-class
IoU was background `0.9981`, paper `0.9858`, hand `0.9440`, ink `0.7269`.
These are synthetic validation results, not a measurement on an independent
real-camera holdout. ONNX-vs-PyTorch parity was not run in Kaggle
(`onnx_validation: null`).

## Maintainer review on real camera frames

The repository maintainer also ran this pilot through the project's real-frame
harness during review. On the 49-frame set (43 positive frames, 6 negative
frames, with ground-truth page polygons for 11 frames), the pilot detected
`42/43` positive frames and produced false positives on `6/6` negative frames.
Its page-quadrilateral IoU averaged `0.478` (minimum `0.000`). The production
v3 detector, run through this PR's `draw/neural.py`, scored `43/43` positives,
`0/6` negative-frame false positives, and page-quadrilateral IoU `0.957`
(minimum `0.845`) on the same harness, as reported by the maintainer.

On all three frames containing a white box, the pilot classified the box as
paper, with a predicted mask covering about `19%` of the frame and presence
score `+1.00`; the production v3 detector did not declare a page on those
frames. This is a serious regression for page detection. The pilot remains an
experimental, opt-in artifact and is not a replacement candidate for v3.

These figures are the maintainer's review-harness results on the author's
recorded camera set, not an independent holdout or evidence of generalization
to other cameras, desks, or lighting. The harness has only 6 negative frames
and 11 annotated page polygons; treat the numbers as a regression warning for
this set, not as a broad performance estimate.

## Artifact integrity

`model.onnx` SHA-256:

```text
2ca5ea38210f8f1c7b4a6e981de89c0c434437341a18b7317c6733daba11c908
```

The ONNX input is `image`, `[N,3,512,512]`, ImageNet-normalized float32. It has
one output, `logits`, with four class channels at one-quarter input resolution.
The app adapter is responsible for deriving the page mask from paper and ink
classes while excluding hand pixels.

## App activation

Merging this pilot does not make it the app's default model and does not replace
the current model selection. To test this artifact explicitly from the
repository root, run:

```bash
python main.py --neural models/kaggle-real-pilot-v1/model.onnx
```
