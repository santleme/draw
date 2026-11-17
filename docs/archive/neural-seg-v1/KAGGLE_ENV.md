# Training environment and compatibility

This document describes the current contract of the pipeline. The full training
has not yet been run in this proposal; local dependencies and checks do not
replace a run with GPU and real data.

## Reproducible environment

Use Python 3.11 or 3.12 and the pinned direct versions in
`training/requirements.txt`. The notebook does not install packages: it fails in preflight
if a mandatory dependency is missing or has a different version. For `torch`,
install the version `2.11.0` wheel compatible with the machine's CUDA runtime.

| Package | Version | Use |
|---|---:|---|
| torch | 2.11.0 | Model, training, AMP and export |
| transformers | 4.57.1 | Pre-trained SegFormer-B0 |
| albumentations | 2.0.8 | Paired image and mask augmentations |
| numpy | 1.26.4 | Data and metrics |
| opencv-contrib-python | 4.11.0.86 | Images, masks and preprocessing |
| onnx | 1.23.0 | Export |
| onnxruntime | 1.31.1 | Validation of the exported artifact |

The notebook receives `training/` and the datasets as Kaggle inputs; it does not clone the
repository or download dependencies. The `nvidia/mit-b0` backbone must be in
cache or accessible on the first `from_pretrained` call.

## What the code executes

- Geometric augmentation applies the same parameters to image and mask;
  masks use nearest-neighbor. Photometry changes only the image.
- Training uses bf16 on CUDA when available, scaled fp16 as fallback and
  fp32 on CPU. `gradient_accumulation_steps` controls optimizer steps.
- The scheduler does linear warmup followed by cosine decay, in optimizer
  steps.
- The `val` split drives early stopping and checkpoint selection. The `test`
  split is evaluated once after loading the best checkpoint and does not participate in
  selection.
- The export wrapper resizes SegFormer logits to the input
  resolution. For 512×512 input, the contract is `image` `[N,3,512,512]` and
  `logits` `[N,4,512,512]`.
- ONNX validation compares shape, maximum absolute logit difference and minimum
  argmax IoU. `ok` passed the limits, `degraded` failed, `skipped` could not
  compare. `skipped` never means the ONNX is validated.
- Training metrics use `tools/seg_metrics.py`, including 255 pixels and
  ink tolerance aggregated over all evaluated images.

## Known limits

- `last.pt` and `best.pt` save model weights, but the pipeline does not yet resume
  optimizer, scheduler or epoch state after interruption.
- The current real dataset has 719 frames from two sessions, used only in training.
  The masks are pseudo-labels with thumbnail review; thin edges have not yet
  been reviewed pixel by pixel. There is no independent real holdout.
- No full training was run and no model produced in this proposal.
  Training results with the pseudo-labels must be marked as
  exploratory; use a new real session for independent evaluation.
- GPU, Kaggle quota and weight cache must be confirmed in the execution
  runtime. A CPU smoke test does not prove GPU use or model quality.
