# Running training on Kaggle

This runbook describes running the pipeline in `training/train.py`. The code and
`training/config.json` are the source of the contract; this document does not assert that a
full training or a model has already been produced.

## Prepare the environment

Use Python 3.11 or 3.12 and the dependencies in `training/requirements.txt`, with the
PyTorch wheel compatible with the runtime's CUDA GPU. The notebook preflights
versions and fails if any direct dependency is missing or divergent.
The `nvidia/mit-b0` backbone must be in cache or accessible on the first
run.

Attach to the notebook:

- the synthetic dataset with `train/`, `val/`, `test/` and, if available,
  `split_manifest.json`;
- the private real pseudo-label package, when available. The two current real
  sessions enter training only via `--real-train-dir`.

Do not use these two sessions as holdout. The current real masks are automatic
proposals and still need pixel-by-pixel review; results with them are
exploratory. Reserve a new real session for independent evaluation.

## Run

Confirm the GPU appears in the runtime, that the dataset paths exist and
that `training/config.json` is available. The command below uses the
config's hyperparameters:

```bash
python training/train.py \
  --config training/config.json \
  --data-dir /kaggle/input/<dataset>/dataset/synthetic \
  --real-train-dir /kaggle/input/<real-data>/pseudo_training_candidate \
  --out-dir /kaggle/working/out
```

The `--real-train-dir` argument is optional. To check pairing, `255`
masks, loss and structure without training, run:

```bash
python training/train.py --config training/config.json --smoke
```

Training uses paired geometric augmentation, nearest-neighbor on masks,
linear warmup and cosine decay per optimizer update, gradient
accumulation per the config and bf16/fp16 on GPU or fp32 on CPU. Validation drives
early stopping and best-checkpoint selection. The test split is computed a
single time after selection; never use test to adjust weights or thresholds.

## Artifacts and criteria

The output directory contains checkpoints, `classes.json`, `metrics.json`,
`training_config.json`, `split_manifest.json` and, if export works,
`draw_segmentation_v1.onnx`. The ONNX contract is input `image` `[N,3,512,512]`
and logits `logits` `[N,4,512,512]`; the wrapper resizes the SegFormer output
to the input resolution.

`metrics.json` records per-class metrics, ignored pixels, ink with
spatial tolerance aggregated over all images, test-split metrics and
ONNX validation state. Validation compares shape, maximum logit
difference and minimum argmax IoU against PyTorch. `skipped`, `failed` or `degraded`
does not mean the ONNX was validated.

Do not publish real-quality claims without label review and evaluation
on an independent real holdout. The pipeline and local tests do not replace a
full GPU training run, checking the ONNX artifact, or visual
evaluation of the 15 scenarios defined by the project.
