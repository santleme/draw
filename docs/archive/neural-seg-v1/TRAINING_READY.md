# Training operational status

> **Code prepared for review; full training and model quality not yet
> validated.** This PR delivers tools and an execution contract, not a
> checkpoint or performance metrics.

## Prerequisites

1. Python 3.11 or 3.12 with the packages pinned in `training/requirements.txt`.
2. Synthetic dataset in `train/`, `val/` and `test/`, each split with image
   and mask pairs. The test split is kept out of checkpoint selection.
3. If using real data, pass the candidate mask folder with
   `--real-train-dir`; the directory contains only pairs intended for training.
4. `nvidia/mit-b0` weights accessible via cache or internet on first load.
5. For full training, a compatible GPU; the notebook prints device and
   versions before starting.

## Commands

Structure smoke test, 255 masks, loss and output contract:

```bash
python training/train.py --config training/config.json --smoke
```

Full training with separate synthetic evaluation:

```bash
python training/train.py \
  --config training/config.json \
  --data-dir dataset/synthetic \
  --real-train-dir captures/triage/pseudo_training_candidate \
  --out-dir out
```

The real directory is optional. Without it, training uses only synthetic data.

## Artifacts and reading the results

The pipeline writes `metrics.json`, `training_config.json`, `split_manifest.json`,
`classes.json`, checkpoints and, when exportable, `draw_segmentation_v1.onnx`.
`metrics.json` contains validation metrics at the top level and
test-split metrics in the `test` field. The ONNX has explicit validation status; `skipped`
or `failed` cannot be described as validated.

Before claiming performance on real data, review the masks pixel by pixel,
run the tests and the preflight of all pairs, train and reserve a new
real session as holdout. See `REAL_DATA_STATUS.md` for the status of the
current capture tranche.
