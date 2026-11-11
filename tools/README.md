# tools/ — operator tools

Scripts that operate the Draw pipeline from outside the app: fetching
and validating models, building and labeling data, judging candidates
and replaying recorded behavior. The app never imports from here; most
tools import from `draw/` instead.

Paths in this folder are referenced by the README, by the CI path
filters and by the training launchers, and the two
`replay_reregister*.py` files are additionally hash-anchored (their
bytes are part of a paired-comparison contract). Add new tools here;
do not rename existing ones casually.

## Models

- `fetch_models.py` — downloads the official models into `models/`,
  verified against `models/MANIFEST.json`.
- `validate_onnx.py` — ONNX validator for the segmentation models.
- `export_segformer.py` — re-exports a SegFormer checkpoint without
  retraining and compares resolutions.
- `merge_checkpoints.py` — checkpoint fusion (WiSE-FT / model
  merging) across the model lineage.

## Judging and evaluation

- `judge.py` — the single executable entry point for judging Draw
  candidates.
- `judge_candidate.py` — judges one sheet-detector candidate in a
  single command.
- `eval_real_frames.py` — evaluation harness on real frames; the gate
  for neural models.
- `seg_metrics.py` — semantic-segmentation metrics.
- `benchmark_native_contours.py` — synthetic micro-benchmark of the
  classic contour path (no ONNX).

## Data and labeling

- `gen_synthetic.py` — reproducible synthetic-data generator.
- `build_training_dataset.py` — packages mask proposals into a
  training dataset, with a strict integrity preflight.
- `label_frames.py` — labels captured frames with the classic sheet
  detector.
- `label_ccby.py` — teacher labeling of CC BY videos (multi-setup
  pool).
- `classify_frames.py` — ranks captured frames for human review; it
  creates no labels.
- `select_review_frames.py` — builds a compact, auditable review
  queue from `classify_frames.py` output.
- `make_review_sheets.py` — splits the review queue into visual
  batches.
- `propose_semantic_masks.py` — conservative, explicitly unverified
  semantic-mask proposals.
- `convert_smartdoc.py` — converts the SmartDoc 2015 dataset into
  Draw's layout.
- `kaggle_review_calibration.py` — Kaggle-side calibration of the
  real-capture review pool.

## Replay rulers

- `replay_reregister.py` and `replay_reregister_v2.py` — the
  re-registration ruler: moving the notebook must leave the canvas
  identical (F-07/F-09/F-11). Frozen, hash-anchored files.
- `replay_orchestration.py` — orchestration replay: union (before)
  versus layered canvas (after).
- `replay_anchor_migration.py` — sustained-landing replay for
  intra-generation anchor migration.

## Vector bridge

- `export_vectors.py` — confirmed canvas to polylines to metric
  SVG + DXF; the paper-to-CAD bridge prototype. Sample outputs live
  in `examples/vector-bridge/`.
- `vectorize.py` — legacy offline vectorizer (own A4 constants, not
  format-aware).

## Latency

- `step0_latency.py` — glass-to-ink latency statistics, per stage.
- `cadence_fusion.py` — One Euro filter cadence experiment.

## Subpackages and support

- `runner/` — the experiment runner (collect, cycle, watch, archive)
  with its own [README](runner/README.md) and its own tests, which
  the root `pytest` run includes.
- `selflearn/` — operators of the offline learning loop around
  `draw/selflearn/`: `judge_gate.py` (promotion gate),
  `process_sessions.py`, `train_round.py`, `measure_latency.py`.
- `run_log.py` — per-run JSONL execution log.
- `helper_receiver.py` — WSL-side receiver for the Windows helper
  protocol (see `bridge/`).
