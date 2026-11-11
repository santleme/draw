# draw/selflearn: the loop that learns on its own (Loop A, slice 1)

Draw in production is only inference with frozen weights. This package
is the beginning of the closed offline loop that changes that: the app collects what
it sees, the monitor points out where the network probably erred, an ordered
queue decides what is worth correcting, an offline teacher generates
pseudo-labels with veto, and a training round in the UniMatch
V1 style (EMA teacher, weak-to-strong) learns from labeled + pseudo.
Learning happens between sessions, never inside the app in real time.

## The loop

```
live session
  -> [0 MONITOR]  monitor.py: entropy/margin rank suspicion; ink
     that vanishes between passes is an error signal; the classic
     witness cross-checks the network and separates FP from FN
     candidates
  -> [1 COLLECT]  collect.py: ExampleRecorder stores warp +
     probability map + a manifest line with sha256, session, model,
     and origin (each example's lineage)
  -> [2 SELECT]   select.py: auditable score (TTA disagreement +
     temporal dispersion + region impurity + novelty + risk
     stratum), per-session cap, and a guaranteed random quota
  -> [3 TEACHER]  teacher.py: temporal vote (median of aligned
     passes) and pseudo-labels with the witness's veto: ink without
     darkening goes to ignore, never to background; darkening the
     network missed also goes to ignore
  -> [5 TRAIN]    tools/selflearn/train_round.py: a round with full
     replay of the labeled examples in every batch, EMA teacher
     (initial ramp, 0.999 cap) labeling the weak view, student on
     the strong view, supervised BCE+Dice loss + confidence-masked
     consistency. Requires lineage and an independent holdout,
     preserves warm start, evaluates full pages only at the end,
     and exports ONNX + metrics.json + candidate_real.json.
     See docs/TREINO.md
```

## Modules

- `monitor.py` pure suspicion and divergence functions (step 0).
- `collect.py` example recorder in dataset format (U1).
- `select.py` active learning queue with reasons per item (U2).
- `teacher.py` temporal aggregation and pseudo-label with veto (U3).
- `drift.py` accepted-set drift monitor (v2 item 2): per-class PSI/KS of
  what each round accepted against a reference profile
  (`--accepted-reference`). Pre-registered thresholds: PSI >= 0.10
  ATTENTION, >= 0.25 DRIFT; KS p < 0.01 only raises to ATTENTION; small
  samples are INSUFFICIENT. A signal to the gate's human review, never a
  verdict change.
- `conformal.py` per-class (Mondrian) split conformal judge (v2 item 4),
  calibrated on independent labels from `--calibration-dir` (sessions
  that neither train nor belong to the sealed holdout). A pixel enters
  the consistency loss only if its set is a singleton agreeing with the
  pseudo-label.
- `review.py` + `tools/selflearn/review_holdout.py` sealed-holdout review:
  `status`, `mark` (requires `--accept-draft` or a corrected `--label`)
  and `export` (complete sessions only, in `--holdout-dir` format).
- clDice/ROI (v2 item 3) live with the other metrics in
  `tools/seg_metrics.py`; experimental, read by no gate.
- `tools/selflearn/train_round.py` the training round (minimum U5).
  `--smoke` exercises the recipe on CPU with its own synthetic dataset:
  supervised loss falls, EMA moves, lineage/holdout guards run, and
  the ONNX is checked against PyTorch. It is not real-world quality
  evidence.

Tests: `tests/test_selflearn.py`.

## Loop v2: the six corrections of 2026-11-02

The second slice of the loop closes Lucas's corrections on the
first: the engine existed, what was missing was the real raw material and the
right gates.

- `session.py` (U1) automatic session collection, wired into the app:
  each generation records reference (first warp; in legacy, the warp
  of the C key), final warp, final canvas and `manifest.json` with
  sha256 of the files and of the models, keys with relative time and
  pass telemetry. Local only
  (`captures/sessions/<session_id>/`, outside git); another
  person's desk requires consent. Whatever the `--save-dir`, the
  save directory and each session folder get a `.gitignore` with
  `*`, so raw frames never enter git (`draw/safe_io.py`).
- Enforcement in code, not only here: `collect.py` refuses rows
  without `session_id`/`origin`; `select.rank_examples` requires an
  explicit `holdout_session_ids` and runs `guard_selection_pool`;
  `tools/selflearn/train_round.py --split-ledger captures/pool/split.json`
  refuses holdout pages from unsealed sessions and keeps every sealed
  session out of training. Consent and human promotion remain
  procedural.
- `split.py` (U2) train x holdout split **by session**,
  deterministic by hash, stratified so that the holdout includes
  easy sessions and sessions with blank pages. Guards that raise
  `LeakageError`: a holdout session does not enter training, uncertainty
  selection, pseudo-labeling or threshold tuning; the
  holdout label is independent (final photo - reference, reviewed by
  hand, `draft_independent_label` + `guard_holdout_example`) and
  a pseudo-label never enters it. Correction of the prediction (with its
  anchoring bias) is training, never holdout.
- `gate.py` + `tools/selflearn/judge_gate.py` (U3) the promotion
  gate: who decides is the sealed real replay; DIBCO IoU comes out
  labeled as diagnostic and promotes no one (round 4
  won on it and lost on the real one). Below the pre-registered
  minimum n (default: 5 sessions and 25 holdout pages), the
  verdict is INCONCLUSIVE with the n values printed - the replay of ~48
  old files still has no statistical power.
- `process.py` + `tools/selflearn/process_sessions.py`: the
  translation step over the recorded sessions — integrity check
  against the manifests, the sealed train x holdout split
  persisted in `captures/pool/split.json` (a sealed session never
  unseals; the fraction climbs the ladder until the holdout
  reaches the gate minimum), independent-label drafts for the
  holdout and the pool scoreboard against the gate minimums.
- Latency telemetry (U0): `_PageCleanWorker` exposes
  `telemetry_snapshot()` (submits, latest-wins replacements,
  submit/start/end/publication timestamps per pass) without changing
  behavior, and `tools/selflearn/measure_latency.py` measures the
  stages over recorded warps.

Loop v2 tests: `tests/test_selflearn_session.py`,
`tests/test_selflearn_split.py`, `tests/test_selflearn_gate.py`,
`tests/test_page_clean_telemetry.py`.

## What is NOT implemented yet

- U4: minimal correction interface (edit the prediction with brush and
  scribble, budget in minutes and gestures).
- Historical stress set and automatic pre-registration of the round in the
  gate (the current gate judges primary/worst-group/n; promotion
  of a model remains a human decision).
- PrevMatch-lite (vote of checkpoints from previous rounds).
- Loop B (larger offline teacher with per-channel distillation to the
  student) and Loop C (masked pre-training of the backbone).
- Integration with the Kaggle runner (tools/runner) and the automatic
  upload of the versioned dataset.
- Per-tile change detection in the app (real-time front of the
  addendum): the monitor here is offline, not wired to the app.
