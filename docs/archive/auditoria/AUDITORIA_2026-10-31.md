# Heavy audit of Draw — with a focus on the AI model

**Date:** 2026-10-31. **Audited base:** `main` at `5d64146` (fresh
clone from GitHub), plus the complete laboratory (`draw-lab`, `draw-bridge`) where
training runs, rulers and reports live. **Where it ran:** Julia's VM — no
latency measurement from this audit transfers to Lucas's Ryzen; the
VM times cited are only proof of execution. **Method:** code reading
with `file:line` evidence, measured ONNX metadata inspection
(onnx 1.23.2 / onnxruntime 1.31.1 / OpenCV 5.0.0), execution of the
test suite of the clone and reading of all training/judgment reports
from the laboratory. A number with no declared source = a report number, not
remeasured.

> **Base caveat.** The audit froze the code at `5d64146`.
> **During the audit, main advanced** to `686e29a` (2026-10-31
> 16:34), which pushes restart unit 3 (`draw/page_clean.py`
> new + changes in `draw/app.py`: page-clean as the default ink,
> heuristic behind `--legacy-ink`). The unit 3 code **was not
> audited line by line** — page-clean enters through the laboratory
> reports (`page-clean-v0/UNIT3_REPORT.md`) — and the sequence
> (default first, real judgment afterwards) is itself a finding
> (R2/R8).

---

## 1. Executive summary

The project has changed in size: there are **11 models** in the lineage (sheet
v1–v4, 4-class pilot, ink U-Net v1/v2, paired ink v3,
paired flagship, page-clean v0, plus the CornerNet precursor), an
experiment runner (`tools/runner`), a judgment bridge
(`draw-bridge`, partly absorbed by the runner) and dozens of laboratory
reports. **Only one model is in production** (sheet v3) and **only one
became the default without real judgment** (page-clean v0 — it entered main
during this audit, `686e29a`).

The three structural problems the audit finds, in order:

1. **Reproducibility**: the production model is not in the repo, there is no
   integrity check on loading, and the module for the `--ink-model` flag
   was never pushed. Whoever clones the project does not reproduce the system that
   runs on Lucas's desk (R1, R8).
2. **Judgment**: the internal metric **never** predicted the real scene — in
   none of the families, with one documented case per family (§3.2). The
   sheet v3 is in production **with no number from another setup**; the
   external per-setup protocol designed on 2026-10-29 was never
   executed. Page-clean has **zero** real measurement and latency on the Ryzen
   never measured (R2, R4).
3. **Ink latency is structural**: four full-page ink formulations measure 323–355 ms on the Ryzen; the flagship measures 431.3 ms,
   above the 400 cap. Added to detection, the frame estimate is
   ~428–459 ms (parts measured separately; integration in the same loop
   was never measured). It is not a problem of one model — it is of the
   "full page per frame" formulation (R3).

What is **healthy** and the audit confirms: the discipline of real
rulers (harness, replay, scene ruler) has already saved the project from three
bad swaps (sheet v2, ink v1, sheet v4); the bridge cascade correctly encodes
the learning (internal only detects death; the ruler is what ranks); and the
published geometric guards hold the measured jitter
(TR.x std 129.6 → 65.6 px in the recorded series).

---

## 2. Model inventory (condensed lineage)

Complete detail, with n and the ruler for each number, in the laboratory
(`ink-flagship/REPORT.md`, `ink-model/BUILD_REPORT*.md`,
`neural/v*/TRAIN_REPORT.md`/`RUN_REPORT.md`, `page-clean-v0/*.md`).

| Model | What it is | Judgment (main ruler) | Status |
|---|---|---|---|
| Sheet v1 | SegFormer-B0, mask + presence | MIDV held-out IoU 0.987 (200/96); Lucas holdout n=3 IoU 0.92 | Superseded |
| Sheet v2 | Identical; corrected data (school + SmartDoc) | Raw model ties/wins; **through the app: 41/43** — DO NOT SWAP | Dormant |
| **Sheet v3** | Identical; rotating SmartDoc + COCO negatives + copy-paste | Harness 43/43, 0 FP, IoU 0.962/0.841 (11 masks); training 44.7 min (v2: 3h51) | **PRODUCTION** (`draw_seg_v3_dyn.onnx`, 384 px dynamic) |
| Sheet v4 | SegFormer-B0 multitask + corner heatmaps | By the SPEC: presence **7/43**; corner head collapsed (peaks 0.09–0.15) — DO NOT SWAP | Failed |
| 4-class pilot (PR #7) | SegFormer-B0 background/paper/hand/ink | Synthetic IoU 0.9137; **real: 42/43, FP 6/6, quad IoU 0.478** | Dormant, opt-in |
| Ink U-Net v1 | U-Net 483k params, full page | Synthetic 17/18 rulers green; **real replay: 0/8 regions, median P(ink) 0.0000 on real** | Failed for swap |
| Ink U-Net v2 | Same U-Net; corrected appearance + real pseudo-labels | Partial replay: coverage 99.3%, bubbles 54.4%; printed leakage 1.41% (ruler 0.5%) | Failed for swap |
| Paired ink U-Net v3 | Pair (before, after) → new ink | Synthetic strata ok; **scene ruler: shadow fails by 5–15×** | Dormant |
| Paired ink flagship | Paired SegFormer-B0 + occlusion head | Scene: shadow **0.00%** false, recall under shadow 0.804; clean recall **0.755 < 0.85**; latency **431.3 ms** | Pending (iterate protocol v6) |
| **Page-clean U-Net v0** | Rectified page → ink mask | Synthetic→synthetic: IoU **0.800** × 0.177 for the threshold (48 pages). **Real: nothing measured** | **Default in the app (main `686e29a`)**, real judgment pending (unit 4) |
| CornerNet (precursor) | Small U-Net, corner heatmaps | Synthetic only (error 11.3 px, n=40) | Superseded |

**Name collision (real confusion risk):** "v3" names three
unrelated things — the SegFormer sheet, the paired ink U-Net and the
valid run of the flagship (version 3 of its kernel). Likewise "v1/v2" of ink ×
of sheet. See R9.

---

## 3. Findings

### 3.1 The production model does not live in the repo (high severity)

- `models/` in the repo contains **only the experimental pilot** (15.2 MB).
  The production `draw_seg_v3_dyn.onnx` lives only in the laboratory and on
  Lucas's machine. Measured in this audit: dyn = 15,191,109 B,
  sha256 `751e74da…` (matches the laboratory record), symbolic
  input `[1,3,H,W]`, 1,585 nodes, 3,714,402 parameters; the fixed v3
  (512) has 959 nodes and the same parameters.
- `_load_session` (`draw/neural.py:153-171`) **checks no integrity
  whatsoever**: any file with the right name is accepted. There is no
  model manifest (name + hash + ruler that approved it) in the repo.
- **Inverted policy**: the artifact versioned in git is precisely the
  experimental model that cannot become the default, while the production one
  stays outside.
- The `--ink-model` flag is **dead in a fresh clone**: `draw/app.py`
  imports `draw/ink_model.py` behind a try/except because the module was never
  pushed — the app warns "not present in this package" and continues
  on the heuristic. A flag without a module is silent debt.

### 3.2 The internal metric never predicted the scene (high severity, pattern)

One documented case per family, all in the laboratory reports:

- **Ink v1**: 17/18 synthetic rulers green → real replay with
  median P(ink) **0.0000** over the true ink.
- **Pilot**: synthetic paper recall 0.9888 → in real use, the mask
  covers 29–34% of the frame where the sheet dominates; IoU 0.9137 → 0.478.
- **Sheet v4**: the internal score **rose until epoch 20** while the
  corner head (the decisive output of the SPEC) was already collapsing on the
  training's own held-out (error 173.4/1000 × 44.3/1000 for the mask
  fit). The internal ruler had no term capable of failing the head
  that ruled the path.
- **Flagship × paired v3**: parity in the synthetic strata; the
  scene ruler separates the two by 5–15× at the shadow gates.

Corollary already measured twice: when the internal ruler and the scene ruler
disagree, **the scene ruler is the one that decides** (sheet v2 and ink v1). The
bridge cascade (`tools/runner` + `draw-bridge`) already encodes this —
internal only detects a dead run; the mini-ruler ranks (proven on 5/6
key metrics); full rulers judge; the real holdout is the final
gate. The audit recommends **formalizing this rule as a written swap
policy** (R4), because today it lives in bridge documents, not
in the decision path of the main repo.

### 3.3 Generalization: the missing number is from another setup (high)

- Sheet v3 is in production **with no number outside Lucas's
  desk**. The external per-setup protocol (3–5 setups, 40–60 frames per
  setup, per-setup gate: presence ≥95%, 0 FP, IoU ≥0.85, worst setup
  as the headline) was designed in `research-v3/REPORT.md` §3.3 on
  2026-10-29 and **was never executed**.
- The real holdout for the sheet is **3 frames from a single session** (one
  desk, one camera), with a ±20 px ground truth. The 6 negatives of the 49
  frames took part in presence training (oversampling ×12): every
  "0 FP on its negatives" is an **in-sample** reading. The
  out-of-sample evidence against FP is the MIDV held-out (96 negatives).
- The white box remains the known FP: the raw v3 fires weakly on it
  (logit +0.41 to +0.98 at 512 px) and v4 fired strongly
  (+4.84 to +5.39) — one of the reasons for v4's DO NOT SWAP.
- Page-clean **has no real measurement**: IoU 0.800 is
  synthetic→synthetic, exactly the position ink v1 was in
  when it failed in real use. And its latency was measured only on this VM
  (median 1,542 ms/page, mosaic of 256 crops without overlap —
  a VM number, not transferable, but indicating that the mosaic strategy
  needs its own measurement on the Ryzen before any conclusion).
  During this audit page-clean became the default path in main
  (`686e29a`): the live test (unit 4) stopped being an entry gate
  and became judgment of a default already in force — recommendation
  R2 becomes more urgent, not less.

### 3.4 Latency: the whole account (high)

Corrected and accepted account (Ryzen measurements, laboratory reports):

| Portion | ms |
|---|---|
| Neural detection (sheet v3 dyn, 384) | 91.0 |
| Heuristic ink | 23.1 |
| Accumulator | 8.7 |
| Display warp | 2.6 |
| **Summed components** | **125.4** |
| Undecomposed residue in the latency audit | ~1.8 |
| **Measured detection frame** | **127.2** |

- Neural ink on a full page, four formulations: **323–355 ms**.
  Flagship: **431.3 ms**. Swapping the heuristic for the neural one, the estimated frame
  goes to **~428–459 ms** (sum of separately measured parts; the
  integration of the two models in the same loop **was never measured**),
  above the 400 cap and far from the 250 goal.
- **Glass-to-ink** latency (camera → visible ink) was never
  measured end to end, with any model. There is a kit for this
  (`draw-lab/desk-session/latency_probe.py`), never run on the desk.
- Known output not yet executed as a design: **decoupling ink from
  detection** — rare detection (every N frames) with tracked homography
  between them (XFeat/ECC measured in the laboratory), ink
  running at its own cadence. Without decoupling, making detection cheaper
  delays ink along with it, because today the two only run on the detection frame
  (every 5 frames, `draw/app.py`).
- Real micro-costs in the neural path (code): unnecessary softmax
  in the legacy branch (prob > 0.5 ⇔ paper logit > background logit; today there is
  `np.exp` over 2×512×512 per call), two whole-frame conversions
  per detection (BGR→RGB in predict + BGR→GRAY in refinement, with the
  resize only after converting) and `_refit_corners` in O(points×4)
  over the contour without prior simplification.

### 3.5 Fragile ONNX contract in the adapter (medium)

The detector accepts two contracts (legacy 2 outputs × 4-class pilot)
and the contract tests pass — but robustness is uneven:

- **Hardcoded input name `"image"`** (`draw/neural.py:573`).
  `get_inputs()` is consulted only for the shape. A future export with
  input `input`/`x` breaks on the first frame, not on loading.
- **Legacy branch does not validate channels**: it slices `logits[0,0]`/`[0,1]` as
  background/paper without checking that `mask_logits` has exactly 2 channels.
  A hybrid 4-channel export with presence would be read **silently**
  as background × class-1 — a wrong mask with no error at all. The 4-class
  branch, by contrast, validates shape (asymmetry).
- **4-class branch requires the output list to be exactly
  `["logits"]`** (`neural.py:586`): any auxiliary output becomes a
  fatal `ValueError` on the first frame, instead of degradation.
- **Resolution reads only `shape[2]` and assumes square**
  (`neural.py:192-198`): a rectangular export would be squashed to
  square without warning.
- **Pilot presence is synthetic** (`neural.py:593-601`): with no trained
  head, the adapter fabricates +1.0/−10.0 from an area ratio ≥1%. The
  −10.0 vetoes the classic path with the force of a strong trained logit — but it is
  an area threshold, not calibration; and the +1.0 lets the pilot's known
  FPs (white box with presence +1.00; FP on 6/6 negatives in the real harness) pass without friction, because the veto only blocks
  negatives. Safe today **only because the pilot is opt-in**.
- **The pilot is not dynamic in space**: input `[N,3,512,512]` — only
  the batch is symbolic; it always runs at 512 (confirmed by probe), and the
  mask is born at ¼ resolution (128) enlarged with NEAREST: a cell of
  ~10×5.6 px in a 720p frame feeding the corner fit.

### 3.6 Geometric guards: specific holes (medium)

The published guards (`_fit_plausible`, `_SheetTracker`) work on the
recorded series, but the code audit found holes:

- **`_fit_plausible` accepts NaN**: every comparison with NaN is false, so
  no rule fires and the guard **approves** a quad with NaN. It also lacks
  a minimum area and a ratio between adjacent edges (a long thin strip
  passes).
- The public function `fit_quad` **skips the entire guard** — whoever calls
  through it receives a quad with no plausibility at all.
- Renewal of `_SheetTracker` by presence alone can hold an
  old quadrilateral **indefinitely** if the model insists on
  "paper" without the fit closing: a temporal cap for the frozen
  quad is missing.
- Dead state: the tracker's `_pending_hits` is written in 5 places and
  read in none. Global detector singleton with no session close
  and with edge memory shared between instances; the edge memory is not
  cleared on an empty frame/negative presence (inheritance between
  scenes is decided by a single IoU ≥ 0.7).

### 3.7 Training runs: the bugs live in the unexercised path (medium, pattern)

Four runs, four bugs that only appeared on a long run or a path not
tested before launch:

| Run | Bug | Cost |
|---|---|---|
| Sheet v1 | school-notebooks with **0 samples** (path bug; 600 read errors in the log) | Whole run without the source v2 would show mattered |
| Sheet v3, attempt 1 | `Image.draft()` enlarging masks | MIDV IoU 0.476; run discarded |
| Flagship v2 | Scheduler with effective LR ~1e-8 | **4h02 of GPU** on an invalid run; the self-test existed and did not assert the effective LR |
| Sheet v4 | COCO **silently absent** (`coco.ok=false`): 400 scene negatives and 300 backgrounds became MIDV background | The judged run is not the designed run |

Sibling pattern: truncated downloads and dying drivers became a manual
ritual in at least 3 runs, and ink v1 **did not save a torch
checkpoint** (ONNX only) — v2 had to train from scratch. The new runner
(`tools/runner`) already attacks part of this (collection with resume + sha256,
collapse watch, candidate file), and v6 was already born with a
self-test asserting the effective LR. What is missing is the **data preflight**: asserting
counts per source (>0 for every source declared in the recipe) and the
export contract, before epoch 1.

### 3.8 Test suite is not hermetic (medium)

Measured in this audit, on a fresh clone (OpenCV 5.0.0, environment same
as Lucas's): `pytest tests/` **dies on collection** with INTERNALERROR
because `tests/test_accumulator_stamps.py` executes `sys.exit(2)` on
import when a manual file is missing (`/tmp/ink_pre_stamp.py`, a pre-change copy
of `draw/ink.py` that the operator must prepare by hand).
Ignoring that module: **44 passed, 3 skipped**, with a
`PytestUnhandledThreadExceptionWarning` (exception in an unhandled capture
thread in the test). A test that requires manual preparation cannot live
in `tests/` with a `test_` prefix without becoming a graceful `skip` — in a clean
clone, the whole suite looks broken.

### 3.9 Documentary and version coherence (low/medium)

- The decision to switch to sheet v3 was taken **outside** the
  report of its own run (which recommended "do not switch for now",
  conditioned on an external set that never existed). The
  v4 report already treats it as production: there is a documentary leap with no
  recorded act linking the two.
- "MIDV held-out" is a **different protocol** between sheet generations:
  v1–v3 use n=200/96 (IoU 0.982–0.988); v4 uses a 10% split, n=100
  (0.786). The numbers are not comparable with each other, but they live side by side
  in the tables without the caveat alongside.
- The "17 of 18" score for ink means different things in v1
  (red outside the cells) and in v2 (two reds). The
  BUILD_REPORT_V2 §7 says "latency not measured" and the addendum in the same
  file measures it — whoever reads only the section takes away the wrong conclusion.
- The page-clean IoUs 0.800 and 0.8465 are different sets/aggregations
  (48 whole pages × wrapper proof on 20 pages of the training protocol)
  treated as references to each other.

---

## 4. Prioritized recommendations

**R1 — Version the real production.** Bring the
`draw_seg_v3_dyn.onnx` into the repository (or to a canonical dataset/release
with a link), with a `models/MANIFEST.json` manifest: name, sha256,
ruler that approved it, date. `_load_session` checks the manifest hash
when it exists and **warns loudly** on divergence. Push
`draw/ink_model.py` or remove the `--ink-model` flag from main — a flag without a
module does not stay.

**R2 — Execute the per-setup protocol before the next swap.**
3–5 external setups (person + desk + webcam + room), 40–60 frames
per setup, per-setup gates and **worst setup as the main number** —
for sheet v3 (which is in production without this number) and for
page-clean, whose live test (unit 4) now judges a default that
already entered main during this audit.

**R3 — Treat ink latency as a formulation problem.**
(a) Measure the sheet + neural ink integration **in the same loop** on the Ryzen
(the current account is a sum of parts). (b) Measure page-clean on the Ryzen with the
current mosaic strategy and with overlap — 1,542 ms/page on the VM
decides nothing, but calls for measuring. (c) Design the decoupling
detection × ink with tracked homography (the output Lucas proposed
on 2026-10-30), because no full-page U-Net fits in the frame.
(d) Measure glass-to-ink with the `latency_probe.py` of the desk kit.

**R4 — Written swap policy in the repo.** No model swap without:
green scene ruler + real holdout (sealed, when it exists) + latency
measured on the target within the cap. The internal metric serves only to detect
a dead run. This is already the practice; it is missing being written where the decision
happens.

**R5 — Harden the adapter contract (cheap, one afternoon).**
Use `get_inputs()[0].name` in the feed; validate channels/shape in both
branches; accept 4 classes by the presence of `logits` (not by list
equality); read H and W separately and validate square-or-dynamic on
loading. Each item with a test in the existing `test_neural_onnx_contract.py`.

**R6 — Close the guard holes.** `isfinite` + minimum area + ratio
of adjacent edges in `_fit_plausible`; public `fit_quad` goes through
the guard; temporal cap for the frozen quad in `_SheetTracker`;
remove `_pending_hits` (dead state) or tie it to a real read.

**R7 — Hermetic suite.** `test_accumulator_stamps.py` becomes a graceful
skip without the manual file (or leaves `tests/` for `tools/`);
handle the capture-thread exception in the test. Verifiable goal:
fresh clone → green `pytest` with no manual preparation.

**R8 — Audit unit 3 and keep one source of truth.** The unit 3 diff
(`draw/page_clean.py`, 111 lines, + `draw/app.py` in
`686e29a`) entered main during this audit and was not covered by
it: it becomes the first unit of a follow-up audit, with the
mosaic latency on the Ryzen as a mandatory item (R3b). And the
local mirrors (snapshot without git, Lucas's tree) diverge from main to
the point that whole lines differ (`draw/neural.py`: 655 lines on
main × ~1,070 in the mirror) — a future audit needs to know which tree
is the product.

**R9 — Model registry with canonical names.** Adopt in
`tools/runner/candidates.jsonl` (and in every new report) collision-free names: `sheet-seg-v3`, `ink-paired-unet-v3`, `ink-flagship-v3`,
`pageclean-unet-v0` — and retire the bare "vN", which has already named three
different models.

**R10 — Run preflight in the runner.** Before epoch 1: assert
sample counts per source declared in the recipe (every source > 0),
effective LR after the scheduler's first step, and the export contract
(1-batch test with the ONNX loaded by the real adapter).
Always save a torch checkpoint — ONNX alone does not allow resuming.

**R11 — Measured micro-optimizations, afterwards.** Remove the softmax from the
legacy branch, convert color after the resize (or in a single pass) and
simplify the contour before `_refit_corners` — in this order, each
with a Ryzen before/after measurement. Nothing here changes without a number.

**R12 — Pilot presence must never reuse logit semantics.**
If the pilot (or any model without a presence head) is a candidate
one day, the area-derived signal gets its own name and threshold,
documented as a heuristic — and the veto on the classic path starts using that
threshold, not the trained models' `PRESENCE_VETO_LOGIT`.

---

## 5. What this audit did not do

- It did not remeasure any ruler (harness, replay, scene) — the verdicts are
  those of the laboratory reports, with source and n declared.
- It did not measure latency on the Ryzen (VM does not transfer) nor run a
  full-page eval on this VM (house rule: one eval at a time, and none
  was needed for the findings).
- It did not audit the cursor/Windows bridge (`bridge/`), the hackathon Replit app,
  nor `draw-bridge` beyond what the runner absorbed.
- Unit 3 code (`686e29a`) entered main during the audit
  and was not audited line by line; page-clean was audited through the
  laboratory reports (R8).

*Report generated at Lucas's request, on a separate branch, for review.
No code was changed by this audit.*
