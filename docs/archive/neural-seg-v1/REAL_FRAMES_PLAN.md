# Plan for using real frames — session 20261029_130013

> Status: **plan, not execution.** No mask was annotated here.
> Scope: only decide HOW this session enters (or does not enter) training
> of the 4-class segmentation model. Without inventing labels.

## Updated decision — 2026-10-30

At the user's request, the **719 productive frames** selected from the two
real sessions will be prepared as `real_train`, keeping each session whole
in training. The historical recommendation below to preserve a holdout remains a
methodological guideline, but was superseded for this tranche by the explicit
request to train with the real data. Validation/testing remains synthetic;
there is no independent real holdout in these two sessions. Reserve a new session
for the next real evaluation.

The image tags already reviewed are not pixel-by-pixel masks. Only include a
frame in the training package after pairing its approved mask, checking
dimensions/checksum and validating values `{0,1,2,3,255}`; `255` may only be used
as ignored because the trainer now excludes it from loss and metrics.

## 1. What the session really is (verified, not assumed)

- **Origin:** `origin/main`, commit `38d6293` ("Update camera capture and
  dataset collection"); the frames entered in commit `11d8127`
  ("Add camera training frame session").
- **Path:** `captures/training/session_20261029_130013_453419/`
- **Contents:** **2,518 JPG frames + `frames.csv`** (2,519 lines with
  header), 66 MB in total.
- **CSV format:** `frame_index,timestamp_ns,filename,width,height`.
  All frames: **640×480**, without exception.
- **Duration:** timestamps from `1793289614045309270` to
  `1793289781830593109` ns → **167.8 s ≈ 2 min 48 s**, that is,
  **≈ 15 fps**. It is a continuous video, not loose photos.
- **Scene (samples seen, not just numbers):** same table, same camera,
  keyboard always in the top-left corner. The content changes over the course of
  the take:
  - start: blank paper with a small drawn square, blurred hand
    passing by;
  - middle: printed paper upside down ("CASA DOS ROLAMENTOS")
    with handwriting, hand + pen over the paper;
  - end: headphones covering a large part of the paper.
  Average brightness varies from ~72 to ~104 between samples; the average difference
  between consecutive sampled frames is low (median 2.3), with
  peaks of up to ~37 on object/movement changes. That is: **a
  continuous take, a single environment, but with real events**
  (paper change, hand, pen, occlusion by object).

## 2. The rule that decides everything: single session = single source

**Never split this session by frame between train and test.**

Neighbouring frames are almost identical: same table, same camera, same
angle, same lighting, probably the same hand. If 1,500 frames
go to train and 1,018 to test, the "test" measures memorisation
of this table, not generalisation. The metric would look nice and be false.

For this reason, the whole session goes to **a single destination**:

| Route | Destination of the whole session | Cost | What it proves |
|---|---|---|---|
| **A — Real holdout (recommended)** | `real_holdout`, outside training and outside any augmentation | Zero for qualitative use; ~1–2 h to annotate an evaluation subset (see §4) | The synthetic × real gap, on the real table, without contaminating training |
| **B — Training, only after manual annotation** | `real_train`, as an additional source to the synthetic set | High: 5–40 h of annotation depending on the subset (§4), plus the risk of overfitting to this table | That the model learns from a real photo — but does NOT prove generalisation to another table/hand/light |

Recommendation: **route A now; route B only if more sessions come**
(other tables, other hands, other lights). With a single session,
its value as a holdout is greater than as training: it is the only
real thermometer the project has, and training on it burns the thermometer.

Important detail: even on route B, real validation must come from
**another session** (another table or another take). Training and validating on the
same take, even with different frames, is still leakage.

## 3. What can be done WITHOUT complete manual annotation

### 3.1 Paper pseudo-label from the classic detector — only as pre-annotation, never as truth

There is already `tools/label_frames.py`: it runs `detect_sheet` on each frame
and writes `labels.json` + overlays with the quadrilateral in green for
human review. It is the right tool for generating **candidates**.

But, in this session, the classic detector is weak — measured, not guessed:

- In 26 samples (1 every 100 frames), the current detector
  (`detect_sheet_with_confidence`, version on the branch) found paper in
  **only 2, both with verdict `blank_maybe`**. In the others, `None`.
- That is, a classic pseudo-label would cover ~8% of the samples and only
  the easy cases: a dataset trained on this would learn "paper = only
  when the classic detector already finds it", the exact bias the neural
  model should overcome.

And there is a format problem: the classic detector returns a **quadrilateral**,
not a per-pixel mask. Filling the quadrilateral as `paper`
would label as paper everything inside it — including hand, pen and
headphones. To become a mask, the
hand/object regions would have to be subtracted by hand, which is already manual annotation.

**Permitted use:** generate candidates + overlays, review one by one,
correct the corners, and only then use as a paper label. The overlay
approved by a human becomes a label; the unreviewed overlay becomes
nothing.

### 3.2 Validation of `paper_presence` only, with no label at all

It is possible to run the model (or the classic detector) on the 2,518 frames and plot
paper presence over time, with no mask at all:

- presence should stay high on the blank paper and on the printed paper;
- it should drop (or become unstable) when the headphones cover the paper;
- sharp drops without a visible event are suspicious and worth inspecting.

This is a useful and cheap sanity test. **It is not IoU, it is not
a per-class metric and it does not replace the annotated holdout.** It serves to
find gross failures, not to approve the model.

### 3.3 What CANNOT be done without a mask

- **Training `hand`:** there is no reliable ready-made hand segmentation
  here. Cutting out a hand by colour/skin or by FreiHAND does not transfer
  to this table without review, and the result would teach the model to
  copy the pseudo-labeller's errors.
- **Training `ink`:** classic ink extraction (adaptive
  threshold) is precisely what the model must surpass; using it as a
  label freezes the classic detector's errors (failure under hand, under shadow,
  on thin strokes) inside the new model.
- **Training `paper` by filled quadrilateral:** see §3.1 — hand and
  objects inside the quadrilateral would become paper.
- **"Completing" the mask by temporal propagation without review:**
  propagating the quad by homography/optical flow between annotated
  keyframes reduces cost, but breaks exactly on the interesting
  events (paper change, headphones entering, hand covering a corner).
  Propagation only with human review of the propagated frames.

## 4. Concrete annotation plan (when / if annotating)

### 4.1 How many frames are worth it

**Do not annotate all 2,518.** Neighbouring frames are duplicates for
training/evaluation purposes. Annotating everything would cost weeks and would not buy
diversity.

**Recommended stratified sampling:**

1. **Temporal base:** 1 frame every 2 s ≈ **84 frames** (or 1 per
   second ≈ 168, if there is budget). Use `timestamp_ns` from
   `frames.csv` to sample by time, not by blind index.
2. **Events, always included:** paper change, hand entering,
   hand covering a corner, hand covering the centre, hand over a stroke, visible pen,
   headphones covering the paper, paper partially out of
   frame (if any), start and end of the take.
3. **Viable minimum for a quantitative holdout: 60–100 frames**
   covering all the situations above. Below that, real IoU is
   noise.
4. **Sensible ceiling for this session: 150–200 frames.** Above that,
   the gain is marginal and the money/time is better spent on a
   **second session** (another table, another hand, another light).

Cost estimate per frame (polygons by hand, with review):
paper (4 points) ~30–60 s; hand (outline) ~1–3 min; ink (tracing
the visible strokes) ~2–5 min. Total **~3–8 min/frame** →
100 frames ≈ **5–13 h**; 200 frames ≈ **10–27 h**. That is why
sampling, and not total annotation, is the plan.

### 4.2 How to annotate

- **Tool:** LabelMe (polygons, free, exports JSON per
  image) or CVAT (better for video, with interpolation between
  keyframes — but interpolation needs review on events).
  Do not create a new tool just for this; the bottleneck is the human
  eye, not the software.
- **Output format:** single-channel PNG mask, **at the original
  640×480 resolution**, same base name as the frame + `_mask.png`
  (e.g. `frame_00001251_mask.png`), values **only in {0, 1, 2, 3}**:
  `0=background, 1=paper, 2=hand, 3=ink`. At training time, the
  mask follows the same model preprocessing (squashed to
  512×512, `INTER_NEAREST` — never bilinear, which would create
  fractional classes).
- **Manifest:** a JSON alongside the masks with, per frame:
  `frame`, `annotator`, `date`, `source_session`
  (`session_20261029_130013_453419`), `split` (the whole session in a
  single split — see §2) and `notes` (e.g. "headphones cover 60% of the paper").
  Without a manifest, the masks become orphan files.
- **Quality control:** 10% of the masks reviewed by a
  second person (or by the annotator themselves the next day), with
  focus on hand edges and thin strokes. A mask with a value outside
  {0,1,2,3} is an error and the batch is rejected — same rule as training.

### 4.3 Label rules (the same as the project's, plus two cases from this session)

- **Visible rule, without exception:** what the camera sees. Hand over
  stroke = `hand` (2), never `ink`. Paper under the hand or under the headphones =
  is not visible paper: it is `hand` or `background`, respectively.
- **Pen:** `background`, not `hand` and never `ink`. The pen is an
  occluding object, not part of the hand and not a stroke. (If the user
  decides otherwise, record the decision in the manifest — but do not
  mix the two conventions in the same batch.)
- **Printed paper:** the paper area is `paper` (1), **but the printed text
  is NOT `ink`**. `ink` (3) is only what the user added
  by hand (writing and drawing). If the annotator traces the printed text
  as ink, the model learns "printed text = user stroke",
  the opposite error to the Draw objective. This is the easiest point to
  get wrong in this batch and the most important to leave explicit in
  the annotator briefing.
- **Headphones, keyboard, table:** `background` (0), always.

## 5. How this fits into the existing pipeline

- Training (`training/train.py`) already accepts a split with
  `images/` + `masks/`; a real holdout enters as a separate directory,
  evaluated by `tools/seg_metrics.py` (IoU/Dice/precision/
  recall per class + ink with 2–3 px tolerance), **with numbers
  reported separately from the synthetic ones**, never mixed.
- The training `split_manifest.json` must record this session as
  **a single source** (`source_id = session_20261029_130013_453419`,
  `split = real_holdout` or `real_train`, depending on the route), with the
  IDs of the annotated frames. If one day the session is split by
  mistake, the manifest is the alarm.
- Repository note: this session is already committed in `main`
  (66 MB). **Future sessions must not go to raw git** — the repo
  swells with each take. The path is a private dataset on Kaggle (as
  already done with the synthetic set) or another dataset storage, with
  the repo keeping only CSVs, manifest and checksums. Follow the procedure
  in [`REAL_CAPTURE_PUBLISHING.md`](REAL_CAPTURE_PUBLISHING.md) to
  split, publish and verify the parts. This document
  is on branch `feature/neural-seg-v1`; on merging, reconcile
  the `captures/` path with whatever is in `main`.

## 6. Decisions only the user can make

1. **Route A (holdout) or route B (training)?** Recommendation: A.
2. **How many frames to annotate?** Recommendation: 60–100 for a quantitative
   holdout, ceiling of 200, never all 2,518.
3. **Pen = `background`?** Recommendation: yes (see §4.3).
4. **Printed text ≠ `ink`?** Recommendation: yes, and leave this
   written in the annotator briefing.
5. **Will there be more real sessions?** If so, from which tables/hands/
   lights — this decides whether route B makes sense later, and which
   session becomes the real validation.
6. **Who annotates and with which tool?** LabelMe or CVAT, with the
   10% review from §4.2. Without this answer, the plan does not leave
   the paper — and, meanwhile, the qualitative route A (§3.2) can already
   run at zero cost.
