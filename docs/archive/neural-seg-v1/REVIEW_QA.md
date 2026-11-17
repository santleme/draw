# QA — Draw neural segmentation v1

Role of this document: independent QA checklist. This agent does NOT
correct other agents' files; divergences are pointed out here
for the owning agent to correct. Scope verified: only data, generation
synthetic, training, validation, ONNX and documentation. Nothing about webcam,
integration into the app, tracking, memory, real-time occlusion,
interface, CAD or simulators — that is Codex's, later.

## Fixed contract (the yardstick)

- Classes, in this order, without reordering: 0 = background, 1 = paper,
  2 = hand, 3 = ink.
- ONNX input: RGB, 512×512, ImageNet normalization
  (mean 0.485/0.456/0.406, std 0.229/0.224/0.225), same as `draw/neural.py`.
- ONNX v1 output: logits `[1, 4, 512, 512]`; masks by argmax.
- Visible label rule: what the camera sees. Hand over stroke = hand,
  never ink. `ink_full_mask` is NOT a training class, it is only to evaluate
  future memory.
- `paper_presence`: documented decision — derived from the paper mask,
  not an auxiliary head (see INTEGRATION.md).
- Splits by original source (hand/paper/background), 80/10/10, never a random
  split after generating.
- Metrics: IoU/Dice/Precision/Recall per class, never global accuracy
  alone; ink with spatial tolerance 2–3 px, always alongside exact IoU,
  never replacing it.
- Fixed battery: 15 scenarios (clean paper, with drawing, heavily drawn,
  hand covering corner/centre/stroke, hand partially out of frame, light background,
  dark background, strong shadow, paper partially out of frame, hand without paper,
  white object without paper, thin pencil stroke, thick pen stroke).

## Checklist

- [ ] Classes 0–3 identical in generator, training, metrics, ONNX and docs
- [ ] Training masks only with values 0..3, one channel, PNG
- [ ] 512×512 input + ImageNet in training, ONNX and integration docs
- [ ] Hand over stroke labels hand in generator, docs and tests
- [ ] `ink_full` separate, never as a class or training input
- [ ] Split by source, with seeds and IDs recorded, without leakage
- [ ] Negatives present (empty table, hand without paper, white object,
      light rectangle, book, box, monitor, small/out-of-frame paper)
- [ ] Per-class metrics + ink with documented 2–3 px tolerance
- [ ] Battery of 15 scenarios with predictions saved per version
- [ ] ONNX vs PyTorch validation (maximum difference + IoU) recorded
- [ ] Separate real holdout, never in training, separate metrics
- [ ] No file outside scope touched (app, neural.py, webcam, CAD)
- [ ] Tests running without GPU and without webcam

## Results — final pass

Verified on 2026-10-29, branch `feature/neural-seg-v1`, files still
not committed by the agents (work in progress, no QA commit).
First pass: generator and tests still absent; second pass, after
the generator appeared, all tests below were run.

### Files found (final pass)

- `tools/seg_metrics.py` — present
- `tools/validate_onnx.py` — present
- `tools/gen_synthetic.py` — present (second pass)
- `tools/check_freihand.py` — present
- `training/train.py` — present
- `training/config.json` — present
- `notebooks/train_segformer_kaggle.ipynb` — present (14 cells)
- `tests/test_gen_synthetic.py` — present (10 tests)
- `tests/test_seg_metrics.py` — present (13 tests)
- `docs/neural-seg-v1/INTEGRATION.md` — present
- `docs/neural-seg-v1/README.md` — present
- `docs/neural-seg-v1/HAND_SOURCES_ALT.md` — present (dataset agent)
- `docs/neural-seg-v1/KAGGLE_RUNBOOK.md` — present
- `docs/neural-seg-v1/DATASET_FREIHAND.md` — ABSENT until the end of this
  pass (dataset agent still working; `check_freihand.py`
  already exists and references this document)

### Consistency verified by reading

- Classes: `seg_metrics.py` (`CLASSES = background, paper, hand, ink`,
  0–3), `train.py` (`CLASSES = {0: background, 1: paper, 2: hand,
  3: ink}`), `config.json` (same order), INTEGRATION.md and README.md
  (same order) — CONSISTENT.
- Shape/normalization: INTEGRATION.md (input `[1, 3, 512, 512]`,
  ImageNet mean/std identical to `draw/neural.py`, output
  `[1, 4, 512, 512]`), `config.json` (512×512, same normalization),
  `train.py` (same normalization) — CONSISTENT.
- Hand over stroke: `seg_metrics.py` (docstring: visible pixel is hand,
  never ink), `train.py` (same rule), `config.json` (same rule),
  INTEGRATION.md (same rule) — CONSISTENT.
- `paper_presence`: INTEGRATION.md and `config.json` agree — derived
  from the paper mask, not an auxiliary head; single-head model.
- Ink with tolerance: `seg_metrics.py` implements disk dilation
  (default 3 px, within the 2–3 px range); `train.py` implements dilation
  by cv2 at radii 2 and 3; `config.json` describes the same methodology
  (dilation, radii 2 and 3). Equivalent methodologies, different implementations
  (pure NumPy vs cv2) — acceptable, but the values must be
  compared on the same test case before mixing numbers from the two
  modules. See the divergences section.
- Battery of 15 scenarios: `seg_metrics.py` has exactly the 15 IDs of the
  project, in order — CONSISTENT.
- Class absent on both sides: `seg_metrics.py` and `train.py` use the
  0/0 = 1.0 convention — CONSISTENT (affects the mean; `mean_iou_present`
  exists in `seg_metrics.py` to compensate).

## Tests run

Without GPU, without webcam, in the Draw venv:

```
.venv/bin/python -m pytest tests/test_gen_synthetic.py tests/test_seg_metrics.py -q
23 passed in 2.01s
```

These are 10 generator tests (determinism with the same seed, mask only with
classes 0–3, hand over stroke becomes hand and not ink, `ink_full` contains
visible + occluded, negatives without paper, split without source leakage,
generation in a temporary directory, placeholder hand marked, scene with two
hands, empty hand directory) and 13 metrics/ONNX tests (perfect prediction
= 1.0, empty prediction = 0 for present classes, ink displaced 2 px
passes tolerance but not exact IoU, empty cases, invalid
masks, battery with exactly 15 scenarios, saving predictions,
aggregate, comparison of identical outputs, wrong shape, identical dummy ONNX
ok, degraded flagged, skipped without libraries).

Additional direct verification, outside the agents' tests:

```
seg_metrics: classes ('background', 'paper', 'hand', 'ink'), 15 cenários
perfeito mean_iou = 1.0
vazio vs gt com ink: iou ink = 0.0
ink deslocado 2 px: iou ink = 0.0, tolerância 3 px: precision = recall = f1 = 1.0
train.py: CLASSES {0: background, 1: paper, 2: hand, 3: ink},
  ImageNet mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
  ink com tolerância raios 2 e 3: precision = recall = 1.0 no mesmo caso
```

Generator run by QA in `/tmp/draw-qa-gen` (not in the repo):

```
.venv/bin/python tools/gen_synthetic.py --out /tmp/draw-qa-gen --train 4 --val 2 --test 2
8 cenas, máscaras 512×512 com valores só em [0, 1, 2, 3]
arquivos por cena: scene_*.png, scene_*_mask.png, scene_*_ink_full.png
```

## Divergences found

1. **BLOCKING — generated dataset layout ≠ layout expected by
   training.** The generator writes, per split, flat files:
   `<split>/scene_*.png`, `<split>/scene_*_mask.png`,
   `<split>/scene_*_ink_full.png`. Whereas `training/train.py`
   (`pair_images_masks`) and `training/config.json` (`layout` section) and the
   notebook (which counts `DATA / split / "images"`) expect
   `<split>/images/<name>.png`, `<split>/masks/<same-name>.png` and
   `<split>/ink_full/<name>.png`. Without a conversion step or an
   adjustment on one of the sides, `pair_images_masks(DATA / "train")` fails
   with "split without images/ or masks/". The generator tests pass
   because they test the generator's own layout, not the training layout.
   Owner of the fix: generation or training agent, by agreement — not
   corrected by this QA.
2. **Resolved in verification — two implementations of the ink
   tolerance metric.** `seg_metrics.py` (NumPy, Euclidean disk, default
   3 px) and `train.py` (cv2, ellipse, radii 2 and 3) gave the same
   result on the tested case (ink displaced 2 px: precision and recall
   1.0 in both). Keep the observation: compare again if either
   changes, before mixing numbers from the two modules.
3. **Pending — `DATASET_FREIHAND.md` absent.** The dataset agent
   already delivered `tools/check_freihand.py` (which expects
   `training/rgb`, `training/mask` and `evaluation/rgb`, 224×224 mask,
   and warns about leakage by background variant) and
   `HAND_SOURCES_ALT.md` (FreiHAND as 1st choice if `training/mask/`
   exists, EgoHands as reserve no. 1, RHD as reserve no. 2, rest
   not to use until licence confirmed), but the main audit
   report was still not in the repo at the end of this pass. Without it,
   the final decision "FreiHAND direct vs alternative" is not closed
   in this QA.
4. **Watch — `save_predictions` vs metrics.** `seg_metrics.py`
   documents that the manifest records what was saved and that the
   metrics must be saved by the caller together with the manifest; if
   no caller does this, version comparison loses the
   metrics. The notebook generates `metrics.json` in training, but the link
   between the battery's `predictions.json` and this `metrics.json` must be
   confirmed in the full training review.
5. **Watch — split with few sources.** The generator documents that,
   with few sources, everything goes to train and no other split is
   generated from them; this is honest, but means that a small
   demonstration dataset does not prove the 80/10/10 split. Require, in the
   real dataset, sufficient sources per split before accepting the
   manifest.

## Out of scope — verified

None of the files read in this pass touches webcam, `draw/app.py`,
`draw/neural.py`, tracking, memory, interface, CAD or simulators.
Other agents' files are only read here, never edited.
