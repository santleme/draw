# Integration of the v1 segmentation model (4 classes) with Draw

> Interface document. Not an implementation.
> Who integrates is Codex, later, in `draw/neural.py`.
> This document does not change the current contract and does not simplify the model
> to fit it: the 4-class model is what it is, and the integration
> is what adapts to it.

## 1. What changes and what does not change

### Current contract (phase 2, in `draw/neural.py`, base `e2c866f`)

- Input: camera frame, squashed to **512×512** (no letterbox),
  converted from BGR to RGB, divided by 255 and normalized with the
  ImageNet mean and standard deviation.
- ONNX output: two tensors, `mask_logits` with **2 channels**
  (background, paper) and `presence_logit` (1 value).
- Decision: positive presence (`presence_logit > 0`) **and** quadrilateral
  that closes. Without one of the two, the result is `(None, None)`.
- Mask: probability of the paper channel (softmax of 2 channels) with
  threshold **0.5**, resized to the frame size with
  `INTER_NEAREST`, in uint8 0/255.
- Quadrilateral: largest contour of the mask, minimum area of 1% of the frame,
  epsilon ladder of the classic detector (`draw/paper.py`),
  refit of each edge as a line over the contour points
  (maximum accepted displacement of 100 px), subpixel refinement only on the
  corners inside the frame, and requirement of a convex quadrilateral.
  All of this already exists in `neural.py:fit_quad` and **must not be
  rewritten** — it must be reused.
- Lazy ONNX session: `onnxruntime` is only imported on first
  use; without it, explicit error. The model path arrives by
  argument or via the `DRAW_NEURAL_MODEL` variable. The binary stays
  outside the repository.

### New contract (v1, 4 classes)

- The model now segments **what the camera sees, pixel by pixel**,
  into 4 classes. Nothing here is "what is underneath the hand" — see
  the visible-label rule in section 6.
- The only output of ONNX v1 is segmentation. **There is no `presence_logit`
  in ONNX v1** (decision in section 4). Whoever needs the presence
  signal derives it from the paper mask, deterministically.
- The quadrilateral pipeline (`fit_quad`) remains identical. What
  changes is only the origin of the mask: before, the paper channel of a
  2-channel model; now, the paper channel of a 4-channel model.

## 2. ONNX v1 input

| Item | Value |
|---|---|
| Input name | `image` (keep the same name as the current contract) |
| Format | RGB, not BGR. The frame comes from the camera in BGR (OpenCV) and **must** be converted with `cv2.COLOR_BGR2RGB` before entering the model |
| Resizing | Squashed to **512×512** with `INTER_LINEAR`, exactly as today. No letterbox, no crop, no aspect preservation. This distorts the image proportions, and that is how the model is trained and evaluated — changing this in the integration would break the model |
| Scale | Divide by 255.0 (float32) |
| Normalization | ImageNet, same as current: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Layout | NCHW, float32, batch 1 |
| Final shape | `[1, 3, 512, 512]` |

Order of operations, with no shortcut: BGR → RGB → resize 512×512 →
`/255` → `(x - mean) / std` → transpose to CHW → batch.

## 3. ONNX v1 outputs and class order

| Item | Value |
|---|---|
| Output name | `logits` (sole output of ONNX v1) |
| Shape | `[1, 4, 512, 512]`, float32, logits (not probabilities) |
| Post-processing | softmax over the channel axis (dimension 1) and then `argmax` over the same axis |

Class order, fixed and documented in `classes.json`:

```json
{
  "0": "background",
  "1": "paper",
  "2": "hand",
  "3": "ink"
}
```

**Do not reorder, do not rename, do not remove a channel.** Any model
that changes this order is a different model and needs a new
version of the document, not a silent adjustment in the integration.

## 4. Decision: `paper_presence` — derived from the mask, not an auxiliary head

### Documented decision

**`paper_presence` in v1 is derived from the `paper` mask. There is no
auxiliary presence head in ONNX v1.**

Exact definition, for Codex to implement:

```text
paper_prob   = softmax(logits, dim=canais)[:, 1, :, :]   # canal paper
paper_mask   = argmax(logits, dim=canais) == 1            # 512×512, bool
paper_fraction = paper_mask.mean()   # fração de pixels paper, entre 0 e 1
quad         = fit_quad(paper_mask em 0/255, frame)
paper_presence = 1  se paper_fraction >= 0,01 e quad is not None
                 0  caso contrário
```

The threshold of 0.01 (1% of the frame) is the same as the minimum contour area
already used in `fit_quad` (`_MIN_CONTOUR_AREA_RATIO`). It is not a new
number: it is the same criterion, applied twice in a coherent way
(global fraction and largest contour).

Besides the boolean, the integration must expose `paper_fraction` as a
continuous presence-confidence score (useful for logs, debug, and
for the evaluation harness). The boolean is the decision; the fraction is the
measure.

### Pros and cons considered

**Derived from the mask (chosen):**

- Pros: a single source of truth. Presence and mask never
  disagree (there is no "the model says there is paper, but the mask
  is empty", nor the reverse). Simpler ONNX, with a single
  output. Deterministic and reproducible behavior from the
  segmentation. The threshold is interpretable (fraction of visible paper).
- Cons: if the mask fails, presence fails with it — there is no
  second signal to rescue it. A very small or very
  occluded sheet may fall below 1% even though present; that is
  a known limit, and the correct response is to improve the
  segmentation, not invent a second signal that contradicts it.

**Auxiliary head (rejected for v1):**

- Pros: an independent signal, which could rescue cases where the
  mask fails, and which reproduces the format of the current contract
  (`presence_logit`) without changing the signature.
- Cons: two signals that can diverge, and then the integration
  needs an arbitrary tie-break rule (which counts when
  they disagree?). More complex training (auxiliary loss weight,
  logit calibration). Risk that the head learns a shortcut
  ("there is something white and rectangular") that the mask, more demanding,
  has not learned — exactly the false positive that the dataset's
  negatives exist to avoid.
- Verdict: compatibility with the old format does not justify
  a second signal that may contradict the first. **The model
  is not simplified to keep the old contract; the contract
  is adapted to the model.** If, with metrics in hand, the derived
  signal proves insufficient, an auxiliary head may be evaluated
  as v2, as an additional output, without breaking segmentation —
  never as a silent substitute for the mask.

## 5. Thresholds

| Where | Value | Why |
|---|---|---|
| Paper mask (for the quadrilateral) | `argmax == 1` | In 4 classes, argmax is the model's decision. It is the direct equivalent of `prob > 0.5` of the 2-channel model (there, prob > 0.5 implied argmax) |
| Confident mask (optional, for debug/visualization) | `argmax == 1` **and** `prob_paper >= 0.5` | Shows only pixels where the model is confident; do not use for the quadrilateral, so as not to shrink the sheet |
| `paper_presence` | `paper_fraction >= 0.01` **and** quadrilateral closes | Same minimum-area criterion already used in `fit_quad` |
| Hand and ink masks | `argmax == 2` and `argmax == 3` | No additional threshold in v1; if future calibration requires one, document it as a new version |
| Quadrilateral | current `fit_quad` rules, unchanged | Largest contour, area ≥ 1%, epsilon ladder, edge refit (≤ 100 px), subpixel only inside the frame, convex |

Do not create a new threshold in the integration without recording here the reason
and the metric that justifies it.

## 6. How to extract each mask

All masks come from the same `argmax`, at 512×512 resolution, and
only then are resized to the size of the original frame,
with `INTER_NEAREST` (never linear interpolation on a binary mask,
which would create false intermediate values), in uint8 0/255.

```text
seg        = argmax(softmax(logits), dim=canais)   # [512, 512], valores 0..3
paper_mask = (seg == 1)   # folha visível
hand_mask  = (seg == 2)   # mão visível
ink_mask   = (seg == 3)   # traço visível
# background = (seg == 0), ignorado pelo Draw
# cada máscara: resize para (frame_w, frame_h), INTER_NEAREST, * 255
```

Important points:

- The three masks are in **original frame coordinates**,
  not rectified-sheet coordinates. The warp to the sheet (homography) comes
  later, from the `paper_mask` quadrilateral, and is the work
  of the integration/app — not of this document and not of the model.
- `hand_mask` and `ink_mask` are new v1 outputs. The current
  contract does not have them. The integration must expose them (see section 7),
  but the current app does not need to use them until Codex decides how.
- The masks are mutually exclusive by construction (argmax):
  a pixel is exactly one class. This is what implements the
  visible-label rule (below).

### Visible-label rule (mandatory, no exception)

Segmentation represents **what the camera sees**, not what is
underneath:

```text
papel visível → paper
traço visível → ink
mão visível   → hand
restante      → background
```

If the hand covers a stroke, the final pixel is `hand`, **not** `ink`.
The model must not be trained to "see through the hand".
The full drawing mask (`ink_full_mask`, with the stroke
including where the hand hides it) is a separate artifact of synthetic
generation, **not** a training class, **not** in the ONNX
and **must not** be used as a label. It will serve, in the future,
to evaluate Draw's memory system — outside this scope.

## 7. Paper → current contract conversion (what Codex will do later)

Step by step, reusing what already exists:

1. Run ONNX v1 on the frame (section 2) and obtain `logits`
   `[1, 4, 512, 512]`.
2. Softmax + argmax → `paper_mask` 512×512 (section 6).
3. Resize `paper_mask` to the frame size with
   `INTER_NEAREST`, in uint8 0/255.
4. Call `fit_quad(mask, frame)` — the same as today, without
   rewriting: largest contour, epsilon ladder, edge refit,
   subpixel inside the frame, convex.
5. Compute `paper_fraction` and `paper_presence` by the definition
   in section 4.
6. Return, in the current contract, `(corners, "paper")` when
   `paper_presence == 1`, and `(None, None)` otherwise —
   exactly as `detect_sheet_with_confidence` does today.
7. Expose, in addition to the current contract (without breaking it), the
   new masks: a new method (suggestion: `predict_masks(frame)`)
   returning `paper_mask`, `hand_mask`, `ink_mask` and
   `paper_fraction`, in frame coordinates. The old method
   continues to exist and continues returning only `(corners, verdict)`.

### What Codex will need to change in `draw/neural.py` (list, without implementing here)

- [ ] `predict`: read **one** output (`logits`, 4 channels) instead
      of two; remove reading of `presence_logit`; replace
      the 2-channel softmax with 4-channel softmax + argmax.
- [ ] Extract `paper_mask`, `hand_mask` and `ink_mask` from argmax,
      with `INTER_NEAREST` resize to frame size.
- [ ] Compute `paper_presence` by the definition in section 4
      (fraction ≥ 0.01 and quadrilateral closes), instead of
      `presence_logit > 0`.
- [ ] Keep intact, untouched: `fit_quad`, `_approx_quad`,
      `_refit_corners`, `_refine_inside`, the lazy session,
      the explicit error without `onnxruntime`, the path by argument
      or `DRAW_NEURAL_MODEL`, and the fallback to the classic
      detector when the neural one sees no sheet (this is in
      `draw/app.py`, not here, and continues to hold).
- [ ] Keep the input exactly as in section 2 (name `image`,
      512×512 squashed, ImageNet). Any divergence here is
      an integration bug, not an "adjustment".
- [ ] Update the module docstring to the v1 contract and
      point to this document.
- [ ] Validate with the evaluation harness (`tools/eval_real_frames.py`)
      before considering the integration done, and compare the
      metrics with the phase 2 model — without that comparison, the
      integration is not validated.

### What Codex will NOT do in this step (outside his scope here too)

- Not change the app, webcam, temporal tracking, stroke memory,
  real-time occlusion logic, interface, CAD or
  simulators. The integration is only swapping the neural detector for the
  v1 model, behind the same corners + verdict contract, plus
  the new masks exposed.

## 8. v1 delivery files (reference for the integration)

The files below are delivered by other agents of this project
(training, data). The integration must expect them with exactly these
names and formats; if any arrives differently, that is a divergence to
resolve, not a detail to improvise on the spot:

| File | Content |
|---|---|
| `draw_segmentation_v1.onnx` | v1 model, input `image` `[1,3,512,512]`, output `logits` `[1,4,512,512]` |
| `classes.json` | `{"0": "background", "1": "paper", "2": "hand", "3": "ink"}` |
| `metrics.json` | Per-class metrics (IoU and Dice per class, precision, recall) and ink metric with spatial tolerance |
| `training_config.json` | Architecture, input, loss, seeds, hyperparameters — everything to reproduce training |
| `split_manifest.json` | Which original sources (hands, sheets, backgrounds) went to train/val/test, with seeds |

Before integrating, validate the ONNX: run the original model and the ONNX
on the same frames and compare the outputs; record the comparison.
A conversion that degrades segmentation is not delivered.

## 9. Known limits of this interface

- The model sees the frame squashed to 512×512: a very thin stroke may
  disappear in downscaling. The ink metric with tolerance (2–3 px)
  exists for this, and the integration must not "fix" thin stroke
  with its own threshold.
- A very small or very occluded sheet may fall below 1% of
  paper pixels and therefore have `paper_presence == 0`, even
  though partially visible. This is the defined behavior,
  not a bug to work around in the integration.
- `hand_mask` and `ink_mask` are of the frame, not of the rectified sheet.
  Whoever needs them on the sheet must apply the quadrilateral's homography —
  outside the model, outside this document.
- This document describes the interface. Metrics, samples, and the
  model itself are delivered by the other agents and documented
  in this folder's README.
