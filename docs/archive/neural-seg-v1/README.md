# Draw neural segmentation v1 — background / paper / hand / ink

Semantic segmentation model with 4 classes, trained on Kaggle,
for the Draw pipeline:

```text
frame
  ↓
SegFormer-B0 (transfer learning)
  ↓
background / paper / hand / ink
```

Then, in Draw (outside this project), Codex will use it as follows:

```text
paper      → área da folha e homografia
hand       → região temporariamente bloqueada
ink        → desenho visível
background → ignorado
```

When the hand covers the drawing, the model returns `hand` — never
"sees through the hand". Draw's temporal memory, which preserves
the last reliable state of the stroke until the hand leaves, is future
work, outside this project.

## Focus of this project

Exclusively:

```text
câmera
→ identificar a folha
→ identificar a mão
→ identificar o traço
→ manter o desenho digital coerente
```

Do not consider CAD, simulators or future integrations in this
project. Muse's scope here is only the model and its documentation.

## Muse's scope (only this)

```text
dados
→ geração sintética
→ treinamento
→ validação
→ exportação ONNX
→ documentação
```

## What is explicitly OUT (and who will do it later)

Everything below is later work, by **Codex**, not by this project.
None of this is implemented here, neither in this folder, nor in this branch,
beyond the interface documentation:

- webcam
- integration of the model into the application (`draw/neural.py`, `draw/app.py`)
- temporal tracking
- stroke memory
- real-time occlusion logic
- interface
- CAD
- simulators

The only bridge to this future work is
[`INTEGRATION.md`](INTEGRATION.md), which documents how Codex
should replace the current neural detector with the v1 model, without
Muse implementing that replacement.

## Classes

| Index | Class | Meaning |
|---|---|---|
| 0 | `background` | Everything that is not paper, hand or stroke |
| 1 | `paper` | Visible paper |
| 2 | `hand` | Visible hand |
| 3 | `ink` | Visible stroke |

Label rule, without exception: the segmentation represents what the
camera sees. Hand over stroke = `hand`, never `ink`. The mask
of the complete drawing (`ink_full_mask`, with the stroke including where
the hand hides it) is an optional artifact of synthetic generation,
**not** a training class and **does not** enter the ONNX — it will serve
in the future to evaluate Draw's memory.

## Priority deliverables (mandatory)

1. **Reproducible synthetic generator** — FreiHAND hands + paper +
   backgrounds/tables + strokes + augmentations, with reproducible seeds,
   and splits made **before** generation (original sources divided
   between train/val/test, so that variations of the same hand, paper or
   background do not leak between sets).
2. **Reproducible Kaggle notebook** — SegFormer-B0 with transfer
   learning, RGB 512×512 input, 4-class output.
3. **Complete training configuration** (`training_config.json`).
4. **Split definition** (`split_manifest.json`, with seeds and IDs
   of the sources per set; initial suggestion 80/10/10).
5. **Metrics per class** (`metrics.json`) — IoU per class,
   Dice for paper/hand/ink, precision and recall. Never global accuracy
   alone.
6. **Ink metric with tolerance** — stroke precision/recall
   with spatial tolerance of 2–3 px (dilation, distance transform
   or skeleton), because a thin stroke displaced by a few pixels has
   artificially low IoU even when visually correct.
7. **Evaluation on the real holdout** — small set of real frames
   from Draw, manually annotated, separate, never used in training
   nor in augmentation. Synthetic and real metrics reported
   **separately**; a large drop on real data is a domain gap to
   document, not to hide.
8. **Visual samples** — fixed battery of at least 15 scenarios
   (clean paper, paper with drawing, heavily drawn paper, hand
   covering corner/centre/stroke, hand partially out of frame, light/
   dark background, strong shadow, paper partially out of frame, hand without
   paper, white object without paper, thin pencil stroke, thick
   pen stroke), with predictions saved for comparison between
   versions.
9. **ONNX model** — `draw_segmentation_v1.onnx`, validated against
   the original model (same frames, outputs compared, without
   significant degradation).
10. **ONNX interface documentation** — [`INTEGRATION.md`](INTEGRATION.md):
    input, shape, normalization, outputs, class order,
    thresholds, extraction of each mask and `paper_presence`.
11. **Compatibility documentation with `draw/neural.py`** —
    also in [`INTEGRATION.md`](INTEGRATION.md): conversion of
    `paper` to the current contract (mask → largest contour →
    quadrilateral) and the list of what Codex will need to change later.

## Decision on `paper_presence` (summary)

Documented in detail in `INTEGRATION.md`, section 4:

**`paper_presence` is derived from the `paper` mask** — fraction of
paper pixels ≥ 0.01 **and** a quadrilateral that closes. **There is no auxiliary
presence head in ONNX v1.** Reason: a single source of
truth, with no risk of presence and mask disagreeing. The
compatibility with the old contract does not justify a second
signal: **the model is not simplified to keep the old contract;
the contract is adapted to the model.**

## Synthetic dataset (size rule)

It is not mandatory to deliver the entire generated dataset if that
exceeds reasonable storage limits. In that case,
deliver generator + seeds + configuration + manifest + representative
sample, so that the complete dataset can be
regenerated.

## Data and licences (to check before training)

- **FreiHAND** (Kaggle, `danieldelro/freihand`): check before
  the full pipeline whether it contains usable RGB images and
  real hand segmentation masks, format, quantity,
  resolution, licence and terms. If it does not contain suitable masks,
  **do not invent labels**: document the problem and use an
  appropriate alternative source, recording licence and origin.
- **Real Draw frames**: provided by Codex, manually
  annotated, only for holdout. Muse does not capture camera.
- Mandatory negatives in the synthetic set: empty table, hand without paper,
  white objects, light rectangles, book, box, monitor,
  light surfaces, very small paper, paper practically
  out of frame — so that the model does not learn that "white or rectangular
  = paper".

## Files in this folder

| File | Contents |
|---|---|
| `README.md` | This document: scope, deliverables, what is out |
| `INTEGRATION.md` | ONNX v1 interface and compatibility with `draw/neural.py` |

The model files (`draw_segmentation_v1.onnx`,
`classes.json`, `metrics.json`, `training_config.json`,
`split_manifest.json`) are delivered by the data and
training agents, not by this document.

## Status

Interface documentation created, with no code changed, no
commit, no push, no merge. The v1 model itself, the generator, the
notebook and the metrics are delivered by the other agents of this
project, in this same branch (`feature/neural-seg-v1`).
