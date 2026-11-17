# FreiHAND on Kaggle — verification for the Draw segmentation model

Verification date: 2026-10-29. Verified with Kaggle CLI 2.2.5
(`.venv/bin/kaggle`), without downloading the whole dataset: paginated
listing (`datasets files`, pages of 200 files, page token)
+ sample download of 3 files to `/tmp/freihand-sample`
(2 RGB + 1 mask), + official metadata (`datasets metadata`).
The Kaggle token is NOT in this document and was not displayed in any
step. Scope of this agent: only this report and `tools/check_freihand.py`.
No commit, push, merge, webcam, app, or integration.

## Verified dataset

- Kaggle: `danieldelro/freihand` — "FreiHAND - Hand Pose, Shape and
  Mask Dataset"
- Total size reported: 3,885,249,922 bytes (~3.88 GB)
- Last update on Kaggle: 2022-03-03
- Downloads: 1,756 · votes: 27 · usability: 0.9375 · views: 9,211
- Origin: University of Freiburg, LMB group (Zimmermann et al.,
  ICCV 2019, "FreiHAND: A Dataset for Markerless Capture of Hand Pose
  and Shape from Single RGB Images").

## Direct answers (verified facts, not assumptions)

### Are there usable RGB images? YES
- `evaluation/rgb/00000000.jpg` and `training/rgb/00000000.jpg`
  downloaded and read: **224 × 224 × 3, JPEG, uint8**, real color
  (different channels, distinct per-channel means).
- Count verified by listing: `evaluation/rgb` = **3,960 files**,
  exactly the number in the official description for the evaluation split.
- Training, per the official description: 32,560 unique samples, each
  processed in 4 ways to remove the green background → 130,240 training
  samples. The full listing of the training split was still in
  progress when this report was closed (12,200+ files listed without
  error); the training numbers above are those from the official
  description, not a final count of my own — revalidate with
  `tools/check_freihand.py` on the complete local copy before
  generating the synthetic dataset.

### Are there real hand segmentation masks? YES — only in training
- `training/mask/00000000.jpg` downloaded and read: **224 × 224, JPEG,
  gray replicated across the 3 channels** (identical channels), with
  essentially binary values.
- Measured sample: 91.39% of pixels = 0 (background), 7.60% of pixels
  > 127 (hand). Intermediate values exist only in small quantity
  (1–7 and 248–254), typical of JPEG compression artifacts
  on the hand edge. **Usage rule: binarize with threshold 127**
  (`mask = pixel > 127`); do NOT use the raw JPEG values as a
  multi-class label.
- Partial count verified: **8,038 files in `training/mask`**
  already listed in the first 12,200 entries, before the end of the listing.
- Evaluation: **there is no mask in the evaluation split in this copy**
  (only `evaluation/rgb` in the listed entries). This matches the
  original description — the evaluation annotations were held back for
  the evaluation server — even though the project page mentions
  a later release of the evaluation annotations on the original site.
  Practical consequence: the FreiHAND evaluation split serves for
  visual inspection, NOT for hand metrics with ground truth.

### Mask format
- File: JPEG grayscale (3 identical channels), 224 × 224, 1 mask per
  training RGB image with the **same file name**
  (`training/rgb/NNNNNNNN.jpg` ↔ `training/mask/NNNNNNNN.jpg`).
- It is NOT a mask generated from a mesh in this copy: it is the
  segmentation mask supplied by the dataset. There are not, in the entries
  listed so far, any 3D keypoint, K matrix, scale, or mesh files
  (only `.jpg` and 2 `.json` files at the root, not yet identified —
  check before promising keypoints/mesh from this copy).
  Keypoints/pose/mesh are NOT needed for the Draw pipeline
  (only RGB + mask), so this absence does not block the work.

### Resolution
- RGB and mask: **224 × 224 pixels** (confirmed in the sample and in the
  official description). For the target model (512 × 512 input), the hands
  will be resampled in the synthetic composition — record the
  rescaling in the generator, do not assume 224 as the final input.

### License and terms — READ BEFORE USING IN A PRODUCT
- License on Kaggle: **CC BY-NC 4.0** (Attribution–NonCommercial 4.0
  International).
- Terms of the original dataset, in the official description: **"for research
  purposes only", no warranty, commercial use prohibited, and mandatory
  citation** of the paper (Zimmermann et al., ICCV 2019 — full
  reference in the dataset description).
- Direct implication for Draw: training with FreiHAND is suitable
  for this research stage, with attribution. **If Draw becomes a
  commercial product, the use of a model trained with FreiHAND is a
  license risk to be resolved beforehand (change the hand source or
  obtain authorization)** — this report does not resolve that legal
  question, it only records it. Do not publish redistributed FreiHAND
  images without checking the original terms.
- Synthetic compositions that cut out FreiHAND hands remain
  derivatives of the dataset: same license, same restriction, attribution
  kept in the synthetic dataset manifest.

## Decision

**USE FreiHAND DIRECTLY, only for the hand mask in training.**
No label invention is needed: the real mask exists,
is paired by name with the RGB, and is binarizable with threshold 127.

- Use: `training/rgb` + `training/mask` (training) as raw material
  for the hands in the synthetic scenes.
- Do not use as a metric: `evaluation/rgb` (no mask in this copy).
- Do not promise from this copy: 3D keypoints, K matrix, scale
  and mesh — check the 2 root `.json` files before any use, and,
  if a mesh is needed in the future, seek the original Freiburg source
  (same license/terms), not this Kaggle copy.
- Recorded alternative (only if the direct mask fails on some
  cutout): render a mask from the 3D mesh of the original FreiHAND,
  or use EgoHands (pixel-by-pixel hand masks, research
  use — check the license before downloading). Neither is
  needed now.

## Mandatory rules for the synthetic generator (derived from this)

1. **Split by base hand, never by file.** The original training set has
   32,560 unique hands × 4 background variants. Splitting the files
   randomly would put the same hand (with a different background) in training
   and test. Separate the sources (base hands) first, then generate
   variations per split (80/10/10 suggested in the scope).
2. **Indices restart per split.** `00000000.jpg` exists in training
   and in evaluation and is NOT the same hand. Do not use file name
   as a leak key between splits.
3. **Mask always with threshold 127** after reading the JPEG; store the
   binarized mask in the synthetic dataset, not the raw JPEG.
4. **Scope occlusion rule:** in the composition, where the hand covers
   paper or stroke, the final label is `hand` (what the camera sees).
   Store `ink_full_mask` separately, outside training, only to evaluate
   future memory.
5. **Manifest:** record source (`danieldelro/freihand`, Kaggle),
   license (CC BY-NC 4.0 + research only), ICCV 2019 citation,
   seeds and base hand IDs per split.

## How to reproduce this verification

```bash
# Listagem (sem baixar tudo)
.venv/bin/kaggle datasets files danieldelro/freihand --page-size 1000 --format csv
# Amostra (3 arquivos, /tmp, nunca o dataset inteiro)
.venv/bin/kaggle datasets download danieldelro/freihand -f training/rgb/00000000.jpg --unzip -p /tmp/freihand-sample
.venv/bin/kaggle datasets download danieldelro/freihand -f training/mask/00000000.jpg --unzip -p /tmp/freihand-sample
.venv/bin/kaggle datasets download danieldelro/freihand -f evaluation/rgb/00000000.jpg --unzip -p /tmp/freihand-sample
# Checagem local (não baixa nada)
.venv/bin/python tools/check_freihand.py /tmp/freihand-sample
.venv/bin/python tools/check_freihand.py <raiz-completa-do-freihand>
```

Checker output on the sample (3 files): structure OK, training pairs
1/1, RGB 224×224×3, mask 224×224 with 7.60% coverage
after threshold 127.
