# Real-frame masks for training

## Scope and split

The first real-data tranche is the 719 frames marked `keep_productive` in
`captures/triage/classified_all_candidates.csv`. The 41 frames marked
`exclude_clear_duplicate` are not included. Original captures remain immutable.

The retained frames come from two sessions: 297 at 640×480 and 422 at
1280×720. Per the user's request to train on these captures, both complete
sessions belong to the real training split. Keep session membership intact;
use the synthetic validation/test sets for model selection and reserve a new
real session for an independent real holdout. Do not describe these two
training sessions as a real test set.

The generated package at `captures/triage/pseudo_training_candidate/` is an
**experimental training candidate**, not an accepted ground-truth dataset.
All 719 source/mask pairs received contact-sheet visual triage and 21 clear
large-region errors were corrected. Fine ink boundaries were not reviewed at
full resolution, so the masks remain pseudo-labels. The trainer can consume
them, but any model trained with them must be treated as exploratory.
Structural trainability does not establish semantic correctness.

## Pixel-label contract

Each mask is a single-channel PNG at the source image's original dimensions:

| Value | Meaning |
|---:|---|
| 0 | Visible background, including the pen and other non-hand objects |
| 1 | Visible paper, including printed text |
| 2 | Visible hand; this overrides ink when a hand covers a stroke |
| 3 | Clearly visible user-added ink only |
| 255 | Deliberately unknown pixel, excluded by training loss and metrics |

The trainer now ignores `255` in cross-entropy, Dice loss, pixel counts, and
evaluation. Never map unknown pixels to background or paper. Unknown areas
must remain limited to genuinely ambiguous boundaries, faint marks, and
printed-versus-handwritten regions that cannot be adjudicated. A mask with no
supervised pixels is not a usable example.

Image-level tags such as `user_added_ink` only guide mask proposals; they do
not define pixel labels. SAM/OpenCV/color-threshold/interpolated masks are
drafts until each frame's overlay is checked against its original image. The
current package records thumbnail review separately from pixel-label approval.

## Required outputs and acceptance

Keep the experimental candidate under ignored
`captures/triage/pseudo_training_candidate/`. A fully reviewed release belongs under
ignored `captures/triage/training_ready/`. Each record must link the session/frame ID, original path
and SHA-256, mask path and SHA-256, original width/height, split, annotation
method/version, reviewer status, and per-class/unknown pixel counts.

The tranche is **reviewed training-ready** only when:

1. All 719 retained IDs have exactly one decodable source image and one mask.
2. Image and mask dimensions match; masks are one channel and contain only
   `0, 1, 2, 3, 255`.
3. Every mask overlay has been visually reviewed; all ink-positive, printed
   text, hand-over-ink, and uncertain-boundary cases receive targeted review.
4. No mask is entirely ignored; class support for paper, hand, and ink is
   reported from supervised pixels. Unknown-pixel counts are included in the
   manifest and metrics.
5. The trainer smoke path passes with `255`, and a dataset preflight confirms
   that every pair is accepted before a GPU training run.

The current pseudo-label candidate passes structural preflight and is accepted
by `training/train.py`, including its ignore-index behavior. It passed
thumbnail-level visual triage, but not full-resolution pixel review; use it
only for exploratory training, not for quality claims or model release.

These are pixel masks, not just image-level frame categories. Passing the
data gate makes the frames usable as training input; it does not establish
real-world model accuracy. Report real training and synthetic validation
results separately.
