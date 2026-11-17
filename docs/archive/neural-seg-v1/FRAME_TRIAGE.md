# Frame triage

`tools/classify_frames.py` scans the two original capture sessions and writes
local CSV candidates for review. It does not edit source images, create pixel
masks, or establish ground truth. The queue's copied midpoint images are
excluded because the scanner reads only the two source session directories.

## Fast OpenCV pass

From the repository root:

```bash
python3 tools/classify_frames.py
```

Outputs are written under ignored `captures/triage/`:

- `frames.csv`: one row per scanned source frame, including source identity,
  take and timestamp when present, decode status, brightness, blur proxy,
  rough paper-pixel fraction, and adjacent perceptual-hash distance. The
  slower existing sheet detector is optional (`--sheet-detector`).
- `takes.csv`: per `(session_id, batch_id, take_id)` counts, semantic category
  counts, duplicate/quality counts, and a provisional priority. Batch ID is
  included because take numbers repeat across capture batches.

`higher_priority_candidate`, `lower_priority_near_duplicate`, and
`lower_priority_candidate` are review queues, not accept/reject decisions.
Paper can be useful even when blank, blurred frames can capture a hand or
occlusion event, and a frame without paper may be useful as a negative example.
The `quality_candidate` field ranks sharpness relative to the same session;
it flags the bottom 10% for review rather than applying an absolute rejection
threshold. When enabled, the existing sheet detector is another candidate
signal; its thresholds were not calibrated across these two complete sessions.
The simple white/low-saturation pixel fraction can mistake glare, keyboard
keys, or other pale objects for paper.

For a small benchmark, limit each session:

```bash
python3 tools/classify_frames.py --limit 200 \
  --output captures/triage/sample_frames.csv \
  --takes-output captures/triage/sample_takes.csv
```

## Optional semantic candidates

The optional [OpenCLIP](https://github.com/mlfoundations/open_clip) path scores fixed text prompts for user-made marks,
hands on/away from the paper, blank/printed paper, pens, no-paper background,
and poor quality:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install torch open_clip_torch opencv-contrib-python-headless
python3 tools/classify_frames.py --clip
```

The first run downloads model weights. Similarity scores depend on the prompt
set and are not calibrated probabilities. OpenCLIP sees the whole frame, so it
can confuse a hand on the keyboard with a hand over paper, or user drawing
with printed/faint marks. Keep scores for ranking; a person must verify the
category. The output includes the top score and its margin over the runner-up;
small margins indicate especially uncertain rankings. Text printed on the
sheet remains paper, and a pen is background.

In the first full run, all 91 takes were surfaced as productive candidates;
the model did not find a confidently empty take to discard. The predicted
frame categories were 9,736 paper-with-user-marks candidates, 2,070 printed
paper, 1,601 hands away from paper, 1,075 blank paper, 820 pens without hands,
349 hands over paper, and 194 poor-quality frames. No frame's top category was
no-paper background. Also, 13,294 frames were flagged as adjacent near-
duplicate candidates by pHash; this is a review cue, not safe automatic
deletion, because small drawing changes can happen between neighboring frames.
The top-versus-runner-up similarity margin was below 0.02 for 10,563 frames,
so many rankings are ambiguous. These are model rankings, not verified labels.
The older 640×480 session showed more confusion between faint marks, blank
paper, and hands than the newer session, so keep session identity in every
review and do not use the predictions as training masks.

[CleanVision](https://cleanvision.readthedocs.io/en/stable/tutorials/tutorial.html)
is another existing option for technical image-quality flags and duplicate
discovery. It does not answer whether a take shows paper, a hand, or user-added
drawing, so it is not required by this scanner.

## Current speed evidence

The full OpenCLIP pass processed 15,845 unique frames in 845 seconds (about
18.8 frames/second) on the local CPU. This excludes the first-time package and
model download. It took about 14 minutes after setup, comfortably inside a
five-hour processing window on this machine. Manual visual review and dense
mask annotation are separate work and are not completed by this scan.

## Compact review selection

After the full scan, build a review list without changing or deleting the
original images:

```bash
python3 tools/select_review_frames.py
```

The selector writes four local files under ignored `captures/triage/`:

- `selection_manifest.csv`: one row for every source frame, with checksum,
  nearest selected frame, visual distance, reason, and a disposition.
- `review_queue.csv`: up to 650 first-pass review candidates.
- `overflow_review_queue.csv`: additional candidates that did not fit in the
  first-pass limit; they remain recorded and are not discarded.
- `selection_summary.csv`: counts and selection reasons by take or continuous
  two-second review segment.

It keeps first/middle/last anchors per take (or per two-second segment in the
older continuous recording), adds semantic-category candidates, and looks for
visually distinct frames relative to the last retained representative. For
short `hand_over_paper` events, it picks at most one novelty candidate per
half-second window, requiring pHash distance 8 or greater. The limit of 650 is
a first-pass workload target, not a claim that later candidates are useless.
The overflow stays available for a second pass. Low-confidence model rankings
remain flagged for human review.

The selector output is a review plan, not a training dataset: `pending` means
that a person still needs to inspect the frame. It does not create masks or
verify class labels. Frames outside the queue remain in the source folders and
in the manifest; “not selected” means only that they were left out of this
compact review pass. There is deliberately no option here to delete source
images or remote copies.

To force a known difficult example into the list, use repeatable
`--protect-frame SESSION:INDEX:REASON` arguments. For example:

```bash
python3 tools/select_review_frames.py \
  --protect-frame session_20261029_151126_089557:2666:hand_pen_over_paper
```

The full manifest retains source IDs, hashes, and reasons so reviewers can
trace every selected or unselected frame back to the original capture.

To prepare parallel visual review, generate balanced batches that keep every
take or review segment intact:

```bash
python3 tools/make_review_sheets.py --parts 3 --frames-per-sheet 16
```

This creates contact sheets and blank label templates under ignored
`captures/triage/review_batches/`. Reviewers record visible image-level tags,
confidence, and a decision in `labels_reviewed.csv`. The tags describe what is
visible in a frame; they are not pixel masks and do not make an image
training-ready. Unclear cases stay `uncertain`, and model suggestions remain
unverified.

The first parallel review completed all 650 primary candidates. The merged
`captures/triage/classified_primary_queue.csv` contains 609 `keep_productive`
and 41 `exclude_clear_duplicate` decisions; the excluded entries are only
filtered from candidate use and their source images remain untouched. Reviewers
tagged 557 frames with visible user-added ink; those tags are image-level
review results, not verified pixel masks. Structural validation confirmed all
650 source IDs and same-group duplicate references.

The second tier contains another 110 overflow candidates. After reviewing both
tiers, `captures/triage/classified_all_candidates.csv` contains 760 unique
reviewed frames: 719 `keep_productive` and 41 `exclude_clear_duplicate`. All
source frames remain intact. The combined 663 `user_added_ink` tags are visual
image-level judgments, not dense masks; no candidate is training-ready based
on this review alone.

Latest full selection run: all 15,845 unique sources were decoded and hashed;
650 are in the first-pass queue and 110 remain in overflow, across 91 takes
and 84 older-session review segments. The half-second rule selected frame
2666 for its hand-over-paper event and suppressed its near-duplicate neighbor
2667 from the queue. Both remain represented in the manifest. All queued items
are still marked `pending`; this is filtering for review, not completed human
analysis or training labels.
