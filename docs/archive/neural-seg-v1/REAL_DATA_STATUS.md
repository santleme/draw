# Status of real data for training

## Status of this proposal

This PR adds tools and instructions for preparing real data for
training the segmenter. It **does not deliver a validated real dataset or a
trained model** and does not claim quality or accuracy improvement.

| Item | Status |
|---|---|
| Selected frames | 719 productive frames from two sessions; 41 clear duplicates were excluded. |
| Visual review | Thumbnail review completed; 21 clear errors in large regions were corrected. |
| Pixel-by-pixel masks | Review incomplete. Fine ink edges and ambiguous cases still need review at full resolution. |
| Labels | The candidate masks are pseudo-labels, not ground truth. `255` pixels represent unknown regions and are ignored by the trainer. |
| Data published in this PR | No original capture or generated mask is included. |
| Model training and result | No training was run; no checkpoint or trained model is delivered. |
| Automated tests | After reconciliation: 29 passed and 2 tests dependent on full PyTorch were skipped, since the local environment has no usable PyTorch. The CPU smoke test and syntax/configuration checks passed; full training and ONNX export remain pending. |

## Permitted use at this stage

The mask package may be used for exploratory experiments. Training
results must be identified as exploratory and kept separate from the
synthetic validation metrics. The two sessions used for training do not constitute an
independent real holdout.

## Before claiming data or model validation

1. Review each image/mask pair at full resolution, including ink,
   printed text, hand over ink and ambiguous boundaries.
2. Confirm integrity of the 719 pairs: IDs, hashes, dimensions, single channel and
   permitted values `{0, 1, 2, 3, 255}`; reject masks without
   supervised pixels.
3. Run the automated tests and the dataset preflight.
4. Train and publish metrics with configuration and data traceability.
5. Reserve a new real session for independent evaluation before making
   performance claims on real data.
