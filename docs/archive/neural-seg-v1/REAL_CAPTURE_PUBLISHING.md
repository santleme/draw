# Publishing real capture batches

## Standard

- The frames and archive files (`.tar`) stay on Kaggle in a
  **private** dataset. Git keeps only `frames.csv`, `takes.csv`, the batch
  manifest and the checksums of the published files.
- Do not use `git add -f captures/` to circumvent the `.gitignore`. The fact that
  `captures/` is ignored is intentional.
- Group the JPEGs into `.tar` files of approximately 200 MB, without
  recompressing the images. Prefer closing each part at the boundary between takes;
  record in the manifest which takes and frames are in each part. If a take
  alone exceeds the target size, split it and record the frame intervals.
- Keep parts numbered and reproducible (`part-000.tar`, `part-001.tar`,
  …). Do not include Kaggle credentials, configuration files with tokens or
  loose images in the commit.

## Preparing and checking a batch

1. Preserve the captured session and confirm that `frames.csv` and `takes.csv`
   cover all JPEGs. Do not rename the frames after generating the CSVs.
2. Assemble the parts by take, without compression. Before publishing, check that
   each file can be listed/extracted and generate SHA-256 for each part.
3. Create or update the Kaggle dataset as **private**. Use the division into parts
   of ~200 MB that already worked for the synthetic batches. Keep the slug and the
   dataset version in the batch manifest.
4. After upload, check on Kaggle the `ready` state, the list and the size
   of the parts. Download at least one part and compare its SHA-256 with the local one.
5. Commit the CSVs, the manifest and the checksums. Before the commit,
   confirm with `git status --short` that no JPEG or `.tar` will be included.

The manifest must record at least: `batch_id`, local session path,
date/time, camera, requested and observed resolution/FPS when available,
frame count, complete and partial takes, frame intervals per part,
bytes and SHA-256 of each part, Kaggle slug/version/visibility and any
gaps or capture failures. Use relative paths and never record tokens.

## Fallback: Git

Use this path only when private storage is not available
or when the user explicitly asks for Git. Group JPEGs into `.tar` without
compression, keeping each file below 90 MiB to stay with margin under
GitHub's per-file limit. Commit and push every 10 takes,
recording the take interval in each commit. Check the push before
starting the next group. Git remains public if the repository is public; do not send
personal data without explicit authorisation.

SSH may be chosen for new pushes after confirming that the key is
configured and that `git ls-remote git@github.com:GUZZBR1/draw.git` works. Do not
change the protocol or restart a large push that is already progressing: a
new attempt needs to transfer the objects again.

## Status of the first batch

Batch `batch_01` was committed as individual JPEG frames before this
procedure. Do not rewrite its history or replicate this method in subsequent
batches. The transfer of this batch must be verified by the remote SHA before
being considered complete.
