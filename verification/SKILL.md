---
name: verify-geometry-fidelity
description: Formal (bounded) verification of the Draw pipeline's geometric fidelity. Use before and after any change to draw/paper.py, draw/ink.py, draw/config.py or template/generate_template.py, and whenever a change could move, flip or rotate the reproduced drawing.
---

# Verify geometric fidelity (Draw)

## The property

A stroke drawn on the sheet shows up on the digital canvas in the same place it was drawn, whatever tilt or in-plane rotation the sheet has in front of the camera, within the scope below.

The harm when it breaks: the homography in `draw/paper.py`, or the agreement between `draw/config.py` and the printed template, is wrong and everything the user draws is reproduced displaced, mirrored or rotated, with no visible error anywhere.

## Run it

From the repo root, with the project venv:

```bash
.venv/bin/python verification/run_verification.py
```

One run takes about 2 seconds. Exit 0 and a final `VERIFICACAO OK` line mean the property held. Any other result is a finding, not a script error, until you prove otherwise.

Prove the checker itself still works, whenever you touch `verification/`:

```bash
.venv/bin/python verification/run_verification.py --sabotage
```

This must FAIL (exit 1, most probes failing). The sabotage rotates the homography destinations in `config.MARKER_CENTERS` after the sheet is built, which is how a corner-order bug appears in the real code. A run that passes under `--sabotage` means the verification went blind, and nothing it says can be trusted until that is fixed.

## Scope, stated honestly

Python with OpenCV has no code-level formal verifier, so this checks the real code against generated inputs, exhaustively inside a bounded family:

- Synthetic camera scenes of 1600x1200 px
- 12 marker-mode sheet placements (`--markers`): the selftest base pose, that pose with the sheet rotated in plane by 90, 180 and 270 degrees, and 8 convex quads with deterministic jitter up to 70 px (seed 20261028)
- 10 markerless sheet placements (default mode): the base pose, nested pose over a light desk mat, and the same 8 convex quads with deterministic jitter up to 70 px (seed 20261028)
- 9 ink probes (discs of radius 10) on a 3x3 grid in the usable area (198 probes total across the 22 scenes)
- Per probe: recovered centroid within 5 px of the drawn center, and at least 40 percent of the probe ink recovered in a 30 px window

Not covered: real lighting, blur, partial corner/marker occlusion, a real webcam, curved or folded paper, and anything outside the probe grid. A pass covers this scope, not the whole system. Last full run (2026-10-30): all 198 probes passed, worst centroid error 3.05 px (2.43 px in markerless mode). Under `--sabotage`, 176 of 198 probes failed.

## Reading a failure

Failures print in the code's own terms, scene and probe coordinates. Read them this way:

- Probes missing or displaced in every scene, in a rotated or mirrored pattern: the destination order or `MARKER_CENTERS` no longer matches the template. Compare `draw/config.py` with `template/generate_template.py` first.
- Only jitter scenes failing: detection or warp degrades under stronger perspective. Look at `draw/paper.py`.
- Only one probe failing: suspect the marker mask in `draw/ink.py` eating real ink, or a probe that legitimately moved because the usable area changed on purpose.

A modeling error (wrong probe, wrong scene) and a real bug look identical in the output. Settle it by opening the warped image for the failing scene and looking at where the probe actually landed.

If the check uses a TLA+ model in the future, read every result with the pstack `/statechart` skill, which shows each invariant violation as its shortest path in plain words and draws the state chart when the check passes.

## Coverage record

- `draw/paper.py`: `detect_markers`, `warp_paper`, `detect_sheet_with_confidence`, `_order_corners`, `warp_sheet`
- `draw/ink.py`: `extract_ink` (both `clear_markers=True` and default markerless path with `_separable_veto`)
- `draw/config.py`: `MARKER_IDS`, `MARKER_CENTERS`, `PAPER_W`, `PAPER_H`
- `template/generate_template.py`: `build_template` (the printed sheet the marker scenes are built from)
- Sync commit: `e0b8f360413c4b8cf33c8a462531f33234b8f38e` (updated 2026-10-30 on `arena/691b8788-draw` to cover both marker and markerless pipelines)

When code listed above changes after the sync commit, this verification needs another look before its result means anything. Move the sync commit forward only after you confirm the verification describes the code at the new commit, and record the new worst centroid error here.

## Keeping it true

When a change intentionally moves the property (a new sheet layout, new marker positions, a different probe grid), update `run_verification.py` and this file in the same change, and rerun both modes. Never weaken a bound just to make a failing run pass. If the code really broke the property, report the counterexample and stop.

For bringing the verification back in line after changes that did not recheck it, use the pstack `maintain_formal_verification_skill` skill.
