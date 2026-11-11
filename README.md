<p align="center">
  <img src="docs/icon.svg" width="160" alt="Draw">
</p>

<h1 align="center">Draw</h1>

<p align="center">
  A digitizing tablet without a digitizing tablet.
</p>

A webcam points at the desk. You write and draw on an ordinary sheet of
paper, and Draw reproduces the stroke digitally, live. No special pen,
no special surface, no printed template.

Draw is the capture layer of a larger vision — a real-time bridge from
paper sketches to professional tools such as CAD and simulators — set
out in [VISION.md](VISION.md). This repository is that capture layer:
sheet detection, page rectification, and ink extraction, in real time
on a commodity CPU.

## Status

Draw runs in real time in daily use on the author's desk: one command,
neural sheet detection and the page-clean ink model by default, on any
sheet or notebook page.

The open problems are capture reliability and generalization. Current
models are judged on the author's setup and on public document datasets;
a sealed test set from other desks, cameras, and lighting is the missing
ruler. The current failure map and prioritized evidence gaps are in
[docs/ESTADO_ATUAL.md](docs/ESTADO_ATUAL.md) and
[docs/FALHAS.md](docs/FALHAS.md). Numbers in this README are from the
author's setup unless stated otherwise.

## Quick start

Prerequisites: Python 3.11 or 3.12, a webcam, and the project
dependencies.

```bash
pip install -r requirements.txt
python tools/fetch_models.py   # one-time: downloads the official models
python main.py
```

Model binaries do not live in git. They are distributed by this
repository's `models-2026-11-06` Release and verified against
`models/MANIFEST.json` (sha256 and size) by the fetch script. MediaPipe
is an optional extra, used for hand tracking:

```bash
pip install mediapipe
```

### Windows with WSL

WSL2 does not see the Windows webcam until it is attached with usbipd.
The complete guide is [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md);
the short path, from inside WSL:

```bash
bash scripts/setup_camera_wsl.sh
python main.py
```

If anything fails, diagnose before changing anything else:

```bash
python main.py --check
```

`--check` validates OpenCV, the video device, its permissions, and a
real frame read, naming each failure and its fix.

### Linux with a USB camera

```bash
python main.py
```

## Usage

With the models fetched, `python main.py` is the whole interface: the
neural detector and the page-clean ink model are the defaults, and the
only key you need is `q` (or `Esc`) to quit. The flags below exist for
diagnostics, experiments, and custom models.

| Flag | What it does |
|---|---|
| `--source N` or URL | Local webcam by index (default 0) or MJPEG stream by URL |
| `--format a4\|a5\|a3\|carta\|caderno` | Portrait paper format, fixed for this session (`--format` > `DRAW_FORMAT` > `a4`) |
| `--neural [model.onnx]` | Neural sheet detector with a custom model. Default: `models/draw_seg_v3_dyn.onnx` when present, else `DRAW_NEURAL_MODEL` |
| `--page-clean [model.onnx]` | Page-clean ink model. Default: `DRAW_PAGE_CLEAN_MODEL` or `models/page_clean_unet.onnx` |
| `--legacy-ink` | The pre-neural ink stack (adaptive threshold heuristic, `--ink-model`, `--session-baseline`, `--hand-model`) instead of the page-clean model |
| `--ink-model [model.onnx]` | Learned ink model inside the legacy stack, replacing the heuristic |
| `--hand-model [model.task]` | MediaPipe hand model for the legacy stack. Default: `DRAW_HAND_MODEL` or `models/hand_landmarker.task` |
| `--session-baseline` | Legacy stack only: the page as it is when you press `c` becomes the background, so only strokes made afterwards enter the canvas |
| `--markers` | Printed template sheet with ArUco markers, instead of markerless detection |
| `--check` | Validate the camera step by step and exit |
| `--save-dir folder` | Captures folder (default `captures`) |

Keys inside the app:

- `s` saves the digital canvas as PNG in `captures/`
- `f` saves the current raw frame in `captures/frames/`
- `g` toggles automatic raw-frame capture (one frame every 10)
- `c` sets a new session baseline (legacy stack, with `--session-baseline`)
- `q` or `Esc` exits

The default remains the incumbent **850×1200 A4** grid. Other formats
use a 1200-pixel long edge and `round(1200 * width_mm / height_mm)`:
A5 846×1200, A3 849×1200, Carta 929×1200, Caderno 873×1200.
Only A4 is pinned; its legacy ratio/isotropy error is an explicit exception,
not a relaxed tolerance for other formats. Pixel thresholds are unchanged:
vertical density is about 4.04 px/mm for A4 and 4.30 px/mm for Carta (+6.5%).

After 15 detected sheet quads, a one-time suggestion uses their
corner-wise median and a pinhole homography (centered principal point,
square pixels, zero skew). It reports a ratio class or abstains; it never
changes the chosen format or estimates physical A-series size. Marker
centers are not sheet-boundary quads, so marker mode explicitly abstains.
Startup telemetry records format, grid and mm/px; `format_suggestion`
is a separate event, including abstention reasons.

For metric SVG/DXF export, pass the same format to the vector bridge:

```bash
python main.py --format carta
python tools/export_vectors.py canvas.png --format carta --out-dir vector-out
```

The format-aware bridge reads its uniform mm/px from `draw.config`.
Explicit `--paper-mm W H` remains a mutually exclusive legacy custom-size
override. `tools/vectorize.py` is legacy and still has its own A4 constants;
it is not format-aware and is unchanged.

Environment variables set model paths (`DRAW_NEURAL_MODEL`,
`DRAW_PAGE_CLEAN_MODEL`, `DRAW_INK_MODEL`, `DRAW_HAND_MODEL`) and
thread caps for the ONNX sessions and OpenCV. Model verification is strict by
default: a model whose sha256/size differs from `models/MANIFEST.json`,
or that has no entry there, is refused. `DRAW_MODEL_STRICT=0` or
`--allow-unverified` downgrades that to a warning.

Every run writes a session report (rates, pass timings, worker states)
as JSON under `<save-dir>/session-reports/`, updated every 15 s and on
exit.

## How it works

Per frame: the camera delivers a frame; a detector finds the sheet and
returns its four corners; a homography rectifies the page to a top view
of 850 × 1200 px by default (other formats use the grid above); the ink
stage separates stroke from paper in that view and updates the digital
canvas. Detection and ink run on worker threads (latest-wins); the main
loop never waits for a model pass.

### Sheet detection

The primary detector is a SegFormer-B0 exported to ONNX, run at
384 × 384. It emits a presence logit and a paper mask; the sheet
quadrilateral comes from the mask's largest contour, with each edge
refit as a line and subpixel corner refinement. An edge memory holds
the last reliable line of each side, so a hand covering an edge does
not move the corner. A classic OpenCV detector (geometry candidates
plus a texture gate) is the per-frame fallback when the model sees no
sheet, and remains the detector on a clone without the models.

### Ink

The default ink model is a lightweight U-Net (122 K parameters) that
maps the rectified page to an ink probability map in 256 × 256 tiles.
It suppresses much of the ruling, paper texture, and shadow, but it
does not reliably distinguish handwriting from printed content. New ink
appears immediately as a provisional layer; the confirmed canvas is
what a multi-pass vote ratifies, with a frontier gate that keeps
sub-threshold borders from accumulating into strokes no single pass
saw. With a hand-presence signal, the ink worker rests at zero passes
once the confirmed output is quiet and the hand is absent or still.
Without that signal, it slows to a 1 Hz probe instead. Notebook movement
re-registers geometry before voting, but canvas invariance during movement
remains an open failure (F-07/F-09 in `docs/FALHAS.md`).

Hand presence comes from MediaPipe landmarks, sampled asynchronously.
In the page-clean path, presence controls cadence and rest behavior.
When the hand model and MediaPipe are available, a per-pass hand mask
also marks covered pixels as unknown evidence: they do not vote,
provisional ink is preserved under the hand, and page identity uses
only visible reference ink (or defers when too little is visible).
The model still receives the unmasked warp; visibility follows the
same registration as its probabilities. Without the mask, this
protection is unavailable; real occlusion reliability remains unproven.

### Fallbacks

Without the model files, Draw still runs: classic sheet detection and
the legacy ink stack. That mode exists so a fresh clone is never a
dead end; it is not the product path, and startup says which mode is
active.

## Performance

Measured on the author's machine (Ryzen 7 7700X, no GPU), live
sessions, November 2026:

| Stage | Cost |
|---|---|
| Sheet detection (ONNX, 384 × 384, 3 threads) | ~42 ms per pass |
| Ink pass (full page, 4 threads) | ~105 ms per pass |
| Main loop | ~12 fps (camera-bound: the webcam delivers 13–21 fps) |
| Total CPU while drawing, with the combined thread caps | ~290% |

The thread caps (detector 3, ink 4, OpenCV 4) are defaults chosen by
measurement: uncapped pools reached ~550% CPU while drawing for the
same pass times.

## Models and training

Training runs through versioned GPU/TPU adapters launched from Kaggle
or Colab. The author's real desk frames live in private datasets and
never enter this repository.

- `training/run.py` routes CUDA/CPU to `training/train.py`, XLA to
  `training/train_tpu.py`, and TorchTPU to `training/run_t7.py`.
  Thin launchers live in `training/launchers/`. Detector data comes from
  public document datasets (MIDV-2020, SmartDoc, school-notebooks) plus the
  author's labeled desk frames.
- A model is judged before it is promoted: the gate is the sealed real
  replay and the live desk test, never a synthetic score alone.
  `tools/eval_real_frames.py` is the entry point for judging a
  sheet-detector candidate.
- `draw/selflearn/` holds the offline learning loop (session
  collection, selection, teacher, promotion gate) that prepares
  training rounds between sessions. The app itself only infers;
  nothing trains at runtime.

## Repository layout

This is **not a formal monorepo**: there is no workspace tooling, no
per-package versioning and no `apps/` + `packages/` split. It is one
product whose three systems share a single history and a single CI,
each already isolated in its own folder with its own stack and build:

| System | Where | Stack | Build / entry point |
|---|---|---|---|
| Desktop app (the product) | `draw/` + `main.py` | Python, OpenCV, ONNX | `python main.py` |
| Website | `site/` | React, TypeScript, Vite | `npm run build` inside `site/` |
| Training pipeline | `training/` | Python, PyTorch | `python training/run.py`; Kaggle/Colab launchers inside |

Everything else in the tree supports one of those three. The CI
(`.github/workflows/ci.yml`) already treats them as separate areas:
path filters run the app suite, the training contracts and the site
build only when their own area changes.

- `draw/` — the application: `app.py` (main loop), `capture.py`
  (camera and asynchronous I/O), `sheet_track.py` (sheet tracking),
  `hand_presence.py` (presence worker), `ink_layers.py` (ink worker,
  registration and layered canvas), `neural.py`
  (sheet detector), `paper.py` (classic detector and homography),
  `page_clean.py` (ink model), `ink.py` (legacy ink stack),
  `hand.py` (hand tracking), `view.py` (window composition),
  `session_report.py`, `telemetry.py`, `manifest.py`, `config.py`,
  and `selflearn/` (offline learning loop).
- `models/` — `MANIFEST.json`, the integrity record of the model
  binaries. Binaries themselves are Release assets, not git content.
- `tools/` — operator tools: `fetch_models.py`, `eval_real_frames.py`,
  `label_frames.py`, `judge.py`, and the experiment runner in
  `tools/runner/`. The full index, grouped by job, is
  [tools/README.md](tools/README.md).
- `training/` — versioned training bodies, configs and Kaggle/Colab
  launchers; `notebooks/` retains the earlier Kaggle detector notebook.
  [The training guide](docs/TREINO.md) covers dataset integrity, the
  U-Net holdout contract, ONNX delivery gates, and same-checkpoint
  384/256 evaluation.
- `scripts/` — `setup_camera_wsl.sh`, the WSL camera installer.
- `template/` — generator of the printed sheet used by `--markers`.
- `bridge/` — the Windows helper protocol (`PROTOCOL.md`,
  `windows_helper.py`; receiver in `tools/helper_receiver.py`).
  Experimental; the app does not use it yet.
- `tests/` — the webcam-free suite. `pytest` runs everything,
  including `tools/runner/tests/`.
- `site/` — the project website (React/TypeScript). Its design rules
  are in `design.md` and `AGENTS.md`.
- `docs/` — [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md) and
  [docs/TREINO.md](docs/TREINO.md), the living documents besides
  this README and [VISION.md](VISION.md), plus dated records kept as
  evidence. The index is [docs/README.md](docs/README.md).
- `examples/` — sample vector-bridge outputs (SVG, DXF and the proof
  JSON), kept as reference outputs of `tools/export_vectors.py`.

Root documents, each with one job: [VISION.md](VISION.md) — the
larger vision this capture layer serves;
[SECURITY_MODEL.md](SECURITY_MODEL.md) — the trust zones for code
that touches private captures; `design.md` — the website's design
source of truth; `AGENTS.md` — repository guidance for automated
contributors. `.replit` is the website's static deployment config:
it builds `site/` and serves `site/dist`.

## Development

```bash
pip install -r requirements-dev.txt   # pytest, ruff, onnx (fixtures)
pytest                                 # full suite
python tests/selftest.py               # synthetic-scene pipeline check
```

CI (`.github/workflows/ci.yml`) always runs ruff and lightweight contracts
on pushes/PRs (Python 3.12); path filters select the full suite/selftest,
CPU training contracts and site build. Nightly/manual runs check the full
suite and selftest on Python 3.11 and 3.12, plus a report-only synthetic
re-registration replay.

## Limitations

- Generalization is unproven: no sealed test set from other setups
  exists yet, so today's numbers describe one desk, one camera, one
  light.
- Fine strokes (1–3 px) are the first casualty of rectification
  jitter; the confirmed canvas reduces the damage but does not remove
  the cause.
- To the classic fallback detector, a blank smooth sheet is
  indistinguishable from any white smooth rectangle. The neural
  detector is what separates them, which is why it is the default.

## Roadmap

Done: real-time markerless capture on the author's desk; neural sheet
detection; page-clean ink with a confirmed canvas; hand-presence-aware
worker cadence; combined thread caps and session reports. Page identity,
occlusion handling, erasure, and cross-desk reliability remain open.

In progress: the external test set for generalization; ink quality — a
three-class ink/print candidate intended to separate handwriting from
printed content. No such candidate is promoted to the production path.

Planned: vector strokes (order, path, smoothing); interpretation of
shapes, letters, and diagrams; integration with a first target system
(CAD or simulator); real-time simulation feedback. See
[VISION.md](VISION.md).
