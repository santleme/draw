<p align="center">
  <img src="docs/icon.svg" width="160" alt="Draw">
</p>

<h1 align="center">Draw</h1>

<p align="center">
  A digitizing tablet without a digitizing tablet.
</p>

A webcam points at the desk. The person writes and draws on an ordinary sheet of paper, and Draw reproduces the stroke in digital form, live. No special pen, no special surface, no printed template sheet.

The full vision of the project is in [VISION.md](VISION.md). This document covers the current state, how to run it, and how the system works internally.

## Public website

The website lives in `site/`. To publish it on Replit, choose a **Static**
deployment, use `site/dist` as the public directory, and use the build command
from `.replit`. The site build is independent of the Python training
environment. The workspace Run command remains `python main.py`.

## Status

Draw runs in real time on the author's desk. The everyday detector is the v3 neural model, with the classic fallback detector inside the same process. The digital canvas already accumulates ink between frames and treats the hand as an unknown region.

The honest limit at the moment is generalization. v3 was judged on frames from the author's desk and on public document datasets. It has never been through a test set from another desk, another camera, and another light. That set does not exist yet, and no number in this README measures another setup.

## Quick start

Prerequisites: Python (the author runs 3.12), a webcam, and the project's dependencies.

```bash
pip install -r requirements.txt
python tools/fetch_models.py
```

The second command downloads the official models (sheet detector v3, page-to-clean ink model, and the MediaPipe hand model) into the `models/` folder. The binaries do not live in git: they are distributed by this repository's `models-2026-11-02` Release, and the script checks each file's sha256 and size against `models/MANIFEST.json` before installing. Without this step, Draw runs on the fallbacks: the classic sheet detector and the old ink stack.

### Windows with WSL

WSL2 does not see the Windows webcam by default. No `/dev/video*` exists inside it until the camera is attached.

The complete guide is in [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md). The short path runs from inside WSL.

```bash
bash scripts/setup_camera_wsl.sh
python main.py
```

The script prepares usbipd, attaches the camera to WSL, and checks the result. It also corrects the user's `video` group and turns back on the audio services that it stopped itself.

If something fails, run the diagnosis before touching anything else.

```bash
python main.py --check
```

`--check` validates OpenCV, the existence of `/dev/video*`, the process's permission over the device, and the reading of a real frame. Each failure comes out named, with the fix on the following line.

### Linux with a USB camera

```bash
python main.py
```

### With the neural detector

The official models are distributed by the `models-2026-11-02` Release and registered in `models/MANIFEST.json` (sha256, size, the ruler that approved each one, and the date). A fresh clone downloads everything with a single command, with no credentials:

```bash
python tools/fetch_models.py
python main.py
```

Without the fetch, the app starts on the fallbacks and warns in the log: the classic sheet detector in place of neural v3, and the old ink stack in place of the page-to-clean model. To use your own model, the path still exists:

```bash
python main.py --neural models/draw_seg_v3_dyn.onnx
```

## Usage

| Flag | What it does |
|---|---|
| `--source N` or URL | Local webcam by index (default 0) or MJPEG stream by URL |
| `--neural [model.onnx]` | Neural detector as primary, with the classic one as fallback. Without a path, it reads `DRAW_NEURAL_MODEL` |
| `--ink-model [model.onnx]` | Learned ink model as the source of the mask. Without a path, it reads `DRAW_INK_MODEL`. Without a file or with a load error, the app warns once and continues with the heuristic |
| `--hand-model [model.task]` | MediaPipe hand mask for the accumulative canvas. Without a path, it reads `DRAW_HAND_MODEL` or `models/hand_landmarker.task` |
| `--markers` | Optional mode with the printed template sheet and ArUco markers |
| `--check` | Validates the camera port by port and exits, without opening a window |
| `--save-dir folder` | Captures folder. The default is `captures` |

The variable `DRAW_ORT_THREADS` changes the number of threads of the ONNX session without touching the code. The default is 4, measured on the author's machine.

The hand mask is optional and requires a package outside the standard installation.

```bash
pip install mediapipe
```

Without the package or without the model, the app runs without the mask and warns once.

Keys inside the app:

- `s` saves the digital canvas as PNG in `captures/`
- `f` saves the current raw frame in `captures/frames/`
- `g` turns automatic capture on or off, one raw frame every 10 frames
- `q` or `Esc` exits

The default mode is the markerless one, on any sheet or notebook page. It has been the default since v0.2. The marker mode requires the template sheet generated by `python template/generate_template.py`.


## How it works

The pipeline has four steps per frame. The camera delivers the frame. A detector finds the sheet and returns the four corners. The homography rectifies the sheet to the top view, at 850 by 1200 px. The ink step separates the stroke from the paper in this view and updates the digital canvas.

### Neural detection

The primary detector is a SegFormer-B0 exported to ONNX. It returns two outputs. A presence logit says whether there is paper in the scene. A mask says where the paper is. The sheet's quadrilateral comes from the largest contour of the mask, with an epsilon ladder, refitting each edge as a line, and subpixel refinement at the corners. An edge memory keeps the last reliable line of each side. When the hand covers an edge, the inherited line holds the corner in place instead of projecting the arm's border.

When the model sees no sheet in the frame, the classic detector takes over that frame. When the model sees a sheet and the quadrilateral fitting fails, the mask's minimum-area rectangle enters as a fallback.

The dynamic ONNX accepts any input resolution. The app runs this file at 384 by 384. The original ONNX, of fixed format, imposes its own 512 by 512. The performance section brings the numbers of this trade-off.

### Classic detection

The classic detector is computer vision in OpenCV, in two steps.

The first step proposes candidates by geometry. The frame passes through Canny and the full contour tree. A contour becomes a candidate when it closes a convex quadrilateral, with an area of at least 12% of the frame and an aspect ratio between 0.4 and 2.5. The polygon approximation uses an epsilon ladder from 0.01 to 0.05, because rounded corners and a notebook's spine do not close on a single epsilon. The accepted corners gain subpixel refinement. In the app loop, detection runs spaced out and the corners go through temporal stabilization.

The second step is a texture gate. It measures the shrunken interior of each candidate and returns one of three verdicts.

- `paper` when the interior looks like paper
- `blank_maybe` when the candidate is white enough, but too smooth to assert paper
- rejected when the candidate is not even white enough

The gate measures the white fraction, the internal edge density, and the Laplacian variance. A candidate becomes `paper` when the white fraction passes 0.5, the Laplacian variance passes 30, and one of two texture conditions holds. The edge density needs at least 0.03, or the Laplacian variance needs at least 200. A failed candidate does not end the search. The next contour may still be the sheet.

A `blank_maybe` candidate receives harsher treatment in the app. It needs twice as many consistent detections to lock, 6 instead of 3. While the lock comes from this verdict, the screen shows the low-confidence warning.

The gate exists because of a measured history. The geometry-only version accepted any large convex quadrilateral. In an audit with 10 scenes with no paper at all, it fired on 6. A desk mat, a cardboard box, a monitor, and a white tray became a sheet. The texture gate cut these cases in the measured scenes, with no model at all. The current thresholds are laboratory calibration and need to be redone with real captures.

### Ink and canvas

The default ink path is the page→clean model: a lightweight U-Net (`models/page_clean_unet.onnx`, the w8 model) that takes the rectified sheet and returns the ink mask, separating the stroke from the ruling, the paper texture, and shadow. The binary is distributed via the `models-2026-11-02` Release and registered in `models/MANIFEST.json`; a fresh clone downloads it with `python tools/fetch_models.py`. The model runs on its own thread. New ink first appears as a provisional layer, lighter and on screen only; the confirmed canvas is the union of the masks that the vote across passes confirms, and it is what feeds the cursor and the snapshots. The w8 is the current model, not the ceiling: approved by the lab replay ruler, it remains provisional until a better candidate passes the same ruler and the live test on the author's desk.

Without the model file, or if loading it fails, the app warns pointing to the fetch and falls back to the old ink stack — the same path as the `--legacy-ink` flag. The following paragraphs describe that stack.

An adaptive threshold separates the ink from the paper on the rectified sheet. A separable veto runs alongside. A pixel only becomes ink when it is at least 6 levels below the medians of its entire row and column. The veto kills the notebook's ruling and the paper's texture, which the threshold alone marked as ink.

The old-stack canvas is not a photo of the current frame. It accumulates evidence with hysteresis over a 15-frame window. A pixel enters ink with a visible-sample fraction of 0.45 and leaves with 0.20, with a minimum of 4 visible samples. The stroke survives the hand passing over it. The eraser also works through this path. Erased ink disappears from the canvas after enough visible clean samples, not on the following frame.

With the hand mask on, the region under the hand becomes unknown. It neither gains nor loses ink while it is covered.

The `--ink-model` flag injects a custom ONNX into the old stack; without it, the path described above applies.

## Measured performance

Numbers measured on 2026-10-29 on the author's Ryzen 7 7700X, with the app's full pipeline over real frames. Medians per frame.

| Configuration | Time per frame | Rate |
|---|---|---|
| Standard ONNX Runtime session, fixed model at 512 | 212.8 ms | 4.7 fps |
| 4 intra-op threads, fixed model at 512 | 183.5 ms | 5.4 fps |
| 4 intra-op threads, dynamic model at 384 | 127.2 ms | 7.9 fps |

The session alone explains almost everything. On the ONNX Runtime default, `session.run` marked 248 ms on the same tensor. With 4 threads, it marks 122 ms at 512 and 58.3 ms at 384. With 16 threads, it worsens to 210 ms. Dynamic INT8 was measured and rejected. Its quality held, but the time rose to 366 ms, three times the FP32 of this machine.

The resolution trade-off cost little in quality. On the 49 harness frames, the mask at 384 has a mean IoU of 0.984 against the mask at 512. No presence verdict changed on the positives. The three frames of a white box on the desk, which at 512 were a weak false positive, at 384 became negative.

## Detector training

All training runs on Kaggle, on GPU. The author's real frames stay in private datasets and never enter this repository.

The v3 mixture combines four sources. MIDV-2020 photos brings 1,000 real photos of documents with annotated corners. SmartDoc 2015 brings 24,889 frames with ground truth. The school-notebooks-RU dataset brings notebook pages. The author's 49 desk frames enter with a presence label for all of them and hand-drawn polygons for 11 of them.

The classic detector was once the planned teacher for labeling the frames. On the 49 real frames it labeled zero, because the notebook leaves cropped by the frame's top edge and the four corners never appear together. Human ground truth replaced the teacher.

The harness `tools/eval_real_frames.py` is the gate for any new model. No model enters the app without passing it. v3 marks presence on 43 of 43 positives and 0 false positives on the 6 negatives. On the 11 annotations, the mask IoU is 0.962 on average and 0.841 at minimum at 512. In the production configuration, at 384, presence is 49 of 49 and the IoU is 0.959 on average and 0.851 at minimum.

The v3 kernel took 44.7 minutes on a T4. The v2 kernel took 3 hours and 51 minutes. The difference did not come from smaller data. It came from an async pipeline with workers, mixed precision, batch 16, and a 20-epoch schedule with stopping by ruler, instead of 40 fixed epochs.

## Known limits

- A totally blank and smooth sheet measures the same as any white and smooth rectangle, like a white tray. No interior statistic separates the two cases for the classic detector. What separates them is learned object identity, and that is why the neural one is the primary.
- v3 has never been judged on another person's desk. The external test set is the missing ruler for calling the detector general.
- The Windows capture bridge in `bridge/` has never been tested on a real machine.
- A fine stroke of 1 to 3 px suffers from rectification jitter. The measured mean is 5.4 px per sample in the author's replay. The accumulative canvas reduces the damage, but does not eliminate the cause.

## Architecture

- `draw/` is the main package. `app.py` has the webcam loop and the camera validations. `view.py` has the window composition (camera and canvas side by side) and the warning frames. `paper.py` has the classic detection, the texture gate, and the homography. `neural.py` has the neural detector via ONNX, with presence, mask, and quadrilateral. `page_clean.py` has the ink via the page→clean model. `ink.py` has the ink heuristic and the accumulative canvas. `hand.py` has the MediaPipe hand mask. `fusion.py` has the pen-tip fusion (prototype with a tested contract; the app does not use it yet). `manifest.py` checks the sha256 of the models against `models/MANIFEST.json`. `config.py` keeps the size of the rectified sheet and the marker data. `selflearn/` has the cross-session learning loop (session collection, monitor, selection, teacher, train and holdout split by session, and the promotion gate). The extractor of the learned ink model (`ink_model.py`) is not in this repository: its path is behind `--ink-model`.
- `template/` has the generator of the printable template sheet, used only by the `--markers` mode. The generated PNG does not enter versioning.
- `bridge/` has the Windows capture bridge. An `ffmpeg.exe` on Windows serves the webcam as an MJPEG stream and Draw consumes it via `--source`. The step by step is in [bridge/README.md](bridge/README.md).
- `scripts/` has the installation scripts. Today it keeps `setup_camera_wsl.sh`, the WSL camera installer.
- `tests/` has the webcam-free tests: `selftest.py` validates detection, homography, ink extraction, and the texture gate on synthetic scenes; the `test_*.py` files cover the ONNX adapter (with truly tiny models), the ink, `page_clean`, the manifest, the view composition, and the app's session state. `pytest` runs everything.
- `tools/` has the support tools, like plain operator scripts called by path. `label_frames.py` labels captured frames. `eval_real_frames.py` is the real-frame harness, the gate for any neural model. `cadence_fusion.py` is the cadence bench for tip fusion. The families with internal imports group in `tools/runner/` (experiment cycle) and `tools/selflearn/` (training round, gate judge, and loop latency measurement).
- `training/` and `notebooks/` are the detector's training route: `training/train.py` is the trainer and the notebook is the Kaggle kernel that runs it with `training/` as input. [The training guide](docs/TREINO.md) covers dataset integrity, the U-Net holdout contract, ONNX delivery gates, and same-checkpoint 384/256 evaluation.
- `models/` keeps the integrity record of the binaries in `MANIFEST.json`. A local model (`.onnx`) does not enter versioning.
- `verification/` is the record of the limited formal verification of v0. It keeps the project's skill and the script that proves homography properties with numeric probes. It is a historical record of a completed step, not a pipeline step.
- `docs/` has the installation (the Windows guide with WSL) and the archive of documents from completed eras (`docs/archive/`, with its own index: the 2026-10-31 audit and analysis, the v1 segmentation era, and the 2026-10-30 night summary).
- `main.py` is the entry point. It only calls the app.
- `VISION.md` is the project's vision document.

What no longer lives here: the documents from completed eras left `docs/` for `docs/archive/`; the cadence bench left `tests/` (where the name pretended to be a test) for `tools/`; and one-batch fix scripts were removed, with the history preserved in git.

## Development

Without a webcam nearby, the entire pipeline validates with synthetic scenes.

```bash
pip install -r requirements-dev.txt   # pytest, ruff, onnx (fixtures)
pytest                                 # entire suite, ~6 s
python tests/selftest.py
```

The selftest runs the marker, markerless, nested-sheet, and texture-gate cases. It prints the maximum corner error and the ink recall of each case. All must pass before any publication.

The limited formal verification runs with the command below.

```bash
python verification/run_verification.py
```

To label captured frames:

```bash
python tools/label_frames.py captures/frames dataset/rotulos
```

The output is a `labels.json` with the labeled and rejected frames. Each labeled frame gains a review image in the `overlays` subfolder. The source frames are never modified.

## Roadmap

Done:

- Real-time capture, with a rectified sheet, extracted ink, and a live digital canvas
- Flexible video source, by local index or by stream URL
- Windows camera attached to WSL by the installation script
- Camera diagnosis port by port with `--check`
- Markerless detection as the default mode
- Texture gate with a low-confidence verdict for a smooth sheet
- Paper segmentation trained on Kaggle, with a presence signal, in versions v1, v2, and v3
- Real document datasets and the author's desk frames in training
- Accumulative canvas with hysteresis and an optional hand mask
- Performance package with pinned threads and a dynamic model at 384

In progress:

- External test set, with frames from other setups, to judge the generalization of v3
- Stroke phase, with live fidelity already measured and the cursor bridge to Windows under construction
- Learned ink model, with the formulation closed and the synthetic dataset being assembled

Planned:

- Vector stroke, with order, path, and smoothing of the strokes
- Interpretation of shapes, letters, and diagrams
- Integration with a first target system, CAD or simulator
- Real-time simulation feedback
