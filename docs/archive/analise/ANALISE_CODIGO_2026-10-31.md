# Draw code analysis — 2026-10-31

> Second pass over the repository, after the product audit in
> `docs/auditoria/AUDITORIA_2026-10-31.md` (R1–R12). That audit looked at the
> **product** (protocol, rulers, model, ink latency). This one looks at the
> **code**: architecture, concurrency, I/O contracts, tests, repository
> hygiene and the computational cost of each stage. Where the two intersect, the
> earlier claim is verified here with an experiment, not repeated.
>
> Environment: new worktree at `/home/user/.draw-analysis-wt` (detached at
> `23d3a36`), working tree at `arena/2868ddc2-draw`. 2 vCPU Xeon 2.60 GHz,
> 3 GB of RAM, Python 3.11.2, OpenCV **5.0.0**, numpy 2.4.6, onnxruntime 1.31.1.
> No GPU, no webcam, none of the production models present — what
> could not be measured is stated at the end.

---

## TL;DR

1. **The code is in better shape than its size suggests.** The whole suite
   now runs in 6 s (`91 passed, 6 skipped`), the selftest reconstructs 96%
   of the ink and the geometric verification has no probe above 2.74 px.
   There is no gross design error on the main path.
2. **The real defects are at the edges**: threads that die silently, drawing
   functions that write into the caller's buffer, dead state pretending to be
   logic, a camera check that ignored `--source`, and an ONNX adapter
   that assumed the name `"image"` and square input. All of these appeared in
   experiments, not in reading — they are in section 4 with reproduction.
3. **The biggest performance gains have already been made** in this pass (batching in
   `page_clean`, `multiply` with `dtype=uint8`, logit comparison instead of
   softmax, composition buffer). What remains is refinement: `_update_stamps`
   costs ~40% of `InkAccumulator.update` and can be cut by the ink
   bounding box (section 5).
4. **The biggest debt is not code, it is repository debt**: half the tests
   depended on files that only exist on the author's machine, there was no CI or
   configured `pytest`, and the README documents a module (`draw/ink_model.py`)
   that does not exist. That was addressed (sections 6 and 7).

---

## 1. Scope and method

What was read: the 9 modules of `draw/` (≈ 3,700 lines), `main.py`,
the files in `tests/`, `tools/` (runner included), `training/`,
`bridge/`, `verification/`, `scripts/`, `docs/`, README and VISION.

How it was verified:

| Method | What it catches | Where it is |
|---|---|---|
| Full reading | architecture, contracts, debt | — |
| Instrumented probes | API contracts, aliasing, limits | `docs/analise/probes/probe_core.py` |
| Differential against the old tree | silent raster/geometry regression | `docs/analise/probes/diff_*.py` |
| Micro-benchmarks | cost per stage and per alternative | `docs/analise/probes/bench_*.py` |
| Suite + selftest + verification | whether the set still stands | `pytest`, `tests/selftest.py`, `verification/run_verification.py` |

The worktree was created on purpose for this: the old tree stays frozen at
`23d3a36` and serves as an oracle for the differentials. The working tree received
the fixes and the new tests. The probes were copied to
`docs/analise/probes/` (they run with `python docs/analise/probes/probe_core.py`),
so that what is written here can be redone without the worktree.

---

## 2. Architecture in one page

```
main.py ──> draw.app.main() ── argparse ──> draw.app.run(...)
                                              │
        ┌─────────────────────────────────────┼───────────────────────────┐
        │                                     │                           │
  _LatestFrameCapture                  laco principal              _PngWriter
  (thread, 1 slot,                    (janela + teclado)          (thread, fila)
   latest-wins)                              │
        │                                    │
        │                     ┌──────────────┴─────────────┐
        │                     │                            │
        │            deteccao de folha             tinta (2 caminhos)
        │            ├ neural (ONNX)               ├ padrao: _PageCleanWorker
        │            └ classico (paper.py)         │   (thread, 1 slot,
        │                                          │    lote de 256 px)
        │                     │                    └ legado: InkAccumulator
        │              paper.warp_sheet                  (histerese + carimbos)
        │                     │                            │
        │              _SheetTracker                 render_canvas
        │              (presenca + histerese)              │
        └─────────────────────┴──── _compose_view ──────────┘
```

Three threads, one queue contract per thread:

- **capture**: 1 slot, latest-wins, `Condition`; the consumer never sees accumulated
  lag. Driver failure → `_error` + warning frame.
- **ink** (`page_clean`): 1 pending, latest-wins, **generation** (a number that
  changes with each new sheet) so that a result in flight does not enter the next
  canvas. Model error → `_error`, the thread stops and the loop reports.
- **PNG**: `SimpleQueue`, draining with a cap in `close()`.

The rest is synchronous and deterministic — a good choice: geometry
and ink can be tested with no thread at all, and that is what `tests/selftest.py` does.

**Dependencies**: `cv2`, `numpy`, `onnxruntime` (lazy), `mediapipe`
(optional). No heavy dependency on the import path — only OpenCV, which
already costs ~150 ms.

---

## 3. The previous audit: what is confirmed

| Item | Verification in this pass | State |
|---|---|---|
| **R1** model manifest | implemented in `draw/manifest.py`; `--check` verifies and a wrong hash becomes `AVISO` (or an error with `DRAW_MODEL_STRICT=1`) | **done** |
| **R2** per-setup protocol | `docs/` and `tools/eval_real_frames.py` require the laboratory dataset; not code | still open |
| **R3** ink latency | measured again: `extract_ink` 17.6 ms + `update` 9.9 ms + rest ≈ 46 ms/frame on the heuristic path; the real bottleneck is the model, as the audit said | confirmed |
| **R4** model swap policy | `models/MANIFEST.json` now records `approved_by` and `status`; the policy itself remains human | partial |
| **R5** harden the ONNX adapter | **confirmed by experiment**: an input called `"input"` broke on the first frame (`RuntimeError: feed invalido`); a `[1,3,384,640]` input was squashed to 384×384 without warning | **done** |
| **R6** holes in the geometry guard | **confirmed**: `_fit_plausible` accepted a quadrilateral with `NaN` and a 1000×2 px strip; `_SheetTracker` keeps frozen corners **by design** while the verdict is `paper` without a quadrilateral (see A18) | `NaN` closed; the rest stays documented |
| **R7** hermetic tests | **confirmed**: `pytest tests/` died with `INTERNALERROR` because of `sys.exit(2)` on import in two tests; and `tools/runner/tests/test_cycle.py` did not collect due to an import of `/opt/muse/...` | **done** |
| **R11** micro-optimizations | `state*255` generated an 8 MB `int64` temporary per frame → `np.multiply(..., dtype=uint8)`; measured 0.98 ms → 0.07 ms | **done** |
| **R12** pilot presence semantics | pilot code is not in the repo; `MANIFEST.json` marks the pilot as `experimental`, which already prevents confusion | confirmed |

What the previous audit **did not catch** is in the next section.

---

## 4. New findings

Severity: **A** = loss of data or functionality; **B** = silent failure,
poor diagnosis or leak; **C** = hygiene, future risk.

### A1 — The camera check ignored the requested source (**A**, fixed)

`run_check(source)` always called `cv2.VideoCapture(0)`, even with
`--source 2` or `--source http://…`. Whoever ran `python main.py --check
--source 1` received the verdict for device 0 — the opposite of what the message
said. There was also no model check at all.

Reproduction (old tree):

```python
from draw.app import run_check
run_check(2)   # abria /dev/video0 e aprovava
```

Fix: index → checks `/dev/video{source}` in the node list and fails listing
the ones that exist; URL → `_check_remote_source` with a 5 s deadline for the first
frame; models → `_check_models` against `models/MANIFEST.json`.

### A2 — Capture thread died with a traceback, without warning (**A**, fixed)

`_LatestFrameCapture._pump` only handled a `False` return from `read()`. A driver
that **raises** (stuck V4L2, invalid index in certain versions, GStreamer
backend) killed the thread: `PytestUnhandledThreadExceptionWarning` in the suite,
a loose traceback on the error output and an app that kept looking at the last
frame forever.

Fix: `try/except` around the loop, `_error` saved, and the main loop
shows `_error_view(...)` for 3 s before exiting — the same outcome as
`read()` returning `False`.

### A3 — Ink model failure froze the canvas silently (**A**, fixed)

`_PageCleanWorker._work` called `ink_mask` without `try`. A broadcast
`ValueError` (model output at a different resolution), an ONNX out of memory or a
dead session killed the daemon thread: the canvas stopped at the last frame and **nothing
was printed**. The user saw the drawing stuck and had no way of knowing why.

Fix on two fronts:

1. the worker saves the exception in `_error`, stops and notifies;
2. the main loop reads `ink_worker.error` on each `poll()` and, if there is one, prints
   `A tinta pagina->limpa falhou e a thread parou: … O canvas nao avanca.`,
   shows the warning frame and exits.

In addition, `_output_patch_size` now validates the output contract **on
loading** (`[N,1,256,256]`), so the common case becomes a named error before the
thread exists.

### A4 — `_compose_view` wrote into the caller's canvas (**B**, fixed)

`_fit(image, 640)` returns **the array itself** when the height already fits; the
`putText("folha ao vivo")` was applied to that array. With a 640-high canvas, composing the view painted 554 pixels into the caller's canvas — and into
`_BLANK_SHEET`, which is a module cache and therefore kept the label
forever.

```python
canvas = np.full((640, 850, 3), 255, np.uint8)
app._compose_view(cam, canvas)
# antes: 554 pixels do canvas alterados; depois: 0
```

Fix: defensive copy only in the branch where `_fit` returns the original (zero
cost on the normal path, where the 850×1200 canvas is resized).

### A5 — `InkAccumulator` kept the visibility mask reference (**B**, fixed)

`update(ink_mask, visible_mask)` stuffed `visible_mask` into the window's
`deque` without copying. A caller that recycles the buffer (the natural case for whoever wants
to avoid allocation per frame) rewrote the window's past:

```python
visible = np.ones((1200, 850), bool)
for _ in range(6):
    acc.update(ink, visible, t)
acc.state.sum()            # 10 404 px
visible[:] = False         # reciclagem do buffer
# a janela inteira passa a dizer "nunca visível"
```

The app does not fall into this today (it creates a new `visible` each frame), but the API is
public and the failure mode is silent. Fix: `visible.copy()` when the
array is already boolean (the `> 0` in the other cases already produces a new array).

### A6 — `run()` without `try/finally` (**B**, fixed)

Any exception in the loop (model, GUI, `poll()`) jumped over the epilogue:
camera open, capture daemon thread still reading the device, OpenCV windows
not destroyed. In a process that stays alive (notebook, automated test)
this leaves the device busy.

Fix: epilogue in `finally` — `stop()` on capture, `stop()` on ink,
`close()` on the writer, `destroyAllWindows()`.

### A7 — `_PngWriter` died on the first failing `imwrite` (**B**, fixed)

`cv2.imwrite` returns `False` (disk full, invalid path, missing codec) and
can raise. The thread neither checked nor handled it: it died, all subsequent saves
stayed in the queue and `close()` waited for the drain cap.

Fix: `imwrite` in `try`, result checked, warning printed, thread continues.

### A8 — `_pending_hits` was dead state (**C**, fixed)

Five write sites (`+= 1`, `= 1`, `= 0` on sheet change) and **no use
in a decision**: the counter rose and nobody consulted it. Worse than dead code:
**it looked like** a confirmation rule ("2 hits to accept the sheet") that did not
exist anywhere. The probe counted 5 writes and 1 read — the read is the
`+=` itself; no condition in the function depends on the value. Removed.

### A9 — `hand.py` resolved the model relative to the CWD (**B**, fixed)

`_DEFAULT_CANDIDATES = ("models/hand_landmarker.task",)`. The neural detector and the
`page_clean` resolve from the repository root; the hand does not. Opening the app from
another directory (`cd ~ && python draw/main.py`) lost the hand mask without
warning — the model existed, it was just not found. Now the first candidate is
`<repo>/models/hand_landmarker.task`, with the relative path as a fallback.

### A10 — Opposite return conventions in neighboring functions (**C**, open)

`run_check` returns **0 = success**; `_check_remote_source` returns **1 =
success**. The composition (`return 0 if _check_remote_source(...) and all_ok else
1`) is correct, but reading it is an invitation to invert one of the two in a
refactoring. Suggestion: naming (`ok = ...`) or standardizing on an internal `bool` and
converting only at the output.

### A11 — Two copies of the default model path (**C**, open)

`models/draw_seg_v3_dyn.onnx` appears in `run()` (line ~1054) and in `main()`
(line ~1636), with the same "exists? then use; otherwise, env" logic. Two
copies of a precedence rule is a divergence waiting to happen.
Suggestion: `neural.DEFAULT_MODEL_PATH` + `neural.resolve_model_path()`, as already
exists in `page_clean`.

### A12 — `--hand-model`/`--ink-model`/`--session-baseline` switch on the legacy path (**C**, open)

The logic is correct and is explained in a comment and in a `print`, but it is not in the
`argparse` help: whoever passes `--ink-model` thinking they are on the new path
discovers it in the initialization `print`. Suggestion: one line in each `help` —
"Caminho da pilha antiga; passar esta flag liga --legacy-ink".

### A13 — Runner/`test_watch` threads depended on the author's machine (**B**, fixed)

`tools/runner/runner_kaggle.py` imported `dynamic_credentials` from
`/opt/muse/skills/...` **at the top of the module**: in any clone without that
directory, the module did not import and the runner suite died on collection.
`tools/runner/tests/test_watch.py` read `~/workspace/draw-lab/…` and gave
`FileNotFoundError` likewise. Fix: lazy import with a named error
("credencial do Kaggle indisponível: o conector custom.kaggle não está
instalado nesta máquina") and `pytest.skip` with a reason when the laboratory
history does not exist.

### A14 — Differential tests knocked down pytest collection (**B**, fixed)

`tests/test_accumulator_stamps.py` and `tests/test_accumulator_integer_sparse.py`
call `sys.exit(2)` on import when the frozen copy is not in place — which is reasonable as a script and fatal as a test module (pytest reports
`INTERNALERROR`). Fix: `pytest.skip(..., allow_module_level=True)` with a
reason, keeping `sys.exit(2)` for use as a script.

### A15 — Documentation promises a module that does not exist (**C**, open)

The README says `ink_model.py` has "the learned ink model extractor".
The file does not exist (the `app.py` has a `try: import ... except ImportError`
that assumes it). Alongside: `models/draw_seg_v3_dyn.onnx` is also not in the repo,
which is **deliberate** (R1), but the README does not warn about it. Suggestion: a "what does not come in the clone" note
with `models/README.md`.

### A16 — Model binary inside git (**C**, open, decision)

`models/kaggle-real-pilot-v1/model.onnx` is 15 MB and is versioned. The
`MANIFEST.json` already resolves integrity; the weight remains in history
forever. If the pilot is disposable, it deserves the same policy as the production
models (outside git + manifest).

### A17 — `draw/fusion.py` is used by nobody (**C**, open)

314 lines, no import outside itself. Either it is a forgotten experiment, or it is
the basis of the "fusion mode" that has not yet entered the app. If it is the second, the
`docs/` saying what the plan is are missing; if it is the first, it sponsors confusion.

### A18 — Presence without a quadrilateral holds the old position with no deadline (**B**, open, decision)

`_SheetTracker.update` has an explicit branch for "detected paper, but no
usable quadrilateral closed" (`draw/app.py`, ~line 180): it renews
`_last_seen_at`, stops the loss clock and returns `None`. The comment says why
— the sheet is there, what was missing was the position. The side effect is that the
old **position** keeps applying while this verdict lasts, with no cap:

```python
tracker = _SheetTracker()
tracker.update(quad, "paper", True, 0.0)          # folha travada
for t in range(1, 101):
    tracker.update(None, "paper", True, float(t))  # 100 s de "papel sem quad"
tracker.corners is not None                        # True
```

If the physical sheet moved (or was swapped) and the neural fit starts failing
due to occlusion, the warp keeps being done at the old coordinates and the ink
cropped from them enters the canvas. It is not a typing bug — it is a policy without
a deadline. Suggestion: a cap (e.g.: 3–5 s of presence without quad → `event = "drop"`,
or forcing detection every frame while it lasts), with the number measured on the
replay video.

### A19 — Ink from the previous sheet reappeared on the next sheet (**A**, fixed)

The accumulated canvas (`accumulated_canvas`) was not zeroed when the tracker
announced a new sheet: the first pass of the new generation only arrived ~250 ms
later, and in that interval the window showed the ink of the sheet that had left.
With a sequence of 210 synthetic frames (sheet → empty desk → sheet
again), the probe counted **22 frames with ink from the old sheet** in the panel after
the sheet returned (the first of them with 11,235 px). After the fix, the same
probe counts **0**.

The defect matters because the panel is the only visual confirmation that the app
"saw" the sheet swap: showing the previous drawing makes the user think the
new drawing did not take.

### A20 — `page_clean.ink_mask` materializes the whole page in float32 (**C**, open)

`logits` (850×1200×4 = 4 MB) + `prob` (4 MB) + `exp` per pixel. The comparison
`1/(1+e^-x) > t` is equivalent to `x > log(t/(1-t))`, which eliminates 4 MB and the
`exp` (~2 ms of 1,360 ms on the VM — small, but free and without changing the output
outside the rounding edge).

---

## 5. Optimizations

### 5.1 Already applied in this pass

| Change | Before | After | Gain |
|---|---|---|---|
| `page_clean`: mosaic batches in one `session.run` (dynamic batch) | 20 calls, ~500 ms | 1 call, ~301 ms | **1.66×** (realistic-size synthetic U-Net; bit-identical output) |
| `neural.predict`: `resize` before `cvtColor` + logit comparison in place of softmax | 5.5 ms (pre/post) | 2.6 ms | **2.1×** |
| `InkAccumulator`: `np.multiply(state, 255, dtype=uint8)` | 0.98 ms, 8 MB temporary | 0.07 ms | 14× on the step |
| `_compose_view` with `hstack` + fixed final size | double resize | 1 resize | — |

### 5.2 Cost per stage (this VM, 1280×720, page 850×1200)

```
detector clássico (detect_sheet_with_confidence)     11,1 ms
warp 720p -> 850x1200                                 3,4 ms
extract_ink (heurística)                             17,6 ms   <- _separable_veto 9,1
InkAccumulator.update                                 9,9 ms   <- _update_stamps ~5
render_canvas em buffer                               2,5 ms
_compose_view                                         2,1 ms
------------------------------------------------------------
quadro do caminho legado, sem rede                    46,4 ms  (~21 fps de teto)
```

The default path (page→clean model) is another story: ~1,360 ms per page
on this VM, 168 ms on the author's Ryzen — which is why it runs on its own thread, with
a 250 ms cadence and mask union.

### 5.3 What can still be gained (in order of return)

1. **`_update_stamps` cut by the ink bounding box** (~5 ms → ~1.5 ms per
   frame on the legacy path). The profile shows 40% of `update` in this function:
   five full-page boolean indexings per frame. It suffices to track the
   box of `state`/`first_seen` and operate only on it — the blank page pays
   nothing. It needs a differential against the current tree before entering.
2. **`extract_ink`/`_separable_veto`** (9.1 ms of 17.6): two separable
   medians per frame. If the veto is applied on a reduced version
   (e.g.: 1/2) and re-expanded, the cost falls ~4×; it is a product decision (it changes the
   mask), so it asks for a ruler, not just a benchmark.
3. **`page_clean` without `exp`** (A20): 2 ms for free.
4. **Larger batch in `page_clean`**: the batch of 20 is already the default when the file
   declares a dynamic batch; measuring on the Ryzen with the real model is the next step,
   with `DRAW_PAGE_CLEAN_BATCH` as the valve.

---

## 6. Quality and hygiene

**Before this pass**

- `pytest tests/` → `INTERNALERROR` on collection (two files with `sys.exit(2)`).
- `pytest tools/runner/tests` → import error (`/opt/muse/...`).
- No CI running on `main`: the only workflow GitHub lists
  (`.github/workflows/ci.yml`) does not exist in the repository tree — it was
  triggered from a branch that was never merged. There was also no
  `pyproject.toml` nor development requirement.
- `ruff` with the standard set (`E4,E7,E9,F`): **6 errors** (imports and dead
  variables).
- ONNX contract tests nonexistent; the adapter was only exercised with the
  real model, which is not in the repository.

**After**

```
$ pytest -q
91 passed, 6 skipped in 5.9s

$ ruff check .
All checks passed!
```

What went in:

- `pyproject.toml` with `testpaths` (`tests` + `tools/runner/tests`) and the
  `ruff` selection;
- `.github/workflows/tests.yml`, with the YAML validated with `yaml.safe_load`
  before the commit (a step name with `:` unquoted knocked down the first
  run, with zero jobs);
- `requirements-dev.txt` (pytest, ruff, onnx to generate the fixtures);
- `.github/workflows/tests.yml` (matrix 3.11/3.12, `libgl1`, lint, suite,
  selftest);
- `tests/onnx_fixtures.py`: generates genuinely tiny ONNX models (without torch)
  to test the adapter on the real onnxruntime;
- new tests: ONNX contract (14), `page_clean` (6), app session state
  (3, with scripted capture), manifest (9), accumulator contract (5),
  view composition (6).

The 6 skips are declared and intentional: two are the differential tests that
need the frozen copy in `/tmp` or in the laboratory; two are the training
pairing, which requires the full PyTorch; two are the training histories that
only exist on the author's machine. `pytest -q -rs` prints the reason for each one.

---

## 7. What changed in this branch

```
16 arquivos rastreados alterados, 1 014 inserções, 348 remoções
+ draw/manifest.py, models/MANIFEST.json, pyproject.toml,
  requirements-dev.txt, .github/workflows/tests.yml
+ tests/onnx_fixtures.py, test_manifest.py, test_page_clean.py,
  test_app_session_state.py, test_ink_accumulator_contract.py,
  test_compose_view.py
+ docs/analise/ (este documento e as sondas que o sustentam)
```

| File | Change |
|---|---|
| `draw/app.py` | `run_check` honors `--source` and checks models; `_error_view`/`_message_view`; capture and ink errors reported; epilogue in `finally`; PNG writer does not die; `_pending_hits` removed; `_compose_view` does not write into the caller's canvas; canvas zeroed on sheet change |
| `draw/neural.py` | contract read from the file (name, resolution, outputs); rectangular/mixed input refused with a named error; 4-channel `mask_logits` by `argmax`; `NaN` refused in `_fit_plausible`; softmax swapped for logit comparison |
| `draw/page_clean.py` | output contract validated on loading; batch per `session.run` with `DRAW_PAGE_CLEAN_BATCH` |
| `draw/ink.py` | visibility mask copy; `render_canvas` with `dtype=uint8` (and optional `out=`, exercised in test, not yet used by the loop) |
| `draw/manifest.py` | **new** — sha256/size against `models/MANIFEST.json`, strict mode |
| `draw/hand.py` | hand model resolved from the repository root |
| `tools/`, `training/` | lazy credential import, lint, `--onnx` used in the message |

All of this was checked against the old tree:

- `diff_ink.py`: 60 frames, canvas/state/stamps **identical**;
- `diff_neural.py`: 5 scenes (positive, negative, rotated, empty, noise),
  mask/verdict/presence/corners **identical**;
- `diff_pageclean.py`: mask **identical** (batch × block by block);
- `selftest`: 96.03% recall; `verification`: worst error 2.74 px.

---

## 8. Suggested priorities

| # | Action | Why | Cost |
|---|---|---|---|
| 1 | Commit this branch (fixes + tests + CI) | closes A1–A9, A13–A14 and R7 | low |
| 2 | `models/README.md` saying what does not come in the clone (A15, A16) | everyone's first hour cloning | low |
| 3 | `run_check` test with a fake source (V4L2 not testable; URL yes) | A1 comes back easily in a refactoring | low |
| 4 | Decide the fate of `fusion.py` (A17) | 314 lines of ambiguity | low |
| 5 | `_update_stamps` by box (5.3.1) | ~3.5 ms/frame on the legacy path | medium |
| 6 | Ruler for `_separable_veto` at half resolution (5.3.2) | ~7 ms/frame, but changes the mask | medium |
| 7 | Standardize `run_check`/`_check_remote_source` return (A10) | reading trap | low |

---

## Limits of this analysis

- **No webcam and no production models** (`draw_seg_v3_dyn.onnx`,
  `page_clean_unet.onnx`, `hand_landmarker.task`): the real ONNX latency on
  the target was not measured; the `page_clean` numbers use a realistic-size synthetic U-Net
  and the neural detector numbers use a fake session (pre/post only).
- **No MediaPipe installed**: `hand.py` was read and adjusted, not executed.
- **Kaggle runner**: without the `custom.kaggle` connector and without a credential, the
  network path does not run; only the chaining (`cycle`) was exercised, with a fake
  runner.
- **OpenCV 5.0.0**: the code was written for 4.x. Nothing broke on the exercised paths,
  but the CI matrix pins what is in `requirements.txt`; it is worth
  testing 4.11 explicitly before supporting both.
