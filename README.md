<p align="center">
  <img src="docs/icon.svg" width="160" alt="Draw">
</p>

<h1 align="center">Draw (legacy)</h1>

<p align="center">
  A digitizing tablet without a digitizing tablet.
</p>

> **This repository is the legacy implementation.** Active development
> moved to **[santleme/draw-core](https://github.com/santleme/draw-core)**,
> a clean rebuild of the same capture pipeline — sheet detection, page
> rectification, ink extraction — as a tested monorepo.

## What lives where

- **draw-core** — the maintained runtime (`src/draw`), the lab tooling
  (`lab/` — capture/scenario evaluation, falsifier, replay, parity),
  the bridge receiver, the website, CI, and the reviewed model manifest.
  Start there.
- **draw (this repo)** — the original codebase, kept for reference:
  experiment history, desk notes, the simulator and training corpora,
  archived model candidates under `models/lab/`, and the docs that
  record *why* the pipeline looks the way it does
  ([VISION.md](VISION.md), [docs/FALHAS.md](docs/FALHAS.md),
  [docs/ESTADO_ATUAL.md](docs/ESTADO_ATUAL.md),
  [docs/SIMULATOR.md](docs/SIMULATOR.md)).

Production model binaries are not committed anywhere: hashes and
provenance live in `draw-core/assets/manifest.json`, and the approved
`.onnx` / `.task` files ship in the
[draw-core releases](https://github.com/santleme/draw-core/releases).

The previous project README remains in git history (`git show` any
commit before this notice) for the full quick-start and architecture
description.
