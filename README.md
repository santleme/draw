# Draw (legacy — archived)

A digitizing tablet without a digitizing tablet.

> **This repository is archived.** Active development moved to
> **[santleme/draw-core](https://github.com/santleme/draw-core)** —
> a clean, tested rebuild of the same capture pipeline (sheet
> detection, page rectification, ink extraction) plus the lab
> tooling, the website, and the reviewed model manifest.
>
> Training, datasets and the self-learning cycle live in
> **[santleme/draw-training](https://github.com/santleme/draw-training)**.
> The Windows→WSL2 camera transport lives in
> **[santleme/hvcam](https://github.com/santleme/hvcam)**.

## The old code

Everything that was here — the full runtime, lab, docs, training
scripts and experiment history — is preserved at the
[`legacy-final`](https://github.com/santleme/draw/releases/tag/legacy-final)
tag:

```bash
git clone --branch legacy-final https://github.com/santleme/draw.git
```

Design rationale and failure notes worth keeping are in
`VISION.md`, `docs/FALHAS.md`, `docs/ESTADO_ATUAL.md` and
`docs/SIMULATOR.md` on that tag.

## Models

Production `.onnx` binaries were never committed here; they are
distributed via GitHub Release with hashes recorded in
`models/MANIFEST.json` (on the tag) and now delivered through
`santleme/draw-core`'s `assets/manifest.json`.
