"""Micro-benchmark sintético: findContours + fit, sem ONNX, resize ou subpixel.

Uso: python tools/benchmark_native_contours.py --iterations 500 --warmup 50
O resize de referência é feito antes da medição. Não mede a latência
fim a fim, nem substitui um replay com capturas reais.
"""

import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter_ns

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw.neural import fit_quad  # noqa: E402


def _masks():
    quad = np.int32([[70, 40], [310, 48], [325, 340], [55, 330]])
    clean = np.zeros((384, 384), np.uint8)
    cv2.fillPoly(clean, [quad], 255)
    jagged = clean.copy()
    for start, end in zip(quad, np.roll(quad, -1, axis=0)):
        for i, t in enumerate(np.linspace(0.08, 0.92, 24)):
            point = np.rint((1 - t) * start + t * end).astype(int)
            cv2.circle(jagged, tuple(point), 1, 255 if i % 2 else 0, -1)
    occluded = clean.copy()
    cv2.circle(occluded, (190, 335), 35, 0, -1)
    return {"clean": clean, "jagged": jagged, "occluded": occluded}


def benchmark(iterations=500, warmup=50):
    results = []
    for name, native in _masks().items():
        full = cv2.resize(native, (1280, 720), interpolation=cv2.INTER_NEAREST)
        paths = (
            ("native", native, {"scale_x": 1280 / 384, "scale_y": 720 / 384}),
            ("full_res", full, {}),
        )
        for _ in range(warmup):
            for _, mask, kwargs in paths:
                fit_quad(mask, **kwargs)
        samples = {label: [] for label, _, _ in paths}
        # Alterna a ordem para não favorecer sistematicamente um caminho.
        for iteration in range(iterations):
            for label, mask, kwargs in paths[:: 1 if iteration % 2 else -1]:
                start = perf_counter_ns()
                quad = fit_quad(mask, **kwargs)
                elapsed = (perf_counter_ns() - start) / 1e6
                if quad is None or not np.isfinite(quad).all():
                    raise RuntimeError(f"fit inválido: {name}/{label}")
                samples[label].append(elapsed)
        metrics = {}
        for label, mask, _ in paths:
            metrics[label] = {
                "shape": list(mask.shape),
                "pixels": int(mask.size),
                "median_ms": float(np.median(samples[label])),
                "p95_ms": float(np.percentile(samples[label], 95)),
            }
        results.append(
            {
                "case": name,
                **metrics,
                "median_speedup": metrics["full_res"]["median_ms"]
                / metrics["native"]["median_ms"],
                "native_median_under_3ms": metrics["native"]["median_ms"] < 3.0,
            }
        )
    return {
        "source": "synthetic",
        "scope": "findContours + approxPolyDP + robust edge fit + analytical quad upscale",
        "excluded": ["ONNX", "mask resize", "full-res cornerSubPix", "temporal memory"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "iterations": iterations,
        "warmup": warmup,
        "target_median_ms": 3.0,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations deve ser positivo e warmup não negativo")
    print(json.dumps(benchmark(args.iterations, args.warmup), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
