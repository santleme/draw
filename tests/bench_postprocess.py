"""Microbenchmark sintético; não mede session.run nem o modelo.

Execute da raiz: .venv/bin/python tests/bench_postprocess.py
Os pares de saída são (altura, largura). Geração de logits e warmup
ficam fora da medição; os mesmos logits finitos são reutilizados.
"""

import argparse
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw.neural import _INK_CLASS, _INPUT_SIZE, _PAPER_CLASS  # noqa: E402


def argmax_mask(logits, output_shape):
    classes = np.argmax(logits[0], axis=0)
    mask_small = ((classes == _PAPER_CLASS) | (classes == _INK_CLASS)).astype(np.uint8)
    height, width = output_shape
    return (
        cv2.resize(mask_small, (width, height), interpolation=cv2.INTER_NEAREST) * 255
    )


def direct_mask(logits, output_shape):
    background, paper, hand, ink = logits[0]
    # Para logits finitos, preserva o desempate do argmax pelo menor índice:
    # fundo vence papel/tinta; papel vence mão; mão vence tinta.
    page = ((paper > background) & (paper >= hand)) | (
        (ink > background) & (ink > hand)
    )
    height, width = output_shape
    return (
        cv2.resize(
            page.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        )
        * 255
    )


def benchmark(logits, output_shape, iterations=500, warmup=25):
    methods = (("argmax", argmax_mask), ("direct", direct_mask))
    np.testing.assert_array_equal(
        argmax_mask(logits, output_shape), direct_mask(logits, output_shape)
    )
    for _ in range(warmup):
        for _, method in methods:
            method(logits, output_shape)
    samples = {name: [] for name, _ in methods}
    for iteration in range(iterations):
        # Alterna a ordem para reduzir viés de aquecimento/drift entre métodos.
        for name, method in methods[:: 1 if iteration % 2 == 0 else -1]:
            start = time.perf_counter()
            method(logits, output_shape)
            samples[name].append((time.perf_counter() - start) * 1000.0)
    return {
        name: {
            "mean_ms": float(np.mean(times)),
            "p95_ms": float(np.percentile(times, 95)),
        }
        for name, times in samples.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=25)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations precisa ser positivo e warmup não negativo")

    print(
        f"Python {platform.python_version()} | NumPy {np.__version__} | OpenCV {cv2.__version__}"
    )
    print(
        f"OpenCV threads={cv2.getNumThreads()} | iterations={args.iterations} | warmup={args.warmup}"
    )
    print("Synthetic finite float32 logits, warm cache; sizes are HxW.")
    print("Includes binarization, uint8 conversion, INTER_NEAREST resize and *255.")
    print(
        "Excludes input generation, presence reduction, preprocessing and session.run."
    )
    print(f"{'logits':>9} {'output':>10} {'method':>8} {'mean_ms':>10} {'p95_ms':>10}")
    rng = np.random.RandomState(0)
    for size in (_INPUT_SIZE, 512):
        logits = rng.randn(1, 4, size, size).astype(np.float32)
        for output_shape in ((720, 1280), (512, 512)):
            results = benchmark(logits, output_shape, args.iterations, args.warmup)
            for name, stats in results.items():
                print(
                    f"{size:4}x{size:<4} {output_shape[0]:4}x{output_shape[1]:<5} "
                    f"{name:>8} {stats['mean_ms']:10.3f} {stats['p95_ms']:10.3f}"
                )
            saving = results["argmax"]["mean_ms"] - results["direct"]["mean_ms"]
            print(f"  direct saving (mean): {saving:.3f} ms/frame")


if __name__ == "__main__":
    main()
