"""Regressao diferencial do InkAccumulator otimizado.

Compara a implementacao atual (draw.ink) contra a referencia canonica
em float32 + carimbos densos (tests._ink_reference) sobre sequencias
deterministicas e aleatorias semeadas, e exige canvas, estado e os
dois planos de carimbo byte-identicos depois de CADA frame.

Cobertura deliberada: fracoes de fronteira exatas (9/20 na
entrada, 4/20 na saida, com janela 20), denominadores impares
via oclusao de mao, borracha, e carimbos de borda (t=0, wrap de
uint32, t repetido).

Uso: python tests/test_accumulator_integer_sparse.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import ink as ink_new  # noqa: E402
from tests import _ink_reference as ink_old  # noqa: E402

H, W = 1200, 850


def run_pair(frames, label, **kwargs):
    """Alimenta os dois acumuladores e compara tudo a cada frame."""
    acc_old = ink_old.InkAccumulator(**kwargs)
    acc_new = ink_new.InkAccumulator(**kwargs)
    ratios = set()
    for i, (observed, visible, t_ms) in enumerate(frames):
        canvas_old = acc_old.update(observed, visible, t_capture_ms=t_ms)
        canvas_new = acc_new.update(observed, visible, t_capture_ms=t_ms)
        assert canvas_old.dtype == canvas_new.dtype
        assert np.array_equal(canvas_old, canvas_new), (
            f"{label} frame {i}: canvas divergiu")
        assert np.array_equal(acc_old.state, acc_new.state), (
            f"{label} frame {i}: estado divergiu")
        assert np.array_equal(
            acc_old.first_seen_ms(), acc_new.first_seen_ms()), (
            f"{label} frame {i}: first_seen divergiu")
        assert np.array_equal(
            acc_old.last_update_ms(), acc_new.last_update_ms()), (
            f"{label} frame {i}: last_update divergiu")
        judged = acc_new._vis_counts >= acc_new._min_visible
        pairs = zip(
            acc_new._ink_counts[judged].tolist(),
            acc_new._vis_counts[judged].tolist(),
        )
        ratios.update(pairs)
    print(f"{label}: {len(frames)} frames, canvas+estado+carimbos "
          f"identicos em todos; {len(ratios)} pares (ink, vis) julgados")
    return acc_new, ratios


def deterministic_frames():
    """Mesma geometria do teste dos carimbos: traco, mao, borracha."""
    rng = np.random.default_rng(42)
    frames = []
    for i in range(60):
        observed = np.zeros((H, W), dtype=np.uint8)
        observed[100:112, 100:300] = 255
        if i >= 30:
            observed[100:112, 180:240] = 0
        if i >= 45:
            observed[500:512, 400:600] = 255
        ys = rng.integers(0, H, size=40)
        xs = rng.integers(0, W, size=40)
        observed[ys, xs] = 255
        visible = np.ones((H, W), dtype=bool)
        if 20 <= i < 30:
            visible[90:130, 130:170] = False
        frames.append((observed, visible, 1000 + 33 * i))
    return frames


def boundary_frames():
    """Sondas isoladas atravessando as fracoes de fronteira.

    Janela 20. Cada sonda e um pixel so, longe dos outros para a
    dilatacao de 1 px nao misturar vizinhos. P1 entra exatamente
    em 9/20 e sai exatamente em 4/20. P2 para em 8/20 e nunca
    entra. P3 fica visivel so 9 frames na janela, com 4 de tinta:
    4/9 < 0,45, nunca entra.
    """
    probes = {
        "P1": (100, 100),
        "P2": (100, 200),
        "P3": (100, 300),
    }
    frames = []
    for i in range(45):
        observed = np.zeros((H, W), dtype=np.uint8)
        visible = np.ones((H, W), dtype=bool)
        y, x = probes["P1"]
        if 11 <= i <= 19:
            observed[y, x] = 255
        y, x = probes["P2"]
        if 12 <= i <= 19:
            observed[y, x] = 255
        y, x = probes["P3"]
        visible[90:110, 290:310] = i >= 11
        if 11 <= i <= 14:
            observed[y, x] = 255
        frames.append((observed, visible, 5000 + 33 * i))
    return frames, probes


def random_frames(seed, n_frames=80):
    """Blobs de tinta e de mao nascendo e sumindo, com ruido."""
    rng = np.random.default_rng(seed)
    blobs = []
    frames = []
    for i in range(n_frames):
        if rng.random() < 0.25 and len(blobs) < 6:
            y = int(rng.integers(0, H - 60))
            x = int(rng.integers(0, W - 60))
            blobs.append([y, x, int(rng.integers(5, 40)),
                          int(rng.integers(5, 40)),
                          int(rng.integers(15, 50))])
        observed = np.zeros((H, W), dtype=np.uint8)
        alive = []
        for y, x, h, w, ttl in blobs:
            if ttl > 0:
                observed[y:y + h, x:x + w] = 255
                alive.append([y, x, h, w, ttl - 1])
        blobs = alive
        ys = rng.integers(0, H, size=25)
        xs = rng.integers(0, W, size=25)
        observed[ys, xs] = 255
        visible = np.ones((H, W), dtype=bool)
        if rng.random() < 0.5:
            y = int(rng.integers(0, H - 120))
            x = int(rng.integers(0, W - 120))
            visible[y:y + 100, x:x + 100] = False
        t_ms = 2000 + 33 * i + int(rng.integers(0, 5))
        frames.append((observed, visible, t_ms))
    return frames


def edge_time_frames():
    """Carimbos de borda: t=0, t repetido, wrap do uint32."""
    ts = [0, 0, 1, 7, 7, 2**31, 2**32 - 1, 2**32 + 5, 12, 12,
          10**6, 10**6 + 33]
    frames = []
    for i, t_ms in enumerate(ts):
        observed = np.zeros((H, W), dtype=np.uint8)
        if i < 9:
            observed[300:312, 300:500] = 255
        visible = np.ones((H, W), dtype=bool)
        if 4 <= i < 7:
            visible[290:330, 350:420] = False
        frames.append((observed, visible, t_ms))
    return frames


def main():
    run_pair(deterministic_frames(), "deterministica")

    frames, probes = boundary_frames()
    acc, ratios = run_pair(frames, "fronteira (janela 20)", window=20)
    assert (9, 20) in ratios, "fronteira de entrada 9/20 nao foi julgada"
    assert (4, 20) in ratios, "fronteira de saida 4/20 nao foi julgada"
    assert (8, 20) in ratios, "vizinho 8/20 nao foi julgado"
    assert (4, 9) in ratios, "denominador impar 4/9 nao foi julgado"
    y, x = probes["P1"]
    assert not acc.state[y, x], "P1 deveria ter saido em 4/20 no fim"
    y, x = probes["P2"]
    assert not acc.state[y, x], "P2 (max 8/20) nao podia ter entrado"
    print("fronteira: 9/20 entra, 4/20 sai, 8/20 e 4/9 ficam fora")

    run_pair(random_frames(7), "aleatoria semente 7")
    run_pair(random_frames(1234), "aleatoria semente 1234")
    run_pair(edge_time_frames(), "tempos de borda")
    print("DIFERENCIAL OK: implementacao nova identica a anterior")


def test_accumulator_integer_sparse():
    main()


if __name__ == "__main__":
    main()
