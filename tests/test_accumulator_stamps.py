"""Teste dos carimbos temporais do InkAccumulator (Bloco 2 da noite).

Duas frentes:

1. Regressão de raster: uma sequência sintética determinística de
   máscaras de tinta (semente fixa), com máscara de mão num trecho e
   um episódio de borracha, roda na implementação de referência em
   float32 (tests/_ink_reference.py) e na atual (draw/ink.py). Os
   canvas de todos os frames têm de ser byte-idênticos: carimbo e
   otimizações não podem mudar o raster.

2. Invariantes dos carimbos na implementação nova: first_seen <=
   last_update onde há tinta carimbada; first_seen estável enquanto
   o pixel permanece tinta; last_update avança quando a evidência se
   renova; os dois zeram quando o pixel sai do canvas; pixel sob a
   mão não ganha carimbo novo.

Uso: python tests/test_accumulator_stamps.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import ink as ink_new  # noqa: E402
from tests import _ink_reference as ink_old  # noqa: E402

H, W = 1200, 850
N_FRAMES = 60
T0_MS = 1000
DT_MS = 33

# Geometria da sequência (espaço do papel).
STROKE_A = (slice(100, 112), slice(100, 300))
ERASE_B = (slice(100, 112), slice(180, 240))
STROKE_C = (slice(500, 512), slice(400, 600))
HAND = (slice(90, 130), slice(130, 170))
P_HAND = (105, 150)   # tinta de A coberta pela mão na fase 2
P_ERASE = (105, 200)  # tinta de A apagada na fase 3
P_C = (505, 500)      # tinta de C, desenhada na fase 4


def build_sequence():
    """(observado, visível, t_ms) por frame, determinístico."""
    rng = np.random.default_rng(42)
    frames = []
    for i in range(N_FRAMES):
        observed = np.zeros((H, W), dtype=np.uint8)
        observed[STROKE_A] = 255
        if i >= 30:
            observed[ERASE_B] = 0  # borracha: some a tinta de B
        if i >= 45:
            observed[STROKE_C] = 255
        # Ruído determinístico: 40 pontos avulsos por frame.
        ys = rng.integers(0, H, size=40)
        xs = rng.integers(0, W, size=40)
        observed[ys, xs] = 255
        visible = np.ones((H, W), dtype=bool)
        if 20 <= i < 30:
            visible[HAND] = False  # mão cobre parte de A
        frames.append((observed, visible, T0_MS + DT_MS * i))
    return frames


def main():
    frames = build_sequence()
    acc_old = ink_old.InkAccumulator()
    acc_new = ink_new.InkAccumulator()

    snapshots = []  # (state, first_seen, last_update) por frame, impl. nova
    frames_divergentes = 0
    for observed, visible, t_ms in frames:
        canvas_old = acc_old.update(observed, visible)
        canvas_new = acc_new.update(observed, visible, t_capture_ms=t_ms)
        if canvas_old.dtype != canvas_new.dtype or not np.array_equal(
            canvas_old, canvas_new
        ):
            frames_divergentes += 1
        snapshots.append(
            (
                acc_new.state.copy(),
                acc_new.first_seen_ms(),
                acc_new.last_update_ms(),
            )
        )

    print(f"regressao de raster: {len(frames)} frames comparados, "
          f"{frames_divergentes} divergentes")
    assert frames_divergentes == 0, "canvas divergiu da implementacao anterior"
    assert np.array_equal(acc_old.state, acc_new.state)
    print("REGRESSAO OK: canvas byte-identico em todos os frames")

    # Invariante global: first_seen <= last_update onde ha tinta carimbada,
    # e todo pixel de tinta desta sequencia entrou visivel, logo carimbado.
    viol_ordem = 0
    viol_sem_carimbo = 0
    for state, fs, lu in snapshots:
        viol_ordem += int(((fs > 0) & (fs > lu)).sum())
        viol_sem_carimbo += int((state & (fs == 0)).sum())
    print(f"invariante first<=last: {viol_ordem} violacoes")
    print(f"tinta sem carimbo: {viol_sem_carimbo} ocorrencias")
    assert viol_ordem == 0
    assert viol_sem_carimbo == 0

    # first_seen estavel enquanto o pixel permanece tinta e visivel.
    viol_estabilidade = 0
    for i in range(1, len(frames)):
        state_prev, fs_prev, _ = snapshots[i - 1]
        state_now, fs_now, _ = snapshots[i]
        visible_now = frames[i][1]
        fica = state_prev & state_now & visible_now
        viol_estabilidade += int((fica & (fs_now != fs_prev)).sum())
    print(f"invariante first_seen estavel: {viol_estabilidade} violacoes")
    assert viol_estabilidade == 0

    # Mao: carimbos de P_HAND congelam entre o frame 19 e o 29.
    _, fs19, lu19 = snapshots[19]
    state29, fs29, lu29 = snapshots[29]
    assert state29[P_HAND], "pixel sob a mao deveria permanecer tinta"
    assert fs29[P_HAND] == fs19[P_HAND] and lu29[P_HAND] == lu19[P_HAND], (
        "carimbo mudou sob a mao"
    )
    print(f"mao: P_HAND congelado (first={fs19[P_HAND]}, last={lu19[P_HAND]}) "
          "nos frames 20-29")

    # Renovacao: last_update de P_C acompanha o t dos frames finais e
    # first_seen e o t do frame em que C entrou no canvas.
    state_f, fs_f, lu_f = snapshots[-1]
    entrada_c = next(
        i for i, (state, _, _) in enumerate(snapshots) if state[P_C]
    )
    t_entrada_c = T0_MS + DT_MS * entrada_c
    assert state_f[P_C]
    assert fs_f[P_C] == t_entrada_c, (fs_f[P_C], t_entrada_c)
    assert lu_f[P_C] == T0_MS + DT_MS * (N_FRAMES - 1)
    print(f"renovacao: P_C entrou no frame {entrada_c} "
          f"(first={fs_f[P_C]}), last_update={lu_f[P_C]} "
          f"= t do ultimo frame")

    # Borracha: P_ERASE sai do canvas e os carimbos zeram no mesmo frame.
    saida = next(
        i
        for i in range(30, N_FRAMES)
        if snapshots[i - 1][0][P_ERASE] and not snapshots[i][0][P_ERASE]
    )
    _, fs_s, lu_s = snapshots[saida]
    assert fs_s[P_ERASE] == 0 and lu_s[P_ERASE] == 0
    state_f, fs_f, lu_f = snapshots[-1]
    assert not state_f[P_ERASE] and fs_f[P_ERASE] == 0 and lu_f[P_ERASE] == 0
    print(f"borracha: P_ERASE saiu no frame {saida} com carimbos zerados")

    # Acessores devolvem copias.
    copia = acc_new.first_seen_ms()
    copia[0, 0] = 424242
    assert acc_new.first_seen_ms()[0, 0] != 424242
    print("acessores: devolvem copias")

    # Caminho sem t_capture_ms: carimbo vem do relogio de processamento.
    acc_none = ink_new.InkAccumulator()
    obs = np.zeros((H, W), dtype=np.uint8)
    obs[STROKE_A] = 255
    for _ in range(6):
        acc_none.update(obs)
    assert int((acc_none.first_seen_ms() > 0).sum()) > 0
    print("sem t_capture_ms: carimbos preenchidos pelo relogio de "
          "processamento (medem processamento, nao captura)")

    # Custo de memoria dos dois planos.
    bytes_planos = (
        acc_new.first_seen_ms().nbytes + acc_new.last_update_ms().nbytes
    )
    print(f"memoria dos carimbos: {bytes_planos} bytes "
          f"({bytes_planos / 2**20:.2f} MiB) para {H}x{W} uint32 x2")
    assert bytes_planos == 2 * H * W * 4

    print("TESTE DOS CARIMBOS OK")


def test_accumulator_stamps():
    main()


if __name__ == "__main__":
    main()
