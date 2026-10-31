"""Testes do módulo draw.fusion: portão, One Euro, fusão, caneta.

Python puro mais numpy, sem câmera e sem MediaPipe. Rodável com:

    ~/workspace/draw-lab/venv-eval/bin/python tests/test_fusion.py

Os números impressos são de bancada sintética (trajetórias e ruído
gerados aqui, semente fixa). Eles documentam o comportamento do
código, não o desempenho no ao vivo: isso segue não medido.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw.fusion import (
    OneEuroFilter2D,
    PenTipFusion,
    landmark_on_sheet,
    point_in_expanded_quad,
)

FAILURES = []
_COLLECT_ONLY = False


def check(name, condition, detail=""):
    if condition:
        print("ok - %s %s" % (name, detail))
    else:
        FAILURES.append(name)
        print("FALHOU - %s %s" % (name, detail))
        if not _COLLECT_ONLY:
            raise AssertionError(f"{name} {detail}".strip())


def test_gate():
    # Dentro da folha (850x1200) vale; fora não vale.
    check("portao dentro", landmark_on_sheet((425.0, 600.0)))
    check("portao fora (pe, y=1500)", not landmark_on_sheet((400.0, 1500.0)))
    check("portao fora (outra mao, y=2070)",
          not landmark_on_sheet((300.0, 2070.0)))
    # Margem: y=1230 está 30 px abaixo da borda inferior.
    check("portao margem 60 aceita y=1230",
          landmark_on_sheet((400.0, 1230.0), margin_px=60.0))
    check("portao margem 0 recusa y=1230",
          not landmark_on_sheet((400.0, 1230.0), margin_px=0.0))
    check("portao margem 60 recusa y=1270",
          not landmark_on_sheet((400.0, 1270.0), margin_px=60.0))
    check("portao None nao vale", not landmark_on_sheet(None))
    # Quadrilátero genérico inclinado, margem perpendicular uniforme.
    quad = [(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)]
    check("quad generico dentro",
          point_in_expanded_quad((200.0, 200.0), quad, 0.0))
    check("quad generico fora sem margem",
          not point_in_expanded_quad((200.0, 320.0), quad, 0.0))
    check("quad generico fora com margem 30",
          point_in_expanded_quad((200.0, 320.0), quad, 30.0))
    # Ordem horária também funciona.
    quad_cw = list(reversed(quad))
    check("quad horario dentro",
          point_in_expanded_quad((200.0, 200.0), quad_cw, 0.0))
    check("quad horario fora",
          not point_in_expanded_quad((200.0, 320.0), quad_cw, 0.0))


def test_one_euro_still_and_step():
    rng = np.random.default_rng(42)
    fps = 30.0
    # Trecho parado sintético: ponto fixo + ruído gaussiano de 8 px.
    n_still = 90
    truth = np.array([300.0, 300.0])
    samples = truth + rng.normal(0.0, 8.0, size=(n_still, 2))
    filt = OneEuroFilter2D(mincutoff=0.5, beta=0.002, dcutoff=1.0)
    out = np.array(
        [filt.filter(i / fps, tuple(p)) for i, p in enumerate(samples)]
    )
    # Descarta as 10 primeiras amostras (assentamento do filtro).
    raw_std = samples[10:].std(axis=0)
    filt_std = out[10:].std(axis=0)
    check(
        "one euro reduz jitter parado",
        bool(np.all(filt_std < raw_std)),
        "(n=%d, std bruto x/y=%.2f/%.2f px, filtrado x/y=%.2f/%.2f px)"
        % (n_still - 10, raw_std[0], raw_std[1], filt_std[0], filt_std[1]),
    )
    # Degrau de 150 px em x: o atraso até 50% do degrau é finito.
    filt2 = OneEuroFilter2D(mincutoff=0.5, beta=0.002, dcutoff=1.0)
    t_step = 1.0
    crossed_t = None
    for i in range(90):
        t = i / fps
        x = 100.0 if t < t_step else 250.0
        ox, _ = filt2.filter(t, (x, 100.0))
        if crossed_t is None and t >= t_step and ox >= 175.0:
            crossed_t = t
    lag = None if crossed_t is None else crossed_t - t_step
    check(
        "one euro segue degrau com atraso finito",
        lag is not None and 0.0 < lag < 2.0,
        "(atraso ate 50%% do degrau: %s)"
        % ("nao medido" if lag is None else "%.3f s a 30 fps" % lag),
    )


def test_pen_up_irregular_times():
    fusion = PenTipFusion(offset=(0.0, 0.0))
    landmark = (400.0, 500.0)
    # Delta fresco em t=0: caneta abaixada.
    state = fusion.update(0.0, landmark, (410.0, 500.0))
    check("caneta abaixa com delta fresco", state.pen_down)
    # Tempos irregulares: 0,9 s de silêncio mantém a caneta abaixada.
    state = fusion.update(0.37, landmark, None)
    check("silencio 0,37 s mantem", state.pen_down)
    state = fusion.update(0.9, landmark, None)
    check("silencio 0,9 s mantem", state.pen_down)
    # 1,1 s de silêncio levanta.
    state = fusion.update(1.1, landmark, None)
    check("silencio 1,1 s levanta", not state.pen_down)
    # Delta novo reabaixa, e o relógio de silêncio recomeça dele.
    state = fusion.update(2.05, landmark, (415.0, 500.0))
    check("delta novo reabaixa", state.pen_down)
    state = fusion.update(2.9, landmark, None)
    check("silencio 0,85 s apos delta mantem", state.pen_down)
    state = fusion.update(3.2, landmark, None)
    check("silencio 1,15 s apos delta levanta", not state.pen_down)


def test_fusion_correction_and_gate():
    # Sem delta, o cursor converge para landmark + offset.
    plain = PenTipFusion(offset=(0.0, 0.0))
    with_delta = PenTipFusion(offset=(0.0, 0.0))
    landmark = (200.0, 200.0)
    delta = (300.0, 200.0)
    state_plain = state_delta = None
    for i in range(30):
        t = i / 30.0
        state_plain = plain.update(t, landmark, None)
        state_delta = with_delta.update(t, landmark, delta)
    check(
        "fusao corrige em direcao ao delta",
        state_delta.x > state_plain.x + 20.0,
        "(sem delta x=%.1f, com delta x=%.1f, alvo da mistura=250.0)"
        % (state_plain.x, state_delta.x),
    )
    # Landmark fora da folha não cria posição nem move o cursor.
    gated = PenTipFusion(offset=(0.0, 0.0))
    state = gated.update(0.0, (400.0, 1500.0), None)
    check("landmark fora da folha nao posiciona", not state.has_position)
    state = gated.update(0.1, (400.0, 600.0), None)
    check("landmark dentro posiciona", state.has_position)
    # Hover: sem landmark, segura a última posição e marca stale
    # depois do timeout (2,0 s, valor não medido).
    state = gated.update(0.5, None, None)
    check("hover segura posicao", state.has_position and not state.position_stale)
    state = gated.update(3.0, None, None)
    check("hover velho marca stale", state.has_position and state.position_stale)


def main():
    global _COLLECT_ONLY
    FAILURES.clear()
    _COLLECT_ONLY = True
    try:
        test_gate()
        test_one_euro_still_and_step()
        test_pen_up_irregular_times()
        test_fusion_correction_and_gate()
    finally:
        _COLLECT_ONLY = False
    if FAILURES:
        print("\n%d teste(s) falharam: %s" % (len(FAILURES), FAILURES))
        return 1
    print("\nTODOS OS TESTES DE FUSAO OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
