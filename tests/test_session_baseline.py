"""Testes de SessionBaselineInk, align_to_reference e LUT do InkAccumulator."""

from fractions import Fraction
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import config, ink  # noqa: E402


def _ruled_page(bg=212):
    page = np.full((config.PAPER_H, config.PAPER_W, 3), bg, dtype=np.uint8)
    for y in range(110, config.PAPER_H - 60, 44):
        cv2.line(page, (0, y), (config.PAPER_W, y), (bg - 26, bg - 26, bg - 26), 2)
    cv2.line(
        page, (95, 0), (95, config.PAPER_H), (bg - 22, bg - 22, bg - 22), 2
    )
    return page


def test_baseline_ignores_occluded_arm_during_capture():
    page = _ruled_page(212)
    session = ink.SessionBaselineInk()
    for i in range(12):
        sample = page.copy()
        vis = np.ones((config.PAPER_H, config.PAPER_W), dtype=bool)
        if i < 10:
            # Braço escuro parado durante 10 das 12 amostras do C.
            sample[400:900, 300:750] = 65
            vis[400:900, 300:750] = False
        session.add_sample(sample, vis)
    assert session.ready
    clean_gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY).astype(np.float32)
    err = float(
        np.abs(
            session.reference[500:800, 400:700] - clean_gray[500:800, 400:700]
        ).max()
    )
    assert err <= 1.0, f"braco contaminou a referencia: erro max {err}"


def test_realignment_suppresses_ruled_line_false_ink_and_lighting_shift():
    page = _ruled_page(212)
    session = ink.SessionBaselineInk()
    for _ in range(12):
        session.add_sample(page)
    assert session.ready

    # Deslocamento de 2 px sem traço novo: pauta não pode virar tinta.
    m_shift = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.5]], dtype=np.float32)
    shifted = cv2.warpAffine(
        page,
        m_shift,
        (config.PAPER_W, config.PAPER_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    obs_shift = session.observe(shifted)
    # Ignora faixa de 4 px da borda replicada.
    interior = obs_shift[8:-8, 8:-8]
    assert int((interior > 0).sum()) == 0, (
        f"falsa tinta a 2 px de deslocamento: {int((interior > 0).sum())} px"
    )

    # Queda global de luz de -8% sem traço novo: 0 px de falsa tinta.
    dimmed = np.clip(page.astype(np.float32) * 0.92, 0, 255).astype(np.uint8)
    obs_dim = session.observe(dimmed)
    assert int((obs_dim > 0).sum()) == 0

    # Salto > 10 px: align_to_reference reporta ok=False sem quebrar.
    gray_ref = session.reference
    m_big = np.array([[1.0, 0.0, 15.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    big_shift = cv2.warpAffine(
        gray_ref, m_big, (config.PAPER_W, config.PAPER_H)
    )
    _, _, _, ok = ink.align_to_reference(gray_ref, big_shift)
    assert not ok


def test_hand_occlusion_does_not_erase_faint_pencil_via_gain_bias():
    """Regressão do Gargalo #3: mão escura não pode inflar o ganho global."""
    bg = 210
    page = np.full((config.PAPER_H, config.PAPER_W, 3), bg, dtype=np.uint8)
    # Textura sutil determinística para o realinhamento ter fase definida.
    rng = np.random.default_rng(11)
    tex = rng.integers(-2, 3, size=(config.PAPER_H, config.PAPER_W), dtype=np.int16)
    page = np.clip(page.astype(np.int16) + tex[:, :, None], 0, 255).astype(np.uint8)

    session = ink.SessionBaselineInk()
    for _ in range(12):
        session.add_sample(page)

    # Traço de lápis fraco: contraste de ~14 níveis (limiar é 12,0).
    drawn = page.copy()
    drawn[220:228, 120:520] = np.clip(
        drawn[220:228, 120:520].astype(np.int16) - 14, 0, 255
    ).astype(np.uint8)

    # Referência sem mão: ~6 linhas internas sobrevivem ao GaussianBlur(3x3)
    # em contraste limítrofe de 14 níveis contra limiar 12,0 (~2388 px).
    obs_clean = session.observe(drawn)
    clean_hits = int((obs_clean[220:228, 120:520] > 0).sum())
    assert clean_hits >= 2200

    # Mão/antebraço escuro cobrindo ~58% da página (fora do traço).
    vis = np.ones((config.PAPER_H, config.PAPER_W), dtype=bool)
    drawn[500:, :] = 55
    vis[500:, :] = False

    # Nota: se a mediana fosse medida sem mascarar `vis`, mais da metade
    # da página valeria 55, empurrando o ganho para o teto 1,18 e
    # clareando o traço de 196 para ~231 (0 px detectados!).
    obs = session.observe(drawn, visible=vis)
    stroke_hits = int((obs[220:228, 120:520] > 0).sum())
    hand_leaks = int((obs[500:, :] > 0).sum())
    assert stroke_hits >= int(0.95 * clean_hits), (
        f"traco fraco apagado pelo vies da mao no ganho: "
        f"{stroke_hits} vs {clean_hits} px sem mao"
    )
    assert hand_leaks == 0


def test_lut_exhaustively_matches_exact_rational_predicate():
    for window, enter, exit_t, min_vis in [
        (15, 0.45, 0.20, 4),
        (20, 0.45, 0.20, 4),
        (30, 0.50, 0.25, 5),
    ]:
        acc = ink.InkAccumulator(
            shape=(16, 16),
            window=window,
            enter=enter,
            exit=exit_t,
            min_visible=min_vis,
        )
        en_f = Fraction(enter).limit_denominator(100)
        ex_f = Fraction(exit_t).limit_denominator(100)
        for v in range(window + 1):
            for k in range(v + 1):
                want_en = (v >= min_vis) and (k * en_f.denominator >= v * en_f.numerator)
                want_ex = (v >= min_vis) and (k * ex_f.denominator <= v * ex_f.numerator)
                got_en = bool(k >= int(acc._enter_lut[v]))
                got_ex = bool(k + 1 <= int(acc._exit_lut[v]))
                assert got_en == want_en, (window, v, k, got_en, want_en)
                assert got_ex == want_ex, (window, v, k, got_ex, want_ex)

    # Fallback para window > 254 continua funcionando.
    acc_big = ink.InkAccumulator(shape=(32, 32), window=300, min_visible=2)
    assert not acc_big._use_u8_lut
    obs = np.full((32, 32), 255, dtype=np.uint8)
    out = acc_big.update(obs, t_capture_ms=100)
    assert out.dtype == np.uint8
    out2 = acc_big.update(obs, t_capture_ms=200)
    assert int((out2 > 0).sum()) == 32 * 32


def main():
    test_baseline_ignores_occluded_arm_during_capture()
    test_realignment_suppresses_ruled_line_false_ink_and_lighting_shift()
    test_hand_occlusion_does_not_erase_faint_pencil_via_gain_bias()
    test_lut_exhaustively_matches_exact_rational_predicate()
    print("TESTES DE SESSION BASELINE E LUT OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
