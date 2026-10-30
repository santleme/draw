"""Medição da captura do papel por condição (etapa 3 do Senado).

Estes testes travam a régua, não o resultado desejado: a cena base
tem que continuar detectando, os negativos têm que continuar
recusados, e o harness tem que ser determinístico. As variantes que
hoje falham (sombra, luz baixa, kraft, canto oculto) são registradas
como lacunas medidas, com comentário explícito: a próxima fatia pode
( e deve ) mudar esses vereditos, atualizando estes testes junto
com a justificativa medida. Nenhum limiar é alterado aqui.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.measure_paper import measure_all  # noqa: E402


def _by_id(results):
    return {r.id: r for r in results}


def test_harness_returns_all_variants():
    results = measure_all()
    ids = _by_id(results)
    expected = {
        "base", "gradiente", "sombra", "brilho", "luz_baixa",
        "inclinacao_extrema", "canto_oculto", "papel_kraft",
        "papel_amarelo_palido", "aninhada",
        "negativo_tapete_de_mesa", "negativo_caixa_de_papelao",
        "negativo_monitor", "negativo_bandeja_branca",
        "negativo_folha_lisa",
    }
    assert set(ids) == expected
    for result in results:
        assert result.verdict in (None, "paper", "blank_maybe")
        assert result.false_positive is False


def test_base_and_robust_variants_still_detect():
    ids = _by_id(measure_all())
    for variant_id in ("base", "gradiente", "inclinacao_extrema",
                       "papel_amarelo_palido", "aninhada"):
        result = ids[variant_id]
        assert result.detected, variant_id
        assert result.corner_error_max_px <= 12.0, variant_id
        assert result.ink_recall >= 0.90, variant_id


def test_negatives_are_never_paper():
    ids = _by_id(measure_all())
    for variant_id, result in ids.items():
        if variant_id.startswith("negativo_"):
            assert not result.detected, variant_id
            assert result.verdict != "paper", variant_id


def test_measured_gaps_are_recorded_not_hidden():
    # Lacunas medidas na etapa 3 (valores observados em 2026-10-28):
    # sombra e luz baixa são recusadas pelo portão de textura, não
    # pela geometria: a fração de branco usa V > 180 absoluto, e a
    # sombra derruba para ~0,25 e a luz baixa para 0,00. Kraft é
    # recusado por desenho do portão (S alto). Brilho detecta, mas
    # o recall cai para ~52%. Canto oculto detecta um quadrilátero
    # errado (erro ~131 px, recall ~3%): falha insegura, não ausência
    # de detecção. Se uma fatia futura corrigir qualquer um destes,
    # ela atualiza este teste com os novos números medidos.
    ids = _by_id(measure_all())
    assert not ids["sombra"].detected
    assert not ids["luz_baixa"].detected
    assert not ids["papel_kraft"].detected
    assert ids["brilho"].detected
    assert ids["brilho"].ink_recall < 0.90
    assert ids["canto_oculto"].detected
    assert ids["canto_oculto"].corner_error_max_px > 12.0


def test_harness_is_deterministic():
    first = [(r.id, r.verdict, r.detected) for r in measure_all()]
    second = [(r.id, r.verdict, r.detected) for r in measure_all()]
    assert first == second


def test_variant_scenes_are_valid_images():
    # Guardrail contra variante degenerada (tela preta, por exemplo):
    # a cena base e a de gradiente têm conteúdo claro suficiente.
    from tools.measure_paper import place_sheet, white_sheet, _gradient
    from draw import config
    scene = place_sheet(white_sheet(), np.array(
        [(430, 130), (1150, 210), (1080, 1120), (330, 1040)],
        dtype=np.float32))
    assert scene.shape == (1200, 1600, 3)
    assert float(np.mean(scene)) > 80.0
    assert float(np.mean(_gradient(scene))) > 60.0
    assert config.PAPER_W == 850 and config.PAPER_H == 1200
