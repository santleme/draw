"""Testes da guarda combinada de luz (draw/light_guard.py).

Regras travadas aqui, todas medidas nas cenas sintéticas do
selftest / measure_paper, sem mudar nenhum limiar de paper/ink/light:
- Com a flag ligada, se "off" já detectar, fica com "off" (base).
- Se "off" não detectar, tenta "background" (sombra, luz baixa).
- Resultado de "background" é descartado quando o interior do quad,
  medido no frame ORIGINAL, é majoritariamente saturado (kraft:
  fração S < 60 ~ 0,01, bem abaixo de 0,5).
- Resultado de "background" é descartado quando o quad é de quadro
  inteiro (área > 90% do frame e 4 cantos perto das bordas).
- A margem de borda é relativa: 2,5% do menor lado do frame.
- Resgates preservados com "background" (off falha primeiro, guarda
  devolve "paper"/"background" com erro <= 12 px): sombra forte,
  luz baixa moderada com ruído, luz baixa limpa, gradiente forte e
  invertido, papel cinza 180.
- Cena aninhada continua resolvida por "off" (modo vencedor "off").
- Luz baixa forte com ruído (0,35x + ruído sigma 8, seed 0) vira
  recusa segura: "background" devolve um quad de quadro inteiro
  (erro ~515 px) e a guarda o descarta, em vez de devolver o quad
  deslocado.
- Negativos via guarda não viram "paper": tapete, caixa, monitor,
  bandeja branca e folha lisa.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from draw import config  # noqa: E402
from draw import light as light_module  # noqa: E402
from draw import light_guard as guard_module  # noqa: E402
from draw import paper as paper_module  # noqa: E402
import selftest as selftest_module  # noqa: E402
from tools.measure_paper import (  # noqa: E402
    BASE_DST,
    _shadow,
    draw_on_sheet,
    place_sheet,
)


def _apply_shadow(scene):
    out = scene.astype(np.float32)
    height, width = out.shape[:2]
    gradient = np.linspace(0.45, 1.0, width)[None, :, None]
    out = out * gradient
    yy, xx = np.mgrid[0:height, 0:width]
    blob = np.exp(
        -(((xx - 700) ** 2) / (2 * 300 ** 2)
          + ((yy - 500) ** 2) / (2 * 250 ** 2))
    )
    out = out * (1 - 0.5 * blob)[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def test_off_wins_when_it_already_detects():
    scene, _, corners_gt = selftest_module.build_markerless_scene()
    corners, verdict, mode = guard_module.detect_sheet_with_light_guard(
        scene, "background"
    )
    assert verdict == "paper" and mode == "off"
    assert float(np.linalg.norm(corners - corners_gt, axis=1).max()) <= 12.0


def test_background_rescues_shadow_and_low_light():
    scene, _, corners_gt = selftest_module.build_markerless_scene()
    for variant in (
        _apply_shadow(scene),
        np.clip(scene.astype(np.float32) * 0.35, 0, 255).astype(np.uint8),
    ):
        corners_off, _ = paper_module.detect_sheet_with_confidence(variant)
        assert corners_off is None
        corners, verdict, mode = guard_module.detect_sheet_with_light_guard(
            variant, "background"
        )
        assert verdict == "paper" and mode == "background"
        assert float(np.linalg.norm(corners - corners_gt, axis=1).max()) <= 12.0


def test_off_flag_is_unchanged():
    scene, _, _ = selftest_module.build_markerless_scene()
    variant = _apply_shadow(scene)
    corners, verdict, mode = guard_module.detect_sheet_with_light_guard(
        variant, "off"
    )
    assert (corners, verdict, mode) == (None, None, "off")


def test_fallback_uses_supported_normalization_mode(monkeypatch):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    corners = np.zeros((4, 2), dtype=np.float32)
    detections = iter(((None, None), (corners, "paper")))
    modes = []
    monkeypatch.setattr(
        paper_module,
        "detect_sheet_with_confidence",
        lambda image: next(detections),
    )
    monkeypatch.setattr(
        light_module,
        "normalize_light",
        lambda image, mode: modes.append(mode) or image,
    )
    monkeypatch.setattr(guard_module, "guard_passes", lambda image, quad: True)

    result = guard_module.detect_sheet_with_light_guard(frame, "background")

    assert result[0] is corners
    assert result[1:] == ("paper", "background")
    assert modes == ["background"]


def test_kraft_detected_by_background_is_discarded():
    kraft = draw_on_sheet(
        np.full((config.PAPER_H, config.PAPER_W, 3), (140, 180, 210),
                dtype=np.uint8)
    )
    scene = place_sheet(kraft, BASE_DST)
    # Sem a guarda, background detectaria kraft como papel: a recusa
    # abaixo é da guarda, não de ausência de detecção.
    bg_corners, bg_verdict = paper_module.detect_sheet_with_confidence(
        light_module.normalize_light(scene, "background")
    )
    assert bg_verdict == "paper" and bg_corners is not None
    assert guard_module.unsaturated_fraction(scene, BASE_DST) < 0.5
    assert not guard_module.guard_passes(scene, bg_corners)
    corners, verdict, mode = guard_module.detect_sheet_with_light_guard(
        scene, "background"
    )
    assert (corners, verdict, mode) == (None, None, "off")


def test_edge_margin_is_relative_to_frame_size():
    assert guard_module.edge_margin_px((1200, 1600, 3)) == 30.0
    assert guard_module.edge_margin_px((600, 800, 3)) == 15.0
    assert guard_module.edge_margin_px((2400, 3200, 3)) == 60.0


def test_full_frame_quad_detection():
    shape = (1200, 1600, 3)
    full = np.array(
        [(0, 0), (1599, 0), (1599, 1199), (0, 1199)], dtype=np.float32
    )
    assert guard_module.is_full_frame_quad(shape, full)
    # Área ainda acima de 90%, mas um canto longe de todas as bordas:
    # não é quadro inteiro.
    one_corner_in = np.array(
        [(100, 100), (1599, 0), (1599, 1199), (0, 1199)],
        dtype=np.float32,
    )
    assert not guard_module.is_full_frame_quad(shape, one_corner_in)
    # Folha normal da cena base: área bem abaixo de 90%.
    scene, _, corners_gt = selftest_module.build_markerless_scene()
    assert not guard_module.is_full_frame_quad(shape, corners_gt)
    assert guard_module.guard_passes(scene, corners_gt)
    assert not guard_module.guard_passes(scene, full)


def _apply_gradient_range(scene, low, high):
    width = scene.shape[1]
    factor = np.linspace(low, high, width, dtype=np.float32)
    return np.clip(
        scene.astype(np.float32) * factor[None, :, None], 0, 255
    ).astype(np.uint8)


def _apply_low_light_noisy(scene, factor, sigma=8.0, seed=0):
    # Mesma construção de ruído de tools/measure_paper._low_light
    # (default_rng(0), gaussiana sigma 8), com o fator explícito.
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, scene.shape)
    return np.clip(
        scene.astype(np.float32) * factor + noise, 0, 255
    ).astype(np.uint8)


def _apply_strong_shadow(scene):
    out = scene.astype(np.float32)
    height, width = out.shape[:2]
    gradient = np.linspace(0.30, 1.0, width)[None, :, None]
    out = out * gradient
    yy, xx = np.mgrid[0:height, 0:width]
    blob = np.exp(
        -(((xx - 700) ** 2) / (2 * 300 ** 2)
          + ((yy - 500) ** 2) / (2 * 250 ** 2))
    )
    out = out * (1 - 0.6 * blob)[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _assert_background_rescue(variant, corners_gt):
    """Off falha primeiro; a guarda resgata via background, sem
    deslocar os cantos (erro <= 12 px, a régua do selftest)."""
    corners_off, _ = paper_module.detect_sheet_with_confidence(variant)
    assert corners_off is None
    corners, verdict, mode = guard_module.detect_sheet_with_light_guard(
        variant, "background"
    )
    assert verdict == "paper" and mode == "background"
    assert float(np.linalg.norm(corners - corners_gt, axis=1).max()) <= 12.0


def test_nested_scene_recovered_via_off():
    scene, _, corners_gt = selftest_module.build_nested_scene()
    # Off já detecta a folha aninhada, então a guarda fica com off:
    # background sequer é tentado (e, aqui, nem detectaria).
    corners_off, verdict_off = paper_module.detect_sheet_with_confidence(
        scene
    )
    assert verdict_off == "paper" and corners_off is not None
    corners, verdict, mode = guard_module.detect_sheet_with_light_guard(
        scene, "background"
    )
    assert verdict == "paper" and mode == "off"
    assert float(np.linalg.norm(corners - corners_gt, axis=1).max()) <= 12.0


def test_strong_shadow_rescued_by_background():
    scene, _, corners_gt = selftest_module.build_markerless_scene()
    # Sombra da bateria de papel (elipse x0,5) e sombra mais forte
    # (gradiente 0,30-1,0 + mancha x0,6): ambas resgatadas.
    for variant in (_shadow(scene), _apply_strong_shadow(scene)):
        _assert_background_rescue(variant, corners_gt)


def test_moderate_low_light_with_noise_rescued_by_background():
    scene, _, corners_gt = selftest_module.build_markerless_scene()
    # Moderada com ruído: 0,50x + gaussiana sigma 8, seed 0.
    # Limite documentado, não escondido: em 0,45x com o mesmo ruído
    # o quad de background sai deslocado (~65 px), então essa
    # variante mais forte NÃO é travada aqui como resgate limpo.
    _assert_background_rescue(
        _apply_low_light_noisy(scene, 0.50), corners_gt
    )


def test_clean_low_light_rescued_by_background():
    scene, _, corners_gt = selftest_module.build_markerless_scene()
    variant = np.clip(
        scene.astype(np.float32) * 0.35, 0, 255
    ).astype(np.uint8)
    _assert_background_rescue(variant, corners_gt)


def test_strong_and_inverted_gradients_rescued_by_background():
    scene, _, corners_gt = selftest_module.build_markerless_scene()
    # Forte 0,30-1,0 e invertido 1,0-0,30: off falha nos dois.
    # Gradientes mais suaves (0,65-1,0; 1,0-0,55) já detectam em
    # off, então não provam resgate e não entram aqui.
    for variant in (
        _apply_gradient_range(scene, 0.30, 1.0),
        _apply_gradient_range(scene, 1.0, 0.30),
    ):
        _assert_background_rescue(variant, corners_gt)


def test_gray_paper_180_rescued_by_background():
    # Papel cinza 180 (abaixo do branco 248 da cena base): off
    # recusa pelo portão de textura (V > 180 absoluto), background
    # resgata sem a guarda descartar (não é kraft saturado).
    gray = draw_on_sheet(
        np.full((config.PAPER_H, config.PAPER_W, 3), 180,
                dtype=np.uint8)
    )
    scene = place_sheet(gray, BASE_DST)
    _assert_background_rescue(scene, BASE_DST)


def test_full_frame_low_light_noisy_is_safe_refusal():
    scene, _, corners_gt = selftest_module.build_markerless_scene()
    # Luz baixa forte com ruído, seed 0 (a mesma seed da bateria):
    # off falha, mas background devolve um quad de quadro inteiro
    # (área ~96% do frame, erro ~515 px). A guarda descarta esse
    # quad e devolve a recusa de off, não o quad deslocado.
    variant = _apply_low_light_noisy(scene, 0.35)
    corners_off, _ = paper_module.detect_sheet_with_confidence(variant)
    assert corners_off is None
    bg_corners, bg_verdict = paper_module.detect_sheet_with_confidence(
        light_module.normalize_light(variant, "background")
    )
    assert bg_verdict == "paper" and bg_corners is not None
    assert guard_module.is_full_frame_quad(variant.shape, bg_corners)
    assert not guard_module.guard_passes(variant, bg_corners)
    corners, verdict, mode = guard_module.detect_sheet_with_light_guard(
        variant, "background"
    )
    assert (corners, verdict, mode) == (None, None, "off")


def test_negatives_have_no_new_false_positive_via_guard():
    # Todos os negativos da bateria de papel (tapete, caixa,
    # monitor, bandeja branca) mais a folha lisa: nenhum pode virar
    # "paper" com a guarda ligada. blank_maybe continua permitido:
    # detect_sheet só devolve cantos para veredito "paper", e a
    # guarda nunca promove blank_maybe a paper.
    scenes = selftest_module.build_negative_scenes()
    scenes["folha lisa"] = selftest_module.build_blank_sheet_scene()
    assert set(scenes) == {
        "tapete de mesa", "caixa de papelao", "monitor",
        "bandeja branca", "folha lisa",
    }
    for name, negative in scenes.items():
        _, verdict, _ = guard_module.detect_sheet_with_light_guard(
            negative, "background"
        )
        assert verdict != "paper", f"falso positivo via guarda: {name}"
