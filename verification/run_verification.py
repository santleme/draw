"""Verificacao formal (limitada) da fidelidade geometrica do Draw.

Propriedade, em uma frase: um traco desenhado na folha aparece no canvas
no mesmo lugar onde foi desenhado, tanto no modo com marcadores quanto no
modo padrao sem marcadores, dentro do escopo abaixo.

Dano se quebrar: a homografia (draw/paper.py), a ordenacao de cantos ou a
correspondencia entre draw/config.py e a folha-modelo esta errada e tudo
o que o usuario desenha e reproduzido deslocado, espelhado ou girado, sem
erro visivel.

Escopo verificado (e so isso):
- cenas sinteticas de 1600x1200 (a "mesa" vista pela camera);
- modo com marcadores (12 posicionamentos x 9 sondas = 108 sondas):
  a posicao base do selftest, girada no plano em 90, 180 e 270 graus, e
  8 quadrilateros convexos com jitter deterministico de ate 70 px;
- modo padrao sem marcadores (10 posicionamentos x 9 sondas = 90 sondas):
  a posicao base, folha aninhada sobre tapete claro e os mesmos 8
  quadrilateros com jitter deterministico de ate 70 px (sem rotacoes de
  90/180/270 graus, pois folha lisa sem marcador nao define orientacao
  intrinseca alem da geometria da cena);
- 9 sondas (discos de tinta de raio 10) numa grade 3x3 da area util;
- por sonda: centroide recuperado a no maximo 5 px do centro desenhado
  e pelo menos 40% da tinta da sonda recuperada na janela de 30 px.

Fora do escopo: iluminacao real, desfoque, oclusao parcial dos cantos ou
marcadores, webcam de verdade, e qualquer coisa fora das sondas.

Uso:
  python verification/run_verification.py            # tem que passar
  python verification/run_verification.py --sabotage # tem que FALHAR
O modo --sabotage gira o destino da homografia nos dois caminhos
(MARKER_CENTERS e destino de warp_sheet), simulando o erro classico de
ordem dos cantos. Um verificador que nunca falhou nao prova nada.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draw import config, ink, paper  # noqa: E402
from template.generate_template import build_template  # noqa: E402

PROBES = [(x, y) for y in (350, 600, 850) for x in (300, 425, 550)]
PROBE_RADIUS = 10
WINDOW = 30
MAX_CENTROID_ERROR_PX = 5.0
MIN_RECOVERED_FRACTION = 0.40

SCENE_W, SCENE_H = 1600, 1200
BASE_QUAD = np.array(
    [(430, 130), (1150, 210), (1080, 1120), (330, 1040)], dtype=np.float32
)
NESTED_QUAD = np.array(
    [(460, 180), (1120, 250), (1050, 1080), (380, 1000)], dtype=np.float32
)
SHEET_CORNERS = np.array(
    [(0, 0), (config.PAPER_W, 0), (config.PAPER_W, config.PAPER_H),
     (0, config.PAPER_H)],
    dtype=np.float32,
)


def marker_placements():
    """Os 12 posicionamentos do modo com marcadores."""
    yield "markers/base", BASE_QUAD.copy()
    for k in (1, 2, 3):
        yield f"markers/rot-{k * 90}deg", np.roll(BASE_QUAD, k, axis=0)
    rng = np.random.default_rng(20261028)
    for i in range(8):
        quad = BASE_QUAD + rng.integers(-70, 71, size=BASE_QUAD.shape)
        yield f"markers/jitter-{i}", quad.astype(np.float32)


def markerless_placements():
    """Os 10 posicionamentos do modo padrao sem marcadores."""
    yield "sem-marc/base", BASE_QUAD.copy(), False
    yield "sem-marc/aninhada", NESTED_QUAD.copy(), True
    rng = np.random.default_rng(20261028)
    for i in range(8):
        quad = BASE_QUAD + rng.integers(-70, 71, size=BASE_QUAD.shape)
        yield f"sem-marc/jitter-{i}", quad.astype(np.float32), False


def build_marker_scene(quad):
    sheet = build_template()
    for x, y in PROBES:
        cv2.circle(sheet, (x, y), PROBE_RADIUS, (0, 0, 0), thickness=-1)
    matrix = cv2.getPerspectiveTransform(SHEET_CORNERS, quad)
    scene = np.full((SCENE_H, SCENE_W, 3), 60, dtype=np.uint8)
    placed = cv2.warpPerspective(sheet, matrix, (SCENE_W, SCENE_H))
    mask = cv2.warpPerspective(
        np.full(sheet.shape[:2], 255, dtype=np.uint8), matrix,
        (SCENE_W, SCENE_H),
    )
    scene[mask > 0] = placed[mask > 0]
    return scene


def build_markerless_scene(quad, nested=False):
    sheet = np.full(
        (config.PAPER_H, config.PAPER_W, 3), 248, dtype=np.uint8
    )
    for x, y in PROBES:
        cv2.circle(sheet, (x, y), PROBE_RADIUS, (0, 0, 0), thickness=-1)
    matrix = cv2.getPerspectiveTransform(SHEET_CORNERS, quad)
    scene = np.full((SCENE_H, SCENE_W, 3), 60, dtype=np.uint8)
    if nested:
        cv2.ellipse(
            scene, (800, 600), (770, 545), 0, 0, 360, (150, 150, 150), -1
        )
        cv2.rectangle(scene, (1100, 0), (1560, 640), (60, 60, 60), -1)
    placed = cv2.warpPerspective(sheet, matrix, (SCENE_W, SCENE_H))
    mask = cv2.warpPerspective(
        np.full(sheet.shape[:2], 255, dtype=np.uint8), matrix,
        (SCENE_W, SCENE_H),
    )
    scene[mask > 0] = placed[mask > 0]
    return scene


SABOTAGE = False


def check_marker_scene(name, quad):
    scene = build_marker_scene(quad)
    if not SABOTAGE:
        warped, markers = paper.warp_paper(scene)
        return _check_warped(name, warped, f"marcadores: {markers}", True)
    original = dict(config.MARKER_CENTERS)
    try:
        config.MARKER_CENTERS.update(
            {i: original[(i + 1) % 4] for i in original}
        )
        warped, markers = paper.warp_paper(scene)
        return _check_warped(name, warped, f"marcadores: {markers}", True)
    finally:
        config.MARKER_CENTERS.update(original)


def check_markerless_scene(name, quad, nested=False):
    scene = build_markerless_scene(quad, nested=nested)
    corners, verdict = paper.detect_sheet_with_confidence(scene)
    # Com apenas 9 discos de raio 10 (0,28% da folha) sobre branco liso,
    # o portao de textura pode classificar como "paper" ou "blank_maybe";
    # nos dois casos os cantos refinados sao devolvidos e precisam manter
    # a fidelidade geometrica da homografia.
    if corners is None or verdict not in ("paper", "blank_maybe"):
        return [f"{name}: folha nao detectada (veredito: {verdict})"], 0.0
    if not SABOTAGE:
        warped = paper.warp_sheet(scene, corners)
    else:
        sabotaged_dst = np.roll(SHEET_CORNERS, 1, axis=0)
        warped = paper.warp_sheet(scene, corners, destination=sabotaged_dst)
    return _check_warped(name, warped, f"veredito: {verdict}", False)


def _check_warped(name, warped, detail, clear_markers):
    if warped is None:
        return [f"{name}: folha nao detectada ({detail})"], 0.0
    recovered = ink.extract_ink(warped, clear_markers=clear_markers)
    failures = []
    max_error = 0.0
    expected_area = np.pi * PROBE_RADIUS ** 2
    for x, y in PROBES:
        window = recovered[y - WINDOW:y + WINDOW, x - WINDOW:x + WINDOW]
        ys, xs = np.nonzero(window > 0)
        fraction = len(xs) / expected_area
        if fraction < MIN_RECOVERED_FRACTION:
            failures.append(
                f"{name}: sonda ({x},{y}) recuperou {fraction:.0%} da tinta"
            )
            continue
        error = float(np.hypot(xs.mean() + x - WINDOW - x,
                               ys.mean() + y - WINDOW - y))
        max_error = max(max_error, error)
        if error > MAX_CENTROID_ERROR_PX:
            failures.append(
                f"{name}: sonda ({x},{y}) deslocada {error:.1f}px "
                f"(limite {MAX_CENTROID_ERROR_PX:.0f}px)"
            )
    return failures, max_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sabotage", action="store_true",
                        help="gira destinos da homografia para falhar")
    args = parser.parse_args()

    global SABOTAGE
    SABOTAGE = bool(args.sabotage)
    if SABOTAGE:
        print("SABOTAGEM ATIVA: destinos da homografia girados em 90 graus")

    all_failures = []
    worst = 0.0
    total_scenes = 0

    for name, quad in marker_placements():
        total_scenes += 1
        failures, max_error = check_marker_scene(name, quad)
        worst = max(worst, max_error)
        status = "ok" if not failures else "FALHOU"
        print(f"  cena {name:<20} {status} (9 sondas, "
              f"erro max {max_error:.2f}px)")
        all_failures.extend(failures)

    for name, quad, nested in markerless_placements():
        total_scenes += 1
        failures, max_error = check_markerless_scene(
            name, quad, nested=nested
        )
        worst = max(worst, max_error)
        status = "ok" if not failures else "FALHOU"
        print(f"  cena {name:<20} {status} (9 sondas, "
              f"erro max {max_error:.2f}px)")
        all_failures.extend(failures)

    total = total_scenes * len(PROBES)
    if all_failures:
        print(f"\nVERIFICACAO FALHOU: {len(all_failures)} de {total} "
              f"sondas fora da propriedade")
        for failure in all_failures[:12]:
            print(f"  - {failure}")
        return 1
    print(f"\nVERIFICACAO OK: {total} sondas dentro da propriedade "
          f"(erro de centroide <= {MAX_CENTROID_ERROR_PX:.0f}px, "
          f"tinta recuperada >= {MIN_RECOVERED_FRACTION:.0%}; "
          f"pior erro observado {worst:.2f}px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
