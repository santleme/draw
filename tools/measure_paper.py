"""Mede a captura do papel por condição, sem mudar nenhum limiar.

Etapa 3 do Senado: antes de ajustar qualquer limiar de detecção ou
de tinta, medir. Este script gera variantes sintéticas a partir das
cenas do selftest (sombra, gradiente, brilho, luz baixa, papel
colorido, inclinação extrema, canto oculto) e reporta, por variante:
veredito do portão de textura, detecção, erro máximo de canto e
recall de tinta na mesma região do selftest. Os negativos existentes
(tapete, caixa, monitor, bandeja, folha lisa) são reportados junto,
para que nenhum ganho de recall esconda um falso positivo.

Isto é medição, não portão: a saída é 0 sempre que o harness roda
até o fim, mesmo que uma variante falhe. Uma variante que falha é
dado, não erro do script. Os testes em tests/test_paper_variants.py
travam só o que não pode regredir: a cena base detecta, e os
negativos continuam recusados.

Sem webcam, sem janela, determinístico (seed fixa para o ruído).

Uso:
  python tools/measure_paper.py [--format markdown|json] [--report out.md]
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draw import config, ink, paper  # noqa: E402
from tests.selftest import (  # noqa: E402
    build_blank_sheet_scene,
    build_negative_scenes,
    build_nested_scene,
)

# Cantos da folha na cena base, os mesmos do selftest.
BASE_DST = np.array(
    [(430, 130), (1150, 210), (1080, 1120), (330, 1040)],
    dtype=np.float32,
)
# Inclinação extrema: folha vista bem de lado, mas ainda com área
# acima do mínimo de 12% e aspecto dentro de 0,4-2,5.
TILT_DST = np.array(
    [(620, 280), (1280, 180), (1180, 1100), (560, 1020)],
    dtype=np.float32,
)
# Região de comparação do selftest (linhas 200-1100, colunas 200-650).
REGION = (slice(200, 1100), slice(200, 650))


def draw_on_sheet(sheet):
    """O mesmo desenho do selftest: círculo, linha e X."""
    cv2.circle(sheet, (425, 500), 110, (0, 0, 0), 6)
    cv2.line(sheet, (280, 800), (570, 800), (0, 0, 0), 6)
    cv2.line(sheet, (320, 950), (530, 1080), (0, 0, 0), 6)
    cv2.line(sheet, (530, 950), (320, 1080), (0, 0, 0), 6)
    return sheet


def place_sheet(sheet, dst):
    """Cola a folha sobre a mesa escura, como o selftest faz."""
    scene = np.full((1200, 1600, 3), 60, dtype=np.uint8)
    src = np.array(
        [(0, 0), (config.PAPER_W, 0), (config.PAPER_W, config.PAPER_H),
         (0, config.PAPER_H)],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    placed = cv2.warpPerspective(sheet, matrix, (1600, 1200))
    mask = cv2.warpPerspective(
        np.full(sheet.shape[:2], 255, dtype=np.uint8), matrix, (1600, 1200)
    )
    scene[mask > 0] = placed[mask > 0]
    return scene


def white_sheet():
    return draw_on_sheet(
        np.full((config.PAPER_H, config.PAPER_W, 3), 248, dtype=np.uint8)
    )


def _gradient(scene):
    factor = np.linspace(0.65, 1.0, scene.shape[1], dtype=np.float32)
    return np.clip(
        scene.astype(np.float32) * factor[None, :, None], 0, 255
    ).astype(np.uint8)


def _shadow(scene):
    out = scene.copy()
    blotch = np.zeros(scene.shape[:2], dtype=np.uint8)
    cv2.ellipse(blotch, (750, 600), (420, 320), 20, 0, 360, 255, -1)
    out[blotch > 0] = (out[blotch > 0].astype(np.float32) * 0.5).astype(
        np.uint8
    )
    return out


def _glare(scene):
    out = scene.copy()
    cv2.circle(out, (760, 560), 130, (250, 250, 250), -1)
    return out


def _low_light(scene):
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 8, scene.shape)
    return np.clip(
        scene.astype(np.float32) * 0.45 + noise, 0, 255
    ).astype(np.uint8)


def _hidden_corner(scene):
    # Cobre o canto inferior-direito da folha com a cor da mesa:
    # simula um objeto (ou mão) escondendo um canto.
    out = scene.copy()
    cv2.rectangle(out, (950, 950), (1250, 1200), (60, 60, 60), -1)
    return out


@dataclass
class VariantResult:
    id: str
    expects_sheet: bool
    verdict: str | None
    detected: bool
    corner_error_max_px: float | None
    ink_recall: float | None
    false_positive: bool

    def as_dict(self):
        return asdict(self)


def _measure(scene, truth_ink, truth_corners, expects_sheet=True):
    corners, verdict = paper.detect_sheet_with_confidence(scene)
    detected = verdict == "paper"
    corner_max = None
    recall = None
    if detected and truth_corners is not None:
        errors = np.linalg.norm(
            np.asarray(corners, dtype=np.float64)
            - np.asarray(truth_corners, dtype=np.float64),
            axis=1,
        )
        corner_max = float(errors.max())
        warped = paper.warp_sheet(scene, corners)
        recovered = ink.extract_ink(warped)
        truth_px = int((truth_ink[REGION] > 0).sum())
        hits = int(
            ((recovered[REGION] > 0) & (truth_ink[REGION] > 0)).sum()
        )
        recall = hits / max(truth_px, 1)
    return VariantResult(
        id="",
        expects_sheet=expects_sheet,
        verdict=verdict,
        detected=detected,
        corner_error_max_px=corner_max,
        ink_recall=recall,
        false_positive=bool(detected and not expects_sheet),
    )


def measure_all():
    """Roda todas as variantes e devolve a lista de resultados."""
    results = []
    base_sheet = white_sheet()
    base_truth = ink.extract_ink(base_sheet)

    def add(variant_id, scene, truth_ink, truth_corners):
        result = _measure(scene, truth_ink, truth_corners)
        result.id = variant_id
        results.append(result)

    base_scene = place_sheet(base_sheet, BASE_DST)
    add("base", base_scene, base_truth, BASE_DST)
    add("gradiente", _gradient(base_scene), base_truth, BASE_DST)
    add("sombra", _shadow(base_scene), base_truth, BASE_DST)
    add("brilho", _glare(base_scene), base_truth, BASE_DST)
    add("luz_baixa", _low_light(base_scene), base_truth, BASE_DST)
    add("inclinacao_extrema",
        place_sheet(base_sheet, TILT_DST), base_truth, TILT_DST)
    add("canto_oculto", _hidden_corner(base_scene), base_truth, BASE_DST)

    # Papéis coloridos: o portão de textura exige branco (S < 60,
    # V > 180) por desenho, então kraft deve ser recusado e amarelo
    # pálido pode passar. Medir, não afrouxar.
    kraft = draw_on_sheet(
        np.full((config.PAPER_H, config.PAPER_W, 3), (140, 180, 210),
                dtype=np.uint8)
    )
    add("papel_kraft", place_sheet(kraft, BASE_DST),
        ink.extract_ink(kraft), BASE_DST)
    pale = draw_on_sheet(
        np.full((config.PAPER_H, config.PAPER_W, 3), (210, 240, 250),
                dtype=np.uint8)
    )
    add("papel_amarelo_palido", place_sheet(pale, BASE_DST),
        ink.extract_ink(pale), BASE_DST)

    # Cena aninhada do selftest, como referência de geometria difícil.
    nested_scene, nested_truth, nested_corners = build_nested_scene()
    add("aninhada", nested_scene, nested_truth, nested_corners)

    # Negativos: nenhum pode virar papel.
    for name, scene in build_negative_scenes().items():
        result = _measure(scene, None, None, expects_sheet=False)
        result.id = f"negativo_{name.replace(' ', '_')}"
        results.append(result)
    blank = _measure(build_blank_sheet_scene(), None, None,
                      expects_sheet=False)
    blank.id = "negativo_folha_lisa"
    results.append(blank)
    return results


def render_markdown(results):
    lines = ["# Medição da captura do papel (etapa 3, sem ajuste de limiar)",
             "",
             "| variante | espera folha | veredito | detectada | "
             "erro canto (máx) | recall tinta | falso positivo |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        corner = (
            f"{r.corner_error_max_px:.2f}px"
            if r.corner_error_max_px is not None else "—"
        )
        recall = f"{r.ink_recall:.2%}" if r.ink_recall is not None else "—"
        lines.append(
            f"| {r.id} | {'sim' if r.expects_sheet else 'não'} | "
            f"{r.verdict or '—'} | {'sim' if r.detected else 'não'} | "
            f"{corner} | {recall} | {'SIM' if r.false_positive else 'não'} |"
        )
    lines.append("")
    detected = [r for r in results if r.expects_sheet and r.detected]
    expected = [r for r in results if r.expects_sheet]
    fps = [r for r in results if r.false_positive]
    lines.append(
        f"Detecção nas variantes com folha: {len(detected)}/{len(expected)}; "
        f"falsos positivos: {len(fps)}."
    )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Mede a captura do papel por condição sintética"
    )
    parser.add_argument("--format", choices=("markdown", "json"),
                        default="markdown")
    parser.add_argument("--report", help="arquivo de saída do relatório")
    args = parser.parse_args(argv)
    results = measure_all()
    if args.format == "json":
        text = json.dumps([r.as_dict() for r in results], indent=2,
                          ensure_ascii=False) + "\n"
    else:
        text = render_markdown(results)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
