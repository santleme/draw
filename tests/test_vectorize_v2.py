"""Testes de tools/vectorize.py v2: grafo O(H*W), limpeza, carimbos e Camada 2."""

import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import vectorize  # noqa: E402


def test_node_indexing_matches_naive_and_scales_linearly():
    h, w = 1200, 850
    mask = np.zeros((h, w), dtype=np.uint8)
    # ~2000 componentes de nós espalhadas pela página:
    for r in range(20, h - 20, 24):
        for c in range(20, w - 20, 20):
            mask[r : r + 2, c : c + 2] = 1
    n_nodes, labels = cv2.connectedComponents(mask, connectivity=8)
    assert n_nodes >= 1800

    t0 = time.perf_counter()
    anchors, node_pixels = vectorize._index_node_components(labels, n_nodes)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert dt_ms < 250.0, f"indexacao O(H*W) demorou demais: {dt_ms:.1f} ms"
    assert len(anchors) == n_nodes - 1

    # Confere equivalência exata em amostra determinística de nós:
    for nid in (1, n_nodes // 2, n_nodes - 1):
        ys, xs = np.nonzero(labels == nid)
        want_anchor = (float(xs.mean()), float(ys.mean()))
        want_pixels = [int(y) * w + int(x) for y, x in zip(ys, xs)]
        assert abs(anchors[nid][0] - want_anchor[0]) < 1e-9
        assert abs(anchors[nid][1] - want_anchor[1]) < 1e-9
        assert node_pixels[nid] == want_pixels


def test_thick_shapes_axis_aligned_rect_circle_lines_and_freehand_wave():
    img = np.full((vectorize.PAPER_H_PX, vectorize.PAPER_W_PX), 255, dtype=np.uint8)
    # 1. Retângulo alinhado aos eixos (traço 5 px — caso clássico que partia em 8+ pedaços):
    cv2.rectangle(img, (80, 90), (340, 290), 0, thickness=5)
    # 2. Círculo (traço 5 px):
    cv2.circle(img, (610, 210), 95, 0, thickness=5)
    # 3. Duas retas cruzadas ('X', traço 5 px):
    cv2.line(img, (110, 480), (390, 740), 0, thickness=5)
    cv2.line(img, (110, 740), (390, 480), 0, thickness=5)
    # 4. Curva senoidal à mão livre (DEVE continuar como 'polyline', sem alucinar reta/círculo):
    wave_pts = np.array(
        [
            [470 + i * 3, int(round(620 + 55.0 * np.sin(i * 0.14)))]
            for i in range(100)
        ],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.polylines(img, [wave_pts], False, 0, thickness=4)
    # 5. Poeira isolada (1-3 px) que deve sumir em min_area=4:
    img[25, 25] = 0
    img[25, 26] = 0
    img[1150, 810] = 0

    ink = img < 128
    polys = vectorize.vectorize_clean(ink, epsilon=1.5)
    prims = vectorize.classify_primitives(polys)
    by_type = {}
    for p in prims:
        by_type.setdefault(p["type"], []).append(p)

    assert len(prims) == 5, f"esperava 5 tracos limpos, obteve {len(prims)}: {[p['type'] for p in prims]}"
    assert len(by_type.get("rectangle", [])) == 1
    assert len(by_type.get("circle", [])) == 1
    assert len(by_type.get("line", [])) == 2
    assert len(by_type.get("polyline", [])) == 1

    circ = by_type["circle"][0]
    assert abs(circ["r_mm"] - 95.0 * vectorize.MM_PER_PX) < 1.2
    rect = by_type["rectangle"][0]
    dims = sorted([rect["width_mm"], rect["height_mm"]])
    want_dims = sorted([200.0 * vectorize.MM_PER_PX, 260.0 * vectorize.MM_PER_PX])
    assert abs(dims[0] - want_dims[0]) < 2.0
    assert abs(dims[1] - want_dims[1]) < 2.0


def test_stamps_order_strokes_chronologically_and_orient_direction():
    h, w = vectorize.PAPER_H_PX, vectorize.PAPER_W_PX
    img = np.full((h, w), 255, dtype=np.uint8)
    first_seen = np.zeros((h, w), dtype=np.uint32)
    last_update = np.zeros((h, w), dtype=np.uint32)

    # Três retas desenhadas de baixo para cima e da DIREITA para a ESQUERDA:
    strokes = [
        ((700, 950), (180, 950), 1000),  # desenhada primeiro (embaixo, dir->esq)
        ((700, 550), (180, 550), 3000),  # desenhada segundo (meio, dir->esq)
        ((700, 180), (180, 180), 6000),  # desenhada terceiro (topo, dir->esq)
    ]
    for (x0, y0), (x1, y1), base_t in strokes:
        cv2.line(img, (x0, y0), (x1, y1), 0, thickness=3)
        for step in range(101):
            alpha = step / 100.0
            cx = int(round(x0 + alpha * (x1 - x0)))
            cy = int(round(y0 + alpha * (y1 - y0)))
            t_val = np.uint32(base_t + step * 5)
            first_seen[cy - 3 : cy + 4, cx - 3 : cx + 4] = t_val
            last_update[cy - 3 : cy + 4, cx - 3 : cx + 4] = t_val + np.uint32(100)

    with tempfile.TemporaryDirectory() as tmp:
        png_path = Path(tmp) / "canvas.png"
        npz_path = Path(tmp) / "canvas_stamps.npz"
        svg_path = Path(tmp) / "out.svg"
        json_path = Path(tmp) / "out.json"
        cv2.imwrite(str(png_path), img)
        np.savez_compressed(
            npz_path, first_seen_ms=first_seen, last_update_ms=last_update
        )

        rc = vectorize.main(
            [
                str(png_path),
                "--clean",
                "--stamps",
                str(npz_path),
                "--primitives",
                "--out",
                str(svg_path),
                "--json",
                str(json_path),
            ]
        )
        assert rc == 0
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        prims = payload["primitives"]
        assert [p["type"] for p in prims] == ["line", "line", "line"]
        # Ordem cronológica: y decrescente (950 -> 550 -> 180):
        ys_mm = [p["y1_mm"] for p in prims]
        assert ys_mm[0] > ys_mm[1] > ys_mm[2], f"ordem temporal incorreta: {ys_mm}"
        # Orientação direita->esquerda preservada (x1 > x2 em todas as três):
        for p in prims:
            assert p["x1_mm"] > p["x2_mm"], f"sentido temporal invertido: {p}"

        svg_txt = svg_path.read_text(encoding="utf-8")
        assert "<line " in svg_txt


def main():
    test_node_indexing_matches_naive_and_scales_linearly()
    test_thick_shapes_axis_aligned_rect_circle_lines_and_freehand_wave()
    test_stamps_order_strokes_chronologically_and_orient_direction()
    print("TESTES DE VECTORIZE V2 E CAMADA 2 OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
