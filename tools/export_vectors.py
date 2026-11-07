"""Ponte de vetores do Draw: canvas confirmado -> polilinhas -> SVG + DXF.

Protótipo da ponte papel -> CAD: recebe uma imagem na grade da página
(página retificada ou o canvas de tinta acumulado), extrai a máscara de
tinta, afina para o esqueleto, percorre o esqueleto como grafo gerando
polilinhas de centro, simplifica (Douglas-Peucker via cv2.approxPolyDP)
e exporta:

  * SVG com <path> vetoriais reais (sem imagem embutida), em mm;
  * DXF mínimo escrito à mão (seção ENTITIES com LWPOLYLINE), em mm,
    eixo Y para cima como manda o CAD (o SVG mantém Y para baixo,
    como a imagem).

Diferente de tools/vectorize.py, este módulo NÃO usa cv2.ximgproc
(indisponível em vários ambientes, inclusive a VM de laboratório):
o afinamento é Zhang-Suen vetorizado em numpy, neste arquivo.

Fonte da máscara (--mask):
  * model: a máscara do próprio app — draw.page_clean.PageCleanInk
    sobre o page-clean titular (w8): mosaico de blocos 256x256 e
    limiar 0,7, o protocolo e o limiar de produção;
  * otsu: limiar de Otsu invertido sobre o cinza (tinta escura).

Uso:
  python tools/export_vectors.py pagina.png --out-dir saida/
      [--mask model|otsu] [--onnx page_clean_unet.onnx]
      [--epsilon 1.5] [--format a4|a5|a3|carta|caderno]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draw import config  # noqa: E402

DEFAULT_ONNX = (
    Path.home()
    / "workspace/draw-models-2026-11-06/assets/page_clean_unet.onnx"
)

_NEIGHBORS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
_DEGREE_KERNEL = np.array(
    [[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8
)


# ---------------------------------------------------------------- máscara

def ink_mask_model(bgr, onnx_path):
    """Máscara de tinta do app: PageCleanInk (mosaico 256, limiar 0,7).

    Usa draw.page_clean, o mesmo caminho de inferência do Draw —
    inclusive a ordem de canais e o limiar de produção (0,7).
    """
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from draw.page_clean import PageCleanInk

    cleaner = PageCleanInk(model_path=str(onnx_path))
    return cleaner.ink_mask(bgr) > 0


def ink_mask_otsu(bgr):
    """Tinta escura por Otsu invertido; devolve também a polaridade."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    mask = dark > 0
    # Se o "escuro" dominar a página, a polaridade está invertida.
    if mask.mean() > 0.5:
        mask = ~mask
    return mask


def filter_microfragments(ink, min_area=4):
    """Remove componentes conexas menores que `min_area` px."""
    if min_area <= 1:
        return ink.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return np.zeros_like(ink)
    keep = stats[:, cv2.CC_STAT_AREA] >= int(min_area)
    keep[0] = False
    return keep[labels]


# ------------------------------------------------------- afinamento (ZS)

def zhang_suen_thin(ink):
    """Esqueleto de 1 px por Zhang-Suen vetorizado (sem cv2.ximgproc)."""
    img = ink.astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            p2 = p[:-2, 1:-1]
            p3 = p[:-2, 2:]
            p4 = p[1:-1, 2:]
            p5 = p[2:, 2:]
            p6 = p[2:, 1:-1]
            p7 = p[2:, :-2]
            p8 = p[1:-1, :-2]
            p9 = p[:-2, :-2]
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            transitions = sum(
                ((seq[i] == 0) & (seq[i + 1] == 1)) for i in range(8)
            )
            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            if step == 0:
                m = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                m = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            delete = (img == 1) & (b >= 2) & (b <= 6) & (transitions == 1) & m
            if delete.any():
                img[delete] = 0
                changed = True
    return img.astype(bool)


# ------------------------------------------------------------ travessia

def make_8_minimal(skel):
    """Remove pixels de canto L redundantes preservando a conectividade-8.

    O Zhang-Suen deixa degraus 4-conexos (pixel com vizinhos N e E, que
    já são 8-vizinhos entre si): esses triângulos viram falsos nós de
    grau 3 ao longo de traços diagonais e partem curvas simples em
    dezenas de arcos curtos. Remover cada orientação em sequência
    preserva a topologia 8-conexa. (Mesma técnica de
    tools/vectorize.py, reimplementada aqui sem ximgproc.)
    """
    s = skel.copy()
    for rot in range(4):
        p = np.pad(s, 1, constant_values=False)
        c = p[1:-1, 1:-1]
        n = p[0:-2, 1:-1]
        so = p[2:, 1:-1]
        e = p[1:-1, 2:]
        w = p[1:-1, 0:-2]
        nw = p[0:-2, 0:-2]
        ne = p[0:-2, 2:]
        sw = p[2:, 0:-2]
        se = p[2:, 2:]
        if rot == 0:
            rem = c & n & e & (~w) & (~so) & (~sw) & (~(nw & se))
        elif rot == 1:
            rem = c & e & so & (~n) & (~w) & (~nw) & (~(ne & sw))
        elif rot == 2:
            rem = c & so & w & (~n) & (~e) & (~ne) & (~(nw & se))
        else:
            rem = c & w & n & (~so) & (~e) & (~se) & (~(ne & sw))
        s[rem] = False
    return s


def _degree(skel):
    skel_u8 = skel.astype(np.uint8)
    deg = cv2.filter2D(
        skel_u8, cv2.CV_16U, _DEGREE_KERNEL, borderType=cv2.BORDER_CONSTANT
    )
    return (deg * skel_u8).astype(np.int32)


def trace_polylines(skel):
    """Esqueleto -> polilinhas de centro ({points Nx2 (x,y) px, closed})."""
    height, width = skel.shape
    deg = _degree(skel)
    node_mask = skel & (deg != 2)
    n_nodes, labels = cv2.connectedComponents(
        node_mask.astype(np.uint8), connectivity=8
    )
    labels_flat = labels.ravel()
    skel_flat = skel.ravel()

    anchors = {}
    node_pixels = {}
    for nid in range(1, n_nodes):
        ys, xs = np.nonzero(labels == nid)
        if len(xs) == 0:
            continue
        anchors[nid] = (float(xs.mean()), float(ys.mean()))
        node_pixels[nid] = [int(y) * width + int(x) for y, x in zip(ys, xs)]

    def neighbors(flat_idx):
        y, x = divmod(flat_idx, width)
        out = []
        for dy, dx in _NEIGHBORS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                nidx = ny * width + nx
                if skel_flat[nidx]:
                    out.append(nidx)
        return out

    polylines = []
    visited = set()  # arestas (a, b) sem direção

    def walk(start_pixel, first_step, start_nid):
        path = [first_step]
        visited.add((start_pixel, first_step))
        visited.add((first_step, start_pixel))
        prev, cur = start_pixel, first_step
        end_anchor = None
        while labels_flat[cur] == 0:
            nbrs = [n for n in neighbors(cur) if n != prev]
            if len(nbrs) != 1:
                end_anchor = (float(cur % width), float(cur // width))
                break
            nxt = nbrs[0]
            visited.add((cur, nxt))
            visited.add((nxt, cur))
            path.append(nxt)
            prev, cur = cur, nxt
        if end_anchor is None:
            end_anchor = anchors[int(labels_flat[cur])]
        points = [anchors[start_nid]]
        points += [(float(p % width), float(p // width)) for p in path]
        points.append(end_anchor)
        closed = points[0] == points[-1] and len(path) >= 2
        return {"points": np.array(points, dtype=np.float64), "closed": closed}

    for nid, pixels in node_pixels.items():
        has_external = False
        for p in pixels:
            for q in neighbors(p):
                if labels_flat[q] == nid:
                    continue
                has_external = True
                if (p, q) not in visited:
                    polylines.append(walk(p, q, nid))
        if not has_external:
            polylines.append(
                {"points": np.array([anchors[nid]]), "closed": False}
            )

    covered = np.zeros(skel_flat.shape, dtype=bool)
    for a, _b in visited:
        covered[a] = True
    remaining = [
        int(i)
        for i in np.nonzero(skel_flat & ~covered & (labels_flat == 0))[0]
    ]
    seen = set()
    for start in remaining:
        if start in seen:
            continue
        path = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nbrs = [n for n in neighbors(cur) if n != prev]
            if not nbrs:
                break
            nxt = nbrs[0]
            if nxt == start and len(path) >= 3:
                break
            if nxt in seen:
                break
            path.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        points = [(float(p % width), float(p // width)) for p in path]
        polylines.append(
            {
                "points": np.array(points, dtype=np.float64),
                "closed": len(path) >= 3,
            }
        )
    return polylines


# --------------------------------------------------------- simplificação

def simplify_polylines(polylines, epsilon):
    """Douglas-Peucker via cv2.approxPolyDP, linha a linha."""
    out = []
    for line in polylines:
        pts = line["points"]
        if len(pts) <= 2:
            out.append(line)
            continue
        approx = cv2.approxPolyDP(
            pts.reshape(-1, 1, 2).astype(np.float32),
            float(epsilon),
            bool(line["closed"]),
        ).reshape(-1, 2).astype(np.float64)
        if len(approx) >= 2:
            out.append({"points": approx, "closed": line["closed"]})
        else:
            out.append(line)
    return out


def median_stroke_width(ink, skel):
    """Largura típica do traço: 2 x mediana do distanceTransform no esqueleto."""
    if not skel.any():
        return 1.0
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    return float(2.0 * np.median(dt[skel]))


# -------------------------------------------------------------- escritas

def _export_scale(img_w, img_h, paper_w_mm, paper_h_mm, scale=None):
    if scale is not None:
        return scale
    for name, dimensions in config.FORMATS.items():
        if dimensions == (paper_w_mm, paper_h_mm):
            unit = config.mm_per_px(name)
            return unit, unit
    return paper_w_mm / img_w, paper_h_mm / img_h


def write_svg(polylines, out_path, img_w, img_h, paper_w_mm, paper_h_mm,
              stroke_w_mm, scale_mm_per_px=None):
    """SVG com <path> reais em mm (Y para baixo, como a imagem)."""
    sx, sy = _export_scale(
        img_w, img_h, paper_w_mm, paper_h_mm, scale_mm_per_px
    )
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!-- Draw vector bridge: polilinhas de centro do canvas, em mm -->",
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{paper_w_mm}mm" height="{paper_h_mm}mm" '
        f'viewBox="0 0 {paper_w_mm} {paper_h_mm}">',
    ]
    for line in polylines:
        pts = line["points"]
        if len(pts) == 1:
            cx, cy = pts[0][0] * sx, pts[0][1] * sy
            lines.append(
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" '
                f'r="{max(stroke_w_mm / 2, 0.1):.2f}" fill="#000"/>'
            )
            continue
        d = "M " + " L ".join(f"{x * sx:.2f} {y * sy:.2f}" for x, y in pts)
        if line["closed"]:
            d += " Z"
        lines.append(
            f'<path d="{d}" fill="none" stroke="#000" '
            f'stroke-width="{stroke_w_mm:.2f}" stroke-linecap="round" '
            f'stroke-linejoin="round"/>'
        )
    lines.append("</svg>")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dxf(polylines, out_path, img_w, img_h, paper_w_mm, paper_h_mm,
              stroke_w_mm, scale_mm_per_px=None):
    """DXF mínimo (AC1015) escrito à mão: LWPOLYLINE em mm, Y para cima."""
    sx, sy = _export_scale(
        img_w, img_h, paper_w_mm, paper_h_mm, scale_mm_per_px
    )

    def conv(pts):
        return [(x * sx, paper_h_mm - y * sy) for x, y in pts]

    out = [
        "0", "SECTION", "2", "HEADER",
        "9", "$ACADVER", "1", "AC1015",
        "9", "$INSUNITS", "70", "4",
        "0", "ENDSEC",
        "0", "SECTION", "2", "TABLES",
        "0", "TABLE", "2", "LAYER", "70", "1",
        "0", "LAYER", "2", "INK", "70", "0", "62", "7", "6", "Continuous",
        "0", "ENDTAB",
        "0", "ENDSEC",
        "0", "SECTION", "2", "BLOCKS",
        "0", "ENDSEC",
        "0", "SECTION", "2", "ENTITIES",
    ]
    for line in polylines:
        pts = conv(line["points"])
        if len(pts) == 1:
            pts = pts + pts  # LWPOLYLINE degenerada de 1 vértice não é válida
        out += [
            "0", "LWPOLYLINE",
            "100", "AcDbEntity", "8", "INK",
            "100", "AcDbPolyline",
            "90", str(len(pts)),
            "70", "1" if line["closed"] else "0",
            "43", f"{stroke_w_mm:.3f}",
        ]
        for x, y in pts:
            out += ["10", f"{x:.3f}", "20", f"{y:.3f}"]
    out += ["0", "ENDSEC", "0", "EOF"]
    Path(out_path).write_text("\n".join(out) + "\n", encoding="utf-8")


# ------------------------------------------------- prova (round-trip IoU)

def parse_svg_paths(svg_path):
    """Lê de volta os <path> do SVG: lista de (Nx2 mm, closed)."""
    text = Path(svg_path).read_text(encoding="utf-8")
    out = []
    for d in re.findall(r'<path d="([^"]+)"', text):
        nums = [float(v) for v in re.findall(r"-?\d+\.?\d*", d)]
        pts = np.array(nums, dtype=np.float64).reshape(-1, 2)
        out.append((pts, d.strip().upper().endswith("Z")))
    return out


def parse_dxf_polylines(dxf_path):
    """Lê de volta as LWPOLYLINE do DXF: lista de (Nx2 mm, closed)."""
    tokens = [
        t.strip() for t in Path(dxf_path).read_text(encoding="utf-8").splitlines()
    ]
    pairs = list(zip(tokens[0::2], tokens[1::2]))
    out = []
    i = 0
    while i < len(pairs):
        code, value = pairs[i]
        if code == "0" and value == "LWPOLYLINE":
            verts = []
            closed = False
            j = i + 1
            pending_x = None
            while j < len(pairs) and pairs[j][0] != "0":
                c, v = pairs[j]
                if c == "70":
                    closed = v == "1"
                elif c == "10":
                    pending_x = float(v)
                elif c == "20" and pending_x is not None:
                    verts.append((pending_x, float(v)))
                    pending_x = None
                j += 1
            if verts:
                out.append((np.array(verts, dtype=np.float64), closed))
            i = j
        else:
            i += 1
    return out


def rasterize(mm_polylines, shape, sx, sy, thickness_px, flip_y=False,
              paper_h_mm=None):
    """Desenha polilinhas (mm) de volta na grade de pixels da página."""
    raster = np.zeros(shape, dtype=np.uint8)
    for pts_mm, closed in mm_polylines:
        pts = pts_mm.copy()
        if flip_y:
            pts[:, 1] = paper_h_mm - pts[:, 1]
        px = np.round(pts / np.array([sx, sy])).astype(np.int32)
        if len(px) == 1:
            cv2.circle(raster, tuple(px[0]), max(thickness_px // 2, 0), 1, -1)
        else:
            cv2.polylines(
                raster, [px.reshape(-1, 1, 2)], bool(closed), 1,
                thickness=max(int(thickness_px), 1), lineType=cv2.LINE_AA,
            )
    return raster > 0


def iou(a, b):
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else 1.0


# ------------------------------------------------------------------ core

def export_vectors(bgr, mask_source="model", onnx_path=None, epsilon=1.5,
                   paper_mm=None, min_area=4, paper_format="a4"):
    """Imagem na grade da página -> dict com polilinhas, SVG/DXF em memória
    (listas de linhas via writers) e estatísticas da vetorização."""
    img_h, img_w = bgr.shape[:2]
    if paper_mm is None:
        paper_w_mm, paper_h_mm = config.FORMATS[paper_format]
        sx = sy = config.mm_per_px(paper_format)
    else:
        # --paper-mm explicito preserva a API anterior de dimensoes livres.
        paper_w_mm, paper_h_mm = paper_mm
        sx, sy = paper_w_mm / img_w, paper_h_mm / img_h
    if mask_source == "model":
        if onnx_path is None or not Path(onnx_path).exists():
            raise FileNotFoundError(
                f"modelo page-clean nao encontrado: {onnx_path}"
            )
        ink = ink_mask_model(bgr, onnx_path)
    else:
        ink = ink_mask_otsu(bgr)
    ink = filter_microfragments(ink, min_area=min_area)

    skel = make_8_minimal(zhang_suen_thin(ink))
    raw = trace_polylines(skel)
    simplified = simplify_polylines(raw, epsilon)
    width_px = median_stroke_width(ink, skel)

    return {
        "ink": ink,
        "skeleton": skel,
        "raw_polylines": raw,
        "polylines": simplified,
        "stroke_width_px": width_px,
        "stroke_width_mm": width_px * 0.5 * (sx + sy),
        "scale_mm_per_px": (sx, sy),
        "points_raw": int(sum(len(line["points"]) for line in raw)),
        "points_simplified": int(sum(len(line["points"]) for line in simplified)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Exporta uma página/canvas do Draw para SVG + DXF vetoriais."
    )
    parser.add_argument("input", help="PNG na grade da página (warp ou canvas)")
    parser.add_argument("--out-dir", default="vector-out", help="pasta de saída")
    parser.add_argument(
        "--mask", choices=["model", "otsu"], default="model",
        help="fonte da máscara de tinta (padrao: model = page-clean titular)",
    )
    parser.add_argument(
        "--onnx", default=str(DEFAULT_ONNX),
        help="page-clean .onnx para --mask model",
    )
    parser.add_argument(
        "--epsilon", type=float, default=1.5,
        help="epsilon do Douglas-Peucker em px (padrao 1,5)",
    )
    size = parser.add_mutually_exclusive_group()
    size.add_argument(
        "--format", choices=list(config.FORMATS), default="a4",
        help="formato da sessao; escala de draw.config (padrao: a4)",
    )
    size.add_argument(
        "--paper-mm", type=float, nargs=2, default=None,
        metavar=("W", "H"), help="dimensoes livres, API legada (sem --format)",
    )
    args = parser.parse_args(argv)

    bgr = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"erro: nao foi possivel ler {args.input}", file=sys.stderr)
        return 1
    img_h, img_w = bgr.shape[:2]
    paper_w_mm, paper_h_mm = (
        config.FORMATS[args.format] if args.paper_mm is None else args.paper_mm
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem

    t0 = time.perf_counter()
    result = export_vectors(
        bgr,
        mask_source=args.mask,
        onnx_path=args.onnx,
        epsilon=args.epsilon,
        paper_mm=args.paper_mm,
        paper_format=args.format,
    )
    svg_path = out_dir / f"{stem}.svg"
    dxf_path = out_dir / f"{stem}.dxf"
    write_svg(
        result["polylines"], svg_path, img_w, img_h,
        paper_w_mm, paper_h_mm, result["stroke_width_mm"],
        scale_mm_per_px=result["scale_mm_per_px"],
    )
    write_dxf(
        result["polylines"], dxf_path, img_w, img_h,
        paper_w_mm, paper_h_mm, result["stroke_width_mm"],
        scale_mm_per_px=result["scale_mm_per_px"],
    )

    # Prova: relê os DOIS arquivos do disco, rasteriza e compara com a máscara.
    sx, sy = result["scale_mm_per_px"]
    thickness = max(int(round(result["stroke_width_px"])), 1)
    ink = result["ink"]
    raster_svg = rasterize(
        parse_svg_paths(svg_path), ink.shape, sx, sy, thickness
    )
    raster_dxf = rasterize(
        parse_dxf_polylines(dxf_path), ink.shape, sx, sy, thickness,
        flip_y=True, paper_h_mm=paper_h_mm,
    )
    iou_svg = iou(raster_svg, ink)
    iou_dxf = iou(raster_dxf, ink)

    overlay = bgr.copy()
    overlay[raster_svg] = (0, 0, 255)
    cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)
    cv2.imwrite(
        str(out_dir / f"{stem}_mask.png"), (ink.astype(np.uint8) * 255)
    )

    proof = {
        "input": str(args.input),
        "mask_source": (
            f"draw.page_clean.PageCleanInk sobre {args.onnx} "
            "(mosaico 256x256, limiar 0.7 — protocolo do app)"
            if args.mask == "model"
            else "otsu invertido (declarado)"
        ),
        "page_px": [img_w, img_h],
        "paper_mm": [paper_w_mm, paper_h_mm],
        "paper_format": args.format if args.paper_mm is None else None,
        "scale_mm_per_px": result["scale_mm_per_px"],
        "epsilon_px": args.epsilon,
        "ink_px": int(ink.sum()),
        "skeleton_px": int(result["skeleton"].sum()),
        "polylines": len(result["polylines"]),
        "points_before_simplification": result["points_raw"],
        "points_after_simplification": result["points_simplified"],
        "stroke_width_px_median": round(result["stroke_width_px"], 2),
        "svg_bytes": svg_path.stat().st_size,
        "dxf_bytes": dxf_path.stat().st_size,
        "iou_svg_roundtrip": round(iou_svg, 4),
        "iou_dxf_roundtrip": round(iou_dxf, 4),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    (out_dir / f"{stem}_proof.json").write_text(
        json.dumps(proof, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(proof, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
