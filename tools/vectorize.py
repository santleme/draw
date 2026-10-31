"""Vetorização offline do canvas de tinta: raster -> grafo limpo -> polilinhas/primitivas -> SVG/JSON.

Ferramenta offline, sem câmera: recebe o PNG de um canvas acumulado
(850x1200, espaço do papel) — e opcionalmente o arquivo `_stamps.npz`
gerado ao salvar no Draw — e devolve as polilinhas e primitivas
geométricas (Camada 2 de VISION.md: reta, círculo, retângulo, polilinha).

Pipeline:
  1. binarização + filtro opcional de microfragmentos de poeira;
  2. afinamento com cv2.ximgproc.thinning (Zhang-Suen ou Guo-Hall);
  3. travessia indexada O(H*W) do esqueleto como grafo (~225x mais
     rápida que varredura por nó em páginas densas);
  4. poda opcional de esporões guiada por cv2.distanceTransform +
     contração de junções residuais de grau 2;
  5. costura opcional de ramos colineares em junções (ponderada por
     ângulo tangente e continuidade temporal dos carimbos quando
     disponíveis);
  6. ordenação temporal (por first_seen_ms) ou espacial (vizinho mais
     próximo com inversão de sentido para reduzir saltos de caneta);
  7. simplificação Douglas-Peucker + classificação de primitivas
     geométricas (line, circle, rectangle, polygon, polyline) e
     exportação SVG/JSON em milímetros do A4.

Uso:
  python tools/vectorize.py canvas.png [--out saida.svg] [--clean]
      [--stamps canvas_stamps.npz] [--primitives] [--json saida.json]
      [--epsilon 1.5] [--thinning zhangsuen|guohall] [--stats]
  python tools/vectorize.py --selfcheck
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

PAPER_W_PX = 850
PAPER_H_PX = 1200
PAPER_W_MM = 210.0
PAPER_H_MM = 297.0
MM_PER_PX_X = PAPER_W_MM / PAPER_W_PX
MM_PER_PX_Y = PAPER_H_MM / PAPER_H_PX
MM_PER_PX = 0.5 * (MM_PER_PX_X + MM_PER_PX_Y)

THINNING_TYPES = {
    "zhangsuen": cv2.ximgproc.THINNING_ZHANGSUEN,
    "guohall": cv2.ximgproc.THINNING_GUOHALL,
}

_NEIGHBORS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
_DEGREE_KERNEL = np.array(
    [[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8
)


def load_ink(path):
    """Lê um PNG de canvas e devolve (mascara booleana de tinta, polaridade)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"nao foi possivel ler {path}")
    dark = img < 128
    if int(dark.sum()) <= int((~dark).sum()):
        return dark, "escura"
    return ~dark, "clara"


def load_stamps(path):
    """Carrega planos de carimbo (`first_seen_ms`, `last_update_ms`) de um `.npz`."""
    data = np.load(str(path))
    first = data["first_seen_ms"] if "first_seen_ms" in data else None
    last = (
        data["last_update_ms"]
        if "last_update_ms" in data
        else (data["last_seen_ms"] if "last_seen_ms" in data else None)
    )
    return {"first_seen_ms": first, "last_update_ms": last}


def filter_microfragments(ink, min_area=4):
    """Remove componentes conexas minúsculas (< min_area px) antes de afinar."""
    if min_area <= 1:
        return ink.astype(bool, copy=True)
    ink_u8 = ink.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink_u8, connectivity=8
    )
    if n_labels <= 1:
        return np.zeros_like(ink, dtype=bool)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep_lut = areas >= int(min_area)
    keep_lut[0] = False
    return keep_lut[labels]


def thin(ink, thinning="zhangsuen"):
    """Afina a máscara de tinta para um esqueleto de 1 px (booleano)."""
    src = ink.astype(np.uint8) * np.uint8(255)
    skel = cv2.ximgproc.thinning(src, None, THINNING_TYPES[thinning])
    return skel > 0


def _degree(skel):
    """Grau de cada pixel do esqueleto (vizinhança de 8)."""
    skel_u8 = skel.astype(np.uint8)
    deg = cv2.filter2D(
        skel_u8, cv2.CV_16U, _DEGREE_KERNEL, borderType=cv2.BORDER_CONSTANT
    )
    return (deg * skel_u8).astype(np.int32)


def _skeleton_neighbors(flat_idx, skel_flat, width, height):
    """Índices planos dos vizinhos de 8 que também são esqueleto."""
    y, x = divmod(flat_idx, width)
    out = []
    for dy, dx in _NEIGHBORS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < height and 0 <= nx < width:
            nidx = ny * width + nx
            if skel_flat[nidx]:
                out.append(nidx)
    return out


def _index_node_components(labels, n_nodes):
    """Indexa pixels e centróides de todos os nós numa única passada O(H*W)."""
    labels_flat = labels.ravel()
    width = labels.shape[1]
    nz_flat = np.flatnonzero(labels_flat)
    if len(nz_flat) == 0 or n_nodes <= 1:
        return {}, {}

    lbl_vals = labels_flat[nz_flat]
    order = np.argsort(lbl_vals, kind="mergesort")
    sorted_flat = nz_flat[order]
    sorted_lbl = lbl_vals[order]
    counts = np.bincount(sorted_lbl, minlength=n_nodes)
    splits = np.cumsum(counts)

    ys = sorted_flat // width
    xs = sorted_flat % width
    sum_x = np.bincount(sorted_lbl, weights=xs, minlength=n_nodes)
    sum_y = np.bincount(sorted_lbl, weights=ys, minlength=n_nodes)

    anchors = {}
    node_pixels = {}
    for nid in range(1, n_nodes):
        c = int(counts[nid])
        if c == 0:
            continue
        anchors[nid] = (float(sum_x[nid] / c), float(sum_y[nid] / c))
        lo = int(splits[nid - 1])
        hi = int(splits[nid])
        node_pixels[nid] = sorted_flat[lo:hi].tolist()
    return anchors, node_pixels


def traverse_graph(skel, include_topology=False):
    """Percorre o esqueleto como grafo e devolve as polilinhas brutas.

    Cada polilinha é um dict {"points": array Nx2 (x, y) em px,
    "closed": bool}. Com `include_topology=True`, inclui também
    `start_node`, `end_node` e `pixel_indices` para poda e costura O(E).
    """
    height, width = skel.shape
    deg = _degree(skel)
    node_mask = skel & (deg != 2)
    n_nodes, labels = cv2.connectedComponents(
        node_mask.astype(np.uint8), connectivity=8
    )
    labels_flat = labels.ravel()
    skel_flat = skel.ravel()
    anchors, node_pixels = _index_node_components(labels, n_nodes)
    node_degrees = {
        nid: int(deg.ravel()[pix].max()) for nid, pix in node_pixels.items()
    }

    polylines = []
    visited = set()

    def walk_edge(start_pixel, first_step, start_nid, start_anchor):
        path = [first_step]
        visited.add((start_pixel, first_step))
        visited.add((first_step, start_pixel))
        prev, cur = start_pixel, first_step
        end_anchor = None
        end_nid = None
        while labels_flat[cur] == 0:
            nbrs = [
                n
                for n in _skeleton_neighbors(cur, skel_flat, width, height)
                if n != prev
            ]
            if len(nbrs) != 1:
                end_anchor = (float(cur % width), float(cur // width))
                break
            nxt = nbrs[0]
            visited.add((cur, nxt))
            visited.add((nxt, cur))
            path.append(nxt)
            prev, cur = cur, nxt
        if end_anchor is None:
            end_nid = int(labels_flat[cur])
            end_anchor = anchors[end_nid]
        points = [start_anchor]
        points += [(float(p % width), float(p // width)) for p in path]
        points.append(end_anchor)
        closed = points[0] == points[-1] and len(path) >= 2
        item = {"points": np.array(points, dtype=np.float64), "closed": closed}
        if include_topology:
            item["start_node"] = start_nid
            item["end_node"] = end_nid
            item["start_deg"] = node_degrees.get(start_nid, 0)
            item["end_deg"] = node_degrees.get(end_nid, 0)
            item["flat_pixels"] = [start_pixel] + path
        return item

    for nid in range(1, n_nodes):
        if nid not in anchors:
            continue
        anchor = anchors[nid]
        has_external = False
        for p in node_pixels[nid]:
            for q in _skeleton_neighbors(p, skel_flat, width, height):
                if labels_flat[q] == nid:
                    continue
                has_external = True
                if (p, q) in visited:
                    continue
                polylines.append(walk_edge(p, q, nid, anchor))
        if not has_external:
            item = {
                "points": np.array([anchor], dtype=np.float64),
                "closed": False,
            }
            if include_topology:
                item["start_node"] = nid
                item["end_node"] = nid
                item["start_deg"] = 0
                item["end_deg"] = 0
                item["flat_pixels"] = list(node_pixels[nid])
            polylines.append(item)

    covered = np.zeros(skel_flat.shape, dtype=bool)
    for a, b in visited:
        covered[a] = True
        covered[b] = True
    remaining = np.nonzero(skel_flat & ~covered & (labels_flat == 0))[0]
    seen = set()
    for start in remaining:
        start = int(start)
        if start in seen:
            continue
        path = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nbrs = [
                n
                for n in _skeleton_neighbors(cur, skel_flat, width, height)
                if n != prev
            ]
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
        closed = len(path) >= 3
        item = {"points": np.array(points, dtype=np.float64), "closed": closed}
        if include_topology:
            item["start_node"] = None
            item["end_node"] = None
            item["start_deg"] = 2
            item["end_deg"] = 2
            item["flat_pixels"] = path
        polylines.append(item)

    return polylines


def _polyline_arc_length(points):
    if len(points) <= 1:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())


def _make_8_minimal(skel):
    """Remove pixels de canto L redundantes preservando a conectividade-8 exata.

    O afinamento de Zhang-Suen deixa degraus 4-conexos (ex.: pixel central
    com N=1 e E=1 sem vizinhos opostos), onde N e E já são 8-vizinhos entre
    si. Esses triângulos criam falsos nós de grau 3 ao longo de círculos e
    retângulos alinhados aos eixos, partindo laços fechados em dezenas de
    arcos curtos. Remover cada orientação sequencialmente preserva a
    topologia 8-conexa e reduz curvas simples a grau 2 puro.
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


def prune_skeleton_spurs(
    skel, ink, max_spur_ratio=2.4, min_spur_px=7.0, passes=2
):
    """Remove cantos L redundantes e esporões curtos guiado por distanceTransform."""
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    cleaned = _make_8_minimal(skel)
    width = cleaned.shape[1]

    for _ in range(passes):
        edges = traverse_graph(cleaned, include_topology=True)
        incident_count = {}
        for e in edges:
            if e["closed"]:
                continue
            for key in ("start_node", "end_node"):
                nid = e.get(key)
                if nid is not None:
                    incident_count[nid] = incident_count.get(nid, 0) + 1

        removed_any = False
        flat_clean = cleaned.ravel()
        for e in edges:
            if e["closed"] or len(e["points"]) <= 1:
                continue
            u, v = e.get("start_node"), e.get("end_node")
            if u is None or v is None or u == v:
                continue
            du = incident_count.get(u, 0)
            dv = incident_count.get(v, 0)
            if not ((du == 1 and dv >= 3) or (dv == 1 and du >= 3)):
                continue
            j_pt = e["points"][-1] if du == 1 else e["points"][0]
            jx = int(np.clip(round(j_pt[0]), 0, width - 1))
            jy = int(np.clip(round(j_pt[1]), 0, cleaned.shape[0] - 1))
            local_r = float(dt[jy, jx])
            limit = max(float(min_spur_px), float(max_spur_ratio) * local_r)
            if _polyline_arc_length(e["points"]) <= limit:
                # Apaga os pixels do ramo exceto os que pertencem à junção.
                leaf_First = du == 1
                pix = e["flat_pixels"]
                to_clear = pix[:-1] if leaf_First else pix[1:]
                for fidx in to_clear:
                    flat_clean[fidx] = False
                # Limpa também o pixel folha na ponta.
                lp = e["points"][0] if leaf_First else e["points"][-1]
                lx = int(np.clip(round(lp[0]), 0, width - 1))
                ly = int(np.clip(round(lp[1]), 0, cleaned.shape[0] - 1))
                flat_clean[ly * width + lx] = False
                removed_any = True
        if not removed_any:
            break
        cleaned = _make_8_minimal(cleaned)
    return cleaned


def _edge_stamp_At_end(points, at_end, stamp_plane):
    """Lê carimbo mediano perto da ponta (`at_end=False` início, `True` fim)."""
    if stamp_plane is None or len(points) == 0:
        return None
    h, w = stamp_plane.shape
    sample = points[-min(6, len(points)) :] if at_end else points[: min(6, len(points))]
    xs = np.clip(np.round(sample[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(sample[:, 1]).astype(int), 0, h - 1)
    vals = stamp_plane[ys, xs]
    valid = vals[vals > 0]
    if len(valid) == 0:
        return None
    return float(np.median(valid))


def _outward_tangent(points, from_end, lookahead=8):
    """Vetor unitário apontando da ponta para DENTRO da aresta."""
    n = len(points)
    if n < 2:
        return np.array([1.0, 0.0], dtype=np.float64)
    if not from_end:
        idx = min(n - 1, lookahead)
        vec = points[idx] - points[0]
    else:
        idx = max(0, n - 1 - lookahead)
        vec = points[idx] - points[-1]
    norm = float(np.hypot(vec[0], vec[1]))
    if norm < 1e-9:
        vec = (points[-1] - points[0]) if not from_end else (points[0] - points[-1])
        norm = max(float(np.hypot(vec[0], vec[1])), 1e-9)
    return vec / norm


def stitch_junctions(polylines, max_turn_deg=32.0, stamps=None):
    """Costura pares de arestas que atravessam uma junção mantendo tangente contínua.

    Dois ramos incidentes num mesmo nó que saem em sentidos quase opostos
    (`ângulo de desvio <= max_turn_deg`, ex.: as duas metades de cada reta
    de um 'X') são fundidos numa única polilinha contínua. Quando `stamps`
    está disponível, desempata cruzamentos pelo menor salto temporal.
    """
    stamp_plane = None
    if isinstance(stamps, dict):
        stamp_plane = (
            stamps.get("first_seen_ms")
            if stamps.get("first_seen_ms") is not None
            else stamps.get("last_update_ms")
        )
    elif isinstance(stamps, np.ndarray):
        stamp_plane = stamps

    open_edges = []
    passthrough = []
    for item in polylines:
        if item["closed"] or len(item["points"]) < 2:
            passthrough.append(
                {"points": item["points"], "closed": item["closed"]}
            )
        else:
            open_edges.append(
                {
                    "points": item["points"].copy(),
                    "u": item.get("start_node"),
                    "v": item.get("end_node"),
                }
            )

    if not open_edges:
        return passthrough

    # Agrupa incidências por nó compartilhado (ou proximidade <= 2,5 px caso sem id).
    changed = True
    cos_min = math.cos(math.radians(float(max_turn_deg)))
    while changed:
        changed = False
        by_node = {}
        for eid, edge in enumerate(open_edges):
            if edge is None:
                continue
            for is_end, nid in ((False, edge["u"]), (True, edge["v"])):
                if nid is not None:
                    by_node.setdefault(nid, []).append((eid, is_end))

        best_pair = None
        best_cost = 1e18
        for nid, inc in by_node.items():
            if len(inc) < 2:
                continue
            for i in range(len(inc)):
                eid_a, end_a = inc[i]
                ea = open_edges[eid_a]
                if ea is None:
                    continue
                tan_a = _outward_tangent(ea["points"], end_a)
                ta = _edge_stamp_At_end(ea["points"], end_a, stamp_plane)
                for j in range(i + 1, len(inc)):
                    eid_b, end_b = inc[j]
                    if eid_a == eid_b:
                        continue
                    eb = open_edges[eid_b]
                    if eb is None:
                        continue
                    tan_b = _outward_tangent(eb["points"], end_b)
                    # Ramos colineares que atravessam a junção apontam em sentidos opostos:
                    # tan_a · tan_b ~ -1  =>  -tan_a · tan_b >= cos_min.
                    align = float(-np.dot(tan_a, tan_b))
                    if align < cos_min:
                        continue
                    turn_deg = math.degrees(math.acos(min(1.0, max(-1.0, align))))
                    cost = turn_deg
                    if stamp_plane is not None:
                        tb = _edge_stamp_At_end(eb["points"], end_b, stamp_plane)
                        if ta is not None and tb is not None:
                            cost += 0.015 * min(abs(ta - tb), 2000.0)
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = (eid_a, end_a, eid_b, end_b)

        if best_pair is not None:
            eid_a, end_a, eid_b, end_b = best_pair
            ea = open_edges[eid_a]
            eb = open_edges[eid_b]
            pts_a = ea["points"] if end_a else ea["points"][::-1]
            u_new = ea["u"] if end_a else ea["v"]
            pts_b = eb["points"][::-1] if end_b else eb["points"]
            v_new = eb["u"] if end_b else eb["v"]
            merged_pts = np.vstack([pts_a, pts_b[1:]])
            if u_new is not None and u_new == v_new and len(merged_pts) >= 4:
                passthrough.append({"points": merged_pts[:-1], "closed": True})
                open_edges[eid_a] = None
            else:
                open_edges[eid_a] = {
                    "points": merged_pts,
                    "u": u_new,
                    "v": v_new,
                }
            open_edges[eid_b] = None
            changed = True

    for edge in open_edges:
        if edge is not None:
            passthrough.append({"points": edge["points"], "closed": False})
    return passthrough


def annotate_and_order_polylines(polylines, stamps=None):
    """Anota carimbos temporais quando presentes e ordena para minimizar saltos."""
    first_plane = None
    last_plane = None
    if isinstance(stamps, dict):
        first_plane = stamps.get("first_seen_ms")
        last_plane = stamps.get("last_update_ms")
    elif isinstance(stamps, np.ndarray):
        first_plane = stamps

    enriched = []
    for item in polylines:
        pts = item["points"].copy()
        closed = bool(item["closed"])
        t_start = None
        t_end = None
        if first_plane is not None and len(pts) > 0:
            h, w = first_plane.shape
            xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
            ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
            vals = first_plane[ys, xs]
            valid = vals[vals > 0]
            if len(valid) > 0:
                t_start = int(np.percentile(valid, 10))
                lvals = (
                    last_plane[ys, xs] if last_plane is not None else vals
                )
                lvalid = lvals[lvals > 0]
                t_end = int(np.percentile(lvalid, 90)) if len(lvalid) > 0 else t_start
                # Orienta polilinha aberta do menor para o maior carimbo.
                if not closed and len(pts) >= 4:
                    q = max(1, len(pts) // 4)
                    head_v = vals[:q][vals[:q] > 0]
                    tail_v = vals[-q:][vals[-q:] > 0]
                    if len(head_v) > 0 and len(tail_v) > 0:
                        if float(np.median(head_v)) > float(np.median(tail_v)):
                            pts = pts[::-1].copy()
        out = {"points": pts, "closed": closed}
        if t_start is not None:
            out["t_start_ms"] = t_start
            out["t_end_ms"] = t_end
        enriched.append(out)

    if len(enriched) <= 1:
        return enriched

    if any("t_start_ms" in it for it in enriched):
        return sorted(
            enriched,
            key=lambda it: (
                it.get("t_start_ms", 1 << 60),
                float(it["points"][0, 1]),
                float(it["points"][0, 0]),
            ),
        )

    # Sem carimbos: encadeamento guloso pelo extremo mais próximo para
    # minimizar o deslocamento com caneta levantada (saltos no plotter/SVG).
    remaining = list(range(len(enriched)))
    # Começa pelo traço mais próximo da origem superior-esquerda:
    first_idx = min(
        remaining,
        key=lambda i: float(
            np.hypot(enriched[i]["points"][0, 0], enriched[i]["points"][0, 1])
        ),
    )
    remaining.remove(first_idx)
    ordered = [enriched[first_idx]]
    cur_pos = ordered[0]["points"][0 if ordered[0]["closed"] else -1]

    while remaining:
        best_i = None
        best_flip = False
        best_d2 = 1e30
        for i in remaining:
            pts = enriched[i]["points"]
            d0 = float(np.sum((pts[0] - cur_pos) ** 2))
            if d0 < best_d2:
                best_d2 = d0
                best_i = i
                best_flip = False
            if not enriched[i]["closed"] and len(pts) > 1:
                d1 = float(np.sum((pts[-1] - cur_pos) ** 2))
                if d1 < best_d2:
                    best_d2 = d1
                    best_i = i
                    best_flip = True
        remaining.remove(best_i)
        chosen = dict(enriched[best_i])
        if best_flip:
            chosen["points"] = chosen["points"][::-1].copy()
        ordered.append(chosen)
        cur_pos = chosen["points"][0 if chosen["closed"] else -1]
    return ordered


def _dp_open(points, epsilon):
    """Douglas-Peucker iterativo sobre uma polilinha aberta."""
    n = len(points)
    if n <= 2:
        return points
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        pa, pb = points[a], points[b]
        ab = pb - pa
        norm = np.hypot(ab[0], ab[1])
        seg = points[a + 1 : b]
        if norm == 0:
            dists = np.hypot(seg[:, 0] - pa[0], seg[:, 1] - pa[1])
        else:
            cross = np.abs(
                ab[0] * (pa[1] - seg[:, 1]) - (pa[0] - seg[:, 0]) * ab[1]
            )
            dists = cross / norm
        i = int(np.argmax(dists))
        if dists[i] > epsilon:
            keep[a + 1 + i] = True
            stack.append((a, a + 1 + i))
            stack.append((a + 1 + i, b))
    return points[keep]


def simplify(points, epsilon, closed):
    """Douglas-Peucker; em laço fechado, divide em dois arcos pelas âncoras extremas."""
    if len(points) <= 2 or epsilon <= 0:
        return points
    if not closed:
        return _dp_open(points, epsilon)
    centroid = points.mean(axis=0)
    i_a = int(np.argmax(np.hypot(*(points - centroid).T)))
    i_b = int(np.argmax(np.hypot(*(points - points[i_a]).T)))
    if i_a == i_b:
        return _dp_open(points, epsilon)
    lo, hi = min(i_a, i_b), max(i_a, i_b)
    arc1 = points[lo : hi + 1]
    arc2 = np.vstack([points[hi:], points[: lo + 1]])
    s1 = _dp_open(arc1, epsilon)
    s2 = _dp_open(arc2, epsilon)
    return np.vstack([s1, s2[1:-1] if len(s2) > 2 else s2[1:]])


def _fit_circle_kasa(pts):
    """Ajuste algébrico de mínimos quadrados (Kåsa) para (cx, cy, r)."""
    x = pts[:, 0]
    y = pts[:, 1]
    a = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    sol, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    if rank < 3:
        return None
    cx, cy, c0 = float(sol[0]), float(sol[1]), float(sol[2])
    r2 = c0 + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return cx, cy, math.sqrt(r2)


def _merge_collinear_closed_vertices(verts, collinear_tol_deg=16.0):
    """Remove vértices quase rasos (~180°) ao longo das arestas de um polígono fechado."""
    pts = list(verts)
    if len(pts) >= 2 and np.hypot(*(pts[0] - pts[-1])) < 1e-6:
        pts = pts[:-1]
    changed = True
    while changed and len(pts) > 4:
        changed = False
        n = len(pts)
        for i in range(n):
            p_prev = pts[(i - 1) % n]
            p_cur = pts[i]
            p_next = pts[(i + 1) % n]
            v1 = p_cur - p_prev
            v2 = p_next - p_cur
            n1 = float(np.hypot(v1[0], v1[1]))
            n2 = float(np.hypot(v2[0], v2[1]))
            if n1 < 2.0 or n2 < 2.0:
                pts.pop(i)
                changed = True
                break
            cos_turn = float(np.dot(v1, v2) / (n1 * n2))
            turn_deg = math.degrees(math.acos(min(1.0, max(-1.0, cos_turn))))
            if turn_deg <= collinear_tol_deg:
                pts.pop(i)
                changed = True
                break
    return np.array(pts, dtype=np.float64)


def classify_primitive(polyline):
    """Classifica uma polilinha em primitiva geométrica da Camada 2 (VISION.md).

    Retorna um dicionário com `type` em `{"point", "line", "circle",
    "rectangle", "polygon", "polyline"}` e parâmetros métricos em mm/px.
    """
    pts = polyline["points"]
    closed = bool(polyline["closed"])
    meta = {}
    if "t_start_ms" in polyline:
        meta["t_start_ms"] = polyline["t_start_ms"]
        meta["t_end_ms"] = polyline["t_end_ms"]

    if len(pts) == 1:
        return {
            "type": "point",
            "x_mm": round(float(pts[0, 0] * MM_PER_PX_X), 3),
            "y_mm": round(float(pts[0, 1] * MM_PER_PX_Y), 3),
            **meta,
        }

    dense = dense_samples(pts, closed, step=2.0)

    # 1. Reta aberta:
    if not closed:
        p0, p1 = pts[0], pts[-1]
        chord_vec = p1 - p0
        chord_len = float(np.hypot(chord_vec[0], chord_vec[1]))
        arc_len = _polyline_arc_length(pts)
        if chord_len >= 10.0 and arc_len <= 1.05 * chord_len:
            cross = np.abs(
                chord_vec[0] * (p0[1] - dense[:, 1])
                - (p0[0] - dense[:, 0]) * chord_vec[1]
            )
            max_dev = float(np.max(cross) / chord_len)
            if max_dev <= max(2.5, 0.025 * chord_len):
                return {
                    "type": "line",
                    "x1_mm": round(float(p0[0] * MM_PER_PX_X), 3),
                    "y1_mm": round(float(p0[1] * MM_PER_PX_Y), 3),
                    "x2_mm": round(float(p1[0] * MM_PER_PX_X), 3),
                    "y2_mm": round(float(p1[1] * MM_PER_PX_Y), 3),
                    "length_mm": round(float(chord_len * MM_PER_PX), 3),
                    "max_dev_px": round(max_dev, 3),
                    **meta,
                }

    # 2. Círculo (fechado ou quase fechado):
    end_gap = 0.0 if closed else float(np.hypot(*(pts[-1] - pts[0])))
    perim = _polyline_arc_length(
        np.vstack([pts, pts[:1]]) if closed else pts
    )
    if (closed or end_gap <= max(8.0, 0.08 * perim)) and len(dense) >= 16:
        fit = _fit_circle_kasa(dense)
        if fit is not None:
            cx, cy, r = fit
            if r >= 6.0:
                radii = np.hypot(dense[:, 0] - cx, dense[:, 1] - cy)
                rmse = float(np.sqrt(np.mean((radii - r) ** 2)))
                max_err = float(np.max(np.abs(radii - r)))
                angles = np.mod(
                    np.degrees(np.arctan2(dense[:, 1] - cy, dense[:, 0] - cx)),
                    360.0,
                )
                bins = np.unique((angles // 15.0).astype(int))
                # >= 21 de 24 setores de 15° cobertos (>= 315°).
                if (
                    len(bins) >= 21
                    and rmse / r <= 0.055
                    and max_err <= max(3.5, 0.10 * r)
                ):
                    return {
                        "type": "circle",
                        "cx_mm": round(cx * MM_PER_PX_X, 3),
                        "cy_mm": round(cy * MM_PER_PX_Y, 3),
                        "r_mm": round(r * MM_PER_PX, 3),
                        "radial_rmse_px": round(rmse, 3),
                        **meta,
                    }

    # 3. Retângulo fechado:
    if closed and len(pts) >= 4:
        coarse = _dp_open(np.vstack([pts, pts[:1]]), epsilon=3.0)
        corners = _merge_collinear_closed_vertices(coarse)
        if len(corners) == 4:
            sides = [
                corners[(i + 1) % 4] - corners[i] for i in range(4)
            ]
            lengths = [float(np.hypot(s[0], s[1])) for s in sides]
            if min(lengths) >= 10.0:
                angles_ok = True
                for i in range(4):
                    u = -sides[(i - 1) % 4] / lengths[(i - 1) % 4]
                    v = sides[i] / lengths[i]
                    ang = math.degrees(
                        math.acos(min(1.0, max(-1.0, float(np.dot(u, v)))))
                    )
                    if abs(ang - 90.0) > 12.0:
                        angles_ok = False
                        break
                opp0 = abs(lengths[0] - lengths[2]) / max(lengths[0], lengths[2])
                opp1 = abs(lengths[1] - lengths[3]) / max(lengths[1], lengths[3])
                # Garante também que o contorno real segue os 4 lados do retângulo.
                rect_dev = []
                for p in dense:
                    d_min = min(
                        abs(
                            float(
                                (corners[(i + 1) % 4, 0] - corners[i, 0])
                                * (corners[i, 1] - p[1])
                                - (corners[(i + 1) % 4, 1] - corners[i, 1])
                                * (corners[i, 0] - p[0])
                            )
                        )
                        / lengths[i]
                        for i in range(4)
                    )
                    rect_dev.append(d_min)
                if (
                    angles_ok
                    and opp0 <= 0.12
                    and opp1 <= 0.12
                    and max(rect_dev) <= max(4.0, 0.05 * max(lengths))
                ):
                    w_px = 0.5 * (lengths[0] + lengths[2])
                    h_px = 0.5 * (lengths[1] + lengths[3])
                    ang_deg = math.degrees(
                        math.atan2(sides[0][1], sides[0][0])
                    )
                    center = corners.mean(axis=0)
                    return {
                        "type": "rectangle",
                        "center_mm": [
                            round(float(center[0] * MM_PER_PX_X), 3),
                            round(float(center[1] * MM_PER_PX_Y), 3),
                        ],
                        "width_mm": round(float(w_px * MM_PER_PX), 3),
                        "height_mm": round(float(h_px * MM_PER_PX), 3),
                        "angle_deg": round(ang_deg, 2),
                        "corners_mm": [
                            [
                                round(float(c[0] * MM_PER_PX_X), 3),
                                round(float(c[1] * MM_PER_PX_Y), 3),
                            ]
                            for c in corners
                        ],
                        **meta,
                    }

    return {
        "type": "polygon" if closed else "polyline",
        "vertices": len(pts),
        "points_mm": [
            [round(float(x * MM_PER_PX_X), 3), round(float(y * MM_PER_PX_Y), 3)]
            for x, y in pts
        ],
        **meta,
    }


def classify_primitives(polylines):
    """Classifica todas as polilinhas em primitivas geométricas estruturadas."""
    return [classify_primitive(line) for line in polylines]


def vectorize_mask(
    ink,
    epsilon=1.5,
    thinning="zhangsuen",
    *,
    min_area=0,
    prune_spurs=False,
    stitch=False,
    stamps=None,
    order_strokes=False,
):
    """Máscara de tinta -> polilinhas simplificadas (px, espaço do papel)."""
    work = filter_microfragments(ink, min_area=min_area) if min_area > 1 else ink
    skel = thin(work, thinning)
    if prune_spurs:
        skel = prune_skeleton_spurs(skel, work)
    raw = traverse_graph(skel, include_topology=stitch)
    if stitch:
        raw = stitch_junctions(raw, stamps=stamps)
    if order_strokes or stamps is not None:
        raw = annotate_and_order_polylines(raw, stamps=stamps)
    out = []
    for line in raw:
        pts = simplify(line["points"], epsilon, line["closed"])
        item = {"points": pts, "closed": line["closed"]}
        if "t_start_ms" in line:
            item["t_start_ms"] = line["t_start_ms"]
            item["t_end_ms"] = line["t_end_ms"]
        out.append(item)
    return out


def vectorize_clean(ink, epsilon=1.5, thinning="zhangsuen", stamps=None):
    """Atalho recomendado para canvas reais: limpa poeira, poda esporões, costura e ordena."""
    return vectorize_mask(
        ink,
        epsilon=epsilon,
        thinning=thinning,
        min_area=4,
        prune_spurs=True,
        stitch=True,
        stamps=stamps,
        order_strokes=True,
    )


def dense_samples(points, closed, step=1.0):
    """Amostra pontos ao longo da polilinha com espaçamento <= step px."""
    if len(points) == 1:
        return points.copy()
    pts = list(points)
    if closed:
        pts = pts + [pts[0]]
    samples = []
    for a, b in zip(pts[:-1], pts[1:]):
        length = float(np.hypot(*(np.array(b) - np.array(a))))
        n = max(1, int(np.ceil(length / step)))
        for i in range(n):
            t = i / n
            samples.append(np.array(a) * (1 - t) + np.array(b) * t)
    samples.append(np.array(pts[-1], dtype=np.float64))
    return np.array(samples)


def distance_stats(ink, polylines):
    """Distâncias polilinha<->tinta nos dois sentidos, em px e mm."""
    dt_to_ink = cv2.distanceTransform(
        (~ink).astype(np.uint8), cv2.DIST_L2, 5
    )
    samples = [
        dense_samples(line["points"], line["closed"]) for line in polylines
    ]
    samples = np.vstack(samples) if samples else np.zeros((0, 2))
    h, w = ink.shape
    if len(samples) > 0:
        sx = np.clip(np.round(samples[:, 0]).astype(int), 0, w - 1)
        sy = np.clip(np.round(samples[:, 1]).astype(int), 0, h - 1)
        d_fwd = dt_to_ink[sy, sx]
    else:
        d_fwd = np.zeros((0,), dtype=np.float32)

    raster = np.zeros(ink.shape, dtype=np.uint8)
    for line in polylines:
        pts = np.round(line["points"]).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) == 1:
            raster[pts[0, 0, 1], pts[0, 0, 0]] = 1
        else:
            cv2.polylines(
                raster,
                [pts],
                bool(line["closed"]),
                1,
                thickness=1,
                lineType=cv2.LINE_8,
            )
    dt_to_poly = cv2.distanceTransform(
        (raster == 0).astype(np.uint8), cv2.DIST_L2, 5
    )
    d_bwd = dt_to_poly[ink]

    def summary(d):
        if len(d) == 0:
            return {"n": 0}
        return {
            "n": int(len(d)),
            "media_px": float(d.mean()),
            "mediana_px": float(np.median(d)),
            "p95_px": float(np.percentile(d, 95)),
            "max_px": float(d.max()),
            "media_mm": float(d.mean() * MM_PER_PX),
        }

    return {
        "polilinha_para_tinta": summary(d_fwd),
        "tinta_para_polilinha": summary(d_bwd),
    }


def write_svg(polylines, out_path, epsilon, thinning, primitives=None):
    """Escreve o SVG com viewBox em mm do papel A4 (210x297)."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<!-- Draw vectorize: epsilon={epsilon} px, thinning={thinning}, "
        f"escala x={MM_PER_PX_X:.5f} mm/px, y={MM_PER_PX_Y:.5f} mm/px -->",
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{PAPER_W_MM}mm" height="{PAPER_H_MM}mm" '
        f'viewBox="0 0 {PAPER_W_MM} {PAPER_H_MM}">',
    ]
    if primitives is not None:
        for prim in primitives:
            kind = prim["type"]
            if kind == "point":
                lines.append(
                    f'<circle cx="{prim["x_mm"]:.2f}" cy="{prim["y_mm"]:.2f}" '
                    f'r="0.30" fill="#000"/>'
                )
            elif kind == "line":
                lines.append(
                    f'<line x1="{prim["x1_mm"]:.2f}" y1="{prim["y1_mm"]:.2f}" '
                    f'x2="{prim["x2_mm"]:.2f}" y2="{prim["y2_mm"]:.2f}" '
                    f'stroke="#000" stroke-width="0.35" stroke-linecap="round"/>'
                )
            elif kind == "circle":
                lines.append(
                    f'<circle cx="{prim["cx_mm"]:.2f}" cy="{prim["cy_mm"]:.2f}" '
                    f'r="{prim["r_mm"]:.2f}" fill="none" stroke="#000" '
                    f'stroke-width="0.35"/>'
                )
            elif kind == "rectangle":
                coords = " ".join(
                    f"{x:.2f},{y:.2f}" for x, y in prim["corners_mm"]
                )
                lines.append(
                    f'<polygon data-primitive="rectangle" points="{coords}" '
                    f'fill="none" stroke="#000" stroke-width="0.35"/>'
                )
            else:
                coords = " ".join(
                    f"{x:.2f},{y:.2f}" for x, y in prim["points_mm"]
                )
                tag = "polygon" if kind == "polygon" else "polyline"
                lines.append(
                    f'<{tag} points="{coords}" fill="none" stroke="#000" '
                    f'stroke-width="0.30"/>'
                )
    else:
        for line in polylines:
            pts = line["points"]
            if len(pts) == 1:
                cx, cy = pts[0][0] * MM_PER_PX_X, pts[0][1] * MM_PER_PX_Y
                lines.append(
                    f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="0.30" fill="#000"/>'
                )
                continue
            coords = " ".join(
                f"{x * MM_PER_PX_X:.2f},{y * MM_PER_PX_Y:.2f}" for x, y in pts
            )
            tag = "polygon" if line["closed"] else "polyline"
            lines.append(
                f'<{tag} points="{coords}" fill="none" stroke="#000" '
                f'stroke-width="0.30"/>'
            )
    lines.append("</svg>")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def describe(ink, polylines, stats, primitives=None):
    """Resumo textual de uma vetorização."""
    n_points = sum(1 for line in polylines if len(line["points"]) == 1)
    n_closed = sum(1 for line in polylines if line["closed"])
    out = {
        "pixels_tinta": int(ink.sum()),
        "polilinhas": len(polylines),
        "polilinhas_fechadas": n_closed,
        "polilinhas_de_um_ponto": n_points,
        "saltos_de_caneta": max(0, len(polylines) - 1),
        "distancias": stats,
    }
    if primitives is not None:
        counts = {}
        for p in primitives:
            counts[p["type"]] = counts.get(p["type"], 0) + 1
        out["primitivas_contagem"] = counts
    return out


def _selfcheck():
    """Valida tanto o grafo exato de 1 px quanto limpeza + primitivas em 4 px."""
    rect = [(250, 70), (430, 250), (250, 430), (70, 250)]
    img = np.full((PAPER_H_PX, PAPER_W_PX), 255, dtype=np.uint8)
    for a, b in zip(rect, rect[1:] + rect[:1]):
        cv2.line(img, a, b, 0, thickness=1)
    cv2.circle(img, (620, 250), 110, 0, thickness=1)
    cv2.line(img, (150, 700), (450, 1000), 0, thickness=1)
    cv2.line(img, (150, 1000), (450, 700), 0, thickness=1)
    ink = img < 128

    polylines = vectorize_mask(ink, epsilon=1.5)
    n_closed = sum(1 for line in polylines if line["closed"])
    n_open = len(polylines) - n_closed
    print(
        f"polilinhas (1 px bruto): {len(polylines)} (fechadas {n_closed}, "
        f"abertas {n_open}); esperado 6 (2 fechadas, 4 abertas)"
    )
    assert len(polylines) == 6, f"contagem inesperada: {len(polylines)}"
    assert n_closed == 2, f"fechadas inesperadas: {n_closed}"

    def seg_dist(p, a, b):
        a, b = np.array(a, float), np.array(b, float)
        pnt = np.array(p, float)
        ab = b - a
        t = np.clip(np.dot(pnt - a, ab) / np.dot(ab, ab), 0, 1)
        return float(np.hypot(*(pnt - (a + t * ab))))

    def dist_to_shapes(p):
        d_rect = min(
            seg_dist(p, a, b) for a, b in zip(rect, rect[1:] + rect[:1])
        )
        d_circ = abs(float(np.hypot(p[0] - 620, p[1] - 250)) - 110.0)
        d_x = min(
            seg_dist(p, (150, 700), (450, 1000)),
            seg_dist(p, (150, 1000), (450, 700)),
        )
        return min(d_rect, d_circ, d_x)

    samples = np.vstack(
        [dense_samples(line["points"], line["closed"]) for line in polylines]
    )
    dists = np.array([dist_to_shapes(p) for p in samples])
    print(
        f"erro contra a forma geradora: n={len(dists)} amostras, "
        f"média {dists.mean():.3f} px ({dists.mean() * MM_PER_PX:.3f} mm), "
        f"p95 {np.percentile(dists, 95):.3f} px, "
        f"máx {dists.max():.3f} px"
    )
    assert dists.mean() <= 1.5, "erro médio acima do esperado"

    # Bancada 2: traço real de 4 px + poeira de 1-2 px + costura + Camada 2:
    img_thick = np.full((PAPER_H_PX, PAPER_W_PX), 255, dtype=np.uint8)
    for a, b in zip(rect, rect[1:] + rect[:1]):
        cv2.line(img_thick, a, b, 0, thickness=4)
    cv2.circle(img_thick, (620, 250), 110, 0, thickness=4)
    cv2.line(img_thick, (150, 700), (450, 1000), 0, thickness=4)
    cv2.line(img_thick, (150, 1000), (450, 700), 0, thickness=4)
    img_thick[40, 40] = 0
    img_thick[41, 40] = 0
    ink_thick = img_thick < 128

    clean_polys = vectorize_clean(ink_thick, epsilon=1.5)
    prims = classify_primitives(clean_polys)
    kinds = sorted(p["type"] for p in prims)
    print(f"primitivas (4 px limpo): {kinds}")
    assert kinds == ["circle", "line", "line", "rectangle"], (
        f"primitivas inesperadas em traco 4 px: {kinds}"
    )

    stats = distance_stats(ink, polylines)
    print("distâncias raster:", json.dumps(stats, indent=1))
    print("SELFCHECK OK")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Vetoriza um canvas de tinta (PNG) para polilinhas/primitivas/SVG."
    )
    parser.add_argument("input", nargs="?", help="PNG do canvas de tinta")
    parser.add_argument("--out", default=None, help="SVG de saída")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.5,
        help="epsilon do Douglas-Peucker em px (padrao 1,5)",
    )
    parser.add_argument(
        "--thinning",
        choices=list(THINNING_TYPES),
        default="zhangsuen",
        help="algoritmo de afinamento (padrao zhangsuen)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="ativa limpeza completa: remove microfragmentos, poda esporões "
        "do esqueleto, costura junções colineares e ordena os traços",
    )
    parser.add_argument(
        "--stamps",
        default=None,
        metavar="ARQUIVO.npz",
        help="carimbos temporais (`_stamps.npz`) para orientar, costurar e "
        "ordenar os traços na ordem real em que foram desenhados",
    )
    parser.add_argument(
        "--primitives",
        action="store_true",
        help="classifica primitivas geométricas da Camada 2 (line, circle, "
        "rectangle) e emite tags SVG semânticas",
    )
    parser.add_argument(
        "--json",
        default=None,
        metavar="SAIDA.json",
        help="exporta primitivas/polilinhas estruturadas em JSON",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="imprime as distâncias nos dois sentidos",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="roda a bancada sintética e sai",
    )
    args = parser.parse_args(argv)

    if args.selfcheck:
        _selfcheck()
        return 0
    if not args.input:
        parser.error("informe o PNG de entrada ou --selfcheck")

    ink, polarity = load_ink(args.input)
    stamps = load_stamps(args.stamps) if args.stamps else None
    if args.clean:
        polylines = vectorize_clean(
            ink, epsilon=args.epsilon, thinning=args.thinning, stamps=stamps
        )
    else:
        polylines = vectorize_mask(
            ink,
            epsilon=args.epsilon,
            thinning=args.thinning,
            stamps=stamps,
            order_strokes=stamps is not None,
        )

    prims = classify_primitives(polylines) if (args.primitives or args.json) else None
    out = args.out or str(Path(args.input).with_suffix(".svg"))
    write_svg(
        polylines,
        out,
        args.epsilon,
        args.thinning,
        primitives=prims if args.primitives else None,
    )
    stats = distance_stats(ink, polylines)
    summary = describe(ink, polylines, stats, primitives=prims)
    summary["polaridade_tinta"] = polarity
    summary["epsilon_px"] = args.epsilon
    summary["thinning"] = args.thinning
    summary["svg"] = out
    if args.json:
        payload = {"summary": summary, "primitives": prims}
        Path(args.json).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary["json"] = args.json

    if args.stats:
        print(json.dumps(summary, indent=1, ensure_ascii=False))
    else:
        print(
            f"{args.input}: {summary['polilinhas']} polilinhas "
            f"({summary['polilinhas_fechadas']} fechadas), "
            f"{summary['saltos_de_caneta']} saltos de caneta, "
            f"dist. média polilinha->tinta "
            f"{stats['polilinha_para_tinta'].get('media_px', float('nan')):.3f} px, "
            f"tinta->polilinha "
            f"{stats['tinta_para_polilinha'].get('media_px', float('nan')):.3f} px "
            f"-> {out}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
