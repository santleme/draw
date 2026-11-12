"""Simulador procedural de escrita do Draw (frente 103).

Gerador de corpus com gabarito matemático EXATO: cada traço nasce
como trajetória paramétrica no espaço da folha (1000×1414), então a
máscara de tinta verdadeira M_GT(t) de cada frame é conhecida por
construção — não por rotulação. Uma câmera virtual compõe a folha
sobre frames REAIS de mesa vazia (corpora de negativos do
laboratório) com homografia H_t conhecida e salva por frame, e o
corpus sai no formato que o tools/replay_lab.py consome
(``frames/`` + arquivos ao lado), acrescido do ``sidecar.json``
com GT, bandas, máscara de impressão e eventos.

A auditoria (``audit``) roda o corpus pelo laço REAL
(draw.loop.SheetInkLoop + trabalhadora página→limpa real em
pump(), o mesmo arranjo síncrono do replay_lab) com a pilha de
produção — detector de folha FT-dyn (SHA e20a89f0…) e tinta v6
(SHA 613e250f…) — e compara o canvas confirmado com o GT.

O simulador é camada de REGRESSÃO (quebra aparece em segundos,
com número exato), NÃO portão de promoção: ele não modela textura
do papel interagindo com a caneta, show-through do verso, blur de
movimento real, artefatos MJPEG nem página enrolada não-planar.
Quem calibra o quanto ele pode afirmar sobre o real é a frente
104.

Uso:
  python tools/sim_writer.py generate --out <dir-corpus>
  python tools/sim_writer.py audit --corpus <dir-corpus> \\
      --out <dir-saida> [--flags DRAW_INK_VOTE_E3=1]
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ------------------------------------------------------------------
# Geometria e constantes
# ------------------------------------------------------------------
SHEET_W, SHEET_H = 1000, 1414          # espaço-mestre da folha
CANVAS_W, CANVAS_H = 850, 1200         # canvas do pipeline (draw.config)
FRAME_W, FRAME_H = 1280, 720           # quadro da câmera virtual
PAPER_GRAY = 243
# Valor de cinza da tinta por contraste (fundo ≈ PAPER_GRAY):
# forte ≈ 218 níveis de contraste, médio ≈ 138, fraco (lápis) ≈ 63.
CONTRASTS = {"strong": 25, "medium": 105, "weak": 180}
# Cantos-base da folha no quadro (perspectiva leve, folha retrato).
BASE_CORNERS = np.array(
    [[425, 25], [855, 40], [865, 695], [415, 680]], np.float64
)
BG_ROOT = (
    Path.home() / "workspace" / "draw-lab" / "train-negatives-v2"
    / "session_raw_capture"
)
SHEET_MODEL = (
    Path.home() / "workspace" / "draw-lab" / "neural" / "v3"
    / "draw_seg_v3_dyn.onnx"
)
INK_MODEL = (
    Path.home() / "workspace" / "draw-lab" / "live-audit-2026-11-10"
    / "frente48" / "results" / "page_clean_finetune_real_v6.onnx"
)
SCENE_FPS = 2.0


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ------------------------------------------------------------------
# Trajetórias
# ------------------------------------------------------------------
def bezier(p0, p1, p2, p3, n=60):
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = np.array([p0, p1, p2, p3], float)
    b = (
        (1 - t) ** 3 * pts[0] + 3 * (1 - t) ** 2 * t * pts[1]
        + 3 * (1 - t) * t ** 2 * pts[2] + t ** 3 * pts[3]
    )
    return b


def cursive_line(x0, y0, width, rng, humps=7):
    """Uma 'linha de escrita': Béziers encadeadas com laços e
    descendentes ocasionais. Devolve lista de polilinhas."""
    strokes = []
    x = x0
    step = width / humps
    pts = []
    for i in range(humps):
        amp = rng.uniform(14, 30)
        p0 = (x, y0)
        p1 = (x + step * 0.3, y0 - amp)
        p2 = (x + step * 0.7, y0 + amp * rng.uniform(0.2, 0.9))
        p3 = (x + step, y0 + rng.uniform(-4, 4))
        seg = bezier(p0, p1, p2, p3, n=36)
        pts.append(seg if i == 0 else seg[1:])
        x += step
        if rng.random() < 0.30:  # laço/descendente: traço separado
            lx = x - step * 0.4
            loop = bezier(
                (lx, y0), (lx - 16, y0 - 44), (lx + 20, y0 - 46),
                (lx + 4, y0 + rng.uniform(18, 40)), n=30,
            )
            strokes.append(loop)
    if pts:
        strokes.insert(0, np.vstack(pts))
    return strokes


def polyline_circle(cx, cy, r, n=90):
    a = np.linspace(0, 2 * math.pi, n)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


def hatch_lines(rect, spacing=30, angle_deg=35):
    """Hachuras diagonais recortadas no retângulo (amostragem +
    corte por dentro/fora, segmentos analíticos por varredura)."""
    x0, y0, x1, y1 = rect
    th = math.radians(angle_deg)
    d = np.array([math.cos(th), math.sin(th)])
    nrm = np.array([-d[1], d[0]])
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    proj = (corners - center) @ nrm
    lines = []
    off = math.floor(proj.min() / spacing) * spacing
    while off <= proj.max():
        base = center + nrm * off
        ts = (corners - base) @ d
        # Interseção do segmento com a caixa: amostra densa e corta.
        t = np.linspace(ts.min() - 50, ts.max() + 50, 400)
        p = base[None, :] + t[:, None] * d[None, :]
        inside = (
            (p[:, 0] >= x0) & (p[:, 0] <= x1)
            & (p[:, 1] >= y0) & (p[:, 1] <= y1)
        )
        if inside.sum() > 5:
            seg = p[inside]
            lines.append(seg[[0, -1]])
        off += spacing
    return lines


class Stroke:
    __slots__ = ("points", "width", "contrast", "band")

    def __init__(self, points, width, contrast, band):
        self.points = np.asarray(points, float)
        self.width = int(width)
        self.contrast = contrast
        self.band = band


def _lengths(points):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def sub_polyline(points, upto):
    """Prefixo da polilinha com comprimento de arco <= upto."""
    cum = _lengths(points)
    if upto <= 0:
        return None
    if upto >= cum[-1]:
        return points
    i = int(np.searchsorted(cum, upto))
    if i == 0:
        return points[:1]
    frac = (upto - cum[i - 1]) / max(1e-9, cum[i] - cum[i - 1])
    tip = points[i - 1] + frac * (points[i] - points[i - 1])
    return np.vstack([points[:i], tip])


# ------------------------------------------------------------------
# Baterias (cenários)
# ------------------------------------------------------------------
def build_strokes(name, rng):
    """(strokes, print_kind, bg_kind, events_dict) por cenário."""
    S = []
    if name in ("cursive", "ruled_cursive", "events"):
        lines = 4 if name != "events" else 3
        for li in range(lines):
            for pts in cursive_line(
                110, 240 + li * 150, 760, rng
            ):
                S.append(Stroke(pts, 3, "strong", "cursive"))
    elif name in ("cad", "grid_cad"):
        S.append(Stroke(
            np.array([[150, 200], [620, 200], [620, 560],
                      [150, 560], [150, 200]], float),
            3, "strong", "cad"))
        S.append(Stroke(polyline_circle(385, 380, 110), 3,
                        "strong", "cad"))
        S.append(Stroke(
            np.array([[150, 700], [620, 700], [620, 1060],
                      [150, 1060], [150, 700]], float),
            2, "strong", "cad"))
        # Linhas de cota com ticks.
        for y, x0, x1 in ((640, 150, 620), (1120, 150, 620)):
            S.append(Stroke(np.array([[x0, y], [x1, y]], float),
                            2, "strong", "cad"))
            for x in (x0, x1):
                S.append(Stroke(
                    np.array([[x, y - 18], [x, y + 18]], float),
                    2, "strong", "cad"))
        S.append(Stroke(polyline_circle(385, 880, 80), 2,
                        "strong", "cad"))
    elif name == "hatch":
        rect = (180, 250, 800, 800)
        S.append(Stroke(
            np.array([[180, 250], [800, 250], [800, 800],
                      [180, 800], [180, 250]], float),
            3, "strong", "hatch"))
        for seg in hatch_lines(rect, spacing=34):
            S.append(Stroke(seg, 2, "strong", "hatch"))
        for pts in cursive_line(180, 980, 620, rng, humps=5):
            S.append(Stroke(pts, 3, "strong", "hatch"))
    elif name == "thin":
        for ri, w in enumerate((1, 2, 3)):
            for ci, contrast in enumerate(
                ("strong", "medium", "weak")
            ):
                band = f"thin_{w}px_{contrast}"
                cx, cy = 140 + ci * 280, 220 + ri * 330
                S.append(Stroke(
                    np.array([[cx, cy], [cx + 180, cy]], float),
                    w, contrast, band))
                S.append(Stroke(
                    np.array([[cx, cy + 50],
                              [cx + 180, cy + 130]], float),
                    w, contrast, band))
                S.append(Stroke(
                    np.array([[cx + 40, cy + 180],
                              [cx + 40, cy + 60]], float),
                    w, contrast, band))
    elif name == "blank_keyboard":
        pass
    else:
        raise ValueError(f"cenário desconhecido: {name}")
    print_kind = {
        "ruled_cursive": "ruled", "grid_cad": "grid",
    }.get(name, "clean")
    bg_kind = "keyboard_mouse" if name in (
        "blank_keyboard", "events") else (
        "objects_pen" if name == "hatch" else "empty_desk")
    events = {}
    if name == "events":
        events = {
            "push_frame": 10, "push_px": 25.0,
            "occlusion_frames": [14, 15, 16],
            "occlusion_center_sheet": [620.0, 390.0],
            "occlusion_radius_sheet": 95.0,
            "rotation_frame": 20, "rotation_deg": 2.0,
        }
    return S, print_kind, bg_kind, events


SCENARIOS = [
    "cursive", "cad", "hatch", "thin", "ruled_cursive",
    "grid_cad", "blank_keyboard", "events",
]
N_FRAMES = {"events": 34}
DEFAULT_FRAMES = 30
SETTLE_FRAMES = 3       # folha em branco antes de escrever
# Folha completa parada no fim: medido na frente 103, o voto
# precisa de passadas extras para confirmar o traço tardio —
# com 3 frames de hold a última fileira da bateria fina mediu
# recall 0,0 e com +2 passadas foi a 0,996 (cadência da
# trabalhadora em descanso, não cegueira do modelo: a prob
# direta no warp via os traços). Hold de 13 frames ≈ 6,5 s de
# cena a 2 fps.
HOLD_FRAMES = 13


# ------------------------------------------------------------------
# Página, impressão e câmera virtual
# ------------------------------------------------------------------
def render_base(print_kind, rng):
    """Página em branco + máscara de impressão.

    Realismo medido na frente 103: página plana e uniforme faz o
    FT-dyn devolver logit ~1,1 e máscara fragmentada; textura
    multiescala + gradiente de iluminação + AO de borda + cast
    quente leve sobem o logit a ~2,3–3,0 e a cobertura da máscara
    no quad de 0,31 para ~0,46 (ainda longe do real ~9,3/0,9+ —
    a distância restante é o gap sintético→real que a F104
    calibra; ver o relatório 103)."""
    noise = rng.normal(0, 2.5, (SHEET_H, SHEET_W))
    page = np.clip(PAPER_GRAY + noise, 0, 255).astype(np.uint8)
    page = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    tex = page.astype(np.float32)
    grad = (
        np.linspace(0.90, 1.03, SHEET_H)[:, None, None]
        * np.linspace(1.0, 0.95, SHEET_W)[None, :, None]
    )
    low = cv2.GaussianBlur(
        rng.normal(0, 9, (SHEET_H, SHEET_W)).astype(np.float32),
        (0, 0), 2)
    fine = rng.normal(0, 4, (SHEET_H, SHEET_W)).astype(np.float32)
    tex = tex * grad + (low + fine)[:, :, None]
    tex[:, :, 0] *= 0.96
    tex[:, :, 2] *= 1.03
    page = np.clip(tex, 0, 255).astype(np.uint8)
    ao = np.ones((SHEET_H, SHEET_W), np.float32)
    ao[:14, :] *= 0.88
    ao[-14:, :] *= 0.88
    ao[:, :14] *= 0.88
    ao[:, -14:] *= 0.88
    page = (page * ao[:, :, None]).astype(np.uint8)
    print_mask = np.zeros((SHEET_H, SHEET_W), np.uint8)
    if print_kind == "ruled":
        for y in range(200, SHEET_H - 120, 64):
            cv2.line(page, (70, y), (SHEET_W - 70, y),
                     (196, 178, 160), 2, cv2.LINE_AA)
            cv2.line(print_mask, (70, y), (SHEET_W - 70, y),
                     255, 2)
    elif print_kind == "grid":
        for x in range(70, SHEET_W - 60, 50):
            cv2.line(page, (x, 120), (x, SHEET_H - 100),
                     (205, 195, 180), 1, cv2.LINE_AA)
            cv2.line(print_mask, (x, 120), (x, SHEET_H - 100),
                     255, 1)
        for y in range(120, SHEET_H - 90, 50):
            cv2.line(page, (70, y), (SHEET_W - 70, y),
                     (205, 195, 180), 1, cv2.LINE_AA)
            cv2.line(print_mask, (70, y), (SHEET_W - 70, y),
                     255, 1)
    return page, print_mask


def draw_ink(page, gt, strokes, reveal_len):
    """Desenha o prefixo revelado de cada traço; GT exato junto."""
    page = page.copy()
    gt = gt.copy()
    offset = 0.0
    for s in strokes:
        total = _lengths(s.points)[-1]
        upto = min(total, max(0.0, reveal_len - offset))
        offset += total
        if upto <= 0:
            continue
        pts = sub_polyline(s.points, upto)
        if pts is None or len(pts) < 2:
            continue
        ip = np.round(pts).astype(np.int32)
        gray = CONTRASTS[s.contrast]
        cv2.polylines(page, [ip], False, (gray, gray, gray),
                      s.width, cv2.LINE_AA)
        cv2.polylines(gt, [ip], False, 255, s.width, cv2.LINE_8)
    return page, gt


def band_masks(strokes):
    """Máscara final (folha inteira desenhada) por banda."""
    out = {}
    for s in strokes:
        m = out.setdefault(
            s.band, np.zeros((SHEET_H, SHEET_W), np.uint8))
        ip = np.round(s.points).astype(np.int32)
        cv2.polylines(m, [ip], False, 255, s.width, cv2.LINE_8)
    return out


def corners_at(frame_idx, events):
    """Cantos da folha no quadro, com os eventos aplicados."""
    corners = BASE_CORNERS.copy()
    if not events:
        return corners
    if frame_idx >= events.get("push_frame", 10 ** 9):
        corners = corners + np.array([events["push_px"], 0.0])
    if frame_idx >= events.get("rotation_frame", 10 ** 9):
        th = math.radians(events["rotation_deg"])
        c = np.array([FRAME_W / 2, FRAME_H / 2])
        rot = np.array([[math.cos(th), -math.sin(th)],
                        [math.sin(th), math.cos(th)]])
        corners = (corners - c) @ rot.T + c
    return corners


def homography(corners):
    src = np.array(
        [[0, 0], [SHEET_W, 0], [SHEET_W, SHEET_H], [0, SHEET_H]],
        np.float64,
    )
    return cv2.getPerspectiveTransform(
        src.astype(np.float32), corners.astype(np.float32)
    ).astype(np.float64)


def load_backgrounds(bg_kind, n):
    """Frames reais de mesa vazia; (frames, nomes, fonte)."""
    files = sorted(BG_ROOT.glob(f"bg_{bg_kind}_*.png"))
    if not files:
        raise FileNotFoundError(
            f"sem fundos bg_{bg_kind}_* em {BG_ROOT}")
    frames, names = [], []
    for i in range(n):
        p = files[i % len(files)]
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"fundo ilegível: {p}")
        frames.append(img)
        names.append(p.name)
    return frames, names


def compose(bg, page, corners, events, frame_idx, rng):
    """Folha sobre o fundo via H; oclusão de mão/caneta se ativa."""
    frame = bg.copy()
    H = homography(corners)
    # Sombra suave da folha.
    shadow = np.zeros((FRAME_H, FRAME_W), np.uint8)
    cv2.fillPoly(
        shadow, [np.round(corners + [7, 9]).astype(np.int32)],
        255)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 8)
    frame = np.where(
        (shadow > 0)[..., None],
        (frame * 0.82).astype(np.uint8), frame,
    ).astype(np.uint8)
    warped = cv2.warpPerspective(page, H, (FRAME_W, FRAME_H))
    alpha = cv2.warpPerspective(
        np.full((SHEET_H, SHEET_W), 255, np.uint8), H,
        (FRAME_W, FRAME_H))
    frame = np.where((alpha > 0)[..., None], warped, frame)
    active_events = []
    if events and frame_idx in events.get("occlusion_frames", []):
        cx, cy = events["occlusion_center_sheet"]
        pt = cv2.perspectiveTransform(
            np.array([[[cx, cy]]], np.float32), H.astype(np.float32)
        )[0, 0]
        # Mão: elipse tom de pele; caneta: traço escuro inclinado.
        cv2.ellipse(frame, (int(pt[0]), int(pt[1])), (95, 55),
                    -18, 0, 360, (150, 180, 215), -1, cv2.LINE_AA)
        cv2.line(frame,
                 (int(pt[0] - 70), int(pt[1] + 55)),
                 (int(pt[0] + 25), int(pt[1] - 30)),
                 (35, 35, 35), 13, cv2.LINE_AA)
        active_events.append("occlusion")
    # Sensor: blur leve + ruído (sem artefato MJPEG — declarado).
    frame = cv2.GaussianBlur(frame, (0, 0), 0.7)
    frame = np.clip(
        frame.astype(np.float32)
        + rng.normal(0, 3.5, frame.shape), 0, 255).astype(np.uint8)
    if events and frame_idx == events.get("push_frame"):
        active_events.append("push")
    if events and frame_idx == events.get("rotation_frame"):
        active_events.append("rotation")
    return frame.astype(np.uint8), H, active_events


# ------------------------------------------------------------------
# Geração do corpus
# ------------------------------------------------------------------
def generate_scenario(out_dir, name, seed=103):
    import zlib

    rng = np.random.default_rng(
        seed + zlib.crc32(name.encode()) % 997)
    strokes, print_kind, bg_kind, events = build_strokes(name, rng)
    n = N_FRAMES.get(name, DEFAULT_FRAMES)
    base_page, print_mask = render_base(print_kind, rng)
    total_len = sum(_lengths(s.points)[-1] for s in strokes)
    bgs, bg_names = load_backgrounds(bg_kind, n)
    scen = Path(out_dir) / name
    (scen / "frames").mkdir(parents=True, exist_ok=True)
    (scen / "gt").mkdir(parents=True, exist_ok=True)
    (scen / "bands").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(scen / "print_mask.png"), print_mask)
    bands = band_masks(strokes)
    if name == "events":
        cx, cy = events["occlusion_center_sheet"]
        r = events["occlusion_radius_sheet"]
        seg = np.zeros((SHEET_H, SHEET_W), np.uint8)
        cv2.circle(seg, (int(cx), int(cy)), int(r), 255, -1)
        full = np.zeros((SHEET_H, SHEET_W), np.uint8)
        for s in strokes:
            ip = np.round(s.points).astype(np.int32)
            cv2.polylines(full, [ip], False, 255, s.width,
                          cv2.LINE_8)
        bands["occluded_segment"] = cv2.bitwise_and(full, seg)
    for band, m in bands.items():
        cv2.imwrite(str(scen / "bands" / f"{band}.png"), m)
    frames_meta = []
    draw_frames = max(1, n - SETTLE_FRAMES - HOLD_FRAMES)
    for i in range(n):
        if i < SETTLE_FRAMES or not strokes:
            frac = 0.0
        else:
            frac = min(
                1.0, (i - SETTLE_FRAMES + 1) / draw_frames)
        page, gt = draw_ink(
            base_page, np.zeros((SHEET_H, SHEET_W), np.uint8),
            strokes, frac * total_len)
        corners = corners_at(i, events)
        frame, H, active = compose(
            bgs[i], page, corners, events, i, rng)
        fname = f"frame_{i:06d}.png"
        cv2.imwrite(str(scen / "frames" / fname), frame)
        cv2.imwrite(str(scen / "gt" / f"gt_{i:06d}.png"), gt)
        frames_meta.append({
            "index": i, "file": fname,
            "gt": f"gt/gt_{i:06d}.png",
            "H_sheet_to_camera": H.tolist(),
            "corners_camera": corners.tolist(),
            "revealed_fraction": round(float(frac), 4),
            "events": active,
        })
    sidecar = {
        "format": "sim_writer_v1",
        "scenario": name,
        "seed": seed,
        "sheet_space": [SHEET_W, SHEET_H],
        "canvas_space": [CANVAS_W, CANVAS_H],
        "frame_space": [FRAME_W, FRAME_H],
        "print_kind": print_kind,
        "print_mask": "print_mask.png",
        "bg_kind": bg_kind,
        "bg_files": bg_names,
        "events_spec": events,
        "bands": {b: f"bands/{b}.png" for b in bands},
        "frames": frames_meta,
        "models_production": {
            "sheet_sha16_expected": "e20a89f0eae7c76d",
            "ink_sha16_expected": "613e250f5f7e15e9",
        },
    }
    (scen / "sidecar.json").write_text(json.dumps(sidecar, indent=1))
    return sidecar


def generate(out_dir, scenarios):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for name in scenarios:
        sc = generate_scenario(out, name)
        index[name] = {
            "frames": len(sc["frames"]),
            "print_kind": sc["print_kind"],
            "bg_kind": sc["bg_kind"],
            "bands": sorted(sc["bands"]),
        }
        print(f"[sim] {name}: {len(sc['frames'])} frames")
    (out / "index.json").write_text(json.dumps(index, indent=1))
    return index


# ------------------------------------------------------------------
# Auditoria pelo laço real
# ------------------------------------------------------------------
def _to_canvas(mask_master):
    r = cv2.resize(
        mask_master, (CANVAS_W, CANVAS_H),
        interpolation=cv2.INTER_AREA)
    return r > 64


def _dil(mask, px=3):
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def diff_panel(scen_dir, canvas_bin, out_path):
    """Prancha de diff no espaço da folha: verde=capturado (TP),
    azul=perdido (FN), vermelho=alucinado/duplicado (FP, fora do
    GT dilatado 3 px). Fundo = página final clareada."""
    side = json.loads((Path(scen_dir) / "sidecar.json").read_text())
    last = side["frames"][-1]
    gt_c = _to_canvas(cv2.imread(
        str(Path(scen_dir) / last["gt"]), cv2.IMREAD_GRAYSCALE))
    gt_d = _dil(gt_c)
    tp = canvas_bin & gt_d
    fn = gt_c & ~_dil(canvas_bin)
    fp = canvas_bin & ~gt_d
    img = np.full((CANVAS_H, CANVAS_W, 3), 246, np.uint8)
    img[fn] = (255, 90, 30)        # azul (BGR)
    img[fp] = (40, 40, 235)        # vermelho
    img[tp] = (40, 180, 40)        # verde
    cv2.putText(
        img,
        f"{side['scenario']} TP={int(tp.sum())} FN={int(fn.sum())} "
        f"FP={int(fp.sum())}",
        (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2,
        cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)
    return {"tp": int(tp.sum()), "fn": int(fn.sum()),
            "fp": int(fp.sum())}


def audit_one(scen_dir, out_dir, flags, detector_kind="real"):
    """Um cenário, um braço, NESTE processo (isolamento de flags).

    detector_kind="real": detecção combinada real (neural+clássica)
    — o laço ponta a ponta. detector_kind="oracle": a trabalhadora
    de detecção devolve o quad VERDADEIRO do sidecar (H_t) e todo o
    resto do laço é o real (rastreador, trabalhadora de tinta,
    voto/acumulador). Costura declarada da frente 103: na câmera
    virtual o FT-dyn erra o quad por 70–112 px (medido), o que
    mistura falha de geometria sintética em toda métrica de tinta;
    o braço-oráculo isola a pilha de tinta — que é onde moram as
    perguntas da E3, das bandas finas e da fuga de impressão. O
    braço real fica como medida do gap sintético→detector."""
    from draw.app import _detect_sheet_combined_traced
    from draw.ink_layers import _PageCleanWorker
    from draw.loop import SheetInkLoop
    from draw.neural import NeuralSheetDetector
    from draw.page_clean import PageCleanInk
    from draw.sheet_track import _gate_info_from_trace

    for kv in flags:
        k, _, v = kv.partition("=")
        os.environ[k.strip()] = v.strip()
    scen_dir, out_dir = Path(scen_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    side = json.loads((scen_dir / "sidecar.json").read_text())
    detector = NeuralSheetDetector(str(SHEET_MODEL))
    detector.load()
    os.environ.setdefault("DRAW_MODEL_STRICT", "0")
    page_clean = PageCleanInk(str(INK_MODEL))
    page_clean.load()

    class Clock:
        dt = 1.0 / SCENE_FPS
        t = 0.0

        def __call__(self):
            return self.t

    clock = Clock()

    class SyncDet:
        def __init__(self):
            self._result = None
            self._seq = 0
            self.error = None
            self.traces = []

        def submit(self, frame, captured_at, frame_index=None):
            corners, verdict, trace = _detect_sheet_combined_traced(
                frame, detector)
            self._seq += 1
            result = (self._seq, corners, verdict, captured_at, 1.0)
            gate = _gate_info_from_trace(trace)
            if gate is not None:
                result = result + (gate,)
            self._result = result
            self.traces.append(trace)

        def poll(self, after_seq):
            if self._result is None or self._result[0] <= after_seq:
                return None
            return self._result

        def take_traces(self, upto_result_seq=None):
            return []

    if detector_kind == "oracle":
        true_corners = {
            fr["index"]: np.asarray(fr["corners_camera"], float)
            for fr in side["frames"]
        }

        class OracleDet:
            def __init__(self):
                self._result = None
                self._seq = 0
                self.error = None
                self.traces = []

            def submit(self, frame, captured_at, frame_index=None):
                corners = true_corners.get(frame_index)
                self._seq += 1
                verdict = "paper" if corners is not None else "none"
                self._result = (
                    self._seq,
                    None if corners is None
                    else corners.astype(np.float32),
                    verdict, captured_at, 1.0,
                )
                self.traces.append({})

            def poll(self, after_seq):
                if self._result is None \
                        or self._result[0] <= after_seq:
                    return None
                return self._result

            def take_traces(self, upto_result_seq=None):
                return []

        det = OracleDet()
    else:
        det = SyncDet()
    worker = _PageCleanWorker(page_clean, None, clock=clock)
    loop = SheetInkLoop(
        detection_worker=det, ink_worker=worker, clock=clock,
        log=lambda *_a: None)
    files = sorted((scen_dir / "frames").glob("*.png"))
    series = []
    seg_gt = None
    if "occluded_segment" in side["bands"]:
        seg_gt = _to_canvas(cv2.imread(
            str(scen_dir / side["bands"]["occluded_segment"]),
            cv2.IMREAD_GRAYSCALE))
    canvas_bin = np.zeros((CANVAS_H, CANVAS_W), bool)
    for idx, path in enumerate(files):
        frame = cv2.imread(str(path))
        clock.t = idx * clock.dt
        loop.step(frame, clock.t * 1000.0)
        worker.pump()
        loop.collect_ink()
        canvas_bin = (
            (loop.clean_mask > 0)
            if loop.clean_mask is not None
            else np.zeros((CANVAS_H, CANVAS_W), bool)
        )
        gt_c = _to_canvas(cv2.imread(
            str(scen_dir / side["frames"][idx]["gt"]),
            cv2.IMREAD_GRAYSCALE))
        true_q = np.asarray(
            side["frames"][idx]["corners_camera"], float)
        quad_err = (
            float(np.linalg.norm(
                np.asarray(loop.tracker.corners, float) - true_q,
                axis=1).max())
            if loop.tracker.corners is not None else None
        )
        series.append({
            "frame": idx,
            "present": loop.tracker.corners is not None,
            "quad_err_max_px": quad_err,
            "generation": loop.clean_generation,
            "passes": worker._passes_total,
            "canvas_px": int(canvas_bin.sum()),
            "gt_px": int(gt_c.sum()),
            "recall_dil3": (
                float((_dil(canvas_bin) & gt_c).sum()
                      / max(1, gt_c.sum())) if gt_c.any() else None
            ),
            "occluded_seg_recall_dil3": (
                float((_dil(canvas_bin) & seg_gt).sum()
                      / max(1, seg_gt.sum()))
                if seg_gt is not None and seg_gt.any() else None
            ),
        })
    np.savez_compressed(
        out_dir / "canvas_final.npz", canvas=canvas_bin)
    # Métricas finais
    gt_last = _to_canvas(cv2.imread(
        str(scen_dir / side["frames"][-1]["gt"]),
        cv2.IMREAD_GRAYSCALE))
    gt_d = _dil(gt_last)
    res = {
        "scenario": side["scenario"],
        "flags": flags,
        "detector": detector_kind,
        "sheet_sha16": sha16(SHEET_MODEL),
        "ink_sha16": sha16(INK_MODEL),
        "frames": len(files),
        "passes_total": worker._passes_total,
        "series": series,
        "final": {},
    }
    fin = res["final"]
    if gt_last.any():
        fin["gt_px"] = int(gt_last.sum())
        fin["canvas_px"] = int(canvas_bin.sum())
        fin["recall_exact"] = float(
            (canvas_bin & gt_last).sum() / gt_last.sum())
        # recall@dil3: fração do GT a ≤3 px de algum px do canvas
        # (o GT dilatado sobre o canvas daria >1 com traço grosso).
        fin["recall_dil3"] = float(
            (_dil(canvas_bin) & gt_last).sum() / gt_last.sum())
        # precision@dil3 (contenção): fração do canvas a ≤3 px
        # do GT.
        fin["precision_dil3"] = float(
            (canvas_bin & gt_d).sum() / max(1, canvas_bin.sum()))
        fin["fp_px"] = int((canvas_bin & ~gt_d).sum())
        border = np.zeros_like(canvas_bin)
        border[:15, :] = border[-15:, :] = True
        border[:, :15] = border[:, -15:] = True
        fin["fp_border_px"] = int(
            (canvas_bin & ~gt_d & border).sum())
        fin["fp_interior_px"] = int(
            (canvas_bin & ~gt_d & ~border).sum())
        fin["growth"] = float(canvas_bin.sum() / gt_last.sum())
        # Deriva: distância dos px do canvas ao GT exato.
        dist = cv2.distanceTransform(
            (~gt_last).astype(np.uint8), cv2.DIST_L2, 5)
        dvals = dist[canvas_bin]
        fin["chamfer_median_px"] = float(np.median(dvals))
        fin["chamfer_p90_px"] = float(np.percentile(dvals, 90))
    else:
        fin["gt_px"] = 0
        fin["canvas_px"] = int(canvas_bin.sum())
        fin["fp_px"] = int(canvas_bin.sum())
        border = np.zeros_like(canvas_bin)
        border[:15, :] = border[-15:, :] = True
        border[:, :15] = border[:, -15:] = True
        fin["fp_border_px"] = int((canvas_bin & border).sum())
        fin["fp_interior_px"] = int((canvas_bin & ~border).sum())
    # Bandas
    fin["bands"] = {}
    for band, rel in side["bands"].items():
        bm = _to_canvas(cv2.imread(
            str(scen_dir / rel), cv2.IMREAD_GRAYSCALE))
        if not bm.any():
            continue
        fin["bands"][band] = {
            "gt_px": int(bm.sum()),
            "recall_exact": float(
                (canvas_bin & bm).sum() / bm.sum()),
            "recall_dil3": float(
                (_dil(canvas_bin) & bm).sum() / bm.sum()),
        }
    # Fuga de impressão: px de pauta/grade (longe da tinta GT)
    # que viraram tinta no canvas.
    if side["print_kind"] != "clean":
        pm = _to_canvas(cv2.imread(
            str(scen_dir / side["print_mask"]),
            cv2.IMREAD_GRAYSCALE))
        free = pm & ~gt_d
        fin["print_px_free"] = int(free.sum())
        fin["print_leak_pct"] = float(
            (canvas_bin & free).sum() / max(1, free.sum()) * 100.0)
    # Eventos: deriva pós-empurrão e recuperação pós-oclusão.
    ev = side.get("events_spec") or {}
    if ev:
        pf = ev["push_frame"]
        pre = series[pf - 1]["recall_dil3"]
        post = series[-1]["recall_dil3"]
        fin["push_recall_pre"] = pre
        fin["push_recall_final"] = post
        fin["push_quad_err_at_push_px"] = series[pf][
            "quad_err_max_px"]
        fin["push_quad_err_final_px"] = series[-1][
            "quad_err_max_px"]
        if "occluded_segment" in side["bands"]:
            bm = _to_canvas(cv2.imread(
                str(scen_dir / side["bands"]["occluded_segment"]),
                cv2.IMREAD_GRAYSCALE))
            fin["occluded_seg_recall_final"] = float(
                (_dil(canvas_bin) & bm).sum() / max(1, bm.sum()))
            res["occluded_segment_gt_px"] = int(bm.sum())
            # Recuperação pós-oclusão: frames desde a revelação
            # (último frame ocluído + 1) até o recall do trecho
            # atingir ≥90% do seu valor final.
            reveal = max(ev["occlusion_frames"]) + 1
            target = 0.9 * fin["occluded_seg_recall_final"]
            rec = None
            for row in series:
                if row["frame"] < reveal:
                    continue
                v = row["occluded_seg_recall_dil3"]
                if v is not None and v >= target:
                    rec = row["frame"] - reveal
                    break
            fin["occlusion_recovery_frames"] = rec
            fin["occlusion_reveal_frame"] = reveal
    panel = diff_panel(scen_dir, canvas_bin,
                       out_dir / "panel_diff.png")
    res["panel_counts"] = panel
    (out_dir / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"scenario": res["scenario"],
                      "final": {k: v for k, v in fin.items()
                                if k != "bands"},
                      "bands": fin.get("bands", {})}, indent=1))
    return res


def audit(corpus, out, flags, scenarios, detector_kind="real"):
    """Roda cada cenário num subprocesso (flags isoladas, ONNX
    sequencial) e consolida."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name in scenarios:
        sub = out / name
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "audit-one", "--corpus", str(Path(corpus) / name),
            "--out", str(sub), "--detector", detector_kind,
        ]
        if flags:
            cmd += ["--flags", ",".join(flags)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:])
            print(r.stderr[-3000:], file=sys.stderr)
            raise SystemExit(f"audit-one falhou: {name}")
        results.append(json.loads(
            (sub / "result.json").read_text()))
    (out / "audit.json").write_text(
        json.dumps(results, indent=1))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", required=True)
    g.add_argument("--scenarios", default=",".join(SCENARIOS))
    a = sub.add_parser("audit")
    a.add_argument("--corpus", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--flags", default="")
    a.add_argument("--detector", default="real",
                   choices=["real", "oracle"])
    a.add_argument("--scenarios", default=",".join(SCENARIOS))
    o = sub.add_parser("audit-one")
    o.add_argument("--corpus", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--flags", default="")
    o.add_argument("--detector", default="real",
                   choices=["real", "oracle"])
    args = ap.parse_args(argv)
    scenarios = [
        s for s in getattr(args, "scenarios", "").split(",")
        if s.strip()
    ]
    if args.cmd == "generate":
        generate(args.out, scenarios)
    elif args.cmd == "audit":
        flags = [f for f in args.flags.split(",") if f.strip()]
        audit(args.corpus, args.out, flags, scenarios,
              detector_kind=args.detector)
    else:
        flags = [f for f in args.flags.split(",") if f.strip()]
        audit_one(args.corpus, args.out, flags,
                  detector_kind=args.detector)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
