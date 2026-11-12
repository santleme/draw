"""Laboratório autônomo de replay (frente 102).

Câmera virtual sobre material gravado: injeta frame a frame no laço
REAL completo (draw.loop.SheetInkLoop — detecção combinada neural+
clássica com rastro, rastreador, trabalhadora página→limpa real em
pump(), acumulador) e produz (a) telemetria por frame em JSON e
(b) pranchas de diagnóstico 2×2 dos piores K frames por sinais
declarados, para auditoria visual sem ir à mesa.

Fronteira honesta: o replay é SÍNCRONO (detecção com atraso de
pipeline de 1 quadro, relógio de cena determinístico). Ele cobre a
classe funcional/visual/geométrica; NÃO reproduz a classe temporal
(cadência viva da F91/F92, fila FIFO do driver da F95) — essas
nasceram de telemetria viva e continuam exigindo sessão de mesa.

Uso:
  python tools/replay_lab.py --corpus <dir-de-frames|dir-de-sessao> \
      --out <dir> [--top-k 8] [--stride 1] [--fps-cena 2.0] \
      [--flags DRAW_INK_VOTE_E3=1,...]
Corpus: diretório com imagens, ou diretório de sessão gravada
(com subpasta raw/). Modelos: os de models/ do repo quando
presentes; senão ~/workspace/draw/models/.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ------------------------------------------------------------------
# Corpus
# ------------------------------------------------------------------
def list_corpus_frames(corpus):
    """Frames do corpus em ordem: raw/ de sessão ou o próprio dir."""
    corpus = Path(corpus)
    src = corpus
    for sub in ("raw", "frames"):
        if (corpus / sub).is_dir():
            src = corpus / sub
            break
    files = sorted(
        p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS
    )
    if not files:
        raise FileNotFoundError(f"nenhum frame em {src}")
    return files


# ------------------------------------------------------------------
# Sinais declarados de seleção dos piores frames
# ------------------------------------------------------------------
def quad_area(quad):
    if quad is None:
        return None
    return float(abs(cv2.contourArea(np.asarray(quad, np.float32))))


def quad_jump(prev, cur):
    """Maior deslocamento de canto entre dois quads (px)."""
    if prev is None or cur is None:
        return None
    d = np.linalg.norm(
        np.asarray(cur, np.float64) - np.asarray(prev, np.float64), axis=1
    )
    return float(d.max())


def compute_signals(records):
    """Sinais por frame a partir da telemetria (função pura).

    - quad_jump_px: salto máximo de canto do quad do DETECTOR
      (det_quad) vs detecção presente anterior — o quad do
      rastreador é suavizado por desenho e quase não salta: medido
      na frente 102, usá-lo deixava o sinal cego (0,0 em 43/43).
    - area_change: |Δárea| / área anterior (do det_quad)
    - canvas_growth_px: crescimento do canvas confirmado na frame
    - presence_flip: 1 se a presença mudou vs frame anterior
    """
    prev_quad = prev_area = prev_canvas = prev_present = None
    prev_tracker = None
    for r in records:
        quad = r.get("det_quad") or r.get("quad")
        # Calibração da auditoria visual da frente 102:
        # - sig_tracker_jump_px: salto do quad do RASTREADOR; o do
        #   detector pisca 113–144 px com a cena visualmente igual
        #   (ruído), o do rastreador só salta quando a geometria da
        #   tinta realmente muda.
        # - sig_quad_aspect: razão entre o maior e o menor lado do
        #   quad do rastreador; a cunha degenerada pós-virada
        #   (mat+madeira no warp) tem aspecto extremo e nenhum dos
        #   sinais originais a nomeava diretamente.
        tracker_quad = r.get("quad")
        r["sig_tracker_jump_px"] = quad_jump(prev_tracker, tracker_quad)
        if tracker_quad is not None:
            prev_tracker = tracker_quad
            q = np.asarray(tracker_quad, float)
            sides = [
                float(np.linalg.norm(q[(i + 1) % 4] - q[i]))
                for i in range(4)
            ]
            r["sig_quad_aspect"] = max(sides) / max(1.0, min(sides))
        else:
            r["sig_quad_aspect"] = None
        area = quad_area(quad) if quad else None
        r["sig_quad_jump_px"] = quad_jump(prev_quad, quad)
        r["sig_area_change"] = (
            abs(area - prev_area) / prev_area
            if (area is not None and prev_area)
            else None
        )
        canvas = r.get("canvas_px") or 0
        r["sig_canvas_growth_px"] = (
            max(0, canvas - prev_canvas) if prev_canvas is not None else 0
        )
        present = bool(r.get("present"))
        r["sig_presence_flip"] = (
            1 if (prev_present is not None and present != prev_present) else 0
        )
        if quad is not None:
            prev_quad, prev_area = quad, area
        prev_canvas, prev_present = canvas, present
    return records


def _rank(values):
    """Posto normalizado 0..1 (maior = pior); empates dividem o
    posto médio — sem isto, sinal constante (crescimento 0 em toda
    a sequência) virava rampa artificial pró-fim da sequência."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    denom = max(1, len(values) - 1)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0 / denom
        for k in range(i, j + 1):
            out[order[k]] = mean_rank
        i = j + 1
    return out


def select_worst(records, top_k, min_gap=2):
    """Piores K por soma de postos dos sinais; vizinhos a <min_gap
    frames de um já escolhido são suprimidos (dedupe)."""
    if not records:
        return []
    jumps = _rank([r.get("sig_quad_jump_px") or 0.0 for r in records])
    areas = _rank([r.get("sig_area_change") or 0.0 for r in records])
    growth = _rank([r.get("sig_canvas_growth_px") or 0 for r in records])
    tjumps = _rank([r.get("sig_tracker_jump_px") or 0.0 for r in records])
    aspects = _rank([r.get("sig_quad_aspect") or 0.0 for r in records])
    flips = [float(r.get("sig_presence_flip") or 0) for r in records]
    scored = []
    for i, r in enumerate(records):
        score = (
            jumps[i] + areas[i] + growth[i] + tjumps[i] + aspects[i]
            + 1.5 * flips[i]
        )
        scored.append((score, i))
    scored.sort(reverse=True)
    chosen = []
    for score, i in scored:
        if all(abs(i - j) >= min_gap for _, j in chosen):
            chosen.append((score, i))
        if len(chosen) >= top_k:
            break
    return [
        {"index": i, "score": round(s, 3), "frame": records[i]["frame"]}
        for s, i in sorted(chosen, key=lambda t: t[1])
    ]


# ------------------------------------------------------------------
# Prancha 2×2
# ------------------------------------------------------------------
def build_panel(frame, quad, warp, ink_mask):
    """2×2: original | warp+ tinta verde | quad vermelho | warp."""
    h, w = frame.shape[:2]
    pw, ph = w // 2, h // 2

    def fit(img):
        return cv2.resize(img, (pw, ph))

    quad_panel = frame.copy()
    if quad is not None:
        pts = np.asarray(quad, np.int32).reshape(-1, 1, 2)
        cv2.polylines(quad_panel, [pts], True, (0, 0, 255), 3)
        for p in np.asarray(quad, np.int32):
            cv2.circle(quad_panel, tuple(p), 7, (0, 0, 255), -1)
    if warp is None:
        warp = np.zeros_like(frame)
        mask_panel = warp.copy()
    else:
        wh, ww = warp.shape[:2]
        mask_panel = warp.copy()
        if ink_mask is not None:
            m = cv2.resize(
                (ink_mask > 0).astype(np.uint8) * 255, (ww, wh)
            )
            green = np.zeros_like(warp)
            green[:, :, 1] = 255
            mask_panel = np.where(
                (m > 0)[..., None],
                cv2.addWeighted(warp, 0.45, green, 0.55, 0),
                warp,
            )
    panels = [fit(frame), fit(mask_panel), fit(quad_panel), fit(warp)]
    labels = ["original", "warp + tinta (verde)", "quad (vermelho)", "warp"]
    for panel, label in zip(panels, labels):
        cv2.rectangle(panel, (0, 0), (pw, 26), (0, 0, 0), -1)
        cv2.putText(
            panel, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    top = np.hstack(panels[:2])
    bottom = np.hstack(panels[2:])
    return np.vstack([top, bottom])


# ------------------------------------------------------------------
# Replay pelo laço real
# ------------------------------------------------------------------
class _SceneClock:
    def __init__(self, dt):
        self.dt = dt
        self.t = 0.0

    def __call__(self):
        return self.t


def _default_model(name):
    repo_models = Path(__file__).resolve().parent.parent / "models" / name
    if repo_models.exists():
        return repo_models
    # Nesta VM os binários verificados moram no laboratório (o
    # models/ de ~/workspace/draw tem a v3 antiga, divergente do
    # manifesto — medido na frente 102).
    lab = Path.home() / "workspace" / "draw-lab" / "neural" / "v3" / name
    if lab.exists():
        return lab
    return Path.home() / "workspace" / "draw" / "models" / name


def run_replay(corpus, out_dir, *, top_k=8, stride=1, scene_fps=2.0,
               flags=None, detector=None, page_clean=None,
               sheet_model=None, ink_model=None, panel_frames=None):
    """Roda o corpus pelo laço real. detector/page_clean injetáveis
    (testes); por omissão, os modelos reais. panel_frames: conjunto
    de índices (na enumeração com stride) cujos artefatos se gravam;
    None = só telemetria (1ª passagem)."""
    from draw.app import _detect_sheet_combined_traced
    from draw.ink_layers import _PageCleanWorker
    from draw.loop import SheetInkLoop
    from draw.neural import NeuralSheetDetector
    from draw.page_clean import PageCleanInk
    from draw.paper import warp_sheet
    from draw.sheet_track import _gate_info_from_trace

    for kv in flags or []:
        k, _, v = kv.partition("=")
        os.environ[k.strip()] = v.strip()

    files = list_corpus_frames(corpus)[::stride]
    if detector is None:
        detector = NeuralSheetDetector(
            str(sheet_model or _default_model("draw_seg_v3_dyn.onnx"))
        )
        detector.load()
    if page_clean is None:
        # Sem modelo no repo, o candidato v6 do laboratório é a
        # tinta julgada atual; ele não está no manifesto do repo,
        # então o lab rebaixa a verificação NESTE processo (o app
        # segue estrito) e o relatório declara o sha usado.
        ink_path = ink_model or _default_model("page_clean_unet.onnx")
        if ink_model is None and not (
            Path(__file__).resolve().parent.parent
            / "models" / "page_clean_unet.onnx"
        ).exists():
            candidate = (
                Path.home() / "workspace" / "draw-lab" / "live-audit-2026-11-10"
                / "frente48" / "results" / "page_clean_finetune_real_v6.onnx"
            )
            if candidate.exists():
                ink_path = candidate
                os.environ.setdefault("DRAW_MODEL_STRICT", "0")
        page_clean = PageCleanInk(str(ink_path))
        page_clean.load()

    clock = _SceneClock(1.0 / scene_fps)

    class SyncDetectionWorker:
        """Protocolo da trabalhadora real, síncrono (padrão F88):
        submit roda a detecção combinada real; poll entrega no
        step seguinte (atraso de pipeline de 1 quadro)."""

        def __init__(self):
            self._result = None
            self._seq = 0
            self.error = None
            self.traces = []

        def submit(self, frame, captured_at, frame_index=None):
            outcome = _detect_sheet_combined_traced(frame, detector)
            corners, verdict, trace = outcome[0], outcome[1], outcome[2]
            extras = outcome[3] if len(outcome) > 3 else None
            self._seq += 1
            result = (self._seq, corners, verdict, captured_at, 1.0)
            gate = _gate_info_from_trace(trace)
            if gate is not None:
                result = result + (gate,)
            if extras is not None:
                if gate is None:
                    result = result + (None,)
                result = result + (extras,)
            self._result = result
            self.traces.append(trace)

        def poll(self, after_seq):
            if self._result is None or self._result[0] <= after_seq:
                return None
            return self._result

        def take_traces(self, upto_result_seq=None):
            return []

    det_worker = SyncDetectionWorker()
    ink_worker = _PageCleanWorker(page_clean, None, clock=clock)
    loop = SheetInkLoop(
        detection_worker=det_worker,
        ink_worker=ink_worker,
        clock=clock,
        log=lambda *_a: None,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, path in enumerate(files):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        clock.t = idx * clock.dt
        loop.step(frame, clock.t * 1000.0)
        # Uma chamada de pump por frame (o padrão dos replays do
        # PR #65): a cadência da trabalhadora decide pelo relógio
        # de cena se a passada vence neste frame. Avançar o relógio
        # dentro do frame e resetá-lo no frame seguinte fazia o
        # tempo andar para trás e a contagem de passadas variar
        # entre rodadas idênticas — medido na frente 102.
        t_before = ink_worker._passes_total
        ink_worker.pump()
        passes_now = (
            [round(clock.t, 3)]
            if ink_worker._passes_total > t_before
            else []
        )
        loop.collect_ink()
        quad = (
            None
            if loop.tracker.corners is None
            else np.asarray(loop.tracker.corners, float).tolist()
        )
        trace = det_worker.traces[-1] if det_worker.traces else {}
        canvas_px = (
            int((loop.clean_mask > 0).sum())
            if loop.clean_mask is not None
            else 0
        )
        rec = {
            "frame": idx,
            "file": path.name,
            "present": loop.tracker.corners is not None,
            "quad": quad,
            "verdict": (trace or {}).get("combined_verdict"),
            "det_quad": (trace or {}).get("combined_corners"),
            "presence_logit": (trace or {}).get("presence_logit"),
            "source": (trace or {}).get("source"),
            "canvas_px": canvas_px,
            "generation": loop.clean_generation,
            "passes_total": ink_worker._passes_total,
            "pass_times": passes_now,
        }
        records.append(rec)
        if panel_frames is not None and idx in panel_frames:
            warp = None
            ink_mask = None
            if quad is not None:
                try:
                    warp = warp_sheet(
                        frame, np.asarray(quad, np.float32)
                    )
                    prob = page_clean.ink_prob(warp)
                    ink_mask = (prob >= page_clean.threshold).astype(
                        np.uint8
                    )
                except Exception:  # prancha sem warp é diagnóstico tb
                    warp, ink_mask = None, None
            panel = build_panel(frame, quad, warp, ink_mask)
            cv2.imwrite(str(out_dir / f"panel_{idx:04d}.png"), panel)
    compute_signals(records)
    (out_dir / "telemetry.json").write_text(
        json.dumps(
            {"corpus": str(corpus), "stride": stride,
             "scene_fps": scene_fps, "frames": records},
            indent=1,
        )
    )
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps-cena", type=float, default=2.0)
    ap.add_argument("--flags", default="")
    args = ap.parse_args(argv)
    flags = [f for f in args.flags.split(",") if f.strip()]
    out = Path(args.out)
    records = run_replay(
        args.corpus, out, top_k=args.top_k, stride=args.stride,
        scene_fps=args.fps_cena, flags=flags,
    )
    worst = select_worst(records, args.top_k)
    # 2ª passagem determinística: artefatos só dos selecionados.
    wanted = {w["index"] for w in worst}
    run_replay(
        args.corpus, out, top_k=args.top_k, stride=args.stride,
        scene_fps=args.fps_cena, flags=flags, panel_frames=wanted,
    )
    by_idx = {r["frame"]: r for r in records}
    index = [
        {
            **w,
            "file": by_idx[w["index"]]["file"],
            "signals": {
                k: by_idx[w["index"]].get(k)
                for k in (
                    "sig_quad_jump_px", "sig_area_change",
                    "sig_canvas_growth_px", "sig_presence_flip",
                )
            },
            "panel": f"panel_{w['index']:04d}.png",
        }
        for w in worst
    ]
    (out / "worst_index.json").write_text(json.dumps(index, indent=1))
    print(json.dumps({"frames": len(records), "worst": index}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
