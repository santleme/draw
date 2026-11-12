"""Avalia canvas vs GT ou dirige o laço real do Draw sem interface.

O GT entra apenas depois da captura: não alimenta detector, warp ou tinta.
O modo headless usa tempo de vídeo, detecção síncrona e pump() da worker
real; mede qualidade funcional offline, não latência/throughput de webcam.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pixel_metrics(prediction, truth):
    """Comparação exata, sem dilatação, tolerância ou alinhamento por conteúdo.

    Denominador vazio vale 1 (nenhum erro desse tipo); IoU de duas
    máscaras vazias vale 1. As contagens tornam a convenção auditável.
    """
    prediction, truth = np.asarray(prediction) > 0, np.asarray(truth) > 0
    if prediction.ndim != 2 or prediction.shape != truth.shape:
        raise ValueError("predição e GT precisam ser máscaras 2D do mesmo tamanho")
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    return {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_pixels": tp + fp,
        "gt_pixels": tp + fn,
    }


def compare_masks(prediction, truth, stroke_masks=(), *, stroke_recall=0.5):
    """Relatório por movimento de caneta: captured, partial, lost, unscorable.

    Recall é calculado em toda a geometria de cada traço, inclusive
    interseções; interseções podem ser explicadas por mais de um traço.
    Não é uma inferência causal de qual passada viu qual traço.
    """
    if not math.isfinite(stroke_recall) or not 0 < stroke_recall <= 1:
        raise ValueError("stroke_recall deve estar em (0, 1]")
    result = pixel_metrics(prediction, truth)
    result["stroke_recall_threshold"] = stroke_recall
    result["strokes"] = []
    for index, mask in enumerate(stroke_masks):
        metrics = pixel_metrics(prediction, mask)
        status = (
            "unscorable"
            if metrics["gt_pixels"] == 0
            else "captured"
            if metrics["recall"] >= stroke_recall
            else "partial"
            if metrics["tp"]
            else "lost"
        )
        result["strokes"].append(
            {
                "index": index,
                "status": status,
                "recall": metrics["recall"],
                "captured_pixels": metrics["tp"],
                "gt_pixels": metrics["gt_pixels"],
            }
        )
    result["stroke_counts"] = {
        status: sum(s["status"] == status for s in result["strokes"])
        for status in ("captured", "partial", "lost", "unscorable")
    }
    return result


class _VideoClock:
    now = 0.0

    def __call__(self):
        return self.now


class _SynchronousDetector:
    """Só adapta a execução; a decisão é a mesma de draw.app."""

    error = None

    def __init__(self, neural):
        from draw.app import _detect_sheet_combined_traced

        self.detect = lambda frame: _detect_sheet_combined_traced(frame, neural)
        self.result = None
        self.sequence = 0

    def submit(self, frame, captured_at, frame_index=None):
        from draw.sheet_track import _gate_info_from_trace

        corners, verdict, trace = self.detect(frame)
        self.sequence += 1
        self.result = (self.sequence, corners, verdict, captured_at, 0.0)
        gate = _gate_info_from_trace(trace)
        if gate is not None:
            self.result += (gate,)

    def poll(self, after_seq):
        if self.result is not None and self.result[0] > after_seq:
            return self.result
        return None

    def take_traces(self, upto_result_seq=None):
        return []


def run_pipeline(
    capture,
    *,
    fps=None,
    neural_model=None,
    page_clean_model=None,
    legacy_ink=False,
):
    """Consome VideoCapture ou SyntheticCapture via SheetInkLoop real.

    Não inicia threads, não descarta frames, não lê propriedades de GT da
    fonte. Sem modelo de tinta, exige --legacy-ink explicitamente. Captura
    continua pertencendo ao chamador, que deve chamar release().
    """
    from draw import config, ink
    from draw.ink_layers import _PageCleanWorker
    from draw.loop import SheetInkLoop
    from draw.neural import NeuralSheetDetector
    from draw.page_clean import PageCleanInk

    if not capture.isOpened():
        raise ValueError("não foi possível abrir a fonte de vídeo")
    fps = float(fps if fps is not None else capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("vídeo sem FPS válido; informe --fps")
    neural = NeuralSheetDetector(neural_model) if neural_model else None
    clock = _VideoClock()
    worker = None
    if not legacy_ink:
        page_clean = PageCleanInk(page_clean_model)
        page_clean.load()
        worker = _PageCleanWorker(page_clean, clock=clock)
    detector = _SynchronousDetector(neural)
    loop = SheetInkLoop(
        detection_worker=detector,
        ink_worker=worker,
        ink_source=ink.extract_ink if legacy_ink else None,
        clock=clock,
        log=lambda *_args: None,
    )
    frame_index = 0
    last_legacy_corners = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        clock.now = frame_index / fps
        outcome = loop.step(frame, clock.now * 1000)
        if worker is not None:
            if worker.pump() and outcome is None:
                outcome = loop.collect_ink()
            if worker.error is not None:
                raise RuntimeError("worker de tinta falhou") from worker.error
        elif loop.corner_history:
            last_legacy_corners = np.median(
                np.stack(loop.corner_history), axis=0
            ).astype(np.float32)
        if outcome is not None:
            raise RuntimeError(f"laço do Draw encerrou: {outcome}")
        frame_index += 1
    if frame_index == 0:
        raise ValueError("fonte de vídeo não contém frames decodificáveis")
    if worker is not None:
        mask = None if loop.clean_mask is None else loop.clean_mask.copy()
        corners = worker._anchor_corners
        passes = worker._passes_total
    else:
        mask = (
            None
            if loop.accumulated_canvas is None
            else cv2.cvtColor(loop.accumulated_canvas, cv2.COLOR_BGR2GRAY) < 128
        )
        corners, passes = last_legacy_corners, None
    return {
        "mask": mask,
        "corners": None if corners is None else np.asarray(corners).copy(),
        "frames": frame_index,
        "fps": fps,
        "ink_passes": passes,
        "detections": detector.sequence,
        "status": "ok" if mask is not None else "no_canvas",
        "mode": "legacy" if legacy_ink else "page_clean",
        "sheet_detector": "neural+classic" if neural is not None else "classic",
        "canvas_shape": [config.PAPER_H, config.PAPER_W],
        "scheduler": "synchronous_video_clock",
    }


def load_canvas(path, kind=None):
    path = Path(path)
    kind = kind or ("mask" if path.suffix.lower() == ".npy" else "image")
    data = (
        np.load(path, allow_pickle=False)
        if path.suffix.lower() == ".npy"
        else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    )
    if data is None:
        raise ValueError(f"não foi possível ler {path}")
    if data.ndim == 3 and kind == "image":
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("canvas precisa ser uma imagem/máscara 2D finita")
    if kind == "image":
        return data < 128
    return data >= 0.5 if np.issubdtype(data.dtype, np.floating) else data > 0


def evaluate_canvas(
    prediction,
    gt_path,
    *,
    canvas_space="page",
    corners=None,
    stroke_recall=0.5,
):
    """Alinha GT por homografia conhecida, nunca por ajuste ao desenho.

    page sem corners pressupõe retificação ideal (bundle necessário).
    page com corners usa a âncora real detectada. scene exige a mesma
    grade de câmera que ink_full. NPY simples não tem traços individuais.
    """
    from draw.paper import perspective_matrix

    gt_path = Path(gt_path)
    bundle = None
    if gt_path.suffix.lower() == ".npz":
        with np.load(gt_path, allow_pickle=False) as archive:
            bundle = {key: archive[key] for key in archive.files}
        if int(bundle["version"]) != 1:
            raise ValueError("versão de GT não suportada")
        truth = bundle["ink_full"]
    else:
        truth = np.load(gt_path, allow_pickle=False)
    if truth.ndim != 2:
        raise ValueError("ink_full precisa ser máscara 2D")
    # Sem canvas adquirido: não inventa geometria; todo o GT é perdido.
    if prediction is None:
        prediction = np.zeros_like(truth)
        canvas_space = "scene"
    prediction = np.asarray(prediction)
    if prediction.ndim != 2:
        raise ValueError("predição precisa ser uma máscara 2D")
    height, width = prediction.shape
    page_matrix = None
    alignment = "scene"
    if canvas_space == "scene":
        if prediction.shape != truth.shape:
            raise ValueError("canvas da cena e ink_full têm grades diferentes")
        if bundle is not None:
            page_matrix = bundle["page_to_scene"]
    elif canvas_space == "page":
        if corners is not None:
            corners = np.asarray(corners, np.float32)
            if corners.shape != (4, 2) or not np.isfinite(corners).all():
                raise ValueError("corners precisa ter formato (4, 2) e valores finitos")
            destination = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            matrix = perspective_matrix(corners, destination)
            if bundle is not None:
                page_matrix = matrix @ bundle["page_to_scene"]
            else:
                truth = cv2.warpPerspective(
                    truth, matrix, (width, height), flags=cv2.INTER_NEAREST
                )
            alignment = "detected_anchor"
        elif bundle is not None:
            ph, pw = bundle["page_ink_full"].shape
            page_matrix = np.diag([width / pw, height / ph, 1.0])
            alignment = "ideal_page"
        else:
            raise ValueError("GT NPY em espaço de página exige --corners; use o bundle")
        if page_matrix is not None:
            truth = cv2.warpPerspective(
                bundle["page_ink_full"],
                page_matrix,
                (width, height),
                flags=cv2.INTER_NEAREST,
            )
    else:
        raise ValueError("canvas_space precisa ser page ou scene")

    def strokes():
        if bundle is None:
            return
        shape = bundle["page_ink_full"].shape
        points, offsets = bundle["points"], bundle["offsets"]
        for index, thickness in enumerate(bundle["thickness"]):
            mask = np.zeros(shape, np.uint8)
            geo = points[offsets[index] : offsets[index + 1]].astype(np.int32)
            cv2.polylines(mask, [geo], False, 255, int(thickness))
            yield cv2.warpPerspective(
                mask, page_matrix, (width, height), flags=cv2.INTER_NEAREST
            )

    report = compare_masks(prediction, truth, strokes(), stroke_recall=stroke_recall)
    report.update(
        alignment=alignment,
        canvas_space=canvas_space,
        shape=[height, width],
        per_stroke_available=bundle is not None,
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--canvas", type=Path)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--save-canvas", type=Path, help="máscara .npy + .corners.npy")
    parser.add_argument("--canvas-space", choices=("page", "scene"), default="page")
    parser.add_argument("--canvas-kind", choices=("mask", "image"))
    parser.add_argument("--corners", type=Path, help="âncora real (4,2), .npy")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--stroke-recall", type=float, default=0.5)
    parser.add_argument("--neural-model", default=os.environ.get("DRAW_NEURAL_MODEL"))
    parser.add_argument("--page-clean-model")
    parser.add_argument("--legacy-ink", action="store_true")
    args = parser.parse_args(argv)
    try:
        pipeline = None
        corners = (
            None if args.corners is None else np.load(args.corners, allow_pickle=False)
        )
        if args.video:
            if args.corners or args.canvas_space != "page":
                raise ValueError("--video usa a âncora detectada, em espaço page")
            capture = cv2.VideoCapture(str(args.video))
            try:
                neural_model = args.neural_model
                default_model = (
                    Path(__file__).resolve().parents[1] / "models/draw_seg_v3_dyn.onnx"
                )
                if neural_model is None and default_model.is_file():
                    neural_model = str(default_model)
                pipeline = run_pipeline(
                    capture,
                    fps=args.fps,
                    neural_model=neural_model,
                    page_clean_model=args.page_clean_model,
                    legacy_ink=args.legacy_ink,
                )
            finally:
                capture.release()
            prediction = pipeline.pop("mask")
            corners = pipeline.pop("corners")
            pipeline["anchor_corners"] = None if corners is None else corners.tolist()
        else:
            prediction = load_canvas(args.canvas, args.canvas_kind)
        report = evaluate_canvas(
            prediction,
            args.gt,
            canvas_space=args.canvas_space,
            corners=corners,
            stroke_recall=args.stroke_recall,
        )
        if pipeline is not None:
            report["pipeline"] = pipeline
        if args.save_canvas:
            if prediction is None:
                raise ValueError("nenhum canvas adquirido para salvar")
            if args.save_canvas.suffix.lower() != ".npy":
                raise ValueError("--save-canvas exige extensão .npy")
            args.save_canvas.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.save_canvas, prediction.astype(np.uint8) * 255)
            if corners is not None:
                np.save(args.save_canvas.with_suffix(".corners.npy"), corners)
        serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(serialized, encoding="utf-8")
    except (ValueError, RuntimeError, OSError, KeyError, cv2.error) as error:
        parser.exit(
            2,
            f"erro: {error}\nPara tinta sem ONNX, escolha --legacy-ink explicitamente.\n",
        )
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
