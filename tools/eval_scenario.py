"""Auditoria por fase do cenário escrever→apagar→redesenhar→mover.

Consome um MP4 do sim_source com bundle GT que declara ``events``,
``stroke_phase`` e ``page_ink_final``, e mede o que cada fase exige do
laço real: a escrita confirma, a região apagada perde pixels
confirmados, o traço redesenhado confirma de novo (o contrato do E3:
pixel apagado não fica provisório para sempre) e a folha movida não
zera o canvas. Cada fotografia usa a âncora que a worker realmente
publicava naquele frame; para cenário simples, sem events, o caminho
é o eval_capture de sempre.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw.paper import perspective_matrix  # noqa: E402
from tools.eval_capture import pixel_metrics, run_pipeline  # noqa: E402


def _stroke_mask(shape, points, thickness):
    mask = np.zeros(shape, np.uint8)
    cv2.polylines(mask, [points.astype(np.int32)], False, 255, int(thickness))
    return mask


def _page_expected(bundle, applied_upto, erased):
    """Tinta esperada na folha: traços aplicados até a fase, menos o
    retângulo já apagado (a borracha corta qualquer traço dentro dele)."""
    shape = bundle["page_ink_full"].shape
    expected = np.zeros(shape, np.uint8)
    points, offsets = bundle["points"], bundle["offsets"]
    for index in range(applied_upto):
        geo = points[offsets[index] : offsets[index + 1]]
        cv2.polylines(
            expected,
            [geo.astype(np.int32)],
            False,
            255,
            int(bundle["thickness"][index]),
        )
    if erased:
        x, y, w, h = (int(v) for v in bundle["erase_rect"])
        expected[y : y + h, x : x + w] = 0
    return expected


def _project(mask, corners, page_to_scene, canvas_hw):
    height, width = canvas_hw
    destination = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = perspective_matrix(np.asarray(corners, np.float32), destination)
    return cv2.warpPerspective(
        mask, matrix @ page_to_scene, (width, height), flags=cv2.INTER_NEAREST
    )


def audit(video, gt_path, *, fps=None, models=(), save_masks=None):
    with np.load(gt_path, allow_pickle=False) as archive:
        bundle = {key: archive[key] for key in archive.files}
    missing = {"events", "stroke_phase", "stroke_erased", "page_ink_final"} - set(
        bundle
    )
    if missing:
        raise ValueError(
            f"bundle sem campos de cenário {sorted(missing)}; use eval_capture"
        )
    events = json.loads(str(bundle["events"]))
    capture = cv2.VideoCapture(str(video))
    try:
        snapshots = {}
        ends = {phase["end"] - 1: phase for phase in events}

        def on_frame(loop, worker, frame_index):
            if frame_index not in ends:
                return
            snapshots[frame_index] = {
                "confirmed": (
                    None if loop.clean_mask is None else loop.clean_mask.copy()
                ),
                "provisional": (
                    None
                    if loop.clean_provisional_mask is None
                    else loop.clean_provisional_mask.copy()
                ),
                "corners": (
                    None
                    if worker is None or worker._anchor_corners is None
                    else np.asarray(worker._anchor_corners).copy()
                ),
            }

        pipeline = run_pipeline(
            capture,
            fps=fps,
            neural_model=models.get("neural"),
            page_clean_model=models.get("page_clean"),
            hand_model=models.get("hand"),
            on_frame=on_frame,
        )
    finally:
        capture.release()

    final_mask = pipeline.pop("mask", None)
    pipeline["final_confirmed_px"] = (
        None if final_mask is None else int((np.asarray(final_mask) > 0).sum())
    )
    corners = pipeline.get("corners")
    pipeline["anchor_corners"] = None if corners is None else corners.tolist()
    pipeline.pop("corners", None)
    canvas_hw = tuple(pipeline["canvas_shape"])
    stroke_phase = bundle["stroke_phase"]
    points, offsets = bundle["points"], bundle["offsets"]
    shape = bundle["page_ink_full"].shape

    def phase_stroke_masks(phase_index):
        masks = []
        for index in np.nonzero(stroke_phase == phase_index)[0]:
            geo = points[offsets[index] : offsets[index + 1]]
            masks.append(_stroke_mask(shape, geo, bundle["thickness"][index]))
        return masks

    checkpoints = []
    applied_upto = 0
    erased = False
    writes_seen = 0
    for phase in events:
        frame = phase["end"] - 1
        if phase["kind"] == "write":
            writes_seen += 1
            # fase write termina com a lista dela toda aplicada
            applied_upto = (
                int((stroke_phase == 0).sum())
                if writes_seen == 1
                else int(stroke_phase.size)
            )
        if phase["kind"] == "erase":
            erased = True
        snap = snapshots.get(frame)
        if snap is None:
            continue
        if save_masks is not None and snap["confirmed"] is not None:
            save_masks.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(save_masks / f"confirmed_{phase['kind']}_{frame}.png"),
                (np.asarray(snap["confirmed"]) > 0).astype(np.uint8) * 255,
            )
        moved = any(p["kind"] == "move" and p["end"] <= phase["end"] for p in events)
        page_to_scene = (
            bundle["page_to_scene_final"] if moved else bundle["page_to_scene"]
        )
        expected = _page_expected(bundle, applied_upto, erased)
        entry = {
            "frame": frame,
            "phase": phase["kind"],
            "phase_start": phase["start"],
        }
        if snap["corners"] is None or snap["confirmed"] is None:
            entry["status"] = "sem âncora/canvas"
            checkpoints.append(entry)
            continue
        corners = snap["corners"]
        confirmed = np.asarray(snap["confirmed"]) > 0
        provisional = (
            np.zeros_like(confirmed)
            if snap["provisional"] is None
            else np.asarray(snap["provisional"]) > 0
        )
        truth = _project(expected, corners, page_to_scene, canvas_hw) > 0
        entry.update(pixel_metrics(confirmed, truth))
        entry["confirmed_px"] = int(confirmed.sum())
        entry["provisional_px"] = int(provisional.sum())
        x, y, w, h = (int(v) for v in bundle["erase_rect"])
        if w and h:
            rect_mask = np.zeros(shape, np.uint8)
            rect_mask[y : y + h, x : x + w] = 255
            rect_canvas = _project(rect_mask, corners, page_to_scene, canvas_hw) > 0
            entry["confirmed_in_erased_px"] = int((confirmed & rect_canvas).sum())
        redraw_masks = phase_stroke_masks(1)
        if redraw_masks:
            redraw = np.zeros(shape, np.uint8)
            for mask in redraw_masks:
                redraw |= mask
            redraw_canvas = _project(redraw, corners, page_to_scene, canvas_hw) > 0
            denom = max(1, int(redraw_canvas.sum()))
            entry["redraw_confirmed_recall"] = (
                float((confirmed & redraw_canvas).sum()) / denom
            )
            entry["redraw_provisional_px"] = int((provisional & redraw_canvas).sum())
        checkpoints.append(entry)

    final = next((c for c in reversed(checkpoints) if "iou" in c), {})
    report = {
        "video": str(video),
        "checkpoints": checkpoints,
        "pipeline": pipeline,
        "verdict": {
            "final_iou": final.get("iou"),
            "final_recall": final.get("recall"),
            "final_confirmed_px": final.get("confirmed_px"),
            "redraw_confirmed_recall": final.get("redraw_confirmed_recall"),
            "redraw_provisional_px": final.get("redraw_provisional_px"),
            "residual_confirmed_in_erased_px": final.get("confirmed_in_erased_px"),
        },
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--neural-model")
    parser.add_argument("--page-clean-model")
    parser.add_argument("--hand-model")
    parser.add_argument(
        "--save-masks",
        type=Path,
        help="grava a máscara confirmada de cada checkpoint como PNG",
    )
    args = parser.parse_args(argv)
    report = audit(
        args.video,
        args.gt,
        fps=args.fps,
        models={
            "neural": args.neural_model,
            "page_clean": args.page_clean_model,
            "hand": args.hand_model,
        },
        save_masks=args.save_masks,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
