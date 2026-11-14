"""Bateria de cenários do simulador — um comando, uma tabela.

Roda o laço real (`run_pipeline`) em cada cenário capturado e
consolida o que a rodada precisa saber: métricas de pixel/traço,
auditoria por fase nos cenários com events, e a trilha de decisões
de identidade (same_page/new_page/deferências) do worker. É o loop
de feedback autônomo: `--profile hardened` liga a guarda de
cobertura e o resgate de identidade em todos os cenários e mede
se endurecer regressiona algum caso.

    python tools/eval_battery.py --profile hardened
    python tools/eval_battery.py --only sim-lift,desk-ghost3
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.eval_capture import (  # noqa: E402
    evaluate_canvas,
    pixel_metrics,
    run_pipeline,
)
from tools import eval_scenario  # noqa: E402

NEURAL = "models/draw_seg_v3_dyn.onnx"

# Cenários capturados. O GT decide a auditoria: bundles com
# ``events`` passam pela avaliação por fase (a mesma lógica do
# eval_scenario), os simples pela métrica de canvas do
# eval_capture. ``env`` sobrepõe o perfil quando o cenário pede um
# comportamento específico.
BATTERY = {
    "sim": {},
    "sim-long": {},
    "sim-e3": {},
    "sim-e3-nomove": {},
    "sim-hand": {},
    "sim-lift": {},
    "sim-lift0": {},
    "sim-lift10": {},
    "desk-ghost3": {},
    "desk-return": {},
}

# O perfil endurecido liga as proteções da rodada da mesa real:
# a guarda de cobertura (âncora-fantasma) e o resgate de
# identidade por conteúdo (drop→retorno extremo).
PROFILES = {
    "default": {},
    "hardened": {
        "DRAW_ANCHOR_COVERAGE_GUARD": "1",
        "DRAW_IDENTITY_ECC_ALIGN": "1",
        # Par travado: a máscara de mão do seg só sai no extras
        # quando a cobertura está ligada; a oclusão consome ela.
        # Medido no sim-hand: 421k → 36k px confirmados.
        "DRAW_SHEET_COVERAGE": "1",
        "DRAW_SEG_HAND_OCCLUSION": "1",
    },
}

# Variáveis de comportamento que a bateria controla — restauradas
# depois de cada cenário para um perfil não vazar no seguinte.
TOGGLED_ENV = (
    "DRAW_ANCHOR_COVERAGE_GUARD",
    "DRAW_IDENTITY_ECC_ALIGN",
    "DRAW_ECC_REFINE",
    "DRAW_IDENTITY_CHAMFER",
    "DRAW_SHEET_COVERAGE",
    "DRAW_SEG_HAND_OCCLUSION",
)


def _identity_summary(events):
    """Decisões da vigia de identidade na ordem em que saíram."""
    out = []
    for event in events:
        decision = event.get("decision")
        if decision is None:
            continue
        entry = str(decision)
        if decision == "same_page" and event.get("align_converged"):
            entry += f"({event.get('align_mode')})"
        out.append(entry)
    return out


def _load_bundle(gt_path):
    with np.load(gt_path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _run(name, spec, captures, models):
    video = captures / f"{name}.mp4"
    gt = captures / f"{name}.gt.npz"
    bundle = _load_bundle(gt)
    phased = "events" in bundle
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"{name}: não abriu {video}")

    snapshots = {}
    identity_events = []
    if phased:
        events = json.loads(str(bundle["events"]))
        ends = {phase["end"] - 1 for phase in events}
    else:
        ends = set()

    events_seen = [0]

    def on_frame(loop, worker, frame_index):
        if worker is not None:
            all_events = getattr(worker, "_identity_events", None) or []
            if len(all_events) < events_seen[0]:
                events_seen[0] = 0  # lista reiniciada por geração
            identity_events.extend(list(all_events)[events_seen[0]:])
            events_seen[0] = len(all_events)
        if frame_index in ends:
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

    try:
        pipeline = run_pipeline(
            capture,
            neural_model=models.get("neural"),
            page_clean_model=models.get("page_clean"),
            hand_model=models.get("hand"),
            on_frame=on_frame,
        )
    finally:
        capture.release()

    result = {
        "name": name,
        "phased": phased,
        "frames": pipeline["frames"],
        "ink_passes": pipeline["ink_passes"],
        "identity_decisions": _identity_summary(identity_events),
    }

    if phased:
        # Auditoria por fase com os helpers do eval_scenario — o
        # mesmo veredito, sem rodar a captura de novo.
        events = json.loads(str(bundle["events"]))
        canvas_hw = tuple(pipeline["canvas_shape"])
        stroke_phase = bundle["stroke_phase"]

        checkpoints = []
        applied_upto = 0
        erased = False
        writes_seen = 0
        for phase in events:
            frame = phase["end"] - 1
            if phase["kind"] == "write":
                writes_seen += 1
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
            moved = any(
                p["kind"] in ("move", "return") and p["end"] <= phase["end"]
                for p in events
            )
            page_to_scene = (
                bundle["page_to_scene_final"]
                if moved
                else bundle["page_to_scene"]
            )
            expected = eval_scenario._page_expected(
                bundle, applied_upto, erased
            )
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
            truth = (
                eval_scenario._project(
                    expected, corners, page_to_scene, canvas_hw
                )
                > 0
            )
            entry.update(pixel_metrics(confirmed, truth))
            entry["confirmed_px"] = int(confirmed.sum())
            checkpoints.append(entry)

        final = eval_scenario._final_checkpoint(checkpoints, events)
        result["checkpoints"] = checkpoints
        result["verdict"] = {
            "status": final.get("status"),
            "final_iou": final.get("iou"),
            "final_recall": final.get("recall"),
            "final_confirmed_px": final.get("confirmed_px"),
        }
    else:
        metrics = evaluate_canvas(
            pipeline["mask"],
            gt,
            canvas_space=(
                "page" if pipeline["mask"] is not None else "scene"
            ),
            corners=pipeline.get("corners"),
        )
        result["verdict"] = {
            "status": pipeline["status"],
            "final_iou": metrics["iou"],
            "final_recall": metrics["recall"],
            "final_confirmed_px": (
                0
                if pipeline["mask"] is None
                else int(np.count_nonzero(pipeline["mask"]))
            ),
            "strokes": metrics.get("stroke_counts"),
        }

    return result


def _format_row(result):
    verdict = result["verdict"] or {}
    iou = verdict.get("final_iou")
    recall = verdict.get("final_recall")
    decisions = result["identity_decisions"]
    last_decision = decisions[-1] if decisions else "-"
    deferrals = sum(1 for d in decisions if d.startswith("deferred"))
    return (
        f"{result['name']:<16} "
        f"{(verdict.get('status') or '-'):<18} "
        f"{iou if iou is not None else float('nan'):>5.2f} "
        f"{recall if recall is not None else float('nan'):>5.2f} "
        f"{verdict.get('final_confirmed_px') or 0:>7} "
        f"{deferrals:>3} {last_decision}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="hardened",
        help="conjunto de flags aplicado a todos os cenários",
    )
    parser.add_argument(
        "--only",
        help="cenários separados por vírgula (default: todos)",
    )
    parser.add_argument("--captures", type=Path, default=Path("captures"))
    parser.add_argument("--out", type=Path, help="grava o relatório JSON")
    parser.add_argument("--neural-model", default=NEURAL)
    parser.add_argument("--page-clean-model")
    parser.add_argument(
        "--hand-model",
        default="/nonexistent",
        help=(
            "modelo .task de mão — default desligado: o MediaPipe "
            "cria duas instâncias (máscara + presença) e trava a "
            "bateria em deadlock intermitente no dispatcher; a mão "
            "sintética dos sims não é detectável de qualquer forma"
        ),
    )
    args = parser.parse_args(argv)

    wanted = (
        set(BATTERY)
        if not args.only
        else {name.strip() for name in args.only.split(",") if name.strip()}
    )
    missing = wanted - set(BATTERY)
    if missing:
        raise SystemExit(f"cenários desconhecidos: {sorted(missing)}")

    env = dict(PROFILES[args.profile])
    env.setdefault("DRAW_MODEL_STRICT", "0")
    saved = {key: os.environ.get(key) for key in TOGGLED_ENV}
    saved["DRAW_MODEL_STRICT"] = os.environ.get("DRAW_MODEL_STRICT")

    models = {
        "neural": args.neural_model,
        "page_clean": args.page_clean_model,
        "hand": args.hand_model,
    }
    results = []
    os.environ.update(env)
    try:
        for name in BATTERY:
            if name not in wanted:
                continue
            spec = dict(BATTERY[name])
            scenario_env = spec.pop("env", {})
            saved_scenario = {
                key: os.environ.get(key) for key in scenario_env
            }
            os.environ.update(scenario_env)
            try:
                print(f"== {name} ({args.profile})", flush=True)
                results.append(_run(name, spec, args.captures, models))
            finally:
                for key, value in saved_scenario.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    print()
    print(
        f"{'cenário':<16} {'status':<18} {'IoU':>5} {'rec':>5} "
        f"{'conf_px':>7} {'def':>3} última decisão"
    )
    for result in results:
        print(_format_row(result))

    report = {
        "profile": args.profile,
        "env": env,
        "results": results,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
