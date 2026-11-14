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
# identidade por conteúdo (drop→retorno extremo). O default
# fixa as mesmas chaves em "0" — sem isso uma flag exportada no
# ambiente vazava para a rodada e a linha de base media o
# comportamento endurecido.
PROFILES = {
    "default": {
        "DRAW_ANCHOR_COVERAGE_GUARD": "0",
        "DRAW_IDENTITY_ECC_ALIGN": "0",
        "DRAW_SHEET_COVERAGE": "0",
        "DRAW_SEG_HAND_OCCLUSION": "0",
        "DRAW_INK_REMOVAL": "0",
        "DRAW_INK_MOTION_GATE": "0",
        "DRAW_REG_CONTENT_GUARD": "0",
        "DRAW_SHEET_MERGE_SPLIT": "0",
    },
    "hardened": {
        "DRAW_ANCHOR_COVERAGE_GUARD": "1",
        "DRAW_IDENTITY_ECC_ALIGN": "1",
        # Par travado: a máscara de mão do seg só sai no extras
        # quando a cobertura está ligada; a oclusão consome ela.
        # Medido no sim-hand: 421k → 36k px confirmados.
        "DRAW_SHEET_COVERAGE": "1",
        "DRAW_SEG_HAND_OCCLUSION": "1",
        # Retração de tinta apagada: sim-hand tem 10/17 traços
        # apagados no GT — sem ela ficam confirmados para sempre.
        "DRAW_INK_REMOVAL": "1",
        # Portões de prudência no registro: confirmar nada novo
        # durante deslocamento de quad, e o juiz de conteúdo
        # segura o canvas quando a pose não assentou. Não curam
        # o fantasma rotacionado do sim-hand (medido: fp igual
        # com e sem) — ficam porque o custo é zero e o risco
        # que cobrem é real.
        "DRAW_INK_MOTION_GATE": "1",
        "DRAW_REG_CONTENT_GUARD": "1",
        # Dois papéis na cena (mesa real com caderno): corta o
        # blob fundido no entalhe da junção e veste a âncora só
        # no maior. Medido no gen-002: união → folha só.
        "DRAW_SHEET_MERGE_SPLIT": "1",
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
    "DRAW_INK_REMOVAL",
    "DRAW_INK_MOTION_GATE",
    "DRAW_REG_CONTENT_GUARD",
    "DRAW_SHEET_MERGE_SPLIT",
    "DRAW_SHEET_LK_PROPAGATE",
    "DRAW_ANCHOR_FOLLOW",
)


# ---------- geração autônoma de cenários ----------
#
# O loop de feedback fecha aqui: em vez de só re-jogar os captures
# fixos, --generate N amostra cenários da distribuição realista
# (texto aleatório, mão, borracha, lift/return, props) e grava MP4
# + GT via sim_source — depois avalia pelo MESMO caminho (_run),
# sem branch de código diferente do caso curado.

_TEXTS = [
    "Draw v6",
    "reunião 15h",
    "lista mercado",
    "ideia nova",
    "ok google",
    "hello world",
    "desenho",
    "nota fiscal",
    "lembrar amanhã",
    "rascunho 42",
    "teste tinta",
    "quarta feira",
]

# Fotos reais de mesa do laboratório — quando existem, metade dos
# cenários gerados usa foto de fundo em vez do procedural. O glob é
# configurável (--desk-photos / DRAW_DESK_PHOTOS_GLOB): sem ele a
# distribuição metade-foto só existia na máquina do laboratório.
_DESK_PHOTOS_GLOB = os.environ.get(
    "DRAW_DESK_PHOTOS_GLOB",
    "/home/lucas_coimbra/draw-lab/fp-fresh-*/fresh_*.jpg",
)

# Folha que o sim_source produz (size_idx=0 fixo em
# SyntheticCapture.__init__) — o retângulo de borracha é
# amostrado dentro dela.
_GEN_PAGE_W, _GEN_PAGE_H = 360, 510


def _sample_scenario_args(rng, desk_photos):
    """Um cenário da distribuição realista -> argv do sim_source."""
    args = [
        "--text",
        str(rng.choice(_TEXTS)),
        "--seed",
        str(int(rng.integers(1, 10_000))),
        "--fps",
        "15",
        "--warmup-seconds",
        "1",
        "--settle-seconds",
        str(float(rng.choice([5, 8, 10]))),
        "--hold-seconds",
        str(float(rng.choice([2, 4]))),
    ]
    if desk_photos and rng.random() < 0.5:
        args += ["--desk-photo", str(rng.choice(desk_photos))]
    # Mão sem instrumento: rest/sweep/grab — gesto de humano, não
    # só a mão colada na caneta.
    if rng.random() < 0.4:
        args += ["--hand", "--rest-hand-seconds", "2"]
        if rng.random() < 0.5:
            args += ["--sweep-seconds", "2"]
        if rng.random() < 0.3:
            args += ["--grab-hand"]
    # Borracha + redesenho em ~1/3 dos cenários.
    if rng.random() < 0.33:
        # A folha gerada tem 360x510 px (o sim_source fixa
        # size_idx=0): a posição é amostrada primeiro e o tamanho
        # cabe no que sobra — x e w independentes passavam da
        # borda e o _resolve_erase_rect rejeitava o cenário.
        x = int(rng.integers(40, 200))
        y = int(rng.integers(30, 80))
        w = min(int(rng.integers(120, 260)), _GEN_PAGE_W - x)
        h = min(int(rng.integers(30, 70)), _GEN_PAGE_H - y)
        args += [
            "--erase-rect",
            f"{x},{y},{w},{h}",
            "--redraw-text",
            str(rng.choice(_TEXTS)),
        ]
    kind = rng.random()
    if kind < 0.15:
        # Move puro no fim: desliza/gira a folha sem tirar de cena.
        dx, dy = int(rng.integers(-180, 180)), int(rng.integers(-120, 120))
        deg = int(rng.integers(-15, 15))
        args += [f"--move={dx},{dy},{deg}"]
    elif kind < 0.45:
        # Lift & return: a metade volta na mesma pose, a outra
        # metade volta rodada/deslocada — o gesto real de pegar,
        # virar e repousar. Valores negativos via `=` ou o argparse
        # lê como opção.
        dx, dy = int(rng.integers(-450, 450)), int(rng.integers(-350, 350))
        if dx == 0 and dy == 0:
            dx = 300
        args += [f"--lift={dx},{dy}"]
        args += ["--absent-seconds", str(float(rng.choice([1.5, 2.5, 4])))]
        if rng.random() < 0.15:
            args += ["--stay-out"]
        elif rng.random() < 0.55:
            rdx = int(rng.integers(-120, 120))
            rdy = int(rng.integers(-120, 120))
            rdeg = int(rng.integers(-25, 25))
            args += [f"--return-pose={rdx},{rdy},{rdeg}"]
    # Props e pauta: objetos na folha que não são tinta.
    if rng.random() < 0.15:
        args += ["--resting-pen"]
    if rng.random() < 0.15:
        args += ["--paper-style", "ruled"]
    return args


def _capture_artifacts(target):
    """Os quatro arquivos que a exportação do sim_source grava."""
    target = Path(target)
    return (
        target,
        target.with_suffix(".gt.npz"),
        target.with_name(target.stem + "_ink_full.npy"),
        target.with_suffix(".json"),
    )


def _capture_bundle_complete(target):
    """O pacote de exportação inteiro existe e o GT abre.

    O sim_source grava MP4 -> GT -> ink_full -> manifesto, nessa
    ordem: morto no meio, deixa MP4 sem bundle e o _run morre no
    np.load. Um pacote válido tem os quatro artefatos e um GT
    legível.
    """
    artifacts = _capture_artifacts(target)
    if not all(artifact.exists() for artifact in artifacts):
        return False
    try:
        with np.load(artifacts[1], allow_pickle=False) as bundle:
            bundle.files
    except Exception:
        return False
    return True


def _drop_capture_artifacts(target):
    for artifact in _capture_artifacts(target):
        artifact.unlink(missing_ok=True)


def _photo_identity(argv):
    """Identidade das fotos citadas no argv: caminho, tamanho, mtime."""
    photos = {}
    for index, arg in enumerate(argv):
        if arg == "--desk-photo" and index + 1 < len(argv):
            path = Path(argv[index + 1])
            try:
                stat = path.stat()
                photos[str(path)] = [stat.st_size, stat.st_mtime_ns]
            except OSError:
                photos[str(path)] = None
    return photos


_GENERATION_MANIFEST = "battery-manifest.json"


def _generate_captures(count, seed, out_dir, desk_photos_glob=None):
    """Grava `count` captures novos e devolve o dict BATTERY-style."""
    import glob
    import subprocess

    desk_photos = sorted(
        glob.glob(_DESK_PHOTOS_GLOB if desk_photos_glob is None else desk_photos_glob)
    )
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / _GENERATION_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    scenarios = manifest.setdefault("scenarios", {})
    generated = {}
    failed = []
    sim_source = Path(__file__).resolve().parent / "sim_source.py"
    for i in range(count):
        name = f"gen-{i:03d}"
        target = out_dir / f"{name}.mp4"
        argv = _sample_scenario_args(rng, desk_photos)
        entry = scenarios.get(name)
        cache_ok = (
            entry is not None
            and entry.get("argv") == argv
            and entry.get("photos") == _photo_identity(argv)
            and _capture_bundle_complete(target)
        )
        if target.exists() and not cache_ok:
            # Pacote incompleto (geração morta no meio) ou cache sem
            # identidade conferida (manifesto ausente, argv ou foto
            # diferentes): reaproveitar cega a origem dos pixels
            # avaliados.
            _drop_capture_artifacts(target)
        if target.exists():
            # Cache validado: mesmos argumentos amostrados, mesmas
            # fotos, pacote completo — reproduz o cenário.
            generated[name] = {}
            print(f"== {name}: reutilizado", flush=True)
            continue
        cmd = [
            sys.executable,
            str(sim_source),
            "--out",
            str(target),
            *argv,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0 or not _capture_bundle_complete(target):
            _drop_capture_artifacts(target)
            failed.append(name)
            print(f"!! {name}: geração falhou ({proc.returncode}) {proc.stderr[-300:]}")
            continue
        scenarios[name] = {
            "argv": argv,
            "photos": _photo_identity(argv),
        }
        generated[name] = {}
        print(f"++ {name}: {' '.join(argv)}", flush=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if failed:
        # Cenário pedido que não existe não pode virar relatório
        # verde com zero avaliações: --generate 1 --skip-curated
        # saía 0 com results vazio.
        raise SystemExit(
            f"geração falhou para {len(failed)}/{count} cenários: {failed}"
        )
    return generated


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
            identity_events.extend(list(all_events)[events_seen[0] :])
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
        # mesmo veredito, sem rodar a captura de novo. As métricas
        # por checkpoint espelham as de audit(): resíduo na região
        # apagada e recall/provisória do redesenho, que é onde o
        # contrato erase→redraw se lê.
        events = json.loads(str(bundle["events"]))
        canvas_hw = tuple(pipeline["canvas_shape"])
        stroke_phase = bundle["stroke_phase"]
        points, offsets = bundle["points"], bundle["offsets"]
        shape = bundle["page_ink_full"].shape

        def phase_stroke_masks(phase_index):
            masks = []
            for index in np.nonzero(stroke_phase == phase_index)[0]:
                geo = points[offsets[index] : offsets[index + 1]]
                masks.append(
                    eval_scenario._stroke_mask(shape, geo, bundle["thickness"][index])
                )
            return masks

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
                bundle["page_to_scene_final"] if moved else bundle["page_to_scene"]
            )
            expected = eval_scenario._page_expected(bundle, applied_upto, erased)
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
            truth = (
                eval_scenario._project(expected, corners, page_to_scene, canvas_hw) > 0
            )
            entry.update(pixel_metrics(confirmed, truth))
            entry["confirmed_px"] = int(confirmed.sum())
            entry["provisional_px"] = int(provisional.sum())
            x, y, w, h = (int(v) for v in bundle["erase_rect"])
            if w and h:
                rect_mask = np.zeros(shape, np.uint8)
                rect_mask[y : y + h, x : x + w] = 255
                rect_canvas = (
                    eval_scenario._project(rect_mask, corners, page_to_scene, canvas_hw)
                    > 0
                )
                entry["confirmed_in_erased_px"] = int((confirmed & rect_canvas).sum())
            redraw_masks = phase_stroke_masks(1)
            if redraw_masks:
                redraw = np.zeros(shape, np.uint8)
                for mask in redraw_masks:
                    redraw |= mask
                redraw_canvas = (
                    eval_scenario._project(redraw, corners, page_to_scene, canvas_hw)
                    > 0
                )
                denom = max(1, int(redraw_canvas.sum()))
                entry["redraw_confirmed_recall"] = (
                    float((confirmed & redraw_canvas).sum()) / denom
                )
                entry["redraw_provisional_px"] = int(
                    (provisional & redraw_canvas).sum()
                )
            checkpoints.append(entry)

        final = eval_scenario._final_checkpoint(checkpoints, events)
        result["checkpoints"] = checkpoints
        result["verdict"] = {
            "status": final.get("status"),
            "final_iou": final.get("iou"),
            "final_recall": final.get("recall"),
            "final_confirmed_px": final.get("confirmed_px"),
            "redraw_confirmed_recall": final.get("redraw_confirmed_recall"),
            "redraw_provisional_px": final.get("redraw_provisional_px"),
            "residual_confirmed_in_erased_px": final.get("confirmed_in_erased_px"),
        }
    else:
        metrics = evaluate_canvas(
            pipeline["mask"],
            gt,
            canvas_space=("page" if pipeline["mask"] is not None else "scene"),
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
    parser.add_argument(
        "--generate",
        type=int,
        metavar="N",
        help=(
            "gera N cenários novos (sim_source, distribuição realista) "
            "em --captures/battery-gen-<seed> e avalia junto"
        ),
    )
    parser.add_argument(
        "--gen-seed",
        type=int,
        default=103,
        help="semente da amostragem de cenários gerados",
    )
    parser.add_argument(
        "--skip-curated",
        action="store_true",
        help="roda só os cenários gerados (útil com --generate)",
    )
    parser.add_argument(
        "--desk-photos",
        metavar="GLOB",
        help=(
            "glob das fotos de mesa para os cenários gerados "
            "(default: DRAW_DESK_PHOTOS_GLOB ou o diretório do "
            "laboratório) — torna a distribuição metade-foto "
            "portátil entre máquinas"
        ),
    )
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

    battery = dict(BATTERY)
    captures_dir = args.captures
    if args.generate:
        gen_dir = args.captures / f"battery-gen-{args.gen_seed}"
        generated = _generate_captures(
            args.generate, args.gen_seed, gen_dir, args.desk_photos
        )
        for name in generated:
            battery[f"gen/{name}"] = {
                "name": name,
                "captures": gen_dir,
            }
    if args.skip_curated:
        battery = {name: spec for name, spec in battery.items() if "/" in name}

    wanted = (
        set(battery)
        if not args.only
        else {name.strip() for name in args.only.split(",") if name.strip()}
    )
    missing = wanted - set(battery)
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
        for name in battery:
            if name not in wanted:
                continue
            spec = dict(battery[name])
            run_name = spec.pop("name", name)
            run_captures = Path(spec.pop("captures", captures_dir))
            scenario_env = spec.pop("env", {})
            saved_scenario = {key: os.environ.get(key) for key in scenario_env}
            os.environ.update(scenario_env)
            try:
                print(f"== {name} ({args.profile})", flush=True)
                results.append(_run(run_name, spec, run_captures, models))
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
