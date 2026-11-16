"""Falsificador de cenários: busca adversarial sobre o gêmeo digital.

Em vez de cenários escritos à mão, amostra o espaço de parâmetros do
``SyntheticCapture`` — fases, mão, ISP, folha, luz — exporta o MP4,
roda a bancada real (``eval_scenario.audit``, mesmo laço do app) e
pontua a falha. Rodadas elitistas re-amostram mutações dos piores
casos: a busca converge para as combinações físicas que mais quebram
o laço, como um humano tentando derrubar o sistema na mesa — e deixa
vídeo, GT, máscaras por fase e frames da cena no momento da falha.

O placar compõe as falhas documentadas na bancada: IoU final do
canvas, ausência de canvas, resíduo confirmado em região apagada,
recall do redesenho e o pior checkpoint de fase. Config e seed de
cada tentativa vão ao relatório — todo cenário é reproduzível com
``tools/sim_source.py``.

Exemplo::

    python tools/falsify.py --trials 12 --rounds 2 --out captures/falsify
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import gen_synthetic as gs  # noqa: E402
from tools.eval_scenario import audit  # noqa: E402
from tools.sim_source import SyntheticCapture, export_video  # noqa: E402

TEXT_POOL = [
    "ab",
    "draw",
    "Draw ok",
    "ação 123",
    "lógica",
    "ui",
    "R 12,50",
    "Olá, Draw!",
    "viga 3m",
]

# ---------------------------------------------------------------------
# espaço de busca
# ---------------------------------------------------------------------


def _uniform(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def sample_config(
    rng: np.random.Generator, index: int, texts: list[str] | None = None
) -> dict:
    """Sorteia uma configuração de cenário no espaço do falsificador.

    Cada fase/efeito entra com probabilidade independente — o espaço
    inclui desde o warmup trivial (linha de base) até a cena hostil
    completa (mão + folha curvada + ISP + movimento). Durações são
    mantidas curtas para o custo de bancada caber em lote.
    """
    params: dict = {
        "fps": float(rng.choice([8, 10, 12, 15, 20, 30])),
        "stroke_seconds": _uniform(rng, 0.08, 0.45),
        "warmup_seconds": _uniform(rng, 0.4, 1.5),
        "settle_seconds": _uniform(rng, 1.5, 6.0),
        "height": _uniform(rng, 14, 34),
        "background": str(rng.choice(gs.BACKGROUND_SOURCES)),
        "hand": bool(rng.random() < 0.6),
    }
    if rng.random() < 0.55:
        if rng.random() < 0.5:
            params["erase_stroke"] = -1
        else:
            # A folha do simulador é sempre 360×510 (size_idx=0).
            params["erase_rect"] = (
                int(rng.integers(20, 240)),
                int(rng.integers(30, 380)),
                int(rng.integers(40, 120)),
                int(rng.integers(30, 80)),
            )
        params["erase_seconds"] = _uniform(rng, 0.5, 2.0)
        if rng.random() < 0.7:
            params["redraw_text"] = str(rng.choice(TEXT_POOL[:4]))
    params["hold_seconds"] = _uniform(rng, 0.8, 5.0)
    has_motion = False
    if rng.random() < 0.45:
        params["move"] = (
            _uniform(rng, -180, 180),
            _uniform(rng, -180, 180),
            _uniform(rng, -40, 40),
        )
        params["move_seconds"] = _uniform(rng, 0.4, 1.5)
        has_motion = True
    if rng.random() < 0.3:
        params["lift"] = (_uniform(rng, -400, 400), _uniform(rng, -400, 400))
        params["absent_seconds"] = _uniform(rng, 0.8, 3.0)
        params["lift_seconds"] = _uniform(rng, 0.3, 1.2)
        params["return_seconds"] = _uniform(rng, 0.3, 1.2)
        if rng.random() < 0.7:
            params["return_pose"] = (
                _uniform(rng, -80, 80),
                _uniform(rng, -80, 80),
                _uniform(rng, -45, 45),
            )
        has_motion = True
    if has_motion and rng.random() < 0.5:
        params["grab_hand"] = True
    if rng.random() < 0.4:
        params["rest_hand_seconds"] = _uniform(rng, 0.5, 3.0)
    if rng.random() < 0.35:
        params["sweep_seconds"] = _uniform(rng, 0.5, 2.5)
    if rng.random() < 0.4:
        params["hand_scale"] = _uniform(rng, 0.6, 2.2)
    # Gêmeo óptico / físico: cada efeito entra de forma independente.
    if rng.random() < 0.5:
        params["auto_exposure"] = True
    if rng.random() < 0.5:
        params["auto_white_balance"] = True
    if rng.random() < 0.5:
        params["motion_blur"] = True
    if rng.random() < 0.4:
        params["jpeg_quality"] = int(rng.integers(40, 96))
    if rng.random() < 0.45:
        params["hand_shadow"] = True
    if rng.random() < 0.4:
        params["ink_sheen"] = _uniform(rng, 0.2, 0.9)
    if rng.random() < 0.35:
        params["page_curl"] = _uniform(rng, 0.1, 0.7)
    pool = texts or TEXT_POOL
    return {
        "text": str(rng.choice(pool)),
        "seed": int(rng.integers(0, 2**31 - 1)),
        "params": params,
    }


# Limites do contrato do SyntheticCapture — mutação que escapa deles
# não é adversário, é configuração quebrada (score=-1 por erro, não
# por falha do pipeline): a busca perderia gerações inteiras.
_BOUNDS = {
    "fps": (4.0, 60.0),
    "stroke_seconds": (0.05, 0.8),
    "warmup_seconds": (0.0, 4.0),
    "settle_seconds": (0.2, 10.0),
    "hold_seconds": (0.0, 8.0),
    "erase_seconds": (0.1, 5.0),
    "move_seconds": (0.1, 4.0),
    "absent_seconds": (0.1, 8.0),
    "lift_seconds": (0.1, 4.0),
    "return_seconds": (0.1, 4.0),
    "rest_hand_seconds": (0.0, 6.0),
    "sweep_seconds": (0.0, 6.0),
    "height": (6.0, 60.0),
    "hand_scale": (0.3, 4.0),
    "ink_sheen": (0.0, 1.0),
    "page_curl": (0.0, 1.0),
    "jpeg_quality": (10.0, 100.0),
}


def _clamp_numeric(key, value):
    """Corta um número mutado ao intervalo válido do seu parâmetro."""
    if key not in _BOUNDS:
        return value
    low, high = _BOUNDS[key]
    return type(value)(max(low, min(high, float(value))))


def _sanitize_tuple(key, value):
    """Normaliza tuplas mutadas; devolve None quando o gesto morreu.

    ``lift`` fora do domínio é irrecuperável (direção ~0) — melhor
    manter o valor original do que produzir configuração inválida.
    """
    if key == "erase_rect":
        x, y, w, h = (int(round(float(v))) for v in value)
        x, y = max(0, min(359, x)), max(0, min(509, y))
        return (x, y, max(1, min(360 - x, w)), max(1, min(510 - y, h)))
    if key == "lift":
        vec = np.asarray(value, np.float64)
        norm = float(np.linalg.norm(vec))
        return None if norm < 1.0 else tuple(float(v) for v in vec)
    # move/return_pose: (dx, dy, graus) — só precisam ser finitos;
    # limita a deriva a gestos fisicamente reconhecíveis.
    return tuple(
        float(max(-600.0, min(600.0, v)))
        if index < 2
        else float(max(-180.0, min(180.0, v)))
        for index, v in enumerate(value)
    )


def mutate_config(cfg: dict, rng: np.random.Generator) -> dict:
    """Perturba um caso falho: números ±30%, flags ligam/desligam,
    seed re-sorteada — a vizinhança do erro é onde moram as falhas
    correlatas que uma grade solta não visita."""
    out = {
        "text": cfg["text"],
        "seed": cfg["seed"] if rng.random() < 0.7 else int(rng.integers(2**31 - 1)),
        "params": dict(cfg["params"]),
    }
    params = out["params"]
    for key, value in list(params.items()):
        if key == "erase_stroke":
            # Índice categórico: outro índice pode não existir na
            # lista de traços — não é uma vizinhança útil.
            continue
        if isinstance(value, bool):
            if rng.random() < 0.25:
                params[key] = not value
        elif isinstance(value, (int, float)) and rng.random() < 0.4:
            delta = 0.3 * abs(float(value)) + 0.05
            params[key] = _clamp_numeric(
                key, type(value)(float(value) + rng.uniform(-delta, delta))
            )
        elif isinstance(value, tuple) and rng.random() < 0.4:
            mutated = tuple(
                type(v)(float(v) * (1.0 + rng.uniform(-0.3, 0.3)))
                for v in value
            )
            params[key] = _sanitize_tuple(key, mutated) or value
    # Chance de acender um efeito adormecido: a mutação também explora.
    for key, low, high in (
        ("ink_sheen", 0.2, 0.9),
        ("page_curl", 0.1, 0.7),
    ):
        if key not in params and rng.random() < 0.2:
            params[key] = _uniform(rng, low, high)
    for flag in ("auto_exposure", "auto_white_balance", "motion_blur", "hand_shadow"):
        if flag not in params and rng.random() < 0.15:
            params[flag] = True
    return out


# ---------------------------------------------------------------------
# placar
# ---------------------------------------------------------------------


def score_report(report: dict) -> tuple[float, dict]:
    """Pontua a falha: maior = pior. Componentes registrados para o
    diagnóstico — o falsificador precisa dizer COMO quebrou, não só
    que quebrou."""
    verdict = report.get("verdict", {})
    iou = verdict.get("final_iou")
    worst_phase = min(
        (c["iou"] for c in report.get("checkpoints", []) if "iou" in c),
        default=None,
    )
    components = {
        "iou_loss": 1.0 - float(iou) if iou is not None else 1.0,
        "no_canvas": 1.0 if iou is None else 0.0,
        "erase_residual": min(
            1.0, (verdict.get("residual_confirmed_in_erased_px") or 0) / 4000.0
        ),
        "redraw_loss": (
            1.0 - float(verdict["redraw_confirmed_recall"])
            if verdict.get("redraw_confirmed_recall") is not None
            else 0.0
        ),
        # Sem nenhum checkpoint com IoU, nunca houve canvas: é o
        # pior cenário possível, não ausência de sinal.
        "worst_phase_loss": 1.0 - float(worst_phase)
        if worst_phase is not None
        else 1.0,
    }
    score = (
        components["iou_loss"]
        + components["no_canvas"]
        + 0.5 * components["erase_residual"]
        + 0.5 * components["redraw_loss"]
        + 0.5 * components["worst_phase_loss"]
    )
    return score, components


# ---------------------------------------------------------------------
# julgamento
# ---------------------------------------------------------------------


def run_trial(
    cfg: dict,
    workdir: Path,
    tag: str,
    models: dict,
    keep_artifacts: bool,
) -> dict:
    """Gera o vídeo do cenário, roda a bancada real e pontua."""
    video = workdir / f"{tag}.mp4"
    started = time.monotonic()
    try:
        capture = SyntheticCapture(cfg["text"], seed=cfg["seed"], **cfg["params"])
        manifest = export_video(capture, video)
        masks_dir = workdir / f"{tag}_masks" if keep_artifacts else None
        report = audit(
            video, video.with_suffix(".gt.npz"), models=models, save_masks=masks_dir
        )
        score, components = score_report(report)
        if keep_artifacts:
            _save_failure_frames(video, report, workdir / f"{tag}_frames")
        return {
            "tag": tag,
            "score": round(score, 4),
            "components": components,
            "verdict": report["verdict"],
            "config": cfg,
            "frames": manifest["frames"],
            "elapsed_s": round(time.monotonic() - started, 2),
            "artifacts": str(video) if keep_artifacts else None,
        }
    finally:
        # Sem keep_artifacts, MP4/GT/manifesto não têm dono: a falha
        # na auditoria não pode deixar restos para o próximo trial.
        if not keep_artifacts and video.exists():
            for path in (
                video,
                video.with_suffix(".gt.npz"),
                video.with_suffix(".json"),
                video.with_name(video.stem + "_ink_full.npy"),
            ):
                path.unlink(missing_ok=True)


def _save_failure_frames(video: Path, report: dict, out_dir: Path) -> None:
    """Frames da cena no instante de cada checkpoint — a prancha visual
    do erro: a máscara diz o que o modelo viu, o frame diz por quê.

    Leitura sequencial: CAP_PROP_POS_FRAMES sobre mp4v não é seek
    exato — o frame de evidência precisa ser o do checkpoint, não o
    vizinho mais próximo do keyframe.
    """
    checkpoints = {c["frame"]: c for c in report.get("checkpoints", []) if "frame" in c}
    if not checkpoints:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = cv2.VideoCapture(str(video))
    try:
        index = 0
        while checkpoints:
            ok, frame = reader.read()
            if not ok:
                break
            checkpoint = checkpoints.pop(index, None)
            if checkpoint is not None:
                cv2.imwrite(
                    str(
                        out_dir
                        / f"scene_{checkpoint['phase']}_{checkpoint['frame']}.png"
                    ),
                    frame,
                )
            index += 1
    finally:
        reader.release()


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=12, help="população inicial")
    parser.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="rodadas evolutivas: cada elite gera mutações re-avaliadas",
    )
    parser.add_argument(
        "--elite", type=int, default=4, help="piores casos mutados por rodada"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--keep",
        type=int,
        default=5,
        help="quantos piores cenários mantêm vídeo/GT/máscaras/frames",
    )
    parser.add_argument(
        "--texts",
        type=str,
        help="pool de textos separado por | (default: pool interno)",
    )
    parser.add_argument("--neural-model")
    parser.add_argument("--page-clean-model")
    parser.add_argument("--hand-model")
    args = parser.parse_args(argv)
    if args.trials < 1 or args.rounds < 0 or args.elite < 1 or args.keep < 0:
        parser.exit(2, "erro: trials/elite/keep/rounds fora da faixa\n")
    texts = args.texts.split("|") if args.texts else TEXT_POOL
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    workdir = args.out / "trials"
    workdir.mkdir(exist_ok=True)
    models = {
        "neural": args.neural_model,
        "page_clean": args.page_clean_model,
        "hand": args.hand_model,
    }

    results = []
    batch = [sample_config(rng, i, texts) for i in range(args.trials)]
    tag_seq = 0
    for round_index in range(args.rounds + 1):
        evaluated = []
        for cfg in batch:
            tag = f"r{round_index}_t{tag_seq:03d}"
            tag_seq += 1
            try:
                result = run_trial(cfg, workdir, tag, models, keep_artifacts=False)
            except (ValueError, RuntimeError, OSError, cv2.error) as error:
                result = {
                    "tag": tag,
                    "score": -1.0,
                    "error": str(error),
                    "config": cfg,
                }
            evaluated.append(result)
            results.append(result)
            score_txt = result.get("score")
            print(
                f"[{tag}] score={score_txt} "
                f"iou={result.get('verdict', {}).get('final_iou')} "
                f"{cfg['text']!r} seed={cfg['seed']}",
                flush=True,
            )
        if round_index < args.rounds:
            ranked = sorted(
                (r for r in evaluated if r.get("score", 0) >= 0),
                key=lambda r: -r["score"],
            )
            elite = ranked[: args.elite]
            per_elite = max(2, args.trials // args.elite)
            batch = [
                mutate_config(cfg["config"], rng)
                for cfg in elite
                for _ in range(per_elite)
            ]

    valid = [r for r in results if r.get("score", 0) >= 0]
    valid.sort(key=lambda r: -r["score"])
    # Mantém artefatos só dos piores --keep: o relatório já carrega
    # config+seed de todos — qualquer cenário é regenerável.
    for result in valid[: args.keep]:
        tag = result["tag"]
        try:
            rerun = run_trial(
                result["config"], workdir, f"{tag}_keep", models, keep_artifacts=True
            )
            result["artifacts"] = rerun["artifacts"]
        except (ValueError, RuntimeError, OSError, cv2.error) as error:
            result["artifacts_error"] = str(error)
    report = {
        "seed": args.seed,
        "trials": args.trials,
        "rounds": args.rounds,
        "elite": args.elite,
        "evaluated": len(results),
        "worst": valid[: args.keep],
        "all": sorted(results, key=lambda r: -r.get("score", -1)),
    }
    out_path = args.out / "falsify_report.json"
    out_path.write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nrelatório: {out_path}")
    print(f"piores {min(args.keep, len(valid))} de {len(valid)} cenários avaliados")
    for result in valid[: args.keep]:
        v = result.get("verdict", {})
        print(
            f"  {result['tag']}: score={result['score']} "
            f"iou={v.get('final_iou')} status={v.get('status')} "
            f"cfg={json.dumps(_jsonable(result['config']['params']), ensure_ascii=False)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
