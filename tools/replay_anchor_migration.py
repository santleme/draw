"""Replay do pouso sustentado: migracao de ancora intra-geracao.

Prova de replay da flag DRAW_ANCHOR_MIGRATION (draw/ink_layers.py),
no desenho da prova pre-registrada do estudo do pouso (decisao A do
Lucas, 2026-11-10; §8 de
~/workspace/draw-lab/live-audit-2026-11-10/12-estudo-pouso.md).

Fonte: SINTETICA. A pagina e a de `tools/replay_reregister.py`
(`synthetic_pages`, `render_pass`, `base_quad`): o mapa de
probabilidade vive na grade da pagina, a pagina esta num quad
VERDADEIRO na camera e o app retifica com o quad do detector. A
trabalhadora real (`_PageCleanWorker`) roda com um page_clean falso
que devolve o proprio mapa submetido (sem ONNX). Sem os dados reais
do laboratorio, o veredito aqui e de replay sintetico, nao de video.

Series (todas numa geracao so, presenca continua):

- eventos: folha parada; o detector da flips de 1 passada
  (20-60 px por canto) no meio do tremor de 1 px. Esperado: 0 migracoes.
- pouso_40 / pouso_120: desenho, assenta, a folha anda 40/120 px
  da camera em 4 passadas e assenta. Esperado: 1 migracao.
- isca_modo: folha parada; o detector entra num modo espurio
  sustentado (quad deslocado ~30 px) por 6 passadas e volta.
- pousos_3: 3 pousos seguidos com tinta nova entre eles.
- quieto: folha parada, so tremor, nada novo.
- rot_45 / rot_90 (sem portao): pouso com rotacao; o detector
  devolve os cantos em ordem posicional (TL, TR, BR, BL na imagem),
  como o rastreador real.

Bracos: base (flag desligada), migracao (DRAW_ANCHOR_MIGRATION=1) e
migracao + disciplina (+ DRAW_REG_ANCHOR_MEDIAN3=1 e
DRAW_REG_DEADBAND=1). DRAW_ECC_REFINE fica desligado: o julgamento e
sem re-alinhamento, contra o oraculo de substituicao — o canvas que
uma geracao limpa daria com a pagina final (todo o conteudo
desenhado) parada e retificada pelo quad verdadeiro.

Metricas (tolerancia de `TOL_PX` px, a reamostragem sub-pixel das
mascaras migradas):

- duplicacao: px do canvas final fora da faixa do oraculo, sobre os
  px do canvas pre-pouso;
- iou: IoU tolerante (|A perto de B| + |B perto de A|) / (|A| + |B|)
  do canvas final contra o oraculo; iou_strict e o IoU pixel a pixel;
- perda: px do canvas pre-pouso sem correspondente no canvas final;
  retencao = 1 - perda;
- churn (quieto): |canvas final XOR canvas apos assentar| / |final|.

Uso: .venv/bin/python tools/replay_anchor_migration.py [--json out.json]
"""

import argparse
import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from draw import ink_layers as ink  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "replay_reregister", REPO / "tools" / "replay_reregister.py"
)
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)

SHAPE = (240, 170)
BASE = rr.base_quad(SHAPE)
TOL_PX = 1
NOISE_PX = 1.0
FLAGS = (
    "DRAW_ANCHOR_MIGRATION",
    "DRAW_REG_ANCHOR_MEDIAN3",
    "DRAW_REG_DEADBAND",
    "DRAW_ECC_REFINE",
)
ARMS = {
    "base": {},
    "migracao": {"DRAW_ANCHOR_MIGRATION": "1"},
    "migracao_disciplina": {
        "DRAW_ANCHOR_MIGRATION": "1",
        "DRAW_REG_ANCHOR_MEDIAN3": "1",
        "DRAW_REG_DEADBAND": "1",
    },
}
# Diagnostico, sem portao e fora do produto: mesma deteccao do pouso,
# mas a migracao so troca a ancora e deixa o estado na grade em que
# esta (matriz identidade no lugar da inversa do registro). Mede a
# hipotese "a tinta anda com a folha" contra a hipotese "a tinta fica
# no mesmo ponto da camera" que a inversa do registro assume.
DIAGNOSTIC_ARMS = {"diag_reancora_sem_mover": {"DRAW_ANCHOR_MIGRATION": "1"}}
GATES = {
    "dup_max": 0.05,
    "iou_min": 0.90,
    "loss_per_migration_max": 0.03,
    "retention_3_min": 0.91,
    "spurious_iou_min": 0.85,
    "quiet_churn_regression_max": 0.02,
}


class _ProbIsWarp:
    threshold = 0.7

    def ink_prob(self, warped):
        return warped


@contextlib.contextmanager
def _env(values):
    saved = {name: os.environ.get(name) for name in FLAGS}
    try:
        for name in FLAGS:
            os.environ.pop(name, None)
        os.environ.update(values)
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def positional_order(quad):
    """Cantos na ordem posicional da imagem (TL, TR, BR, BL)."""
    quad = np.asarray(quad, np.float32)
    s = quad.sum(axis=1)
    d = quad[:, 1] - quad[:, 0]
    return np.float32(
        [quad[np.argmin(s)], quad[np.argmin(d)], quad[np.argmax(s)], quad[np.argmax(d)]]
    )


# Um passo de cena por passada: folga sobre qualquer cadencia da
# trabalhadora (250 ms ativa, 1 s de sonda), sem relogio de parede.
STEP_SECONDS = 10.0


class Episode:
    """Uma geracao da trabalhadora real, passada a passada.

    A trabalhadora e a `_PageCleanWorker` de producao no caminho
    sincrono compartilhado com tools/replay_reregister.py
    (`make_ink_worker` + `pump()` em relogio de cena): sem thread,
    sem espera por relogio de parede e sem encurtar a cadencia.
    O protocolo deste estudo entrega o quad do detector direto a
    trabalhadora (sem rastreador nem mediana do laco), como no
    pre-registro.
    """

    def __init__(self):
        self.clock = rr.SceneClock()
        self.worker = rr.make_ink_worker(_ProbIsWarp(), self.clock)
        self.generation = self.worker.new_generation()

    def step(self, page, true_quad, detected):
        prob = rr.render_pass(page, true_quad, detected)
        self.clock.now += STEP_SECONDS
        self.worker.submit(prob, self.generation, detected)
        if not self.worker.pump():
            raise RuntimeError("a passada nao executou")
        if self.worker.error is not None:
            raise self.worker.error

    def canvas(self):
        got = self.worker.poll_display(self.generation, 0)
        return np.zeros(SHAPE, np.uint8) if got is None else got[1]

    def close(self):
        return self.worker.telemetry_snapshot()


def _near(mask, radius):
    if radius <= 0:
        return mask > 0
    kernel = np.ones((2 * radius + 1,) * 2, np.uint8)
    return cv2.dilate((mask > 0).astype(np.uint8), kernel) > 0


def tolerant_iou(a, b, radius=TOL_PX):
    a, b = a > 0, b > 0
    total = np.count_nonzero(a) + np.count_nonzero(b)
    if total == 0:
        return 1.0
    hit = np.count_nonzero(a & _near(b, radius)) + np.count_nonzero(
        b & _near(a, radius)
    )
    return hit / total


def strict_iou(a, b):
    a, b = a > 0, b > 0
    union = np.count_nonzero(a | b)
    return 1.0 if union == 0 else np.count_nonzero(a & b) / union


def judge(final, oracle, pre):
    pre_px = max(1, int(np.count_nonzero(pre)))
    dup = np.count_nonzero((final > 0) & ~_near(oracle, TOL_PX)) / pre_px
    loss = np.count_nonzero((pre > 0) & ~_near(final, TOL_PX)) / pre_px
    return {
        "duplication": round(float(dup), 4),
        "iou": round(float(tolerant_iou(final, oracle)), 4),
        "iou_strict": round(float(strict_iou(final, oracle)), 4),
        "loss": round(float(loss), 4),
        "pre_px": pre_px,
        "final_px": int(np.count_nonzero(final)),
        "oracle_px": int(np.count_nonzero(oracle)),
    }


def oracle_canvas(page, passes=8):
    acc = ink._LayeredInkAccumulator(0.7)
    result = None
    for k in range(passes):
        result = acc.add(page[k % len(page)])
    return result["confirmed"]


def _pages(seed, stages):
    drawing, _settled = rr.synthetic_pages(SHAPE, stages=stages, seed=seed)
    return drawing


def _draw(ep, stages, quad, rng, passes_per_stage=4, detect=None):
    detect = detect or (lambda q: rr._noisy(q, rng, NOISE_PX))
    for stage in stages:
        for k in range(passes_per_stage):
            ep.step(stage[k % 3], quad, detect(quad))


def _move(ep, page, start, shift, angle, rng, passes=4, positional=False):
    for i in range(1, passes + 1):
        t = i / passes
        ease = t * t * (3.0 - 2.0 * t)
        quad = rr.transform_quad(start, np.asarray(shift) * ease, angle * ease)
        detected = rr._noisy(quad, rng, NOISE_PX)
        if positional:
            detected = positional_order(detected)
        ep.step(page[i % 3], quad, detected)
    return rr.transform_quad(start, shift, angle)


def _settle(ep, page, quad, rng, passes, positional=False):
    for k in range(passes):
        detected = rr._noisy(quad, rng, NOISE_PX)
        if positional:
            detected = positional_order(detected)
        ep.step(page[k % 3], quad, detected)


def _finish(ep, oracle, pre, extra=None):
    final = ep.canvas()
    snapshot = ep.close()
    out = judge(final, oracle, pre)
    out["migrations"] = snapshot.get("anchor_migrations", 0)
    out["migration_events"] = snapshot.get("anchor_migration_events", [])
    out.update(extra or {})
    return out


def series_events(seed=11):
    stages = _pages(seed, 4)
    rng = np.random.default_rng(seed)
    flips = set(int(i) for i in rng.choice(np.arange(3, 40), size=8, replace=False))
    flips = {i for i in flips if i - 1 not in flips}
    count = [0]

    def detect(quad):
        i = count[0]
        count[0] += 1
        noisy = rr._noisy(quad, rng, NOISE_PX)
        if i in flips:
            noisy = noisy + rng.uniform(20, 60, size=(4, 2)).astype(np.float32) * (
                rng.choice([-1, 1], size=(4, 2))
            )
        return noisy.astype(np.float32)

    ep = Episode()
    _draw(ep, stages, BASE, rng, detect=detect)
    pre = ep.canvas()
    _draw(ep, [stages[-1]] * 6, BASE, rng, passes_per_stage=4, detect=detect)
    return _finish(
        ep, oracle_canvas(stages[-1]), pre, {"flips": len(flips), "expected": 0}
    )


def series_landing(shift_px, seed=21, angle=0.0, positional=False):
    stages = _pages(seed, 4)
    rng = np.random.default_rng(seed + int(shift_px) + int(angle))
    ep = Episode()
    _draw(ep, stages, BASE, rng)
    _settle(ep, stages[-1], BASE, rng, 4)
    pre = ep.canvas()
    target = _move(
        ep, stages[-1], BASE, (shift_px, 0.0), angle, rng, positional=positional
    )
    _settle(ep, stages[-1], target, rng, 14, positional=positional)
    return _finish(ep, oracle_canvas(stages[-1]), pre, {"expected": 1})


def series_spurious(seed=31, mode_passes=6):
    stages = _pages(seed, 4)
    rng = np.random.default_rng(seed)
    offset = np.float32([(30, 4), (34, -3), (28, 6), (31, 2)])
    ep = Episode()
    _draw(ep, stages, BASE, rng)
    _settle(ep, stages[-1], BASE, rng, 4)
    pre = ep.canvas()
    for k in range(mode_passes):
        ep.step(stages[-1][k % 3], BASE, rr._noisy(BASE, rng, NOISE_PX) + offset)
    _settle(ep, stages[-1], BASE, rng, 14)
    return _finish(ep, oracle_canvas(stages[-1]), pre, {"mode_passes": mode_passes})


def series_repeated(seed=41):
    stages = _pages(seed, 7)
    rng = np.random.default_rng(seed)
    ep = Episode()
    _draw(ep, stages[:4], BASE, rng)
    _settle(ep, stages[3], BASE, rng, 4)
    pre = ep.canvas()
    quad = BASE
    losses = []
    for shift, stage in zip(((40, 0), (0, 30), (-50, 10)), stages[4:]):
        before = ep.canvas()
        quad = _move(ep, stage, quad, shift, 0.0, rng)
        _settle(ep, stage, quad, rng, 4)
        after = ep.canvas()
        before_px = max(1, int(np.count_nonzero(before)))
        losses.append(
            round(
                float(np.count_nonzero((before > 0) & ~_near(after, TOL_PX)))
                / before_px,
                4,
            )
        )
        _draw(ep, [stage], quad, rng)
        _settle(ep, stage, quad, rng, 4)
    out = _finish(ep, oracle_canvas(stages[-1]), pre, {"expected": 3})
    out["loss_per_landing"] = losses
    out["retention_after_3"] = round(1.0 - out["loss"], 4)
    return out


def series_quiet(seed=51):
    stages = _pages(seed, 4)
    rng = np.random.default_rng(seed)
    ep = Episode()
    _draw(ep, stages, BASE, rng)
    _settle(ep, stages[-1], BASE, rng, 6)
    settled = ep.canvas()
    _settle(ep, stages[-1], BASE, rng, 30)
    final = ep.canvas()
    out = _finish(ep, oracle_canvas(stages[-1]), settled, {"expected": 0})
    final_px = max(1, int(np.count_nonzero(final)))
    out["churn"] = round(
        float(np.count_nonzero((final > 0) != (settled > 0))) / final_px, 4
    )
    return out


SERIES = {
    "eventos": series_events,
    "pouso_40": lambda: series_landing(40.0),
    "pouso_120": lambda: series_landing(120.0),
    "isca_modo": series_spurious,
    "pousos_3": series_repeated,
    "quieto": series_quiet,
    "rot_45": lambda: series_landing(40.0, angle=45.0, positional=True),
    "rot_90": lambda: series_landing(40.0, angle=90.0, positional=True),
}


def gate(results):
    """Portoes da prova, por braco de migracao."""
    verdict = {}
    base_quiet = results["base"]["quieto"]["churn"]
    for arm, rows in results.items():
        if arm == "base" or arm in DIAGNOSTIC_ARMS:
            continue
        checks = {}
        for name in ("pouso_40", "pouso_120"):
            row = rows[name]
            checks[f"{name}_duplication"] = row["duplication"] <= GATES["dup_max"]
            checks[f"{name}_iou"] = row["iou"] >= GATES["iou_min"]
            checks[f"{name}_loss"] = row["loss"] <= GATES["loss_per_migration_max"]
            checks[f"{name}_one_migration"] = row["migrations"] == 1
        rep = rows["pousos_3"]
        checks["pousos_3_duplication"] = rep["duplication"] <= GATES["dup_max"]
        checks["pousos_3_iou"] = rep["iou"] >= GATES["iou_min"]
        checks["pousos_3_loss_each"] = all(
            loss <= GATES["loss_per_migration_max"] for loss in rep["loss_per_landing"]
        )
        checks["pousos_3_retention"] = (
            rep["retention_after_3"] >= GATES["retention_3_min"]
        )
        checks["pousos_3_three_migrations"] = rep["migrations"] == 3
        checks["eventos_zero_migrations"] = rows["eventos"]["migrations"] == 0
        checks["isca_modo_iou"] = rows["isca_modo"]["iou"] >= GATES["spurious_iou_min"]
        checks["quieto_zero_migrations"] = rows["quieto"]["migrations"] == 0
        checks["quieto_churn"] = (
            rows["quieto"]["churn"] <= base_quiet + GATES["quiet_churn_regression_max"]
        )
        verdict[arm] = {"passed": all(checks.values()), "checks": checks}
    return verdict


@contextlib.contextmanager
def _migration_constants(threshold=None, passes=None):
    saved = (ink._ANCHOR_MIGRATION_DISPLACEMENT_PX, ink._ANCHOR_MIGRATION_PASSES)
    if threshold is not None:
        ink._ANCHOR_MIGRATION_DISPLACEMENT_PX = float(threshold)
    if passes is not None:
        ink._ANCHOR_MIGRATION_PASSES = int(passes)
    try:
        yield
    finally:
        ink._ANCHOR_MIGRATION_DISPLACEMENT_PX, ink._ANCHOR_MIGRATION_PASSES = saved


@contextlib.contextmanager
def _identity_migration():
    real = ink._LayeredInkAccumulator.migrate

    def reanchor_only(self, _matrix):
        return real(self, np.eye(3))

    ink._LayeredInkAccumulator.migrate = reanchor_only
    try:
        yield
    finally:
        ink._LayeredInkAccumulator.migrate = real


def run(series=None, threshold=None, passes=None, diagnostics=False):
    names = list(series or SERIES)
    results = {}
    arms = dict(ARMS)
    if diagnostics:
        arms.update(DIAGNOSTIC_ARMS)
    with _migration_constants(threshold, passes):
        for arm, flags in arms.items():
            patch = (
                _identity_migration()
                if arm in DIAGNOSTIC_ARMS
                else contextlib.nullcontext()
            )
            with _env(flags), patch:
                results[arm] = {name: SERIES[name]() for name in names}
    return results


def _table(results):
    cols = ("migrations", "duplication", "iou", "iou_strict", "loss")
    lines = ["| serie | braco | " + " | ".join(cols) + " | extra |"]
    lines.append("|---" * (len(cols) + 3) + "|")
    for name in next(iter(results.values())):
        for arm, rows in results.items():
            row = rows[name]
            extra = {
                k: row[k]
                for k in ("churn", "loss_per_landing", "retention_after_3")
                if k in row
            }
            lines.append(
                f"| {name} | {arm} | "
                + " | ".join(str(row[c]) for c in cols)
                + f" | {extra or ''} |"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        help="reajuste do limiar (px); padrao _ANCHOR_MIGRATION_DISPLACEMENT_PX",
    )
    parser.add_argument(
        "--passes", type=int, help="reajuste de K; padrao _ANCHOR_MIGRATION_PASSES"
    )
    parser.add_argument(
        "--diagnostics", action="store_true", help="inclui bracos sem portao"
    )
    args = parser.parse_args()
    results = run(
        threshold=args.threshold, passes=args.passes, diagnostics=args.diagnostics
    )
    verdict = gate(results)
    print("source: synthetic")
    print(
        "limiar_px:",
        args.threshold or ink._ANCHOR_MIGRATION_DISPLACEMENT_PX,
        "K:",
        args.passes or ink._ANCHOR_MIGRATION_PASSES,
    )
    print(_table(results))
    for arm, v in verdict.items():
        failed = [k for k, ok in v["checks"].items() if not ok]
        print(f"{arm}: {'APROVADO' if v['passed'] else 'REPROVADO'} {failed}")
    if args.json:
        payload = {
            "source": "synthetic",
            "threshold_px": args.threshold or ink._ANCHOR_MIGRATION_DISPLACEMENT_PX,
            "passes": args.passes or ink._ANCHOR_MIGRATION_PASSES,
            "results": results,
            "verdict": verdict,
        }
        args.json.write_text(json.dumps(payload, indent=2, default=float))


if __name__ == "__main__":
    main()
