"""Régua de replay da orquestração: antes (união) contra depois (camadas).

Roda as sequências reais da régua de 2026-11-01 (sessão de mesa
f01 a f30 e crescimento em página vazia f01 a f03) sobre o cache
de probabilidades do modelo v2, em contagem de passadas, sem
depender do arquivo ONNX nem de threads: cada passada é um mapa
de probabilidade do cache.

- before: o comportamento da main 8b7ec90. União pura das
  máscaras por passada, publicada a cada passada. Nada sai.
- after: a `_PageCleanWorker` desta árvore (passada real, sem
  thread, relógio de cena) e o _LayeredInkAccumulator que ela
  alimenta. A exibição é confirmada + provisória; o canvas dos
  consumidores é só a confirmada; a provisória que não confirma é
  retratada e contada.

A unidade é a passada, não o segundo: segundos desta VM não
dizem nada sobre a máquina de quem usa. Uso:

  python tools/replay_orchestration.py [--out resultado.json]
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import replay_reregister as rr  # noqa: E402

HOME = Path.home()
LAB = HOME / "workspace/draw-lab"
WARPS = LAB / "real-negatives-2026-10-31/proc/warps"
RULER = LAB / "replay-ruler-2026-11-01"
THRESHOLD = 0.7
PASSES_PER_WARP = 3
GROWTH_PASSES = 60


def _load_prob_cache():
    spec = importlib.util.spec_from_file_location("prob_cache", RULER / "prob_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ProbCache()


def load_warp(cache, fid):
    img = cv2.imread(str(WARPS / f"{fid}_warp.png"))
    if img is None:
        raise RuntimeError(f"warp nao encontrado: {fid}")
    return cache.get(img, name=f"warp:{fid}")


class UnionPipeline:
    """A main 8b7ec90: uniao pura, publica tudo, nada sai."""

    def __init__(self):
        self.canvas = None

    def add(self, prob):
        mask = prob > THRESHOLD
        if self.canvas is None:
            self.canvas = mask.copy()
        else:
            self.canvas |= mask
        return {
            "display": self.canvas,
            "confirmed": self.canvas,
            "provisional_px": 0,
            "retracted_px_total": 0,
            "provisional_ever_px": int(np.count_nonzero(self.canvas)),
        }


class _ProbIsWarp:
    threshold = THRESHOLD

    def ink_prob(self, warped):
        return warped


# Um passo de cena por passada: folga sobre qualquer cadencia.
STEP_SECONDS = 10.0


class LayeredPipeline:
    """Esta arvore: exibicao em duas camadas, canvas confirmado.

    A passada e a da `_PageCleanWorker` real, no caminho sincrono
    compartilhado com tools/replay_reregister.py (make_ink_worker +
    pump() em relogio de cena), com uma geracao so e sem cantos: o
    resultado e o do acumulador que a trabalhadora alimenta.
    """

    def __init__(self):
        self._clock = rr.SceneClock()
        self._worker = rr.make_ink_worker(_ProbIsWarp(), self._clock)
        self._generation = self._worker.new_generation()

    def add(self, prob):
        self._clock.now += STEP_SECONDS
        self._worker.submit(prob, self._generation)
        if not self._worker.pump():
            raise RuntimeError("a passada nao executou")
        if self._worker.error is not None:
            raise self._worker.error
        result = self._worker._accumulator.last_result
        confirmed = result["confirmed"] > 0
        provisional = result["provisional"] > 0
        return {
            "display": confirmed | provisional,
            "confirmed": confirmed,
            "provisional_px": result["provisional_px"],
            "retracted_px_total": result["retracted_px_total"],
            "provisional_ever_px": result["provisional_ever_px"],
        }


def _recall(mask_bool, ref):
    if not np.any(ref):
        return 0.0
    return int(np.count_nonzero(mask_bool & ref)) / int(np.count_nonzero(ref))


def run_desk(make_pipeline, probs, masks):
    pipe = make_pipeline()
    dil5 = np.ones((5, 5), np.uint8)
    ref04 = masks["f04"] & ~(cv2.dilate(masks["f03"].astype(np.uint8), dil5) > 0)
    union_ref = np.zeros_like(ref04)
    for fid in [f"f{i:02d}" for i in range(4, 31)]:
        union_ref |= masks[fid]
    event = {
        "display_recall_50_pass": None,
        "display_recall_90_pass": None,
        "confirmed_recall_50_pass": None,
        "confirmed_recall_90_pass": None,
    }
    passes = 0
    event_start = None
    last = None
    empty_end_px = {}
    for index, fid in enumerate([f"f{i:02d}" for i in range(1, 31)]):
        if fid == "f04":
            event_start = passes
        for _ in range(PASSES_PER_WARP):
            last = pipe.add(probs[fid])
            passes += 1
            if event_start is not None:
                relative = passes - event_start
                for layer in ("display", "confirmed"):
                    for level, key in ((0.5, "50"), (0.9, "90")):
                        field = f"{layer}_recall_{key}_pass"
                        if event[field] is None and (
                            _recall(last[layer], ref04) >= level
                        ):
                            event[field] = relative
        if index < 3:
            empty_end_px[fid] = int(np.count_nonzero(last["confirmed"]))
    return {
        "passes_total": passes,
        "empty_segment_confirmed_px": empty_end_px,
        "first_ink_event_f04": {
            "ref_new_ink_px": int(np.count_nonzero(ref04)),
            **event,
        },
        "final": {
            "confirmed_px": int(np.count_nonzero(last["confirmed"])),
            "display_px": int(np.count_nonzero(last["display"])),
            "recall_vs_union_ref": round(_recall(last["confirmed"], union_ref), 4),
            "confirmed_px_outside_ref": int(
                np.count_nonzero(last["confirmed"] & ~union_ref)
            ),
            "provisional_px": last["provisional_px"],
            "retracted_px_total": last["retracted_px_total"],
            "provisional_ever_px": last["provisional_ever_px"],
            "retraction_rate": round(
                last["retracted_px_total"] / max(last["provisional_ever_px"], 1),
                4,
            ),
        },
    }


def run_growth(make_pipeline, probs):
    pipe = make_pipeline()
    fids = ["f01", "f02", "f03"]
    last = None
    for index in range(GROWTH_PASSES):
        last = pipe.add(probs[fids[index % len(fids)]])
    return {
        "passes_total": GROWTH_PASSES,
        "final_confirmed_px": int(np.count_nonzero(last["confirmed"])),
        "final_display_px": int(np.count_nonzero(last["display"])),
        "retracted_px_total": last["retracted_px_total"],
        "provisional_ever_px": last["provisional_ever_px"],
        "retraction_rate": round(
            last["retracted_px_total"] / max(last["provisional_ever_px"], 1),
            4,
        ),
    }


def run_flicker(make_pipeline, probs):
    """Tinta que aparece numa passada e some: o cenario do mobiliario."""
    pipe = make_pipeline()
    last = pipe.add(probs["f04"])
    shown = int(np.count_nonzero(last["display"]))
    for _ in range(5):
        last = pipe.add(probs["f01"])
    return {
        "passes_total": 6,
        "shown_after_first_pass_px": shown,
        "final_confirmed_px": int(np.count_nonzero(last["confirmed"])),
        "final_display_px": int(np.count_nonzero(last["display"])),
        "retracted_px_total": last["retracted_px_total"],
        "provisional_ever_px": last["provisional_ever_px"],
        "retraction_rate": round(
            last["retracted_px_total"] / max(last["provisional_ever_px"], 1),
            4,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cache = _load_prob_cache()
    fids = [f"f{i:02d}" for i in range(1, 31)]
    probs = {fid: load_warp(cache, fid) for fid in fids}
    masks = {fid: probs[fid] > THRESHOLD for fid in fids}
    results = {
        "inputs": (
            "warps reais f01-f30 da mesa, probabilidades do cache "
            "da regua de 2026-11-01 (modelo v2), "
            f"{PASSES_PER_WARP} passadas por warp, unidade = passada"
        ),
        "before_union": {
            "desk": run_desk(UnionPipeline, probs, masks),
            "growth": run_growth(UnionPipeline, probs),
            "flicker": run_flicker(UnionPipeline, probs),
        },
        "after_layered": {
            "desk": run_desk(LayeredPipeline, probs, masks),
            "growth": run_growth(LayeredPipeline, probs),
            "flicker": run_flicker(LayeredPipeline, probs),
        },
    }
    text = json.dumps(results, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
