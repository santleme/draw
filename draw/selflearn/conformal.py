"""Conformal por classe para o juizo dos aceitos (item 4 do self-learning v2).

Terceira perna do juizo triplo, ao lado do veto classico
(teacher.pseudo_label + monitor.classic_witness) e da correcao
humana. Quem gera nao julga: o limiar de aceitacao NAO vem da
confianca que o proprio teacher acha suficiente (INK_THR/BG_THR
fixos), vem de rotulos independentes de um conjunto de calibracao
que nao treina, nao seleciona checkpoint e nao e o holdout selado.

Conformal separado por classe (Mondrian): para cada classe c, o
escore de nao-conformidade de um pixel e s_c = 1 - p_c (p_ink = p,
p_background = 1 - p). Na calibracao, so os pixels cuja verdade e
c entram no quantil q_c, o ceil((n_c + 1)(1 - alpha))-esimo menor
escore. Assim P(c no conjunto | verdade c) >= 1 - alpha vale para
CADA classe, e nao so na media - a tinta, rara, nao fica refem do
fundo dominante.

Conjunto de predicao de um pixel: {c : s_c <= q_c}. O juizo aceita o
pixel como c so quando o conjunto e exatamente {c}; conjunto vazio
ou duplo vira ignore (255). Classe com menos de MIN_CALIBRATION_PIXELS
na calibracao fica sem suporte: nenhum pixel e aceito como ela.

Limites declarados:
- A garantia e marginal sobre pixels permutaveis com a calibracao;
  pixels da mesma pagina sao correlacionados e as sessoes novas
  podem nao ser permutaveis com as antigas (o monitor de deriva
  e quem avisa). Cobertura e meta, nao certificado.
- O quantil e do modelo calibrado (o teacher no inicio da rodada,
  isto e, o incumbente): fine-tune curto mantem o teacher perto
  dele; rodada longa invalida a calibracao.
- Isto restringe o que entra no treino; nao julga promocao. Promover
  continua exigindo holdout selado + teste ao vivo.
"""

import math

import numpy as np

CLASSES = ("background", "ink")
DEFAULT_ALPHA = 0.1
MIN_CALIBRATION_PIXELS = 200
CALIBRATION_PIXEL_CAP = 200000
CALIBRATION_VERSION = 1


def _class_scores(prob):
    p = np.asarray(prob, dtype=np.float64)
    return {"background": p, "ink": 1.0 - p}


def conformal_quantile(scores, alpha):
    """Quantil conformal split: o ceil((n + 1)(1 - alpha))-esimo menor escore.

    Se esse posto passa de n, o quantil e +inf (com n pequeno nao ha
    como garantir 1 - alpha: tudo entra no conjunto, nada vira
    singleton por ele).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha deve estar em (0, 1)")
    s = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    n = s.size
    if n == 0:
        raise ValueError("calibracao sem escores")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return math.inf
    return float(s[rank - 1])


def calibrate(pairs, alpha=DEFAULT_ALPHA, min_pixels=MIN_CALIBRATION_PIXELS):
    """Calibra q_c por classe a partir de (prob, label) independentes.

    pairs: iteravel de (prob HxW em [0, 1], label HxW com 1 tinta,
    0 fundo, 255 ignore). label tem de ser rotulo independente
    (final - referencia revisado, ou anotacao humana), nunca
    pseudo-rotulo do proprio loop. Subamostra por passo fixo acima de
    CALIBRATION_PIXEL_CAP por classe (sem RNG).
    """
    collected = {c: [] for c in CLASSES}
    n_items = 0
    for prob, label in pairs:
        prob = np.asarray(prob, dtype=np.float64)
        label = np.asarray(label)
        if prob.shape != label.shape:
            raise ValueError(f"prob {prob.shape} != label {label.shape}")
        if prob.size and (prob.min() < 0.0 or prob.max() > 1.0):
            raise ValueError("prob fora de [0, 1]")
        bad = ~np.isin(label, (0, 1, 255))
        if bad.any():
            raise ValueError("label so aceita 0, 1 e 255")
        n_items += 1
        scores = _class_scores(prob)
        collected["background"].append(scores["background"][label == 0])
        collected["ink"].append(scores["ink"][label == 1])
    classes = {}
    for c in CLASSES:
        s = np.concatenate(collected[c]) if collected[c] else np.zeros(0)
        n_total = int(s.size)
        if s.size > CALIBRATION_PIXEL_CAP:
            s = s[:: math.ceil(s.size / CALIBRATION_PIXEL_CAP)]
        if s.size < min_pixels:
            classes[c] = {"q": None, "n": n_total, "supported": False}
        else:
            q = conformal_quantile(s, alpha)
            classes[c] = {
                "q": None if math.isinf(q) else q,
                "n": n_total,
                "supported": not math.isinf(q),
            }
    return {
        "version": CALIBRATION_VERSION,
        "alpha": float(alpha),
        "n_items": n_items,
        "min_pixels": int(min_pixels),
        "classes": classes,
        "score": "1 - p_class (Mondrian por classe)",
        "role": "judge_not_generator",
    }


def _check_calibration(calibration):
    if calibration.get("version") != CALIBRATION_VERSION:
        raise ValueError("calibracao conformal em formato desconhecido")
    if set(calibration["classes"]) != set(CLASSES):
        raise ValueError("calibracao conformal sem as classes ink/background")


def prediction_sets(prob, calibration):
    """Mascaras booleanas {classe: classe esta no conjunto do pixel}."""
    _check_calibration(calibration)
    scores = _class_scores(prob)
    sets = {}
    for c in CLASSES:
        block = calibration["classes"][c]
        if not block["supported"]:
            sets[c] = np.zeros(np.shape(prob), dtype=bool)
        else:
            sets[c] = scores[c] <= block["q"]
    return sets


def conformal_accept(prob, calibration):
    """(mask, ignore) no formato de teacher.pseudo_label.

    mask = 255 onde o conjunto e {ink}; ignore = 255 onde o conjunto
    nao e singleton (vazio, duplo ou classe sem suporte).
    """
    sets = prediction_sets(prob, calibration)
    ink_only = sets["ink"] & ~sets["background"]
    bg_only = sets["background"] & ~sets["ink"]
    mask = np.where(ink_only, 255, 0).astype(np.uint8)
    ignore = np.where(ink_only | bg_only, 0, 255).astype(np.uint8)
    return mask, ignore


def coverage(pairs, calibration):
    """Cobertura empirica por classe: fracao com a verdade no conjunto."""
    hits = {c: 0 for c in CLASSES}
    totals = {c: 0 for c in CLASSES}
    singleton = {c: 0 for c in CLASSES}
    for prob, label in pairs:
        label = np.asarray(label)
        sets = prediction_sets(prob, calibration)
        only = {
            "ink": sets["ink"] & ~sets["background"],
            "background": sets["background"] & ~sets["ink"],
        }
        for c, value in (("background", 0), ("ink", 1)):
            sel = label == value
            totals[c] += int(sel.sum())
            hits[c] += int((sets[c] & sel).sum())
            singleton[c] += int((only[c] & sel).sum())
    return {
        c: {
            "n": totals[c],
            "coverage": hits[c] / totals[c] if totals[c] else None,
            "accepted_correct_rate": singleton[c] / totals[c] if totals[c] else None,
        }
        for c in CLASSES
    }
