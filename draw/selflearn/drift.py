"""Monitor de distribuicao dos aceitos (item 2 do self-learning v2).

O loop aceita pseudo-rotulos (tinta e fundo) que passaram pelo
juizo da rodada (limiares do teacher + veto classico; conformal
quando ligado). Se a distribuicao do que e aceito deriva de rodada
para rodada, o loop esta reciclando o proprio vies: o teacher
aceita cada vez mais do que ele mesmo ja acha, e o holdout so ve
isso tarde. Este modulo mede essa deriva POR CLASSE, contra uma
referencia fixa (a primeira rodada ou um conjunto base), e devolve
um status que vai ao juizo - nunca ao gerador.

Quem gera nao julga: o perfil dos aceitos e so observacao (nao
consome RNG, nao muda alvo, perda nem pesos da rodada), os
limiares abaixo sao pre-registrados e o status entra como sinal
para a revisao humana no portao (gate.judge_candidate), sem mudar
veredito nenhum sozinho.

Caracteristicas por classe ("ink", "background"):

- confidence: confianca do teacher nos pixels aceitos daquela
  classe (p para tinta, 1 - p para fundo). PSI sobre a amostra
  agregada de pixels; KS sobre a MEDIA POR ITEM (pixels do mesmo
  patch nao sao independentes - KS por pixel rejeita qualquer
  coisa com n de milhoes).
- accepted_fraction: fracao do item aceita como aquela classe.
  PSI e KS por item.

Limiares (pre-registrados em 2026-11-11; mexer depois de ver uma
rodada e mover a trave - quem muda declara no pre-registro da
rodada seguinte):

- PSI < 0,10: OK; 0,10 <= PSI < 0,25: ATTENTION; PSI >= 0,25: DRIFT
  (faixas usuais de estabilidade de populacao).
- Faixas do PSI: ate 10, mas limitadas pelo n. Sem deriva nenhuma o
  PSI amostral ainda vale ~ (B - 1)(1/n + 1/m) com B faixas - com 60
  itens e 10 faixas isso da 0,3, deriva fantasma. B e o maior valor
  (2..10) que mantem esse vies <= PSI_NULL_BIAS_MAX (0,025, um
  quarto do limiar de atencao); o B usado sai no relatorio.
- KS: p < 0,01 sobe OK para ATTENTION, nunca para DRIFT sozinho:
  com muitos itens o KS acha diferenca trivial significativa; o
  tamanho do efeito quem diz e o PSI. Itens de uma rodada sao
  patches sorteados das mesmas paginas, entao nao sao independentes
  de verdade: o p do KS e otimista - indicio, nao prova.
- Amostra minima: 30 itens por caracteristica de item e 500 pixels
  na amostra agregada; abaixo, a caracteristica e INSUFFICIENT
  (nao e OK: sem amostra nao ha certificado).

Status da classe e da rodada = o pior entre as partes, na ordem
DRIFT > ATTENTION > INSUFFICIENT > OK.
"""

import json
import math
from pathlib import Path

import numpy as np

OK = "OK"
ATTENTION = "ATTENTION"
DRIFT = "DRIFT"
INSUFFICIENT = "INSUFFICIENT"
NO_REFERENCE = "NO_REFERENCE"

PSI_ATTENTION = 0.10
PSI_DRIFT = 0.25
KS_ALPHA = 0.01
PSI_BINS = 10
PSI_NULL_BIAS_MAX = 0.025
PSI_EPS = 1e-4
MIN_ITEMS = 30
MIN_PIXELS = 500

PER_ITEM_PIXEL_CAP = 64
POOLED_PIXEL_CAP = 20000
PROFILE_VERSION = 1

CLASSES = ("ink", "background")
_SEVERITY = {OK: 0, INSUFFICIENT: 1, ATTENTION: 2, DRIFT: 3}


def _worst(statuses):
    statuses = list(statuses)
    if not statuses:
        return INSUFFICIENT
    return max(statuses, key=lambda s: _SEVERITY[s])


def psi(reference, current, bins=PSI_BINS, eps=PSI_EPS):
    """Population Stability Index de current contra reference.

    Bordas nos quantis da referencia (duplicatas removidas, entao
    distribuicao concentrada vira menos faixas, nunca faixa vazia
    por construcao); proporcoes suavizadas por eps.
    PSI = soma (c - r) * ln(c / r).
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cur = np.asarray(current, dtype=np.float64).ravel()
    if ref.size == 0 or cur.size == 0:
        raise ValueError("psi exige amostras nao vazias dos dois lados")
    qs = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    edges = np.unique(np.quantile(ref, qs))
    n_bins = edges.size + 1
    ref_counts = np.bincount(
        np.searchsorted(edges, ref, side="right"), minlength=n_bins
    )
    cur_counts = np.bincount(
        np.searchsorted(edges, cur, side="right"), minlength=n_bins
    )
    r = ref_counts / ref.size + eps
    c = cur_counts / cur.size + eps
    r /= r.sum()
    c /= c.sum()
    return float(np.sum((c - r) * np.log(c / r)))


def _kolmogorov_q(lam):
    """Cauda assintotica Q_KS(lambda) = 2 sum (-1)^(j-1) exp(-2 j^2 lambda^2)."""
    if lam < 0.2:
        return 1.0
    total = 0.0
    for j in range(1, 101):
        term = 2.0 * (-1.0) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam)
        total += term
        if abs(term) < 1e-12:
            break
    return float(min(1.0, max(0.0, total)))


def ks_2samp(reference, current):
    """KS de duas amostras: (D, p) com p assintotico (Stephens).

    NumPy puro, sem SciPy (o projeto nao o instala). D e o sup da
    diferenca entre as CDFs empiricas; p usa
    lambda = (sqrt(ne) + 0.12 + 0.11 / sqrt(ne)) * D, ne = n m / (n + m).
    """
    a = np.sort(np.asarray(reference, dtype=np.float64).ravel())
    b = np.sort(np.asarray(current, dtype=np.float64).ravel())
    if a.size == 0 or b.size == 0:
        raise ValueError("ks_2samp exige amostras nao vazias dos dois lados")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    ne = a.size * b.size / (a.size + b.size)
    sq = math.sqrt(ne)
    return d, _kolmogorov_q((sq + 0.12 + 0.11 / sq) * d)


def psi_bins(n_reference, n_current, max_bins=PSI_BINS):
    """Maior B em [2, max_bins] com vies nulo (B - 1)(1/n + 1/m) <= o teto."""
    inv = 1.0 / n_reference + 1.0 / n_current
    return int(min(max_bins, max(2, math.floor(1 + PSI_NULL_BIAS_MAX / inv))))


def psi_status(value):
    if value >= PSI_DRIFT:
        return DRIFT
    if value >= PSI_ATTENTION:
        return ATTENTION
    return OK


class AcceptedProfile:
    """Perfil dos aceitos de uma rodada, montado item a item.

    add_item recebe a probabilidade do teacher (H, W), o alvo
    aceito (1 tinta / 0 fundo) e o peso (1 aceito / 0 ignore) -
    exatamente o que a rodada ja calculou; nada aqui decide
    aceitacao. A subamostra de pixels e por passo fixo (sem RNG),
    para que observar nao mude a sequencia aleatoria da rodada.
    """

    def __init__(self):
        self._pixels = {c: [] for c in CLASSES}
        self._item_conf = {c: [] for c in CLASSES}
        self._item_frac = {c: [] for c in CLASSES}
        self._pixels_total = {c: 0 for c in CLASSES}
        self.n_items = 0

    def add_item(self, prob, target, weight):
        p = np.asarray(prob, dtype=np.float32).ravel()
        t = np.asarray(target).ravel() > 0.5
        w = np.asarray(weight).ravel() > 0.5
        if not (p.shape == t.shape == w.shape):
            raise ValueError("prob/target/weight com formas diferentes")
        self.n_items += 1
        for name, sel, conf in (
            ("ink", w & t, p),
            ("background", w & ~t, 1.0 - p),
        ):
            values = conf[sel]
            self._item_frac[name].append(float(sel.mean()) if sel.size else 0.0)
            self._pixels_total[name] += int(values.size)
            if values.size:
                self._item_conf[name].append(float(values.mean()))
                stride = max(1, math.ceil(values.size / PER_ITEM_PIXEL_CAP))
                self._pixels[name].append(values[::stride])

    def to_dict(self):
        classes = {}
        for name in CLASSES:
            pooled = (
                np.concatenate(self._pixels[name])
                if self._pixels[name]
                else np.zeros(0, dtype=np.float32)
            )
            if pooled.size > POOLED_PIXEL_CAP:
                pooled = pooled[:: math.ceil(pooled.size / POOLED_PIXEL_CAP)]
            classes[name] = {
                "confidence_sample": [round(float(v), 5) for v in pooled],
                "item_confidence_mean": [round(v, 6) for v in self._item_conf[name]],
                "item_accepted_fraction": [round(v, 6) for v in self._item_frac[name]],
                "accepted_pixels_total": self._pixels_total[name],
            }
        return {"version": PROFILE_VERSION, "n_items": self.n_items, "classes": classes}


def load_profile(path_or_dict):
    if isinstance(path_or_dict, dict):
        profile = path_or_dict
    else:
        profile = json.loads(Path(path_or_dict).read_text(encoding="utf-8"))
    if profile.get("version") != PROFILE_VERSION or "classes" not in profile:
        raise ValueError("perfil de aceitos em formato desconhecido")
    return profile


def _feature(ref, cur, min_n, with_ks):
    report = {"n_reference": len(ref), "n_current": len(cur)}
    if len(ref) < min_n or len(cur) < min_n:
        report.update(psi=None, ks_d=None, ks_p=None, status=INSUFFICIENT)
        return report
    bins = psi_bins(len(ref), len(cur))
    value = psi(ref, cur, bins=bins)
    status = psi_status(value)
    report["psi"] = value
    report["psi_bins"] = bins
    if with_ks:
        d, p = ks_2samp(ref, cur)
        report["ks_d"], report["ks_p"] = d, p
        if status == OK and p < KS_ALPHA:
            status = ATTENTION
    else:
        report["ks_d"] = report["ks_p"] = None
    report["status"] = status
    return report


def compare_profiles(reference, current):
    """Compara o perfil atual com a referencia; devolve o relatorio."""
    ref = load_profile(reference)
    cur = load_profile(current)
    classes = {}
    for name in CLASSES:
        r, c = ref["classes"][name], cur["classes"][name]
        features = {
            "confidence_pixels": _feature(
                r["confidence_sample"], c["confidence_sample"], MIN_PIXELS, False
            ),
            "confidence_item_mean": _feature(
                r["item_confidence_mean"], c["item_confidence_mean"], MIN_ITEMS, True
            ),
            "accepted_fraction": _feature(
                r["item_accepted_fraction"],
                c["item_accepted_fraction"],
                MIN_ITEMS,
                True,
            ),
        }
        classes[name] = {
            "status": _worst(f["status"] for f in features.values()),
            "features": features,
        }
    return {
        "status": _worst(v["status"] for v in classes.values()),
        "classes": classes,
        "n_items": {"reference": ref["n_items"], "current": cur["n_items"]},
        "preregistered": {
            "psi_attention": PSI_ATTENTION,
            "psi_drift": PSI_DRIFT,
            "ks_alpha": KS_ALPHA,
            "psi_max_bins": PSI_BINS,
            "psi_null_bias_max": PSI_NULL_BIAS_MAX,
            "min_items": MIN_ITEMS,
            "min_pixels": MIN_PIXELS,
        },
        "role": "signal_to_judgment",
        "changes_training": False,
    }


def no_reference_report():
    return {
        "status": NO_REFERENCE,
        "role": "signal_to_judgment",
        "changes_training": False,
        "note": "sem perfil de referencia: este accepted_profile.json pode "
        "ser a referencia das rodadas seguintes",
    }


def format_report(report):
    lines = [f"deriva dos aceitos: {report['status']}"]
    if report["status"] == NO_REFERENCE:
        lines.append(f"- {report['note']}")
        return "\n".join(lines)
    pre = report["preregistered"]
    lines.append(
        f"limiares: PSI atencao >= {pre['psi_attention']}, deriva >= "
        f"{pre['psi_drift']}; KS alpha {pre['ks_alpha']}; minimo "
        f"{pre['min_items']} itens / {pre['min_pixels']} pixels"
    )
    for name, block in report["classes"].items():
        lines.append(f"{name}: {block['status']}")
        for feat, f in block["features"].items():
            psi_txt = (
                "n/a" if f["psi"] is None else f"{f['psi']:.4f}, {f['psi_bins']} faixas"
            )
            ks_txt = (
                "" if f["ks_p"] is None else f", KS D={f['ks_d']:.3f} p={f['ks_p']:.3g}"
            )
            lines.append(
                f"  {feat}: {f['status']} (PSI {psi_txt}{ks_txt}; n "
                f"{f['n_reference']} x {f['n_current']})"
            )
    return "\n".join(lines)
