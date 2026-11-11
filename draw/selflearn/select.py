"""Selecao (U2): ordena exemplos coletados para correcao e treino.

O score e uma soma ponderada auditavel: cada item devolvido carrega
o score e a lista de motivos que o compoem. Entropia sozinha nao
entra (concentra em borda); os sinais sao desacordo entre vistas
(TTA), dispersao temporal, impureza de regiao e novidade, com
estrato de risco somando um bonus fixo. A cota aleatoria e garantia
estrutural, nao sorteio de desempate: sem ela a fila estreita a
distribuicao rodada apos rodada.
"""

import random

from .split import guard_selection_pool

W_TTA = 0.35
W_TEMPORAL = 0.25
W_IMPURITY = 0.20
W_NOVELTY = 0.20
STRATUM_BONUS = 0.15
RISK_STRATA = frozenset(
    {"pautado", "mao", "sombra", "movel_texturizado", "tinta_clara"}
)


def _score(item):
    reasons = []
    score = (
        W_TTA * float(item.get("tta_disagreement", 0.0))
        + W_TEMPORAL * float(item.get("temporal_dispersion", 0.0))
        + W_IMPURITY * float(item.get("region_impurity", 0.0))
        + W_NOVELTY * float(item.get("novelty", 0.0))
    )
    if float(item.get("tta_disagreement", 0.0)) >= 0.3:
        reasons.append("tta_disagreement")
    if float(item.get("temporal_dispersion", 0.0)) >= 0.3:
        reasons.append("temporal_dispersion")
    if float(item.get("region_impurity", 0.0)) >= 0.3:
        reasons.append("region_impurity")
    if float(item.get("novelty", 0.0)) >= 0.3:
        reasons.append("novelty")
    risky = RISK_STRATA & set(item.get("strata", set()))
    if risky:
        score += STRATUM_BONUS
        reasons.append("stratum:" + ",".join(sorted(risky)))
    return score, reasons


class RankedExamples(list):
    """Lista devolvida por rank_examples, com o registro da cota.

    `random_quota_requested`: quantos itens a cota pedia.
    `random_quota_unmet`: quantos a cota nao conseguiu preencher
    (0 no caminho normal; > 0 so quando a populacao candidata e
    menor que a cota pedida).
    """

    random_quota_requested = 0
    random_quota_unmet = 0


def rank_examples(
    items, per_session_cap=3, random_quota=0.2, seed=0, *, holdout_session_ids
):
    """Ordena items e aplica teto por sessao e cota aleatoria.

    holdout_session_ids e obrigatorio: a fila so enxerga sessoes de
    treino com linhagem (guard_selection_pool); item de sessao selada
    ou sem session_id/origin levanta LeakageError antes do ranking.

    items: lista de dicts com os sinais (ver modulo) e session_id.
    Retorna nova lista ordenada; cada item ganha "score" (float) e
    "reasons" (lista). Os itens da cota aleatoria ganham o motivo
    "random_quota" e vao ao fim da fila ordenada.

    A cota e definida sobre a populacao candidata: primeiro sorteia
    do overflow do teto por sessao; nao havendo overflow suficiente
    (coleta pequena, todas as sessoes dentro do teto), sorteia da
    propria fila ranqueada — nenhum item novo entra, entao o teto
    por sessao segue respeitado. O que a cota nao preencher fica
    registrado em `random_quota_unmet` da lista devolvida.
    """
    guard_selection_pool(items, holdout_session_ids)
    scored = []
    for item in items:
        score, reasons = _score(item)
        entry = dict(item)
        entry["score"] = score
        entry["reasons"] = reasons
        scored.append(entry)
    scored.sort(key=lambda e: e["score"], reverse=True)

    ranked = []
    per_session = {}
    overflow = []
    for entry in scored:
        session = entry.get("session_id")
        if per_session.get(session, 0) < per_session_cap:
            per_session[session] = per_session.get(session, 0) + 1
            ranked.append(entry)
        else:
            overflow.append(entry)

    n_random = int(round(len(ranked) * random_quota))
    picks = []
    keep = list(ranked)
    if n_random > 0:
        rng = random.Random(seed)
        if overflow:
            sampled = rng.sample(overflow, min(n_random, len(overflow)))
            for entry in sampled:
                pick = dict(entry)
                pick["reasons"] = list(entry["reasons"]) + ["random_quota"]
                picks.append(pick)
            keep = ranked[: len(ranked) - len(picks)]
        missing = n_random - len(picks)
        if missing > 0 and keep:
            chosen = rng.sample(keep, min(missing, len(keep)))
            chosen_ids = {id(entry) for entry in chosen}
            keep = [entry for entry in keep if id(entry) not in chosen_ids]
            for entry in chosen:
                pick = dict(entry)
                pick["reasons"] = list(entry["reasons"]) + ["random_quota"]
                picks.append(pick)
    result = RankedExamples(keep + picks)
    result.random_quota_requested = n_random
    result.random_quota_unmet = n_random - sum(
        1 for entry in result if "random_quota" in entry["reasons"]
    )
    return result
