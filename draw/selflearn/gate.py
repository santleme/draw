"""Portao de promocao (correcao 4 do loop v2).

O IoU DIBCO (0,13-0,22 nas rodadas ate aqui) mede binarizacao de
documento historico, nao tinta nova na mesa. A rodada 4 venceu no
DIBCO e perdeu nos warps reais desenhados: quem promove modelo
aqui e o replay real selado (holdout por sessao de split.py, com
rotulo independente), e o DIBCO entra so como diagnostico
impresso ao lado, incapaz de promover ou de salvar um candidato.

Segundo furo que este modulo fecha: o replay real de hoje tem
~48 arquivos antigos em captures, o que nao da poder estatistico
para separar candidato de incumbente. Abaixo do n minimo
pre-registrado (sessoes E paginas de holdout), o veredito e
INCONCLUSIVO - com os n atuais impressos no resultado. Nunca
PASS: um portao que aprova sem amostra esta medindo ruido e
chamando de ganho.

Pre-registro: min_sessions/min_pages e as margens sao parametros
declarados ANTES de ver o resultado da rodada, e o resultado os
devolve carimbados para a decisao ser auditavel depois.

Regra de parada do loop (ordem de adocao do Lucas, 2026-11-06):
duas rodadas seguidas sem ganho no holdout selado param o loop -
o insumo seguinte e sessao real nova, nao outra rodada de
pseudo-rotulo sobre o mesmo material. Ganho e medido POR SESSAO
e pareado: candidato e campeao julgados nas mesmas sessoes
seladas, e o delta so conta como ganho quando o intervalo de
confianca do bootstrap por sessao exclui o zero. Rodada com n
abaixo do minimo e INCONCLUSIVA e nao conta nem como ganho nem
como falta dele - parar por ruido que nao se consegue medir
seria o mesmo defeito que promover por ele.
"""

import random

PROMOTE = "PROMOTE"
REJECT = "REJECT"
INCONCLUSIVE = "INCONCLUSIVE"

# Piso pre-registrado do loop v2 (2026-11-02). Mudar estes
# numeros depois de ver uma rodada e mover a trave: quem muda
# declara a mudanca no pre-registro da rodada seguinte.
MIN_HOLDOUT_SESSIONS = 5
MIN_HOLDOUT_PAGES = 25


def _check_real(metrics, label):
    for field in ("primary", "worst_group"):
        if field not in metrics:
            raise ValueError(f"{label}: metrica real sem {field!r}")


def judge_candidate(
    candidate_real,
    baseline_real,
    dibco=None,
    min_sessions=MIN_HOLDOUT_SESSIONS,
    min_pages=MIN_HOLDOUT_PAGES,
    promote_margin=0.0,
    worst_group_tolerance=0.0,
    accepted_drift=None,
):
    """Julga um candidato contra o incumbente no replay real.

    candidate_real/baseline_real: dicts com primary (metrica
    principal do replay real, maior melhor), worst_group (a
    mesma metrica no pior grupo) e, no candidato,
    holdout_sessions/holdout_pages (o n do selo). dibco: dict
    com iou do candidato (e opcionalmente do incumbente) - so
    diagnostico.

    Devolve dict com verdict (PROMOTE/REJECT/INCONCLUSIVE),
    reasons, n atual e n exigido, e o bloco dibco_diagnostic
    marcado como incapaz de promover. accepted_drift: o status do
    monitor de deriva dos aceitos da rodada (drift.py, OK/ATTENTION/
    DRIFT/INSUFFICIENT/NO_REFERENCE) - sinal para a revisao humana,
    que nao muda o veredito: deriva nao e prova de piora, e so o
    holdout selado mede ganho.
    """
    _check_real(candidate_real, "candidato")
    _check_real(baseline_real, "incumbente")
    sessions = int(candidate_real.get("holdout_sessions", 0))
    pages = int(candidate_real.get("holdout_pages", 0))

    diagnostic = None
    if dibco is not None:
        diagnostic = {
            "iou": dibco.get("iou"),
            "baseline_iou": dibco.get("baseline_iou"),
            "role": "diagnostic_only",
            "promotes": False,
        }

    result = {
        "verdict": None,
        "reasons": [],
        "n": {"holdout_sessions": sessions, "holdout_pages": pages},
        "required_n": {
            "holdout_sessions": int(min_sessions),
            "holdout_pages": int(min_pages),
        },
        "real": {
            "candidate_primary": candidate_real["primary"],
            "baseline_primary": baseline_real["primary"],
            "candidate_worst_group": candidate_real["worst_group"],
            "baseline_worst_group": baseline_real["worst_group"],
        },
        "dibco_diagnostic": diagnostic,
        "judgment_signals": None,
    }
    if accepted_drift is not None:
        flagged = accepted_drift in ("ATTENTION", "DRIFT")
        result["judgment_signals"] = {
            "accepted_drift": accepted_drift,
            "human_review_required": flagged,
            "changes_verdict": False,
        }
        if flagged:
            result["reasons"].append(
                f"monitor de deriva dos aceitos: {accepted_drift} - a "
                "revisao humana confere o que o loop passou a aceitar "
                "antes do teste ao vivo (sinal, nao muda o veredito)"
            )

    if sessions < min_sessions or pages < min_pages:
        result["verdict"] = INCONCLUSIVE
        result["reasons"].append(
            f"amostra insuficiente: holdout real com "
            f"{sessions} sessoes e {pages} paginas; o minimo "
            f"pre-registrado e {min_sessions} sessoes e "
            f"{min_pages} paginas. Sem poder estatistico, nao ha "
            f"PASS - coletar mais sessoes seladas antes de julgar."
        )
        return result

    if candidate_real["primary"] < (baseline_real["primary"] + promote_margin):
        result["verdict"] = REJECT
        result["reasons"].append(
            f"replay real: primario do candidato "
            f"({candidate_real['primary']}) abaixo do incumbente "
            f"+ margem ({baseline_real['primary']} + "
            f"{promote_margin})"
        )
    if candidate_real["worst_group"] < (
        baseline_real["worst_group"] - worst_group_tolerance
    ):
        if result["verdict"] is None:
            result["verdict"] = REJECT
        result["reasons"].append(
            f"pior grupo regrediu: {candidate_real['worst_group']} "
            f"contra {baseline_real['worst_group']} do incumbente "
            f"(tolerancia {worst_group_tolerance})"
        )

    if result["verdict"] is None:
        result["verdict"] = PROMOTE
        result["reasons"].append(
            "replay real selado: primario >= incumbente + margem "
            "e pior grupo sem regressao, com n suficiente "
            f"({sessions} sessoes, {pages} paginas)"
        )

    if diagnostic is not None and result["verdict"] == REJECT:
        result["reasons"].append(
            "DIBCO e diagnostico e nao entra no veredito: vencer "
            "nele e perder no real e o padrao da rodada 4"
        )
    return result


# Piso pre-registrado da regra de parada (2026-11-06). Mexer
# nestes numeros depois de ver as rodadas e mover a trave.
STOP_ROUNDS_WITHOUT_GAIN = 2
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_SEED = 20261106


def session_bootstrap_delta(
    candidate_by_session,
    baseline_by_session,
    n_boot=BOOTSTRAP_RESAMPLES,
    alpha=BOOTSTRAP_ALPHA,
    seed=BOOTSTRAP_SEED,
):
    """Delta medio candidato - base, com IC por bootstrap de sessao.

    Os dois lados sao dicts sessao -> escore (maior melhor) sobre
    as MESMAS sessoes seladas; o bootstrap reamostra sessoes com
    reposicao e recomputa o delta medio, entao a unidade
    estatistica e a sessao, nunca a pagina (correcao 1 do loop:
    pagina da mesma sessao nao e evidencia independente).

    Devolve (delta, ci_low, ci_high). Conjuntos de sessoes
    diferentes dos dois lados levantam ValueError: delta sem
    pareamento nao mede ganho, mede troca de amostra.
    """
    candidate_sessions = set(candidate_by_session)
    baseline_sessions = set(baseline_by_session)
    if candidate_sessions != baseline_sessions:
        raise ValueError(
            "bootstrap por sessao exige as mesmas sessoes dos dois "
            f"lados; candidato tem {sorted(candidate_sessions)}, "
            f"base tem {sorted(baseline_sessions)}"
        )
    sessions = sorted(candidate_sessions)
    deltas = [
        float(candidate_by_session[s]) - float(baseline_by_session[s]) for s in sessions
    ]
    n = len(deltas)
    delta = sum(deltas) / n
    if n == 0:
        raise ValueError("bootstrap por sessao sem nenhuma sessao")
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    low_idx = int((alpha / 2) * n_boot)
    high_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return delta, means[low_idx], means[high_idx]


def evaluate_rounds(
    rounds,
    min_sessions=MIN_HOLDOUT_SESSIONS,
    stop_after=STOP_ROUNDS_WITHOUT_GAIN,
    n_boot=BOOTSTRAP_RESAMPLES,
    alpha=BOOTSTRAP_ALPHA,
    seed=BOOTSTRAP_SEED,
):
    """Aplica a regra de parada a uma sequencia de rodadas julgadas.

    rounds: lista ordenada de dicts, um por rodada do loop, com
      round              rotulo/numero da rodada (carimbado no resultado)
      candidate_by_session  escores do candidato da rodada por sessao
      baseline_by_session   escores do incumbente por sessao - exigido
                            so na primeira rodada; dela nasce o campeao
                            contra o qual toda rodada e comparada.
    Uma rodada GANHA quando o IC do bootstrap do delta contra o
    campeao exclui o zero por cima; ganhando, o candidato vira o
    campeao e o contador de rodadas sem ganho zera. Rodada sem
    ganho incrementa o contador; no segundo seguido, should_stop.
    Rodada com menos sessoes que min_sessions e INCONCLUSIVA e nao
    mexe nem no contador nem no campeao. Todas as rodadas precisam
    usar o mesmo conjunto de sessoes seladas - holdout que muda no
    meio da sequencia e outro pre-registro, e a funcao levanta
    ValueError em vez de comparar o incomparavel.
    """
    if not rounds:
        raise ValueError("evaluate_rounds sem nenhuma rodada")
    result = {
        "should_stop": False,
        "consecutive_no_gain": 0,
        "champion_round": None,
        "per_round": [],
        "reasons": [],
        "preregistered": {
            "min_sessions": int(min_sessions),
            "stop_after": int(stop_after),
            "n_boot": int(n_boot),
            "alpha": alpha,
        },
    }
    champion_scores = None
    champion_sessions = None
    for entry in rounds:
        label = entry.get("round")
        candidate = entry["candidate_by_session"]
        if champion_scores is None:
            if "baseline_by_session" not in entry:
                raise ValueError(
                    f"rodada {label}: a primeira rodada precisa de "
                    "baseline_by_session (o incumbente de quem o "
                    "campeao herda os escores)"
                )
            champion_scores = entry["baseline_by_session"]
            champion_sessions = set(champion_scores)
            result["champion_round"] = "incumbente"
        record = {"round": label, "n_sessions": len(candidate)}
        if len(candidate) < min_sessions:
            record["outcome"] = INCONCLUSIVE
            record["delta"] = None
            record["ci"] = None
            result["per_round"].append(record)
            result["reasons"].append(
                f"rodada {label}: INCONCLUSIVA - {len(candidate)} "
                f"sessoes seladas abaixo do minimo pre-registrado "
                f"({min_sessions}); nao conta como ganho nem como "
                "falta de ganho"
            )
            continue
        if set(candidate) != champion_sessions:
            raise ValueError(
                f"rodada {label}: sessoes do candidato diferem do "
                "holdout selado das rodadas anteriores; holdout que "
                "muda no meio da sequencia e outro pre-registro"
            )
        delta, ci_low, ci_high = session_bootstrap_delta(
            candidate, champion_scores, n_boot=n_boot, alpha=alpha, seed=seed
        )
        record["delta"] = delta
        record["ci"] = [ci_low, ci_high]
        if ci_low > 0:
            record["outcome"] = "GAIN"
            champion_scores = candidate
            result["champion_round"] = label
            result["consecutive_no_gain"] = 0
            result["reasons"].append(
                f"rodada {label}: ganho sobre o campeao (delta "
                f"{delta:.4f}, IC [{ci_low:.4f}, {ci_high:.4f}] "
                "exclui o zero); candidato vira o campeao"
            )
        else:
            record["outcome"] = "NO_GAIN"
            result["consecutive_no_gain"] += 1
            result["reasons"].append(
                f"rodada {label}: sem ganho sobre o campeao (delta "
                f"{delta:.4f}, IC [{ci_low:.4f}, {ci_high:.4f}] "
                f"inclui o zero); "
                f"{result['consecutive_no_gain']} rodada(s) seguida(s) "
                "sem ganho"
            )
        result["per_round"].append(record)
        if result["consecutive_no_gain"] >= stop_after:
            result["should_stop"] = True
            result["reasons"].append(
                f"PARADA: {stop_after} rodadas seguidas sem ganho no "
                "holdout selado - o proximo insumo e sessao real "
                "nova, nao outra rodada sobre o mesmo material"
            )
            break
    return result


def format_stop_decision(result):
    """Decisao de parada em texto, com os parametros carimbados."""
    pre = result["preregistered"]
    lines = [
        f"parada: {'SIM' if result['should_stop'] else 'nao'} "
        f"(regra: {pre['stop_after']} rodadas seguidas sem ganho; "
        f"bootstrap por sessao n_boot={pre['n_boot']}, "
        f"alpha={pre['alpha']}; minimo de sessoes "
        f"{pre['min_sessions']})",
        f"campeao atual: {result['champion_round']}; rodadas "
        f"seguidas sem ganho: {result['consecutive_no_gain']}",
    ]
    for reason in result["reasons"]:
        lines.append(f"- {reason}")
    return "\n".join(lines)


def format_verdict(result):
    """Veredito em texto, com os n sempre visiveis."""
    n = result["n"]
    req = result["required_n"]
    lines = [
        f"veredito: {result['verdict']}",
        f"holdout real: n={n['holdout_sessions']} sessoes, "
        f"{n['holdout_pages']} paginas (minimo: "
        f"{req['holdout_sessions']} sessoes, "
        f"{req['holdout_pages']} paginas)",
    ]
    real = result["real"]
    lines.append(
        f"primario: candidato {real['candidate_primary']} x "
        f"incumbente {real['baseline_primary']}; pior grupo: "
        f"{real['candidate_worst_group']} x "
        f"{real['baseline_worst_group']}"
    )
    diagnostic = result.get("dibco_diagnostic")
    if diagnostic is not None:
        lines.append(f"DIBCO (so diagnostico, nao promove): IoU {diagnostic['iou']}")
    signals = result.get("judgment_signals")
    if signals is not None:
        lines.append(
            f"deriva dos aceitos (sinal ao juizo): {signals['accepted_drift']}"
        )
    for reason in result["reasons"]:
        lines.append(f"- {reason}")
    return "\n".join(lines)
