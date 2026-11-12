"""Divisao treino x holdout por SESSAO (correcoes 1-3 do loop v2).

Tres regras do Lucas (2026-11-02), aqui viram codigo que levanta
erro, nao convencao:

1. Selecao por incerteza serve para o TREINO, nunca para o
   holdout. O holdout selado e feito de sessoes INTEIRAS -
   incluindo as faceis e as de paginas vazias - e a divisao e por
   sessao, nunca por frame: frames da mesma sessao sao quase
   copias uns dos outros, e separa-los entre treino e holdout
   mede memorizacao, nao generalizacao.

2. Corrigir a predicao do modelo tem vies de ancoragem (a
   tendencia e aceitar o que o modelo desenhou, pior em traço
   fino). Correcao serve para treino. O rotulo do holdout e
   INDEPENDENTE: foto final menos referencia, revisado a mao,
   sem a predicao do modelo na tela. draft_independent_label()
   gera o rascunho dessa subtracao; ele so vira rotulo de holdout
   depois da revisao manual registrada no exemplo.

3. Pseudo-rotulo de teacher da mesma familia repete os erros
   dele. UniMatch entra como regularizacao de consistencia no
   treino; pseudo-rotulo NUNCA entra no holdout, em nenhuma
   funcao deste modulo.

Todo exemplo carrega linhagem obrigatoria: session_id, origin e
split. Origem de pseudo-rotulo e qualquer origin comecando com
"pseudo" (ex.: "pseudo_ema_r3"). As guardas levantam LeakageError
em vez de filtrar em silencio: vazamento que so some da lista
continua vazando na proxima chamada.
"""

import hashlib

TRAIN = "train"
HOLDOUT = "holdout"

# Origens aceitas num exemplo de holdout: somente o rotulo
# independente revisado. Correcao da predicao, pseudo-rotulo e
# sintetico sao todos treino.
HOLDOUT_ALLOWED_ORIGINS = frozenset({"independent_label"})

_DEFAULT_SALT = "draw-holdout-v2"


class LeakageError(ValueError):
    """Uma sessao/exemplo de holdout apareceu onde so treino vale."""


def is_pseudo_origin(origin):
    return isinstance(origin, str) and origin.startswith("pseudo")


def assign_split(session_id, holdout_fraction=0.25, salt=_DEFAULT_SALT):
    """Destino deterministico de uma sessao: hash, nao sorteio.

    O mesmo session_id cai sempre no mesmo lado, em qualquer
    maquina e rodada: rerodar a divisao nunca muda o selo de uma
    sessao que ja foi holdout.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction precisa estar em (0, 1)")
    digest = hashlib.sha256(f"{salt}:{session_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return HOLDOUT if bucket < holdout_fraction else TRAIN


def _session_flags(session):
    """(tem pagina vazia, e sessao facil) de um dict de sessao."""
    empty = bool(session.get("has_empty_pages")) or (session.get("kind") == "empty")
    easy = bool(session.get("easy")) or session.get("kind") == "easy"
    return empty, easy


def split_sessions(sessions, holdout_fraction=0.25, salt=_DEFAULT_SALT):
    """Divide sessoes inteiras em {"train": [...], "holdout": [...]}.

    sessions: dicts com session_id e, quando conhecidos,
    has_empty_pages/easy (ou kind "empty"/"easy"). A atribuicao e
    por hash (assign_split), com um reparo deterministico: se
    existe sessao vazia ou facil e nenhuma caiu no holdout, a de
    menor bucket entre elas muda de lado. Sem o reparo, uma
    amostra pequena podia selar um holdout so de sessoes dificeis
    - que e a selecao por incerteza disfarcada.
    """
    ids = [s["session_id"] for s in sessions]
    if len(ids) != len(set(ids)):
        raise ValueError("session_id duplicado na lista de sessoes")
    by_id = {s["session_id"]: s for s in sessions}
    assignment = {sid: assign_split(sid, holdout_fraction, salt) for sid in ids}

    def bucket(sid):
        digest = hashlib.sha256(f"{salt}:{sid}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    for flag_index in (0, 1):
        flagged = [sid for sid in ids if _session_flags(by_id[sid])[flag_index]]
        if flagged and not any(assignment[sid] == HOLDOUT for sid in flagged):
            candidate = min(flagged, key=bucket)
            assignment[candidate] = HOLDOUT

    result = {TRAIN: [], HOLDOUT: []}
    for sid in ids:
        result[assignment[sid]].append(sid)
    return result


def _require_lineage(example):
    for field in ("session_id", "origin"):
        if not example.get(field):
            raise LeakageError(
                f"exemplo sem linhagem obrigatoria: falta {field!r} em {example!r}"
            )


def assert_no_leakage(train_session_ids, holdout_session_ids):
    overlap = set(train_session_ids) & set(holdout_session_ids)
    if overlap:
        raise LeakageError(f"sessoes nos dois lados da divisao: {sorted(overlap)}")


def _raise_if_holdout(session_ids, holdout_session_ids, context):
    holdout = set(holdout_session_ids)
    bad = sorted(set(session_ids) & holdout)
    if bad:
        raise LeakageError(
            f"{context}: sessao de holdout selado nao pode entrar: {bad}"
        )


def guard_train_example(example, holdout_session_ids):
    """Um exemplo so entra no treino fora do holdout."""
    _require_lineage(example)
    _raise_if_holdout([example["session_id"]], holdout_session_ids, "treino")
    if example.get("split") == HOLDOUT:
        raise LeakageError(f"treino: exemplo marcado como holdout: {example!r}")
    return example


def guard_selection_pool(items, holdout_session_ids):
    """A fila de selecao por incerteza so enxerga sessoes de treino."""
    session_ids = []
    for item in items:
        _require_lineage(item)
        session_ids.append(item["session_id"])
    _raise_if_holdout(session_ids, holdout_session_ids, "selecao por incerteza")
    return items


def guard_pseudo_labeling(examples, holdout_session_ids):
    """Pseudo-rotulo e treino: sessao de holdout nem e rotulada."""
    session_ids = []
    for example in examples:
        _require_lineage(example)
        session_ids.append(example["session_id"])
    _raise_if_holdout(session_ids, holdout_session_ids, "pseudo-rotulacao")
    return examples


def guard_threshold_tuning(session_ids, holdout_session_ids):
    """Calibrar limiar em sessao de holdout e consultar o selo."""
    _raise_if_holdout(session_ids, holdout_session_ids, "tuning de limiar")


def guard_holdout_example(example):
    """Um exemplo so entra no holdout com rotulo independente.

    Exige: origem independent_label (nunca pseudo, nunca correcao
    da predicao), label_source "final_minus_reference" e
    label_reviewed True (a subtracao foi revisada a mao).
    """
    _require_lineage(example)
    if is_pseudo_origin(example["origin"]):
        raise LeakageError(f"holdout: pseudo-rotulo nunca entra: {example!r}")
    if example["origin"] not in HOLDOUT_ALLOWED_ORIGINS:
        raise LeakageError(
            f"holdout: origem {example['origin']!r} nao e rotulo "
            f"independente; aceito: {sorted(HOLDOUT_ALLOWED_ORIGINS)}"
        )
    if example.get("label_source") not in (
        "final_minus_reference",
        "human_from_scratch",
    ):
        raise LeakageError(
            "holdout: o rotulo precisa nascer de final - referencia "
            f"(label_source), nao da predicao: {example!r}"
        )
    if example.get("label_reviewed") is not True:
        raise LeakageError(
            f"holdout: subtracao sem revisao manual registrada: {example!r}"
        )
    return example


def draft_independent_label(final_bgr, reference_bgr, darkening_thr=25):
    """Rascunho do rotulo independente: final - referencia.

    Devolve mascara uint8 0/255 dos pixels que ESCURECERAM da
    referencia para a foto final (tinta nova sobre a pagina como
    ela estava). E um rascunho para revisao manual, nao o rotulo:
    sombra e deslocamento de geometria tambem escurecem, e e a
    revisao (guard_holdout_example, label_reviewed) que separa.
    Nenhuma saida de modelo entra aqui.
    """
    import numpy as np

    final = np.asarray(final_bgr)
    reference = np.asarray(reference_bgr)
    if final.shape != reference.shape:
        raise ValueError(
            f"final e referencia com formas diferentes: "
            f"{final.shape} x {reference.shape}"
        )
    final_gray = final.astype(np.int16)
    reference_gray = reference.astype(np.int16)
    if final.ndim == 3:
        final_gray = final_gray.mean(axis=2)
        reference_gray = reference_gray.mean(axis=2)
    darkened = (reference_gray - final_gray) > darkening_thr
    return np.where(darkened, 255, 0).astype(np.uint8)
