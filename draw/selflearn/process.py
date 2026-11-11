"""Processamento do pool de sessoes gravadas (a traducao do loop).

O gravador (session.py) ja roda ligado no app e acumula sessoes
em captures/sessions/<id>/ - referencia, warp final, canvas final
e manifesto com sha256. O que faltava era a etapa seguinte da
esteira, que este modulo implementa:

1. Varre as sessoes, confere a integridade de cada geracao contra
   o manifesto (arquivo ausente ou sha divergente = geracao
   excluida e problema registrado, nunca silencio).
2. Divide por SESSAO (split.py) com o selo persistido: uma sessao
   que ja foi holdout nunca volta a ser treino. A fracao sobe na
   escada ate o holdout atingir o minimo do portao (gate.py);
   depois disso a fracao fica gravada e nao muda.
3. Para o holdout, gera o RASCUNHO do rotulo independente
   (final - referencia, split.draft_independent_label) - sem
   nenhuma saida de modelo. O rascunho so vira rotulo depois da
   revisao manual registrada em reviews.json; ate la o exemplo
   carrega label_reviewed False e o portao o rejeita, por desenho.
4. Para o treino, indexa os pares (referencia, final) com
   linhagem completa, passando pelas guardas de vazamento.
5. Escreve o placar: sessoes e paginas por lado, progresso contra
   o minimo do portao (sessoes seladas, paginas revisadas).

Tudo local: a raiz de saida padrao e captures/pool/, fora do git.
Sessao sem manifesto e orfa: sem ancora de integridade ela nao
entra em lado nenhum - aparece no relatorio como problema.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from ..safe_io import InputLimitError, ensure_private_dir, read_image, read_json
from . import gate as gate_module
from . import session as session_module
from . import split as split_module

_FRACTION_LADDER = (0.25, 1.0 / 3.0, 0.5, 0.6, 2.0 / 3.0, 0.75, 0.8)
_EMPTY_INK_FRACTION = 0.0005

_FILES = ("reference_warp", "final_warp", "final_canvas")


def _load_json(path):
    return read_json(path)


def _inside(base, rel):
    """Caminho de rel sob base, ou None se rel escapa (absoluto ou ..)."""
    if not isinstance(rel, str) or not rel:
        return None
    candidate = (base / rel).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    return candidate


def scan_sessions(sessions_root):
    """Le as sessoes do disco e confere a integridade.

    Retorna (sessions, problems): sessions e uma lista de dicts
    {session_id, path, generations: [{generation, files, ink_key}]}
    so com geracoes integ ras; problems e uma lista de strings
    descrevendo orfas e geracoes excluidas.
    """
    root = Path(sessions_root)
    sessions = []
    problems = []
    if not root.exists():
        return sessions, [f"raiz de sessoes nao existe: {root}"]
    for session_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        session_id = session_dir.name
        manifest_path = session_dir / "manifest.json"
        if not manifest_path.exists():
            problems.append(f"sessao orfa (sem manifest.json), excluida: {session_id}")
            continue
        try:
            manifest = _load_json(manifest_path)
        except InputLimitError as error:
            problems.append(f"sessao com manifest.json ilegivel, excluida: {error}")
            continue
        entries = (
            manifest.get("generations", []) if isinstance(manifest, dict) else None
        )
        if not isinstance(entries, list):
            problems.append(
                f"sessao com manifest.json malformado, excluida: {session_id}"
            )
            continue
        generations = []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("files"), dict):
                problems.append(f"{session_id}: geracao malformada no manifesto")
                continue
            gen = entry.get("generation")
            files = entry["files"]
            ok = True
            for kind in _FILES:
                rel = files.get(kind)
                expected = entry.get(f"{kind}_sha256")
                if rel is None or expected is None:
                    problems.append(f"{session_id} gen {gen}: manifesto sem {kind}")
                    ok = False
                    continue
                fpath = _inside(session_dir, rel)
                if fpath is None:
                    problems.append(
                        f"{session_id} gen {gen}: caminho fora da sessao em {kind}"
                    )
                    ok = False
                    continue
                if not fpath.is_file():
                    problems.append(f"{session_id} gen {gen}: arquivo ausente {rel}")
                    ok = False
                    continue
                if session_module.sha256_file(fpath) != expected:
                    problems.append(f"{session_id} gen {gen}: sha256 diverge em {rel}")
                    ok = False
            if ok:
                generations.append(
                    {
                        "generation": gen,
                        "files": {
                            kind: str(session_dir / files[kind]) for kind in _FILES
                        },
                    }
                )
        sessions.append(
            {
                "session_id": session_id,
                "path": str(session_dir),
                "generations": generations,
                "models": manifest.get("models") or {},
            }
        )
    return sessions, problems


def _ink_fraction(final_path, reference_path):
    """Fracao de pixels que escureceram (rascunho final - referencia)."""
    final = read_image(final_path, cv2.IMREAD_COLOR)
    reference = read_image(reference_path, cv2.IMREAD_COLOR)
    if final is None or reference is None:
        raise ValueError(f"warp ilegivel: {final_path} / {reference_path}")
    mask = split_module.draft_independent_label(final, reference)
    return float(np.count_nonzero(mask)) / float(mask.size), mask


def _choose_fraction(sessions, min_holdout_sessions):
    """Menor fracao da escada que sela o minimo de sessoes."""
    chosen = _FRACTION_LADDER[-1]
    for fraction in _FRACTION_LADDER:
        result = split_module.split_sessions(sessions, holdout_fraction=fraction)
        chosen = fraction
        if len(result[split_module.HOLDOUT]) >= min_holdout_sessions:
            break
    return chosen


def process_pool(
    sessions_root, out_root, min_holdout_sessions=gate_module.MIN_HOLDOUT_SESSIONS
):
    """Roda a esteira sobre as sessoes gravadas e devolve o placar.

    Escreve em out_root: split.json (selo persistido), index.jsonl
    (uma linha por geracao, com linhagem), os rascunhos do holdout
    em holdout/<sessao>/gen_NNNN/label_draft.png e summary.json.
    """
    out = ensure_private_dir(out_root)

    sessions, problems = scan_sessions(sessions_root)

    # Fracao de tinta por geracao: alimenta a flag de pagina vazia
    # (estratificacao do split) e o placar.
    for session in sessions:
        session["has_empty_pages"] = False
        for generation in session["generations"]:
            ink_fraction, _ = _ink_fraction(
                generation["files"]["final_warp"], generation["files"]["reference_warp"]
            )
            generation["ink_fraction"] = ink_fraction
            if ink_fraction < _EMPTY_INK_FRACTION:
                session["has_empty_pages"] = True

    # Selo: o ledger persistido manda; sessao selada nunca dessela.
    ledger_path = out / "split.json"
    if ledger_path.exists():
        ledger = _load_json(ledger_path)
        fraction = ledger["fraction"]
        sealed = dict(ledger["assignments"])
    else:
        fraction = _choose_fraction(sessions, min_holdout_sessions)
        sealed = {}
    fresh = split_module.split_sessions(sessions, holdout_fraction=fraction)
    assignment = {}
    for side in (split_module.TRAIN, split_module.HOLDOUT):
        for sid in fresh[side]:
            assignment[sid] = side
    for sid, side in sealed.items():
        if sid in assignment and assignment[sid] != side:
            if side == split_module.HOLDOUT:
                raise split_module.LeakageError(
                    f"sessao selada como holdout cairia no treino: {sid}"
                )
            assignment[sid] = side
    split_module.assert_no_leakage(
        [s for s, v in assignment.items() if v == split_module.TRAIN],
        [s for s, v in assignment.items() if v == split_module.HOLDOUT],
    )

    reviews_path = out / "reviews.json"
    reviews = _load_json(reviews_path) if reviews_path.exists() else {}

    holdout_dir = out / "holdout"
    index_lines = []
    counts = {
        "sessions_total": len(sessions),
        "sessions_train": 0,
        "sessions_holdout": 0,
        "pages_total": 0,
        "pages_train": 0,
        "pages_holdout_draft": 0,
        "pages_holdout_reviewed": 0,
    }
    holdout_ids = [s for s, v in assignment.items() if v == split_module.HOLDOUT]
    for session in sessions:
        sid = session["session_id"]
        side = assignment[sid]
        counts[f"sessions_{side}"] += 1
        for generation in session["generations"]:
            counts["pages_total"] += 1
            key = f"{sid}/gen_{generation['generation']:04d}"
            line = {
                "session_id": sid,
                "generation": generation["generation"],
                "split": side,
                "files": generation["files"],
                "ink_fraction": generation["ink_fraction"],
            }
            if side == split_module.HOLDOUT:
                _, mask = _ink_fraction(
                    generation["files"]["final_warp"],
                    generation["files"]["reference_warp"],
                )
                draft_rel = Path(sid) / f"gen_{generation['generation']:04d}"
                draft_path = holdout_dir / draft_rel / "label_draft.png"
                draft_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(draft_path), mask):
                    raise RuntimeError(f"falha ao gravar o rascunho {draft_path}")
                reviewed = bool(reviews.get(key, False))
                line.update(
                    {
                        "origin": "independent_label",
                        "label_source": "final_minus_reference",
                        "label_reviewed": reviewed,
                        "label_draft": str(draft_path),
                    }
                )
                counts["pages_holdout_draft"] += 1
                if reviewed:
                    counts["pages_holdout_reviewed"] += 1
            else:
                line["origin"] = "session_pair"
                split_module.guard_train_example(line, holdout_ids)
                counts["pages_train"] += 1
            index_lines.append(line)

    ledger = {
        "fraction": fraction,
        "assignments": assignment,
        "min_holdout_sessions": min_holdout_sessions,
    }
    with open(ledger_path, "w", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2, sort_keys=True)
    with open(out / "index.jsonl", "w", encoding="utf-8") as handle:
        for line in index_lines:
            handle.write(json.dumps(line, sort_keys=True) + "\n")

    summary = {
        **counts,
        "fraction": fraction,
        "gate": {
            "min_sessions": min_holdout_sessions,
            "min_pages": gate_module.MIN_HOLDOUT_PAGES,
            "sessions_ok": counts["sessions_holdout"] >= min_holdout_sessions,
            "pages_ok": counts["pages_holdout_reviewed"]
            >= gate_module.MIN_HOLDOUT_PAGES,
        },
        "problems": problems,
    }
    with open(out / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def format_summary(summary):
    """Placar legivel do pool, com os problemas no fim (alto e claro)."""
    gate_info = summary["gate"]
    lines = [
        "Pool de sessoes reais",
        f"  sessoes: {summary['sessions_total']} "
        f"(treino {summary['sessions_train']}, "
        f"holdout {summary['sessions_holdout']})",
        f"  paginas: {summary['pages_total']} "
        f"(treino {summary['pages_train']}, "
        f"holdout {summary['pages_holdout_draft']} rascunhos, "
        f"{summary['pages_holdout_reviewed']} revisadas)",
        f"  portao do holdout: sessoes "
        f"{summary['sessions_holdout']}/{gate_info['min_sessions']} "
        f"({'OK' if gate_info['sessions_ok'] else 'FALTA'}), "
        f"paginas revisadas "
        f"{summary['pages_holdout_reviewed']}/{gate_info['min_pages']} "
        f"({'OK' if gate_info['pages_ok'] else 'FALTA'})",
        f"  fracao de holdout selada: {summary['fraction']:.3f}",
    ]
    if summary["problems"]:
        lines.append("  PROBLEMAS:")
        lines.extend(f"    - {p}" for p in summary["problems"])
    return "\n".join(lines)
