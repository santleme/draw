"""Revisao do holdout selado: placar, marcacao e exportacao.

process.py gera os rascunhos holdout/<sessao>/gen_NNNN/label_draft.png
(final - referencia, sem modelo) e le reviews.json, mas nada ajudava
a fechar o ciclo: saber o que falta, registrar a revisao humana com
o rotulo que ela produziu, e entregar ao train_round o diretorio de
holdout no formato que ele exige (input/, mask/, manifest.jsonl).
Este modulo faz so isso. A revisao em si - olhar a pagina e corrigir
o rascunho - e humana; o codigo nunca marca nada sem uma acao
explicita (accept_draft=True ou um rotulo corrigido).

Contrato em disco (dentro do pool, fora do git):
- reviews.json: {"<sessao>/gen_NNNN": true}, o formato que
  process.py ja le (inalterado).
- holdout/<sessao>/gen_NNNN/label_reviewed.png: o rotulo revisado.
  process.py reescreve label_draft.png a cada rodada, entao o
  revisado mora em arquivo proprio e nunca e sobrescrito por ele.
- reviews_log.jsonl: trilha de auditoria (quando, quem, modo
  accept_draft/corrected, sha256 do rascunho e do rotulo).

Sessao "completa" = sessao de holdout com TODAS as paginas
revisadas (o portao conta sessoes inteiras). Exportacao so leva
sessoes completas. Nada aqui promove modelo: o placar diz se o n do
portao foi atingido; o veredito continua em gate.judge_candidate,
com teste ao vivo depois.
"""

import datetime
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from draw.selflearn import gate as gate_module

REVIEWED_LABEL = "label_reviewed.png"
DRAFT_LABEL = "label_draft.png"


def _key(session_id, generation):
    return f"{session_id}/gen_{int(generation):04d}"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_reviews(pool):
    path = Path(pool) / "reviews.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def holdout_pages(pool):
    """Linhas de holdout do index.jsonl, em ordem (sessao, geracao)."""
    index = Path(pool) / "index.jsonl"
    if not index.exists():
        raise FileNotFoundError(
            f"{index} nao existe: rode tools/selflearn/process_sessions.py antes"
        )
    rows = [
        json.loads(line)
        for line in index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("split") == "holdout"]
    return sorted(rows, key=lambda r: (r["session_id"], r["generation"]))


def _gen_dir(pool, session_id, generation):
    return Path(pool) / "holdout" / session_id / f"gen_{int(generation):04d}"


def status(
    pool,
    min_sessions=gate_module.MIN_HOLDOUT_SESSIONS,
    min_pages=gate_module.MIN_HOLDOUT_PAGES,
):
    """Placar do selado por sessao: revisadas, pendentes e o que falta."""
    reviews = _load_reviews(pool)
    sessions = {}
    problems = []
    for row in holdout_pages(pool):
        sid, gen = row["session_id"], row["generation"]
        key = _key(sid, gen)
        block = sessions.setdefault(
            sid, {"pages": 0, "reviewed": [], "pending": [], "missing_label": []}
        )
        block["pages"] += 1
        if bool(reviews.get(key, False)):
            block["reviewed"].append(gen)
            if not (_gen_dir(pool, sid, gen) / REVIEWED_LABEL).is_file():
                block["missing_label"].append(gen)
                problems.append(
                    f"{key}: marcada em reviews.json sem {REVIEWED_LABEL} "
                    "(revisao antiga ou manual); nao completa a sessao nem "
                    "exporta ate remarcar"
                )
        else:
            block["pending"].append(gen)
    for block in sessions.values():
        block["complete"] = not block["pending"] and not block["missing_label"]
    complete = [sid for sid, b in sessions.items() if b["complete"]]
    reviewed_pages = sum(len(b["reviewed"]) for b in sessions.values())
    complete_pages = sum(sessions[sid]["pages"] for sid in complete)
    return {
        "sessions": sessions,
        "sessions_holdout": len(sessions),
        "sessions_complete": len(complete),
        "pages_holdout": sum(b["pages"] for b in sessions.values()),
        "pages_reviewed": reviewed_pages,
        "pages_in_complete_sessions": complete_pages,
        "gate": {
            "min_sessions": int(min_sessions),
            "min_pages": int(min_pages),
            "n_reached": len(complete) >= min_sessions and complete_pages >= min_pages,
            "promotes": False,
        },
        "problems": problems,
    }


def format_status(report):
    gate = report["gate"]
    lines = [
        "holdout selado - revisao",
        f"  sessoes completas: {report['sessions_complete']}/"
        f"{report['sessions_holdout']} (portao pede {gate['min_sessions']})",
        f"  paginas revisadas: {report['pages_reviewed']}/{report['pages_holdout']}"
        f"; em sessoes completas: {report['pages_in_complete_sessions']} "
        f"(portao pede {gate['min_pages']})",
        f"  n do portao atingido: {'sim' if gate['n_reached'] else 'nao'} "
        "(isto nao promove: veredito no portao + teste ao vivo)",
    ]
    for sid, block in sorted(report["sessions"].items()):
        state = "completa" if block["complete"] else "pendente"
        pending = ", ".join(f"gen_{g:04d}" for g in block["pending"]) or "-"
        lines.append(
            f"  {sid}: {len(block['reviewed'])}/{block['pages']} revisadas "
            f"[{state}]; faltam: {pending}"
        )
    for problem in report["problems"]:
        lines.append(f"  ! {problem}")
    return "\n".join(lines)


def _read_binary_label(path):
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"rotulo ilegivel: {path}")
    return np.where(raw > 127, 255, 0).astype(np.uint8)


def mark_reviewed(
    pool,
    session_id,
    generation,
    *,
    accept_draft=False,
    label_path=None,
    reviewer=None,
    now=None,
):
    """Registra a revisao humana de UMA pagina do holdout.

    Exige exatamente uma prova explicita: accept_draft=True (o humano
    olhou e o rascunho esta certo) ou label_path (o rotulo corrigido
    por ele, mesmo tamanho do rascunho). Pagina fora do holdout ou
    inexistente e recusada.
    """
    if bool(accept_draft) == (label_path is not None):
        raise ValueError("use accept_draft=True OU label_path, exatamente um")
    pool = Path(pool)
    rows = {(_key(r["session_id"], r["generation"])): r for r in holdout_pages(pool)}
    key = _key(session_id, generation)
    if key not in rows:
        raise KeyError(f"{key} nao e pagina de holdout neste pool")
    gen_dir = _gen_dir(pool, session_id, generation)
    draft_path = gen_dir / DRAFT_LABEL
    if not draft_path.is_file():
        raise FileNotFoundError(f"rascunho ausente: {draft_path}")
    draft = _read_binary_label(draft_path)
    if accept_draft:
        label = draft
        mode = "accept_draft"
    else:
        label = _read_binary_label(label_path)
        if label.shape != draft.shape:
            raise ValueError(
                f"rotulo {label.shape} com tamanho diferente do rascunho {draft.shape}"
            )
        mode = "corrected"
    out_path = gen_dir / REVIEWED_LABEL
    if not cv2.imwrite(str(out_path), label):
        raise RuntimeError(f"falha ao gravar {out_path}")
    reviews = _load_reviews(pool)
    reviews[key] = True
    (pool / "reviews.json").write_text(
        json.dumps(reviews, indent=2, sort_keys=True), encoding="utf-8"
    )
    stamp = (now or datetime.datetime.now(datetime.timezone.utc)).isoformat()
    entry = {
        "key": key,
        "mode": mode,
        "reviewer": reviewer,
        "at": stamp,
        "draft_sha256": _sha256(draft_path),
        "label_sha256": _sha256(out_path),
        "changed_pixels": int(np.count_nonzero(label != draft)),
    }
    with open(pool / "reviews_log.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def export_holdout(pool, out_dir):
    """Escreve o holdout das sessoes COMPLETAS no formato do train_round.

    out_dir/input/<sessao>__gen_NNNN.png (o warp final, BGR),
    out_dir/mask/<mesmo nome>.png (rotulo revisado) e manifest.jsonl
    com a linhagem que guard_holdout_example exige. Sessao com pagina
    pendente ou sem label_reviewed.png fica de fora, inteira.
    """
    report = status(pool)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} nao esta vazio; exporte para um diretorio novo")
    (out / "input").mkdir(parents=True, exist_ok=True)
    (out / "mask").mkdir(parents=True, exist_ok=True)
    rows = holdout_pages(pool)
    exported, skipped = [], {}
    manifest = []
    for sid, block in sorted(report["sessions"].items()):
        if not block["complete"]:
            skipped[sid] = "paginas pendentes"
            continue
        session_rows = [r for r in rows if r["session_id"] == sid]
        missing = [
            r
            for r in session_rows
            if not (_gen_dir(pool, sid, r["generation"]) / REVIEWED_LABEL).is_file()
        ]
        if missing:
            skipped[sid] = f"sem {REVIEWED_LABEL} em {len(missing)} pagina(s)"
            continue
        for r in session_rows:
            name = f"{sid}__gen_{int(r['generation']):04d}"
            image = cv2.imread(r["files"]["final_warp"], cv2.IMREAD_COLOR)
            label = _read_binary_label(
                _gen_dir(pool, sid, r["generation"]) / REVIEWED_LABEL
            )
            if image is None:
                raise ValueError(f"warp ilegivel: {r['files']['final_warp']}")
            if image.shape[:2] != label.shape:
                raise ValueError(f"{name}: warp e rotulo com tamanhos diferentes")
            cv2.imwrite(str(out / "input" / f"{name}.png"), image)
            cv2.imwrite(str(out / "mask" / f"{name}.png"), label)
            manifest.append(
                {
                    "file": f"{name}.png",
                    "session_id": sid,
                    "generation": r["generation"],
                    "origin": "independent_label",
                    "split": "holdout",
                    "label_source": "final_minus_reference",
                    "label_reviewed": True,
                }
            )
        exported.append(sid)
    (out / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest),
        encoding="utf-8",
    )
    return {
        "sessions_exported": exported,
        "pages_exported": len(manifest),
        "skipped": skipped,
        "gate_n_reached": len(exported) >= gate_module.MIN_HOLDOUT_SESSIONS
        and len(manifest) >= gate_module.MIN_HOLDOUT_PAGES,
    }
