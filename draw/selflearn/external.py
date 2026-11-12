"""Tradução local de páginas externas para a fila humana do self-learning.

Uma importação é um lote imutável em pool próprio. O selo é por sessão/lote,
nunca por frame. Teacher só vê treino; holdout começa com máscara vazia e
exige desenho humano independente, sem proposta do incumbente na tela.
"""

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from draw.safe_io import ensure_private_dir, read_image, read_json
from . import split as splits
from .session import sha256_file

MAX_ROWS = 100_000
MAX_CSV_BYTES = 32 << 20


def _asset(root, relative):
    if not relative:
        raise ValueError("imagem sem caminho")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"imagem ausente ou fora da raiz: {relative}")
    return path


def _true(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def load_catalog(root, manifest=None, batch=None):
    """frames.csv do pool Kaggle ou CSV image_path,session_id/batch_id.

    Sem CSV, todas as imagens da pasta pertencem ao batch explícito.
    Ausência de metadados nunca cria uma sessão independente por imagem.
    """
    root = Path(root).resolve()
    if manifest is not None and not Path(manifest).is_file():
        raise FileNotFoundError(manifest)
    manifest = Path(manifest) if manifest else root / "frames.csv"
    if manifest.exists():
        if manifest.stat().st_size > MAX_CSV_BYTES:
            raise ValueError("CSV acima do limite")
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            rows = []
            for row in csv.DictReader(handle):
                if len(rows) >= MAX_ROWS:
                    raise ValueError("CSV acima do limite de linhas")
                rows.append(row)
    elif batch:
        rows = []
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                rows.append({"image_path": str(path.relative_to(root))})
                if len(rows) > MAX_ROWS:
                    raise ValueError("pasta acima do limite de imagens")
    else:
        raise ValueError("informe --manifest ou --batch para imagens sem catálogo")
    result = []
    for row in rows:
        group = row.get("session_id") or row.get("batch_id") or batch
        if not group:
            raise ValueError(
                "cada imagem exige session_id ou batch_id, nunca split por frame"
            )
        session_id = row.get("session_id")
        if session_id and (
            Path(session_id).name != session_id
            or session_id in (".", "..")
            or any(c in session_id for c in "\\:")
        ):
            raise ValueError("session_id deve ser um nome simples, sem caminho")
        relative = row.get("image_path_in_archive") or row.get("image_path")
        # O pool Kaggle também admite o prefixo images/ (sem busca ambígua por basename).
        if (
            relative
            and not (root / relative).is_file()
            and (root / "images" / relative).is_file()
        ):
            relative = str(Path("images") / relative)
        path = _asset(root, relative)
        result.append(
            {
                "group": str(group),
                "session_id": session_id,
                "relative": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "prior_training": _true(
                    row.get("session_used_for_prior_training", False)
                ),
            }
        )
    if not result:
        raise ValueError("catálogo vazio")
    if len({r["relative"] for r in result}) != len(result):
        raise ValueError("imagem duplicada no catálogo")
    return sorted(result, key=lambda r: (r["group"], r["relative"]))


def translate(
    root,
    out_root,
    dataset,
    *,
    manifest=None,
    batch=None,
    split="auto",
    holdout_fraction=0.25,
    model=None,
    prepare_page=None,
    images_are_pages=False,
    preparation_id=None,
    sealed_split=None,
):
    """Gera input.png + label_draft.png e linhagem; não aprova rótulo algum.

    prepare_page é o adaptador de câmera -> página; None só é aceito quando
    o operador declara images_are_pages. Entradas sem folha são reportadas,
    não viram exemplos vazios disfarçados de páginas.
    """
    if split not in ("auto", "train", "holdout"):
        raise ValueError("split inválido")
    if not images_are_pages and prepare_page is None:
        raise ValueError(
            "frames crus exigem retificação; use --images-are-pages só para páginas"
        )
    root = Path(root).resolve()
    if Path(out_root).resolve().is_relative_to(root):
        raise ValueError("pool de saída deve ficar fora da raiz dos dados brutos")
    catalog = load_catalog(root, manifest, batch)
    seal_path = Path(sealed_split) if sealed_split is not None else root / "split.json"
    seals = read_json(seal_path)["assignments"] if seal_path.exists() else {}
    if sealed_split is not None and not seal_path.is_file():
        raise FileNotFoundError(seal_path)
    if not isinstance(seals, dict) or any(
        v not in ("train", "holdout") for v in seals.values()
    ):
        raise ValueError("selo existente exige assignments train/holdout")
    groups = sorted({r["group"] for r in catalog})
    original_sessions = {r["group"] for r in catalog if r["session_id"]}
    ids = {
        g: g
        if g in original_sessions or g in seals
        else "external-" + hashlib.sha256(f"{dataset}:{g}".encode()).hexdigest()[:24]
        for g in groups
    }
    assignments = {}
    for group in groups:
        prior = any(r["prior_training"] for r in catalog if r["group"] == group)
        side = (
            splits.assign_split(ids[group], holdout_fraction)
            if split == "auto"
            else split
        )
        if ids[group] in seals:
            if split != "auto" and side != seals[ids[group]]:
                raise splits.LeakageError(
                    f"sessão já selada como {seals[ids[group]]}: {group}"
                )
            side = seals[ids[group]]
        if prior:
            if split == "holdout" or seals.get(ids[group]) == "holdout":
                raise splits.LeakageError(f"sessão já usada em treino: {group}")
            side = "train"
        assignments[ids[group]] = side
    # Cópia exata em duas sessões não pode atravessar o selo.
    digest_groups = {}
    for row in catalog:
        owner = digest_groups.setdefault(row["sha256"], row["group"])
        if owner != row["group"]:
            raise splits.LeakageError(
                "imagem idêntica em sessões diferentes; consolide o lote"
            )
    model_path = getattr(model, "model_path", None)
    policy = {
        "dataset": dataset,
        "catalog": catalog,
        "assignments": assignments,
        "sealed_assignments": seals,
        "images_are_pages": images_are_pages,
        "preparation_id": preparation_id,
        "model_sha256": sha256_file(model_path) if model_path else None,
        "threshold": float(getattr(model, "threshold", 0.7)),
    }
    out = Path(out_root)
    ledger = out / "external_import.json"
    if ledger.exists():
        if read_json(ledger) != policy:
            raise splits.LeakageError(
                "lote selado: conteúdo/modelo/split mudou; não sobrescrever revisões"
            )
        return read_json(out / "summary.json")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            "use pool externo novo; não misture com process_sessions.py"
        )
    out = ensure_private_dir(out)
    rows, problems = [], []
    holdout_ids = [sid for sid, side in assignments.items() if side == "holdout"]
    for group in groups:
        sid = ids[group]
        side = assignments[sid]
        for generation, source in enumerate(
            (r for r in catalog if r["group"] == group), 1
        ):
            image = read_image(root / source["relative"], cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"imagem ilegível: {source['relative']}")
            page = image if images_are_pages else prepare_page(image)
            if page is None:
                problems.append(
                    {
                        "session_id": sid,
                        "source": source["relative"],
                        "reason": "no_page",
                    }
                )
                continue
            folder = out / side / sid / f"gen_{generation:04d}"
            folder.mkdir(parents=True, exist_ok=True)
            draft = np.zeros(page.shape[:2], dtype=np.uint8)
            if side == "train":
                if model is None:
                    raise ValueError("treino exige modelo incumbente")
                splits.guard_pseudo_labeling(
                    [{"session_id": sid, "origin": "pseudo_incumbent"}], holdout_ids
                )
                prob = np.asarray(model.ink_prob(page))
                if prob.shape != draft.shape or not np.all(np.isfinite(prob)):
                    raise ValueError("pseudo-máscara com shape/valores inválidos")
                draft = np.where(prob > model.threshold, 255, 0).astype(np.uint8)
            for name, data in (("input.png", page), ("label_draft.png", draft)):
                if not cv2.imwrite(str(folder / name), data):
                    raise RuntimeError(f"falha ao gravar {folder / name}")
            rows.append(
                {
                    "session_id": sid,
                    "generation": generation,
                    "split": side,
                    "origin": "pseudo_incumbent"
                    if side == "train"
                    else "independent_label",
                    "label_source": "incumbent"
                    if side == "train"
                    else "human_from_scratch",
                    "label_reviewed": False,
                    "requires_independent_label": side == "holdout",
                    "files": {"final_warp": str((folder / "input.png").resolve())},
                    "label_draft": str((folder / "label_draft.png").resolve()),
                    "source": {"dataset": dataset, **source},
                    "model_sha256": policy["model_sha256"] if side == "train" else None,
                    "input_sha256": sha256_file(folder / "input.png"),
                }
            )
    summary = {
        "pages_queued": len(rows),
        "sessions": len({r["session_id"] for r in rows}),
        "train": sum(r["split"] == "train" for r in rows),
        "holdout": sum(r["split"] == "holdout" for r in rows),
        "reviewed": 0,
        "problems": problems,
    }
    (out / "index.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    )
    (out / "split.json").write_text(json.dumps({"assignments": assignments}, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    ledger.write_text(json.dumps(policy, indent=2, sort_keys=True))
    return summary
