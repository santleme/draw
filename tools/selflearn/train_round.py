"""Uma rodada auditável de treino do loop que aprende sozinho (U5).

Estilo UniMatch V1 simplificado sobre a U-Net da page-clean:

- Todo batch carrega páginas ROTULADAS com perda supervisionada BCE+Dice.
- O teacher EMA rotula a vista fraca; o aluno treina na vista forte com
  pseudo-rótulos vetados pela testemunha clássica. Desacordo vira ignore.
- O teacher usa EMA com rampa inicial e teto 0.999. Uma rodada posterior
  deve carregar ``--init-checkpoint``; começar do zero exige a decisão
  explícita ``--allow-random-init``.
- Treino, não rotulados e holdout têm linhagem obrigatória. O holdout é
  por sessão, independente e revisado; os guards de draw.selflearn.split
  são executados tanto no CLI quanto dentro de ``run_round``.
- O holdout só é avaliado DEPOIS do treino, no mosaico completo da
  inferência e no threshold fixo de produção (0.7). Não escolhe época,
  pesos nem limiar. A promoção continua em tools/selflearn/judge_gate.py.

Formato de cada diretório:
    input/<arquivo>.png
    mask/<arquivo>.png             (labeled e holdout)
    manifest.jsonl                 (uma linha por arquivo)

O manifesto segue ExampleRecorder: ``file``, ``session_id``, ``origin`` e
``split``. No holdout são obrigatórios ``origin=independent_label``,
``label_source=final_minus_reference`` e ``label_reviewed=true``.

Uso:
    python tools/selflearn/train_round.py --smoke
    python tools/selflearn/train_round.py --labeled-dir D --unlabeled-dir U \
        --holdout-dir H --init-checkpoint out/round_checkpoint.pt --out O

O smoke prova otimização, EMA, linhagem, avaliação e exportação num dataset
sintético mínimo. Ele não é evidência de generalização no mundo real.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from draw.selflearn import conformal, drift  # noqa: E402
from draw.selflearn.gate import MIN_HOLDOUT_PAGES, MIN_HOLDOUT_SESSIONS  # noqa: E402
from draw.selflearn.split import (  # noqa: E402
    HOLDOUT,
    assert_no_leakage,
    guard_holdout_example,
    guard_pseudo_labeling,
    guard_train_example,
    is_pseudo_origin,
)
from draw.selflearn.teacher import pseudo_label  # noqa: E402

BASE = 16
EMA_DECAY = 0.999
EMA_WARMUP_STEPS = 10
INK_THR = 0.9
BG_THR = 0.1
CONS_WEIGHT = 0.5
EVAL_THRESHOLD = 0.7
ONNX_MAX_ABS_DIFF = 1e-3


class Block(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(),
            nn.Conv2d(b, b, 3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.c(x)


class UNet(nn.Module):
    """Espelho da U-Net da page-clean (train_body_v2.py)."""

    def __init__(self, base=BASE):
        super().__init__()
        self.d1 = Block(3, base)
        self.d2 = Block(base, base * 2)
        self.d3 = Block(base * 2, base * 4)
        self.d4 = Block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = Block(base * 8 + base * 4, base * 4)
        self.u2 = Block(base * 4 + base * 2, base * 2)
        self.u1 = Block(base * 2 + base, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        s1 = self.d1(x)
        s2 = self.d2(self.pool(s1))
        s3 = self.d3(self.pool(s2))
        b = self.d4(self.pool(s3))
        y = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.u3(torch.cat([y, s3], 1))
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.u2(torch.cat([y, s2], 1))
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.u1(torch.cat([y, s1], 1))
        return self.head(y)


class PageExample:
    """Página + linhagem mínima exigida pelos guards do loop."""

    def __init__(
        self,
        image,
        mask,
        session_id,
        origin,
        file,
        split="train",
        label_source=None,
        label_reviewed=False,
    ):
        self.image = image
        self.mask = mask
        self.session_id = session_id
        self.origin = origin
        self.file = file
        self.split = split
        self.label_source = label_source
        self.label_reviewed = bool(label_reviewed)

    def lineage(self):
        return {
            "session_id": self.session_id,
            "origin": self.origin,
            "file": self.file,
            "split": self.split,
            "label_source": self.label_source,
            "label_reviewed": self.label_reviewed,
        }


def _guard_lineage(labeled, unlabeled, holdout, previous_holdout_ids=()):
    if not labeled:
        raise ValueError("nenhuma página rotulada")
    if not holdout:
        raise ValueError("holdout independente é obrigatório")
    if (
        previous_holdout_ids is None
        or isinstance(previous_holdout_ids, (str, bytes))
        or any(
            not isinstance(session_id, str) or not session_id
            for session_id in previous_holdout_ids
        )
    ):
        raise ValueError("sessões seladas do checkpoint precisam ser ids de texto")
    for page in list(labeled) + list(unlabeled) + list(holdout):
        if not isinstance(page, PageExample):
            raise TypeError("run_round exige PageExample com linhagem")
    holdout_ids = sorted(
        {page.session_id for page in holdout} | set(previous_holdout_ids)
    )
    train_ids = [page.session_id for page in labeled]
    unlabeled_ids = [page.session_id for page in unlabeled]
    assert_no_leakage(train_ids + unlabeled_ids, holdout_ids)
    for page in list(labeled) + list(unlabeled):
        guard_train_example(page.lineage(), holdout_ids)
    if unlabeled:
        guard_pseudo_labeling([page.lineage() for page in unlabeled], holdout_ids)
    for page in holdout:
        guard_holdout_example(page.lineage())
    return sorted(set(holdout_ids))


def supervised_loss(logits, target, valid=None):
    """BCE+Dice supervisionada; `valid` (0/1) exclui pixels da perda."""
    pos_weight = torch.tensor(8.0, device=logits.device, dtype=logits.dtype)
    p = torch.sigmoid(logits)
    if valid is None:
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        inter = (p * target).sum()
        dice = 1 - (2 * inter + 1) / (p.sum() + target.sum() + 1)
        return bce + dice
    bce_map = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none"
    )
    bce = (bce_map * valid).sum() / valid.sum().clamp(min=1)
    inter = (p * target * valid).sum()
    dice = 1 - (2 * inter + 1) / ((p * valid).sum() + (target * valid).sum() + 1)
    return bce + dice


def shift_with_fill(arr, dy, dx, fill):
    """Translação inteira sem wraparound; a faixa descoberta recebe fill."""
    out = np.full(arr.shape, fill, dtype=arr.dtype)
    h, w = arr.shape[:2]
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    if ys1 > ys0 and xs1 > xs0:
        out[ys0 + dy : ys1 + dy, xs0 + dx : xs1 + dx] = arr[ys0:ys1, xs0:xs1]
    return out


def geo_valid_mask(shape, dy, dx):
    """1 onde o pixel veio do conteúdo original, 0 na faixa descoberta."""
    return shift_with_fill(np.ones(shape, dtype=np.float32), dy, dx, 0.0)


def _background_estimate(img):
    """Mediana dos pixels de borda: estimativa robusta do fundo da página."""
    if img.ndim == 3:
        border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]], axis=0)
        return np.median(border, axis=0).astype(img.dtype)
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    return img.dtype.type(np.median(border))


def geo_view(img, rng, flip=None, dx=None, dy=None, fill=None):
    """Vista geométrica fraca: flip + translação inteira pequena."""
    out = img
    if flip is None:
        flip = rng.random() < 0.5
    if flip:
        out = out[:, ::-1]
    if dx is None:
        dx = int(rng.integers(-3, 4))
    if dy is None:
        dy = int(rng.integers(-3, 4))
    if fill is None:
        fill = _background_estimate(out)
    return shift_with_fill(out, dy, dx, fill), flip, dx, dy


def strong_view(img, rng):
    """Vista forte: mesma geometria, fotometria pesada."""
    out = img.astype(np.float32)
    out = out * float(rng.uniform(0.7, 1.2))
    if rng.random() < 0.7:
        h, w = out.shape[:2]
        gx = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
        gy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        plane = gx if rng.random() < 0.5 else gy
        if rng.random() < 0.5:
            plane = 1.0 - plane
        out = out * (1.0 - 0.4 * plane)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return np.clip(out, 0.0, 1.0)


def consistency_targets(prob, weak_bgr, calibration=None):
    """Alvo/peso da consistência para uma página BGR não rotulada.

    Tinta prevista sem escurecimento e escurecimento não visto viram
    ignore. A testemunha confirma escurecimento, não semântica: impresso,
    pauta e sombra ainda exigem holdout independente e revisão humana.
    Com ``calibration`` (conformal por classe, calibrado em rótulos
    independentes), o pixel só é aceito se o conjunto conformal for
    singleton e concordar com o pseudo-rótulo: juízo triplo, onde o
    veto clássico e o conformal só retiram pixels, nunca acrescentam.
    """
    bgr = (np.clip(weak_bgr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    mask, ignore, _confidence = pseudo_label(prob, bgr, ink_thr=INK_THR, bg_thr=BG_THR)
    target = (mask > 0).astype(np.float32)
    accepted = ignore == 0
    if calibration is not None:
        c_mask, c_ignore = conformal.conformal_accept(prob, calibration)
        accepted &= (c_ignore == 0) & ((c_mask > 0) == (mask > 0))
    weight = accepted.astype(np.float32)
    return target, weight


def _guard_calibration(calibration_pages, labeled, holdout_ids, unlabeled=()):
    """Calibração conformal: rótulo independente, fora do treino e do selado."""
    if not calibration_pages:
        return []
    labeled_ids = {page.session_id for page in labeled}
    labeled_ids |= {page.session_id for page in unlabeled}
    for page in calibration_pages:
        if not isinstance(page, PageExample) or page.mask is None:
            raise TypeError("calibração conformal exige PageExample com máscara")
        if is_pseudo_origin(page.origin):
            raise ValueError(
                f"{page.file}: origem {page.origin!r} é pseudo-rótulo; o juízo "
                "conformal só calibra em rótulo independente"
            )
        if page.session_id in holdout_ids:
            raise ValueError(
                f"sessão {page.session_id} é do holdout selado: não calibra o juízo"
            )
        if page.session_id in labeled_ids:
            raise ValueError(
                f"sessão {page.session_id} treina nesta rodada: calibração "
                "conformal precisa de sessões que não treinam"
            )
    return sorted({page.session_id for page in calibration_pages})


def _manifest_rows(root):
    path = root / "manifest.jsonl"
    if not path.is_file():
        raise ValueError(
            f"{root}: manifest.jsonl obrigatório para provar session_id/origem/split"
        )
    rows = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        name = Path(row.get("file", "")).stem
        if not name:
            raise ValueError(f"{path}:{line_number}: linha sem file")
        if name in rows:
            raise ValueError(f"{path}: file duplicado no manifesto: {name}")
        rows[name] = row
    return rows


def load_pages(root, with_mask, holdout=False, holdout_session_ids=()):
    """Carrega páginas em BGR, como PageCleanInk, com linhagem obrigatória."""
    root = Path(root)
    rows = _manifest_rows(root)
    pages = []
    for path in sorted((root / "input").glob("*.png")):
        row = rows.get(path.stem)
        if row is None:
            raise ValueError(f"{path}: arquivo ausente do manifest.jsonl")
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"imagem ilegível: {path}")
        # Contrato do PageCleanInk: BGR / 255, sem conversão para RGB.
        img = img.astype(np.float32) / 255.0
        mask = None
        if with_mask:
            mpath = root / "mask" / path.name
            raw = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                raise ValueError(f"máscara ausente/ilegível para {path}: {mpath}")
            if raw.shape != img.shape[:2]:
                raise ValueError(
                    f"imagem/máscara com dimensões diferentes em {path}: "
                    f"{img.shape[:2]} x {raw.shape[:2]}"
                )
            mask = (raw > 127).astype(np.float32)
        metadata = dict(row)
        metadata["file"] = path.stem
        metadata.setdefault("split", HOLDOUT if holdout else "train")
        if holdout:
            metadata["split"] = HOLDOUT
            guard_holdout_example(metadata)
        else:
            guard_train_example(metadata, holdout_session_ids)
        page = PageExample(
            img,
            mask,
            session_id=metadata["session_id"],
            origin=metadata["origin"],
            file=path.stem,
            split=metadata["split"],
            label_source=metadata.get("label_source"),
            label_reviewed=metadata.get("label_reviewed") is True,
        )
        pages.append(page)
    if not pages:
        raise ValueError(f"nenhuma página encontrada em {root / 'input'}")
    if not with_mask and not holdout:
        guard_pseudo_labeling([page.lineage() for page in pages], holdout_session_ids)
    return pages


def sample_patch(img, size, rng):
    h, w = img.shape[:2]
    size = min(size, h, w)
    y = int(rng.integers(0, h - size + 1))
    x = int(rng.integers(0, w - size + 1))
    return y, x, size


def to_tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)


def ema_decay_at_step(step, base=EMA_DECAY, warmup=EMA_WARMUP_STEPS):
    """Rampa usual de Mean Teacher: evita ancorar a EMA em pesos aleatórios."""
    if step < 0 or warmup < 1 or not 0.0 < base < 1.0:
        raise ValueError("step/warmup/base de EMA inválidos")
    return min(float(base), float(step + 1) / float(step + warmup))


def _load_student_checkpoint(student, checkpoint, device):
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    state = payload.get("student", payload) if isinstance(payload, dict) else payload
    student.load_state_dict(state)
    previous_holdout = (
        payload.get("holdout_sessions_not_for_training", [])
        if isinstance(payload, dict) and "student" in payload
        else []
    )
    if not isinstance(previous_holdout, (list, tuple)):
        raise ValueError("holdout_sessions_not_for_training precisa ser uma lista")
    return list(previous_holdout)


def _mean3(values):
    return float(np.mean(values[-3:])) if values else None


def _mean_first3(values):
    return float(np.mean(values[:3])) if values else None


def predict_page_probability(model, page, device, patch=256):
    """Inferência BGR em mosaico, com zero-padding igual a PageCleanInk."""
    model.eval()
    h, w = page.image.shape[:2]
    tiles, positions = [], []
    for y in range(0, h, patch):
        for x in range(0, w, patch):
            tile = page.image[y : y + patch, x : x + patch]
            pad_h, pad_w = patch - tile.shape[0], patch - tile.shape[1]
            if pad_h or pad_w:
                tile = np.pad(tile, [(0, pad_h), (0, pad_w), (0, 0)])
            tiles.append(to_tensor(tile))
            positions.append((y, x))
    result = np.zeros((h, w), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(tiles), 16):
            positions_group = positions[start : start + 16]
            batch = torch.stack(tiles[start : start + 16]).to(device)
            probs = torch.sigmoid(model(batch))[:, 0].cpu().numpy()
            for index, (y, x) in enumerate(positions_group):
                hh, ww = min(patch, h - y), min(patch, w - x)
                result[y : y + hh, x : x + ww] = probs[index, :hh, :ww]
    return result


def binary_counts(pred, target):
    """Contagens binárias + suporte de traço fino (largura aproximada <=3 px)."""
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if pred.shape != target.shape:
        raise ValueError("pred/target binários com formas diferentes")
    tp = int(np.count_nonzero(pred & target))
    fp = int(np.count_nonzero(pred & ~target))
    fn = int(np.count_nonzero(~pred & target))
    distance = cv2.distanceTransform(target.astype(np.uint8), cv2.DIST_L2, 3)
    thin = target & (distance <= 1.5)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": int(np.count_nonzero(~pred & ~target)),
        "thin_gt": int(np.count_nonzero(thin)),
        "thin_tp": int(np.count_nonzero(pred & thin)),
        "pixels": int(target.size),
        "gt_ink": int(np.count_nonzero(target)),
    }


def _sum_counts(groups):
    keys = ("tp", "fp", "fn", "tn", "thin_gt", "thin_tp", "pixels", "gt_ink")
    return {key: sum(group[key] for group in groups) for key in keys}


def _metrics_from_counts(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    union = tp + fp + fn
    precision_den, recall_den = tp + fp, tp + fn
    return {
        "iou": tp / union if union else 1.0,
        "dice": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0,
        "precision": (
            1.0
            if precision_den == 0 and recall_den == 0
            else (tp / precision_den if precision_den else 0.0)
        ),
        "recall": (
            1.0
            if recall_den == 0 and precision_den == 0
            else (tp / recall_den if recall_den else 0.0)
        ),
        "thin_ink_recall": (
            counts["thin_tp"] / counts["thin_gt"] if counts["thin_gt"] else None
        ),
        "support": counts,
    }


def evaluate_holdout(model, pages, device, patch=256, threshold=EVAL_THRESHOLD):
    """Avalia páginas seladas uma única vez, sem ajustar modelo ou threshold."""
    page_reports, session_groups = [], {}
    for page in pages:
        prob = predict_page_probability(model, page, device, patch=patch)
        target = page.mask > 0.5
        pred = prob > threshold  # mesmo limiar estrito de PageCleanInk.ink_mask
        counts = binary_counts(pred, target)
        report = _metrics_from_counts(counts)
        page_reports.append(
            {
                "file": page.file,
                "session_id": page.session_id,
                **{key: value for key, value in report.items() if key != "support"},
                "support": counts,
            }
        )
        session_groups.setdefault(page.session_id, []).append(counts)

    total = _sum_counts([report["support"] for report in page_reports])
    micro = _metrics_from_counts(total)
    session_reports = {}
    for session_id, groups in sorted(session_groups.items()):
        session_reports[session_id] = _metrics_from_counts(_sum_counts(groups))
    blank = [r["support"] for r in page_reports if r["support"]["gt_ink"] == 0]
    blank_fp = sum(r["fp"] for r in blank)
    blank_pixels = sum(r["pixels"] for r in blank)
    sessions = sorted(session_groups)
    worst_session = min(sessions, key=lambda sid: session_reports[sid]["iou"])
    candidate_real = {
        "primary": float(np.mean([r["iou"] for r in page_reports])),
        "worst_group": session_reports[worst_session]["iou"],
        "holdout_sessions": len(sessions),
        "holdout_pages": len(page_reports),
    }
    return {
        "evaluation": "sealed_holdout_after_training_only",
        "threshold": threshold,
        "patch": patch,
        "n_sessions": len(sessions),
        "n_pages": len(page_reports),
        "minimum_n_sufficient": (
            len(sessions) >= MIN_HOLDOUT_SESSIONS
            and len(page_reports) >= MIN_HOLDOUT_PAGES
        ),
        "minimum_n_required": {
            "sessions": MIN_HOLDOUT_SESSIONS,
            "pages": MIN_HOLDOUT_PAGES,
        },
        "micro": micro,
        "macro_page_iou": candidate_real["primary"],
        "worst_session": {
            "session_id": worst_session,
            **session_reports[worst_session],
        },
        "sessions": session_reports,
        "empty_pages": {
            "count": len(blank),
            "false_positive_px": blank_fp,
            "false_positive_fraction": blank_fp / blank_pixels if blank_pixels else 0.0,
        },
        "pages": page_reports,
        "candidate_real": candidate_real,
    }


def validate_round_onnx(model, onnx_path, dummy):
    """Compara logits PyTorch/ONNX e testa também o lote dinâmico."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        return {"status": "skipped", "reason": str(exc)}
    model.eval()
    with torch.no_grad():
        expected = model(dummy).detach().cpu().numpy()
    try:
        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        actual = session.run(None, {input_name: dummy.numpy()})[0]
        batch2 = session.run(None, {input_name: dummy.repeat(2, 1, 1, 1).numpy()})[0]
    except Exception as exc:  # noqa: BLE001 - relatório estruturado
        return {"status": "failed", "error": str(exc)}
    max_diff = float(np.max(np.abs(expected - actual)))
    dynamic_ok = batch2.shape[0] == 2 and batch2.shape[1:] == actual.shape[1:]
    ok = expected.shape == actual.shape and max_diff <= ONNX_MAX_ABS_DIFF and dynamic_ok
    return {
        "status": "ok" if ok else "degraded",
        "max_abs_diff": max_diff,
        "max_abs_diff_tolerance": ONNX_MAX_ABS_DIFF,
        "dynamic_batch_ok": bool(dynamic_ok),
        "torch_shape": list(expected.shape),
        "onnx_shape": list(actual.shape),
    }


def run_round(
    labeled,
    unlabeled,
    out_dir,
    steps,
    batch=4,
    patch=256,
    lr=2e-3,
    seed=0,
    log=print,
    holdout=None,
    init_checkpoint=None,
    allow_random_init=False,
    accepted_reference=None,
    calibration_pages=None,
    conformal_alpha=conformal.DEFAULT_ALPHA,
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if steps < 3:
        raise ValueError("steps >= 3 é necessário para as médias first3/last3")
    holdout_ids = _guard_lineage(labeled, unlabeled, holdout or [])
    if init_checkpoint is None and not allow_random_init:
        raise ValueError(
            "rodada fria recusada: use init_checkpoint ou allow_random_init explícito"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")
    student = UNet().to(device)
    if init_checkpoint is not None:
        previous_holdout_ids = _load_student_checkpoint(
            student, init_checkpoint, device
        )
        holdout_ids = _guard_lineage(labeled, unlabeled, holdout, previous_holdout_ids)
        log(f"warm start: {init_checkpoint}")
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    ema_init = {k: v.detach().clone() for k, v in teacher.state_dict().items()}

    calibration_sessions = _guard_calibration(
        calibration_pages or [], labeled, set(holdout_ids), unlabeled
    )
    calibration = None
    if calibration_pages:
        # Calibra no teacher inicial (o incumbente): o limiar vem de
        # rótulo independente, não da confiança do próprio gerador.
        calibration = conformal.calibrate(
            (
                (predict_page_probability(teacher, page, device, patch), page.mask)
                for page in calibration_pages
            ),
            alpha=conformal_alpha,
        )
        calibration["sessions"] = calibration_sessions
        log(f"conformal calibrado em {len(calibration_sessions)} sessões")
    accepted_profile = drift.AcceptedProfile()
    opt = torch.optim.Adam(student.parameters(), lr=lr)

    sup_history, cons_history, coverage_history, ink_history = [], [], [], []
    decay_history = []
    t0 = time.time()
    for step in range(steps):
        imgs, targets, valids = [], [], []
        for _ in range(batch):
            page = labeled[int(rng.integers(0, len(labeled)))]
            y, x, size = sample_patch(page.image, patch, rng)
            img_p = page.image[y : y + size, x : x + size]
            mask_p = page.mask[y : y + size, x : x + size]
            flip = rng.random() < 0.5
            dx = int(rng.integers(-3, 4))
            dy = int(rng.integers(-3, 4))
            img_g, _, _, _ = geo_view(img_p, rng, flip, dx, dy)
            mask_g, _, _, _ = geo_view(mask_p, rng, flip, dx, dy, fill=0.0)
            imgs.append(to_tensor(np.ascontiguousarray(img_g)))
            targets.append(torch.from_numpy(np.ascontiguousarray(mask_g))[None])
            valids.append(torch.from_numpy(geo_valid_mask(mask_p.shape, dy, dx))[None])
        x_lab = torch.stack(imgs).to(device)
        y_lab = torch.stack(targets).to(device)
        v_lab = torch.stack(valids).to(device)
        loss_sup = supervised_loss(student(x_lab), y_lab, v_lab)

        loss = loss_sup
        loss_cons_value = None
        coverage_value = None
        ink_value = None
        if unlabeled:
            weak, strong, weak_np = [], [], []
            for _ in range(batch):
                page = unlabeled[int(rng.integers(0, len(unlabeled)))]
                y, x, size = sample_patch(page.image, patch, rng)
                img_g, _, _, _ = geo_view(page.image[y : y + size, x : x + size], rng)
                weak.append(to_tensor(np.ascontiguousarray(img_g)))
                weak_np.append(np.ascontiguousarray(img_g))
                strong.append(to_tensor(strong_view(img_g, rng)))
            x_weak = torch.stack(weak).to(device)
            x_strong = torch.stack(strong).to(device)
            with torch.no_grad():
                probs = torch.sigmoid(teacher(x_weak))[:, 0].cpu().numpy()
            targets_u, weights_u = [], []
            for i in range(batch):
                t_i, w_i = consistency_targets(probs[i], weak_np[i], calibration)
                accepted_profile.add_item(probs[i], t_i, w_i)
                targets_u.append(torch.from_numpy(t_i)[None])
                weights_u.append(torch.from_numpy(w_i)[None])
            target_u = torch.stack(targets_u).to(device)
            weight_u = torch.stack(weights_u).to(device)
            logits_s = student(x_strong)
            bce = F.binary_cross_entropy_with_logits(
                logits_s, target_u, reduction="none"
            )
            loss_cons = (bce * weight_u).sum() / weight_u.sum().clamp(min=1)
            loss = loss + CONS_WEIGHT * loss_cons
            loss_cons_value = float(loss_cons.detach())
            coverage_value = float(weight_u.mean().detach())
            ink_value = float((target_u * weight_u).sum().detach()) / max(
                float(weight_u.sum().detach()), 1.0
            )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        decay = ema_decay_at_step(step)
        with torch.no_grad():
            for p_t, p_s in zip(teacher.parameters(), student.parameters()):
                p_t.mul_(decay).add_(p_s.detach(), alpha=1 - decay)
            for b_t, b_s in zip(teacher.buffers(), student.buffers()):
                b_t.copy_(b_s)
        sup_history.append(float(loss_sup.detach()))
        if loss_cons_value is not None:
            cons_history.append(loss_cons_value)
            coverage_history.append(coverage_value)
            ink_history.append(ink_value)
        decay_history.append(decay)
        if step % 10 == 0 or step == steps - 1:
            cons_log = "n/a" if loss_cons_value is None else f"{loss_cons_value:.4f}"
            cov_log = "n/a" if coverage_value is None else f"{coverage_value:.3f}"
            log(
                f"step {step}: sup {sup_history[-1]:.4f} "
                f"cons {cons_log} coverage {cov_log} ema {decay:.4f}"
            )

    ema_changed = any(
        not torch.equal(ema_init[k], v)
        for k, v in teacher.state_dict().items()
        if v.dtype.is_floating_point
    )

    # O holdout só aparece aqui: nunca seleciona checkpoint, threshold ou EMA.
    holdout_metrics = evaluate_holdout(
        student, holdout, device, patch=patch, threshold=EVAL_THRESHOLD
    )
    student.eval()
    student.to("cpu")
    checkpoint_path = out_dir / "round_checkpoint.pt"
    torch.save(
        {
            "student": student.state_dict(),
            "seed": seed,
            "steps": steps,
            "patch": patch,
            "input_color_space": "BGR",
            "holdout_sessions_not_for_training": holdout_ids,
        },
        checkpoint_path,
    )

    onnx_path = out_dir / "page_clean_round.onnx"
    dummy = torch.zeros(1, 3, 256, 256)
    try:
        torch.onnx.export(
            student,
            dummy,
            str(onnx_path),
            input_names=["image"],
            output_names=["logits"],
            opset_version=17,
            dynamo=False,
            dynamic_axes={"image": {0: "n"}, "logits": {0: "n"}},
        )
        onnx_check = validate_round_onnx(student, onnx_path, dummy)
    except Exception as exc:  # noqa: BLE001 - preservar checkpoint e métricas
        onnx_check = {"status": "failed", "error": str(exc)}
    delivery_status = (
        "validated" if onnx_check["status"] == "ok" else "blocked_onnx_validation"
    )
    profile_dict = accepted_profile.to_dict()
    (out_dir / "accepted_profile.json").write_text(
        json.dumps(profile_dict), encoding="utf-8"
    )
    if accepted_reference is None:
        drift_report = drift.no_reference_report()
    else:
        drift_report = drift.compare_profiles(accepted_reference, profile_dict)
    (out_dir / "accepted_drift.json").write_text(
        json.dumps(drift_report, indent=2), encoding="utf-8"
    )
    log(drift.format_report(drift_report))
    metrics = {
        "steps": steps,
        "sup_loss_first3": _mean_first3(sup_history),
        "sup_loss_last3": _mean3(sup_history),
        "cons_loss_first3": _mean_first3(cons_history),
        "cons_loss_last3": _mean3(cons_history),
        "pseudo_label_coverage_first3": _mean_first3(coverage_history),
        "pseudo_label_coverage_last3": _mean3(coverage_history),
        "pseudo_label_ink_fraction_first3": _mean_first3(ink_history),
        "pseudo_label_ink_fraction_last3": _mean3(ink_history),
        "semi_supervised_effect_note": "comparar com uma rodada supervisionada equivalente; cobertura/loss não provam ganho causal",
        "ema_changed": bool(ema_changed),
        "ema_decay_base": EMA_DECAY,
        "ema_decay_final": decay_history[-1],
        "ema_warmup_steps": EMA_WARMUP_STEPS,
        "n_labeled": len(labeled),
        "n_unlabeled": len(unlabeled),
        "labeled_sessions": sorted({page.session_id for page in labeled}),
        "unlabeled_sessions": sorted({page.session_id for page in unlabeled}),
        "sealed_holdout_sessions": holdout_ids,
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        "random_init": init_checkpoint is None,
        "input_color_space": "BGR",
        "holdout": holdout_metrics,
        "candidate_real": holdout_metrics["candidate_real"],
        "checkpoint": checkpoint_path.name,
        "onnx": onnx_path.name,
        "onnx_validation": onnx_check,
        "delivery_status": delivery_status,
        "promotion_status": "not_attempted_human_gate_required",
        "accepted_profile": "accepted_profile.json",
        "accepted_drift": {
            "status": drift_report["status"],
            "report": "accepted_drift.json",
            "role": "signal_to_judgment",
        },
        "conformal": calibration,
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (out_dir / "candidate_real.json").write_text(
        json.dumps(holdout_metrics["candidate_real"], indent=2), encoding="utf-8"
    )
    return metrics


def make_synthetic(root, n_labeled=24, n_unlabeled=16, n_holdout=8, size=64, seed=1):
    """Dataset mínimo com sessões e manifestos para exercitar os guards."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    splits = (
        ("labeled", n_labeled, True),
        ("unlabeled", n_unlabeled, False),
        ("holdout", n_holdout, True),
    )
    for split, count, with_mask in splits:
        (root / split / "input").mkdir(parents=True, exist_ok=True)
        if with_mask:
            (root / split / "mask").mkdir(parents=True, exist_ok=True)
        manifest = []
        for i in range(count):
            page = np.full((size, size, 3), 235, dtype=np.uint8)
            page = page + rng.integers(-6, 7, page.shape).astype(np.int16)
            page = np.clip(page, 0, 255).astype(np.uint8)
            for _ in range(rng.integers(2, 5)):
                y = int(rng.integers(6, size - 6))
                x0 = int(rng.integers(4, size // 2))
                x1 = int(rng.integers(size // 2, size - 4))
                thick = int(rng.integers(1, 3))
                cv2.line(
                    page,
                    (x0, y),
                    (x1, y + int(rng.integers(-4, 5))),
                    (30, 30, 30),
                    thick,
                )
            noise = page.astype(np.int16) + rng.integers(-8, 9, page.shape).astype(
                np.int16
            )
            page = np.clip(noise, 0, 255).astype(np.uint8)
            mask = (cv2.cvtColor(page, cv2.COLOR_BGR2GRAY) < 120).astype(np.uint8) * 255
            name = f"p{i:03d}"
            cv2.imwrite(str(root / split / "input" / f"{name}.png"), page)
            if with_mask:
                cv2.imwrite(str(root / split / "mask" / f"{name}.png"), mask)
            if split == "holdout":
                row = {
                    "session_id": f"holdout-s{i % 5}",
                    "origin": "independent_label",
                    "split": "holdout",
                    "label_source": "final_minus_reference",
                    "label_reviewed": True,
                }
            elif split == "labeled":
                row = {
                    "session_id": f"train-s{i % 4}",
                    "origin": "human_correction",
                    "split": "train",
                }
            else:
                row = {
                    "session_id": f"unlabeled-s{i % 4}",
                    "origin": "collected_unlabeled",
                    "split": "train",
                }
            manifest.append({"file": name, **row})
        (root / split / "manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8"
        )
    return root


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--labeled-dir")
    ap.add_argument("--unlabeled-dir")
    ap.add_argument("--holdout-dir")
    ap.add_argument("--init-checkpoint")
    ap.add_argument("--allow-random-init", action="store_true")
    ap.add_argument(
        "--accepted-reference",
        help="accepted_profile.json da rodada de referência (monitor PSI/KS)",
    )
    ap.add_argument(
        "--calibration-dir",
        help="páginas com rótulo independente, fora do treino e do selado, "
        "para o conformal por classe",
    )
    ap.add_argument("--conformal-alpha", type=float, default=conformal.DEFAULT_ALPHA)
    ap.add_argument("--out", default="selflearn_round_out")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps-per-epoch", type=int, default=50)
    args = ap.parse_args(argv)

    if args.smoke:
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="selflearn_smoke_"))
        data = make_synthetic(tmp / "data")
        holdout = load_pages(data / "holdout", with_mask=True, holdout=True)
        holdout_ids = [page.session_id for page in holdout]
        labeled = load_pages(
            data / "labeled", with_mask=True, holdout_session_ids=holdout_ids
        )
        unlabeled = load_pages(
            data / "unlabeled", with_mask=False, holdout_session_ids=holdout_ids
        )
        metrics = run_round(
            labeled,
            unlabeled,
            tmp / "out",
            steps=60,
            batch=4,
            patch=64,
            seed=0,
            holdout=holdout,
            allow_random_init=True,
        )
        print(json.dumps(metrics, indent=2))
        assert metrics["sup_loss_last3"] < metrics["sup_loss_first3"], (
            "perda supervisionada não caiu no smoke"
        )
        assert metrics["ema_changed"], "pesos EMA não se moveram"
        assert metrics["holdout"]["n_pages"] == 8
        assert metrics["pseudo_label_coverage_last3"] is not None
        assert (tmp / "out" / "page_clean_round.onnx").exists()
        assert metrics["delivery_status"] == "validated", metrics["onnx_validation"]
        print("PASS")
        return 0

    if not args.labeled_dir or not args.holdout_dir:
        ap.error("--labeled-dir e --holdout-dir são obrigatórios fora do smoke")
    if not args.init_checkpoint and not args.allow_random_init:
        ap.error("use --init-checkpoint ou autorize --allow-random-init explicitamente")
    holdout = load_pages(args.holdout_dir, with_mask=True, holdout=True)
    holdout_ids = [page.session_id for page in holdout]
    labeled = load_pages(
        args.labeled_dir, with_mask=True, holdout_session_ids=holdout_ids
    )
    unlabeled = []
    if args.unlabeled_dir:
        unlabeled = load_pages(
            args.unlabeled_dir, with_mask=False, holdout_session_ids=holdout_ids
        )
    calibration_pages = None
    if args.calibration_dir:
        calibration_pages = load_pages(
            args.calibration_dir, with_mask=True, holdout_session_ids=holdout_ids
        )
    metrics = run_round(
        labeled,
        unlabeled,
        args.out,
        steps=args.epochs * args.steps_per_epoch,
        holdout=holdout,
        init_checkpoint=args.init_checkpoint,
        allow_random_init=args.allow_random_init,
        accepted_reference=args.accepted_reference,
        calibration_pages=calibration_pages,
        conformal_alpha=args.conformal_alpha,
    )
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["delivery_status"] == "validated" else 3


if __name__ == "__main__":
    sys.exit(main())
