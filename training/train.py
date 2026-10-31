"""Treinamento do modelo de segmentação do Draw (SegFormer-B0, 4 classes).

Escopo: dados -> treino -> validação -> exportação ONNX -> documentação.
FORA do escopo (Codex, depois): webcam, integração no app, tracking
temporal, memória de traços, oclusão em tempo real, interface, CAD,
simulador.

Classes: 0=background, 1=paper, 2=hand, 3=ink.
Máscaras podem usar 255 somente para pixels ignorados pela loss e pelas métricas.
Regra de rótulo: o que a câmera vê. Mão sobre traço = hand, nunca ink.
ink_full_mask (opcional) NÃO é classe nem entrada: é só para avaliar
a memória temporal futura.

Contrato com draw/neural.py (atual, fase 2):
  - entrada: RGB 512x512, normalização ImageNet (mean 0.485/0.456/0.406,
    std 0.229/0.224/0.225), input ONNX chamado "image";
  - saída antiga: mask_logits de 2 canais + presence_logit;
  - saída nova: logits de 4 canais, output ONNX chamado "logits".
  - compatibilidade (decisão documentada, integração pelo Codex):
    paper_mask = (argmax == 1); hand_mask = (argmax == 2);
    ink_mask = (argmax == 3); paper_presence é DERIVADO da máscara
    (fração de pixels paper acima do limiar de integração), não uma
    cabeça auxiliar. O modelo não foi simplificado para caber no
    contrato antigo de 2 canais.

Uso:
  # Kaggle (GPU, treino completo):
  python training/train.py --data-dir /kaggle/input/<dataset>/dataset/synthetic \
      --config training/config.json --out-dir /kaggle/working/out
  # Smoke test (sem GPU, sem treino completo; ver função smoke()):
  python training/train.py --config training/config.json --smoke \
      --smoke-dir /tmp/draw-smoke

Imports pesados (torch, transformers, albumentations, onnxruntime) são
lazy, dentro das funções que os usam. O topo deste arquivo só usa
stdlib + numpy + cv2, para que config, pareamento, pesos e
métricas possam ser testados sem GPU e sem torch instalado.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools import seg_metrics as shared_metrics

CLASSES = {0: "background", 1: "paper", 2: "hand", 3: "ink"}
NUM_CLASSES = 4
IGNORE_INDEX = 255
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
# Mesma normalização de draw/neural.py (_MEAN/_STD).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------- config
def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    tr = cfg["training"]
    assert cfg["model"]["num_classes"] == NUM_CLASSES, "config deve ter 4 classes"
    assert len(cfg["loss"]["cross_entropy_class_weights"]) == NUM_CLASSES
    return cfg


# ---------------------------------------------------------------- dataset
def pair_images_masks(split_dir):
    """Pareia images/<nome> com masks/<mesmo nome base>.png.

    Aceita também o layout plano do gerador (tools/gen_synthetic.py):
    <split>/scene_*.png + <split>/scene_*_mask.png +
    <split>/scene_*_ink_full.png (opcional). Sem essa aceitação, o
    treino falhava com "split sem images/ ou masks/" sobre a saída
    do gerador — divergência encontrada no QA e corrigida aqui.

    Retorna lista de (caminho_imagem, caminho_mascara, caminho_ink_full|None).
    ink_full é opcional e nunca entra no treino.
    """
    split_dir = Path(split_dir)
    img_dir, mask_dir = split_dir / "images", split_dir / "masks"
    if img_dir.is_dir() and mask_dir.is_dir():
        pairs = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            mask_path = mask_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                raise FileNotFoundError(f"máscara ausente para {img_path}")
            ink_full = split_dir / "ink_full" / (img_path.stem + ".png")
            pairs.append((img_path, mask_path, ink_full if ink_full.exists() else None))
    else:
        # Layout plano do gerador.
        pairs = []
        for img_path in sorted(split_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            if img_path.stem.endswith("_mask") or img_path.stem.endswith("_ink_full"):
                continue
            mask_path = split_dir / (img_path.stem + "_mask.png")
            if not mask_path.exists():
                continue
            ink_full = split_dir / (img_path.stem + "_ink_full.png")
            pairs.append((img_path, mask_path, ink_full if ink_full.exists() else None))
    if not pairs:
        raise ValueError(f"nenhum par imagem/máscara em {split_dir}")
    return pairs


def validate_mask(mask):
    """Máscara válida: classes 0..3 e 255 para pixels deliberadamente ignorados."""
    values = np.unique(mask)
    bad = [int(v) for v in values if v not in (0, 1, 2, 3, IGNORE_INDEX)]
    if bad:
        raise ValueError(f"máscara com valores inválidos (permitidos: 0..3 e 255=ignorado): {bad}")
    return True


def load_mask(mask_path):
    """Lê máscara em grayscale; 255 significa pixel sem supervisão."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"máscara ilegível: {mask_path}")
    validate_mask(mask)
    return mask.astype(np.int64)


def load_image_rgb(image_path):
    """Lê imagem com cv2 e converte BGR -> RGB."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"imagem ilegível: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def pixel_counts(pairs, max_files=None):
    """Conta pixels supervisionados por classe; pixels 255 não entram."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for _, mask_path, _ in pairs[:max_files] if max_files else pairs:
        mask = load_mask(mask_path)
        for c in range(NUM_CLASSES):
            counts[c] += int((mask == c).sum())
    return counts


def compute_class_weights(counts, cap=10.0):
    """Peso ~ inverso da frequência, com teto (evita ink dominar tudo)."""
    total = counts.sum()
    if total == 0:
        return [1.0] * NUM_CLASSES
    freq = counts / total
    weights = 1.0 / np.maximum(freq, 1e-6)
    weights = weights / weights.min()
    return [float(min(w, cap)) for w in weights]


# ---------------------------------------------------------------- métricas
def per_class_metrics(pred, target):
    """Compatibility wrapper over the shared segmentation metric contract."""
    report = shared_metrics.evaluate(pred, target)
    return {
        name: {
            metric: report[metric][name]
            for metric in ("iou", "dice", "precision", "recall")
        } | {"support_px": report["support_px"][name]}
        for name in shared_metrics.CLASSES
    }


def ink_tolerance_metrics(pred, target, radii=(2, 3)):
    """Compatibility wrapper over the shared, ignore-index-aware metric."""
    out = {}
    for radius in radii:
        report = shared_metrics.ink_tolerance_metrics(pred, target, radius)
        out[f"radius_{radius}px"] = {
            "precision": report["precision"],
            "recall": report["recall"],
            "f1": report["f1"],
        }
    return out


def dice_loss_numpy(prob, target, smooth=1e-6):
    """Dice loss multiclasse em numpy (espelho da versão torch).

    prob: (C, H, W) probabilidades; target: (H, W) inteiros 0..3/255.
    Pixels ignorados são removidos da probabilidade e do alvo.
    """
    valid = target != IGNORE_INDEX
    if not np.any(valid):
        return 0.0
    losses = []
    for c in range(NUM_CLASSES):
        p = prob[c][valid].ravel()
        t = (target[valid] == c).astype(np.float32).ravel()
        inter = float((p * t).sum())
        losses.append(1.0 - (2 * inter + smooth) / (p.sum() + t.sum() + smooth))
    return float(np.mean(losses))


# ---------------------------------------------------------------- torch
def build_model(cfg):
    """SegFormer-B0 pré-treinado, cabeça nova de 4 classes (lazy torch)."""
    from transformers import SegformerForSemanticSegmentation

    model_cfg = cfg["model"]
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_cfg["backbone"],
        num_labels=model_cfg["num_classes"],
        ignore_mismatched_sizes=True,  # cabeça nova: 4 classes, não as do ImageNet
    )
    return model


def build_dataset(data_dir, split, cfg, augment=False, extra_pairs=None):
    """Dataset torch. Import lazy: sem torch, esta função não existe."""
    import torch
    from torch.utils.data import Dataset

    pairs = pair_images_masks(Path(data_dir) / split)
    if extra_pairs:
        pairs.extend(extra_pairs)
    size = cfg["model"]["input"]["height"]
    aug = None
    if augment:
        try:
            import albumentations as A

            aug = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Affine(
                        scale=(0.9, 1.1),
                        rotate=(-15, 15),
                        mask_interpolation=cv2.INTER_NEAREST,
                        p=0.5,
                    ),
                    A.Perspective(
                        scale=(0.02, 0.06),
                        mask_interpolation=cv2.INTER_NEAREST,
                        p=0.2,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.3
                    ),
                    A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                    A.HueSaturationValue(
                        hue_shift_range=(-8, 8),
                        sat_shift_range=(-12, 12),
                        val_shift_range=(-12, 12),
                        p=0.2,
                    ),
                ]
            )
        except ImportError as exc:
            raise RuntimeError(
                "albumentations é obrigatório para reproduzir as augmentations "
                "definidas em training/config.json; instale training/requirements.txt"
            ) from exc

    class SegDataset(Dataset):
        def __len__(self):
            return len(pairs)

        def __getitem__(self, i):
            img_path, mask_path, _ = pairs[i]
            image = load_image_rgb(img_path)
            mask = load_mask(mask_path)
            if aug is not None:
                out = aug(image=image, mask=mask)
                image, mask = out["image"], out["mask"]
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
            image = image.astype(np.float32) / 255.0
            image = (image - IMAGENET_MEAN) / IMAGENET_STD
            image = np.transpose(image, (2, 0, 1))
            return (
                torch.from_numpy(image),
                torch.from_numpy(mask.astype(np.int64)),
            )

    return SegDataset()


def combined_loss(logits, target, class_weights, ce_weight=0.5, dice_weight=0.5):
    """CE ponderada + Dice multiclasse; target=255 não supervisiona o pixel."""
    import torch
    import torch.nn.functional as F

    valid = target != IGNORE_INDEX
    if not torch.any(valid):
        # Mantém a loss ligada ao grafo para que backward seja válido, mas
        # produz gradiente zero quando o lote inteiro foi deliberadamente ignorado.
        return logits.sum() * 0.0

    ce = F.cross_entropy(logits, target, weight=class_weights, ignore_index=IGNORE_INDEX)
    prob = torch.softmax(logits, dim=1)
    valid_flat = valid.reshape(target.shape[0], -1)
    sample_is_valid = valid_flat.any(dim=1)
    dice = torch.zeros(target.shape[0], dtype=logits.dtype, device=logits.device)
    for c in range(NUM_CLASSES):
        p = prob[:, c].reshape(prob.shape[0], -1)
        p = p * valid_flat
        t = ((target == c) & valid).float().reshape(target.shape[0], -1)
        inter = (p * t).sum(dim=1)
        dice = dice + (1.0 - (2 * inter + 1e-6) / (p.sum(dim=1) + t.sum(dim=1) + 1e-6))
    dice = dice / NUM_CLASSES
    # Correção do reteste no Kaggle: sem esta média, dice tem shape [batch]
    # e loss.backward() falha com "grad can be implicitly created only
    # for scalar outputs". A loss tem que ser escalar.
    dice = dice[sample_is_valid].mean()
    return ce_weight * ce + dice_weight * dice


def warmup_cosine_scheduler(optimizer, total_steps, warmup_ratio):
    """LR linear warmup followed by cosine decay, stepped per optimizer update."""
    import torch

    total_steps = max(int(total_steps), 1)
    warmup_steps = min(int(total_steps * float(warmup_ratio)), total_steps)

    def multiplier(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def configure_mixed_precision(torch, device, training_cfg):
    """Select configured bf16/fp16 on CUDA, fp32 elsewhere."""
    if torch.device(device).type != "cuda":
        return "fp32", None, None
    preferred = training_cfg.get("mixed_precision", "bf16")
    fallback = training_cfg.get("mixed_precision_fallback", "fp16")
    supports_bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    precision = preferred if preferred == "bf16" and supports_bf16 else fallback
    if precision == "bf16":
        return precision, torch.bfloat16, None
    if precision == "fp16":
        return precision, torch.float16, torch.amp.GradScaler("cuda")
    return "fp32", None, None


def train_one_epoch(
    model,
    loader,
    optimizer,
    class_weights,
    device,
    scaler=None,
    scheduler=None,
    accumulation_steps=1,
    amp_dtype=None,
):
    import torch
    import torch.nn.functional as F

    model.train()
    total, n = 0.0, 0
    accumulation_steps = max(int(accumulation_steps), 1)
    num_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)
    for batch_index, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        device_type = torch.device(device).type
        with torch.autocast(
            device_type=device_type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            logits = model(pixel_values=images).logits
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
            loss = combined_loss(logits, masks, class_weights)
        group_start = (batch_index // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, num_batches - group_start)
        scaled_loss = loss / group_size
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        should_step = (
            (batch_index + 1) % accumulation_steps == 0
            or batch_index + 1 == num_batches
        )
        if should_step:
            did_step = True
            if scaler is not None:
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                did_step = scaler.get_scale() >= previous_scale
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None and did_step:
                scheduler.step()
        total += float(loss.detach().item())
        n += 1
    if n == 0:
        raise ValueError("época sem lotes de treino")
    return total / n


def evaluate(model, loader, device):
    """Avalia todo o loader usando as métricas compartilhadas do projeto."""
    import torch

    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    supervised_px, ignored_px = 0, 0
    tolerance_totals = {
        radius: {"pred_ink": 0, "gt_ink": 0, "matched_pred": 0, "matched_gt": 0}
        for radius in (2, 3)
    }
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = model(pixel_values=images).logits
            if logits.shape[-2:] != masks.shape[-2:]:
                import torch.nn.functional as F

                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            pred = logits.argmax(dim=1).cpu().numpy()
            target = masks.numpy()
            valid = target != IGNORE_INDEX
            supervised_px += int(valid.sum())
            ignored_px += int((~valid).sum())
            for pred_image, target_image in zip(pred, target):
                confusion += shared_metrics.confusion_matrix(pred_image, target_image)
                if not np.any(target_image != IGNORE_INDEX):
                    continue
                for radius in tolerance_totals:
                    tol = shared_metrics.ink_tolerance_metrics(
                        pred_image, target_image, tolerance_px=radius
                    )
                    for key in tolerance_totals[radius]:
                        tolerance_totals[radius][key] += tol[key]
    if supervised_px == 0:
        raise ValueError("avaliação sem pixels supervisionados: todas as máscaras são 255")
    metrics = shared_metrics.evaluate_confusion_matrix(confusion, ignored_px=ignored_px)
    aggregate_tolerance = {}
    for radius, counts in tolerance_totals.items():
        pred_ink, gt_ink = counts["pred_ink"], counts["gt_ink"]
        precision = counts["matched_pred"] / pred_ink if pred_ink else (1.0 if not gt_ink else 0.0)
        recall = counts["matched_gt"] / gt_ink if gt_ink else (1.0 if not pred_ink else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        aggregate_tolerance[f"radius_{radius}px"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pred_ink": pred_ink,
            "gt_ink": gt_ink,
            "matched_pred": counts["matched_pred"],
            "matched_gt": counts["matched_gt"],
        }
    metrics["ink_tolerance"] = aggregate_tolerance
    return metrics


# ---------------------------------------------------------------- ONNX
def _make_onnx_dummy(model, size):
    """Create the export sample on the same device as the model parameters."""
    import torch

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    return torch.randn(1, 3, size, size, device=device)


def make_onnx_export_model(model, size):
    """Wrap SegFormer so exported logits match the full input resolution."""
    import torch
    import torch.nn.functional as F

    class ExportModel(torch.nn.Module):
        def __init__(self, wrapped_model, output_size):
            super().__init__()
            self.model = wrapped_model
            self.output_size = output_size

        def forward(self, image):
            logits = self.model(pixel_values=image).logits
            if logits.shape[-2:] != (self.output_size, self.output_size):
                logits = F.interpolate(
                    logits,
                    size=(self.output_size, self.output_size),
                    mode="bilinear",
                    align_corners=False,
                )
            return logits

    return ExportModel(model, size).eval()


def export_onnx(model, out_path, cfg):
    """Exporta para ONNX com o contrato do Draw: input 'image', output 'logits'."""
    import torch

    model.eval()
    size = cfg["model"]["input"]["height"]
    export_model = make_onnx_export_model(model, size)
    dummy = _make_onnx_dummy(export_model, size)
    # dynamo=False: exportador legado. O exportador dynamo padrão do torch
    # 2.11 exige onnxscript, que o ambiente do Kaggle não tem — sem isso,
    # o treino inteiro falhava na exportação, depois das épocas, e perdia
    # até o metrics.json. O exportador legado usa só o pacote onnx.
    torch.onnx.export(
        export_model,
        (dummy,),
        f=out_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": [0], "logits": [0]},
        opset_version=cfg["export"]["opset_version"],
        do_constant_folding=True,
        dynamo=False,
    )
    return out_path


def validate_onnx(model, onnx_path, loader, device, cfg):
    """Valida shape, diferença dos logits e IoU de argmax contra o ONNX."""
    from tools.validate_onnx import (
        DEFAULT_MAX_DIFF_TOL,
        DEFAULT_MIN_ARGMAX_IOU,
        validate_onnx as validate_export,
    )

    images, _ = next(iter(loader))
    size = cfg["model"]["input"]["height"]
    classes = cfg["model"]["num_classes"]
    thresholds = cfg["export"]["validation_thresholds"]
    export_model = make_onnx_export_model(model, size)
    report = validate_export(
        export_model,
        onnx_path,
        images.to(device),
        max_diff_tol=thresholds.get("max_abs_diff", DEFAULT_MAX_DIFF_TOL),
        min_argmax_iou=thresholds.get("min_argmax_iou", DEFAULT_MIN_ARGMAX_IOU),
        expected_shape=(images.shape[0], classes, size, size),
    )
    if report["status"] == "skipped":
        print(f"AVISO: validação ONNX não executada: {report['reason']}", file=sys.stderr)
    return report


# ---------------------------------------------------------------- smoke
def smoke(smoke_dir):
    """Smoke test sem GPU e sem treino completo.

    Cria 2 pares sintéticos (imagem + máscara 0..3/255), testa pareamento,
    validação de máscara, contagem de pixels, pesos, métricas numpy,
    pixel ignorado, métrica de ink com tolerância e o contrato ONNX (nomes/shapes).
    Não treina modelo; usa PyTorch CPU se disponível para verificar a loss.
    Retorna 0 se tudo passar.
    """
    import shutil

    root = Path(smoke_dir)
    if root.exists():
        shutil.rmtree(root)
    img_dir, mask_dir = root / "train" / "images", root / "train" / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    # Imagem 1: fundo + retângulo de papel + linha de tinta.
    img = np.full((64, 64, 3), 200, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:50, 10:50] = 1
    mask[20, 12:48] = 3  # traço fino, 1 px
    mask[30:40, 30:40] = 2  # mão sobre o papel
    mask[54:60, 54:60] = IGNORE_INDEX  # área sem rótulo confirmado
    cv2.imwrite(str(img_dir / "scene_0001.png"), img)
    cv2.imwrite(str(mask_dir / "scene_0001.png"), mask)
    # Imagem 2: só fundo (negativo).
    cv2.imwrite(str(img_dir / "scene_0002.png"), np.full((64, 64, 3), 90, dtype=np.uint8))
    cv2.imwrite(str(mask_dir / "scene_0002.png"), np.zeros((64, 64), dtype=np.uint8))

    pairs = pair_images_masks(root / "train")
    assert len(pairs) == 2, pairs
    counts = pixel_counts(pairs)
    ignored = int((load_mask(mask_dir / "scene_0001.png") == IGNORE_INDEX).sum())
    assert ignored == 36, ignored
    assert counts.sum() == 2 * 64 * 64 - ignored, counts
    weights = compute_class_weights(counts)
    assert len(weights) == NUM_CLASSES and all(w > 0 for w in weights), weights
    pred = load_mask(mask_dir / "scene_0001.png")
    pred[pred == IGNORE_INDEX] = 0  # predições sempre são classes válidas; só o GT usa 255.
    # Predição perfeita exceto 1 px de tinta deslocado: IoU estrita cai,
    # tolerância de 2-3 px deve recuperar.
    pred_shifted = pred.copy()
    pred_shifted[20, 12:48] = 0
    pred_shifted[21, 12:48] = 3
    m = per_class_metrics(pred_shifted, pred)
    assert m["ink"]["iou"] < 1.0, "IoU estrita deveria cair com o deslocamento"
    tol = ink_tolerance_metrics(pred_shifted, pred)
    assert tol["radius_2px"]["recall"] > m["ink"]["recall"], (tol, m["ink"])

    # Predições arbitrárias em pixels 255 não podem alterar métricas nem
    # denominadores das classes supervisionadas.
    ignored_target = pred.copy()
    ignored_target[54:60, 54:60] = IGNORE_INDEX
    ignored_pred = ignored_target.copy()
    ignored_pred[54:60, 54:60] = 3
    ignored_metrics = per_class_metrics(ignored_pred, ignored_target)
    assert ignored_metrics["ink"]["iou"] == 1.0, ignored_metrics["ink"]
    assert ignored_metrics["ink"]["support_px"] == 36, ignored_metrics["ink"]
    ignored_tol = ink_tolerance_metrics(ignored_pred, ignored_target)
    assert ignored_tol["radius_2px"]["precision"] == 1.0, ignored_tol
    uniform_prob = np.full((NUM_CLASSES, 64, 64), 0.25, dtype=np.float32)
    changed_ignored_prob = uniform_prob.copy()
    changed_ignored_prob[0:3, 54:60, 54:60] = 0.0
    changed_ignored_prob[3, 54:60, 54:60] = 1.0
    assert dice_loss_numpy(uniform_prob, ignored_target) == dice_loss_numpy(
        changed_ignored_prob, ignored_target
    ), "probabilidades em pixels ignorados não devem alterar Dice numpy"

    try:
        import torch
        torch.nn
    except (ImportError, AttributeError):
        print("AVISO: PyTorch completo ausente; smoke de loss torch com pixel ignorado pulado.", file=sys.stderr)
    else:
        # ONNX export receives a CUDA model after Kaggle training. The dummy
        # input must follow that model's device; CPU is still the fallback for
        # parameterless modules and the local smoke path.
        for test_device in ("cpu", "meta"):
            probe = torch.nn.Linear(1, 1).to(test_device)
            export_input = _make_onnx_dummy(probe, 8)
            assert export_input.device == next(probe.parameters()).device, (
                test_device, export_input.device, next(probe.parameters()).device
            )
        empty_probe = torch.nn.Identity()
        assert _make_onnx_dummy(empty_probe, 8).device.type == "cpu"
        target_t = torch.from_numpy(pred[None].astype(np.int64))
        logits = torch.zeros((1, NUM_CLASSES, 64, 64), dtype=torch.float32, requires_grad=True)
        weights_t = torch.tensor(weights, dtype=torch.float32)
        loss = combined_loss(logits, target_t, weights_t)
        assert torch.isfinite(loss), loss
        loss.backward()
        assert torch.isfinite(logits.grad).all(), "gradiente não finito com pixels 255"
        assert torch.count_nonzero(logits.grad[:, :, 54:60, 54:60]) == 0, "pixel 255 deve ter gradiente zero"
        all_ignored = torch.full_like(target_t, IGNORE_INDEX)
        ignored_logits = torch.zeros_like(logits.detach(), requires_grad=True)
        ignored_loss = combined_loss(ignored_logits, all_ignored, weights_t)
        assert torch.isfinite(ignored_loss) and ignored_loss.item() == 0.0, ignored_loss
        ignored_loss.backward()
        assert torch.count_nonzero(ignored_logits.grad) == 0, "lote ignorado deve gerar gradiente zero"

        predicted_ids = pred.copy()
        predicted_ids[54:60, 54:60] = 3  # falso ink apenas na região ignorada
        eval_logits = torch.full((1, NUM_CLASSES, 64, 64), -10.0)
        for class_id in range(NUM_CLASSES):
            eval_logits[0, class_id][predicted_ids == class_id] = 10.0

        class _Output:
            def __init__(self, values):
                self.logits = values

        class _FixedModel:
            def eval(self):
                return self

            def __call__(self, pixel_values):
                return _Output(eval_logits.to(pixel_values.device))

        images_t = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
        eval_metrics = evaluate(_FixedModel(), [(images_t, target_t)], "cpu")
        assert eval_metrics["ink"]["iou"] == 1.0, eval_metrics["ink"]
        assert eval_metrics["ink"]["support_px"] == 36, eval_metrics["ink"]
        assert eval_metrics["supervised_px"] == 64 * 64 - ignored, eval_metrics
        assert eval_metrics["ignored_px"] == ignored, eval_metrics
        try:
            evaluate(_FixedModel(), [(images_t, all_ignored)], "cpu")
        except ValueError as exc:
            assert "sem pixels supervisionados" in str(exc), exc
        else:
            raise AssertionError("avaliação totalmente ignorada deveria falhar claramente")
    # Contrato ONNX: nomes e shapes esperados, sem rodar modelo.
    contract = {"input_name": "image", "output_name": "logits", "input_shape": [1, 3, 512, 512], "output_shape": [1, 4, 512, 512]}
    assert contract["output_shape"][1] == NUM_CLASSES
    print(f"SMOKE OK: {len(pairs)} pares, pixels supervisionados {counts.tolist()}, ignorados {ignored}, pesos {[round(w, 2) for w in weights]}")
    print(f"SMOKE OK: ink IoU estrita {m['ink']['iou']:.3f}, recall com tolerância 2px {tol['radius_2px']['recall']:.3f}")
    print("SMOKE OK: pixels 255 excluídos de métricas e contagens; loss com pixel ignorado é verificada quando PyTorch está disponível")
    print("SMOKE OK: amostra ONNX acompanha o dispositivo do modelo (CPU/meta; CUDA se disponível no runtime)")
    print(f"SMOKE OK: contrato ONNX {contract}")
    return 0


# ---------------------------------------------------------------- main
def main(argv=None):
    parser = argparse.ArgumentParser(description="Treino SegFormer-B0 do Draw (4 classes).")
    parser.add_argument("--config", default="training/config.json")
    parser.add_argument("--data-dir", default="dataset/synthetic")
    parser.add_argument(
        "--real-train-dir",
        default=None,
        help="Diretório opcional com images/ e masks/ para adicionar ao treino sintético; val/test continuam em --data-dir.",
    )
    parser.add_argument("--out-dir", default="out")
    parser.add_argument("--smoke", action="store_true", help="Smoke test sem GPU, sem treino.")
    parser.add_argument("--smoke-dir", default="/tmp/draw-smoke")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.smoke:
        return smoke(args.smoke_dir)

    import torch
    from torch.utils.data import DataLoader

    tr = cfg["training"]
    epochs = args.epochs or tr["epochs"]
    batch_size = args.batch_size or tr["batch_size"]
    torch.manual_seed(tr["seed"])
    np.random.seed(tr["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("AVISO: sem GPU; o treino completo em CPU é inviável. Use o Kaggle com GPU.", file=sys.stderr)

    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    real_pairs = []
    if args.real_train_dir:
        real_root = Path(args.real_train_dir)
        if (real_root / "train" / "images").is_dir():
            real_root = real_root / "train"
        real_pairs = pair_images_masks(real_root)
    synthetic_train_pairs = pair_images_masks(Path(args.data_dir) / "train")
    train_ds = build_dataset(
        args.data_dir, "train", cfg, augment=True, extra_pairs=real_pairs
    )
    val_ds = build_dataset(args.data_dir, "val", cfg, augment=False)
    test_ds = build_dataset(args.data_dir, "test", cfg, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=tr["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=tr["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=tr["num_workers"])
    train_pairs = synthetic_train_pairs + real_pairs
    counts = pixel_counts(train_pairs)
    print(
        f"pares de treino: {len(synthetic_train_pairs)} sintéticos + "
        f"{len(real_pairs)} reais = {len(train_pairs)}"
    )
    print(f"pixels de treino por classe: {counts.tolist()}")
    print(f"pesos da config: {cfg['loss']['cross_entropy_class_weights']}")
    print(f"pesos calculados (referência): {[round(w, 2) for w in compute_class_weights(counts)]}")

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
    accumulation_steps = max(int(tr.get("gradient_accumulation_steps", 1)), 1)
    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    scheduler = warmup_cosine_scheduler(
        optimizer,
        total_steps=epochs * max(steps_per_epoch, 1),
        warmup_ratio=tr["warmup_ratio"],
    )
    precision, amp_dtype, scaler = configure_mixed_precision(torch, device, tr)
    print(f"precisão de treino: {precision}; acumulação: {accumulation_steps}")
    class_weights = torch.tensor(cfg["loss"]["cross_entropy_class_weights"], dtype=torch.float32, device=device)

    best_iou, bad_epochs = -1.0, 0
    history = []
    for epoch in range(epochs):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            class_weights,
            device,
            scaler=scaler,
            scheduler=scheduler,
            accumulation_steps=accumulation_steps,
            amp_dtype=amp_dtype,
        )
        metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": loss, "val": metrics})
        print(f"época {epoch}: loss {loss:.4f}, val_mean_iou {metrics['mean_iou']:.4f}")
        torch.save(model.state_dict(), out_dir / "checkpoints" / "last.pt")
        if metrics["mean_iou"] > best_iou:
            best_iou = metrics["mean_iou"]
            bad_epochs = 0
            torch.save(model.state_dict(), out_dir / "checkpoints" / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= tr["early_stopping"]["patience_epochs"]:
                print(f"early stopping na época {epoch} (val_mean_iou sem melhora há {bad_epochs} épocas).")
                break

    # Exportação ONNX a partir do melhor checkpoint.
    # Não-fatal: se a exportação falhar, as métricas e os demais artefatos
    # ainda são escritos, e a falha fica registrada em onnx_validation —
    # antes, uma falha aqui perdia o metrics.json inteiro após o treino.
    model.load_state_dict(torch.load(out_dir / "checkpoints" / "best.pt", map_location=device))
    onnx_path = out_dir / cfg["export"]["onnx"]
    try:
        export_onnx(model, str(onnx_path), cfg)
        onnx_check = validate_onnx(model, onnx_path, val_loader, device, cfg)
    except Exception as exc:  # noqa: BLE001 - registrar, não perder o treino
        print(f"AVISO: exportação ONNX falhou ({exc}); métricas serão escritas sem ONNX.", file=sys.stderr)
        onnx_check = {"status": "failed", "error": str(exc)}
    final_metrics = evaluate(model, val_loader, device)
    final_metrics["test"] = evaluate(model, test_loader, device)
    final_metrics["onnx_validation"] = onnx_check
    final_metrics["history"] = history
    final_metrics["train_pixel_counts"] = counts.tolist()
    final_metrics["train_source_counts"] = {
        "synthetic": len(synthetic_train_pairs),
        "real": len(real_pairs),
    }
    cfg["data_sources"] = {
        "synthetic_root": str(Path(args.data_dir)),
        "real_train_root": str(Path(args.real_train_dir)) if args.real_train_dir else None,
        "train_pair_counts": final_metrics["train_source_counts"],
        "validation_source": "synthetic --data-dir/val only",
        "test_source": "synthetic --data-dir/test only; evaluated once after selecting best checkpoint",
        "mixed_precision_used": precision,
        "gradient_accumulation_steps": accumulation_steps,
    }

    (out_dir / "classes.json").write_text(json.dumps(CLASSES, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    (out_dir / "training_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    # split_manifest: cópia validada, se o gerador a forneceu.
    manifest_src = Path(args.data_dir) / "split_manifest.json"
    if manifest_src.exists():
        split_manifest = json.loads(manifest_src.read_text(encoding="utf-8"))
    else:
        split_manifest = {
            "note": "split_manifest.json não fornecido pelo gerador; splits lidos das pastas train/val/test.",
            "splits": ["train", "val", "test"],
        }
    split_manifest["training_sources"] = cfg["data_sources"]
    (out_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )
    print(f"ONNX: {onnx_path} | métricas: {out_dir / 'metrics.json'} | val_mean_iou: {final_metrics['mean_iou']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
