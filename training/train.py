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
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools import seg_metrics as shared_metrics  # noqa: E402

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
    assert "training" in cfg, "config sem bloco training"
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
            if img_path.stem.endswith(("_mask", "_ink_full", "_change")):
                continue
            if img_path.stem.startswith("tseq_"):
                # Frames de sequência temporal NÃO são cenas soltas:
                # entram no treino só como pares consecutivos
                # (build_pair_dataset), para a dieta de frames — e o
                # val/test — seguir idêntica à receita sem temporal.
                continue
            mask_path = split_dir / (img_path.stem + "_mask.png")
            if not mask_path.exists():
                raise FileNotFoundError(f"máscara ausente para {img_path}")
            ink_full = split_dir / (img_path.stem + "_ink_full.png")
            pairs.append((img_path, mask_path, ink_full if ink_full.exists() else None))
    if not pairs:
        raise ValueError(f"nenhum par imagem/máscara em {split_dir}")
    return pairs


def _assert_disjoint_source_splits(named_splits, context):
    """Rejeita uma mesma fonte em train/val/test de um manifesto."""
    split_sets = {
        split: set(named_splits.get(split, [])) for split in ("train", "val", "test")
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            sample = sorted(overlap)[:5]
            raise ValueError(
                f"vazamento de fontes no manifesto ({context}, {left}/{right}): {sample}"
            )


def validate_split_manifest(manifest):
    """Valida os formatos de split emitidos pelo gerador do Draw.

    ``splits`` guarda fonte -> split -> ids; ``hand_base_splits`` guarda
    split -> ids das mãos base. Manifestos antigos que só listam os nomes
    dos splits continuam aceitos, mas não recebem uma falsa aprovação de
    conteúdo: o retorno registra quantas famílias foram verificadas.
    """
    if not isinstance(manifest, dict):
        raise ValueError("split manifest precisa ser um objeto JSON")
    checked = []
    splits = manifest.get("splits")
    if isinstance(splits, dict):
        for kind, named_splits in splits.items():
            if isinstance(named_splits, dict):
                _assert_disjoint_source_splits(named_splits, str(kind))
                checked.append(str(kind))
    base_splits = manifest.get("hand_base_splits")
    if isinstance(base_splits, dict):
        _assert_disjoint_source_splits(base_splits, "hand_base_splits")
        checked.append("hand_base_splits")
    return {"checked_source_families": checked}


def load_split_manifest(data_dir):
    """Lê split_manifest.json ou o manifest.json nativo do gerador."""
    root = Path(data_dir)
    for name in ("split_manifest.json", "manifest.json"):
        path = root / name
        if path.is_file():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            validation = validate_split_manifest(manifest)
            return manifest, path, validation
    return None, None, {"checked_source_families": []}


def validate_mask(mask):
    """Máscara válida: classes 0..3 e 255 para pixels deliberadamente ignorados."""
    values = np.unique(mask)
    bad = [int(v) for v in values if v not in (0, 1, 2, 3, IGNORE_INDEX)]
    if bad:
        raise ValueError(
            f"máscara com valores inválidos (permitidos: 0..3 e 255=ignorado): {bad}"
        )
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
        }
        | {"support_px": report["support_px"][name]}
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
    """SegFormer-B0 pré-treinado, cabeça nova de 4 classes (lazy torch).

    Honra ``t7.decoder_hidden_size`` da config quando presente (a nota
    da config t7 pedia isso: sem o override, a corrida GPU construía
    decoder 256 enquanto a TPU treinava decoder 128 — arquiteturas
    diferentes sob o mesmo nome). Config sem o bloco = comportamento
    histórico, byte a byte.
    """
    from transformers import SegformerForSemanticSegmentation

    model_cfg = cfg["model"]
    extra = {}
    decoder = (cfg.get("t7") or {}).get("decoder_hidden_size")
    if decoder:
        extra["decoder_hidden_size"] = int(decoder)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_cfg["backbone"],
        num_labels=model_cfg["num_classes"],
        ignore_mismatched_sizes=True,  # cabeça nova: 4 classes, não as do ImageNet
        **extra,
    )
    if (cfg.get("presence") or {}).get("enabled"):
        model = _wrap_presence_head(model)
    init_path = (cfg.get("training") or {}).get("init_checkpoint")
    if init_path:
        report = load_init_checkpoint(model, init_path)
        print(
            f"[init] warm start de {init_path}: {report['loaded']} tensores "
            f"carregados, {report['fresh']} frescos (cabeca nova), "
            f"ignorados por shape: {report['skipped_shape']}"
        )
    return model


def _strip_wrapped_prefix(key):
    """Nome de tensor sem o prefixo do embrulho do modelo vivo.

    O PresenceSegformer guarda o modelo base em ``self.base``: o
    state_dict vivo tem as chaves ``base.segformer...`` /
    ``base.decode_head...`` — mas o checkpoint do modelo antigo foi salvo
    sem o embrulho, com as chaves nuas (``segformer...``). Casar pelo nome
    sem o prefixo, nos dois sentidos; sem isso o warm init carrega 0
    tensores em silêncio (rodada 1 do loop, 2026-11-07: 0/208).
    """
    return key[len("base.") :] if key.startswith("base.") else key


def load_init_checkpoint(model, path, only_prefixes=None):
    """Warm init parcial a partir de um checkpoint anterior.

    Copia todo tensor de mesmo nome (ignorando o prefixo ``base.`` do
    embrulho) e mesma forma; tensores novos do modelo atual (ex.: a
    cabeca de presenca, que o checkpoint antigo nao tem) ficam com a
    inicializacao fresca, e tensores de forma incompativel sao
    ignorados — nunca um load estrito que derrube a corrida por causa
    da cabeca nova. Retorna o relatorio da carga.

    ``only_prefixes`` (Rodada 3): quando dado, só tensores cujo nome
    (sem o prefixo do embrulho) começa com um dos prefixos entram —
    serve para transplantar UMA peça de outro checkpoint (a cabeça de
    presença da Rodada 2) sem tocar no resto do warm init.
    """
    import torch

    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if isinstance(raw.get(key), dict):
                raw = raw[key]
                break
    raw_by_bare = {}
    for key, value in raw.items():
        if torch.is_tensor(value):
            raw_by_bare.setdefault(_strip_wrapped_prefix(key), value)
    if only_prefixes:
        prefixes = tuple(only_prefixes)
        raw_by_bare = {
            key: value for key, value in raw_by_bare.items() if key.startswith(prefixes)
        }
    target = model.state_dict()
    matched = {}
    skipped_shape = []
    for key in target:
        bare = _strip_wrapped_prefix(key)
        if bare not in raw_by_bare:
            continue
        if raw_by_bare[bare].shape == target[key].shape:
            matched[key] = raw_by_bare[bare]
        else:
            skipped_shape.append(key)
    report = {
        "loaded": len(matched),
        "fresh": sum(
            1 for key in target if _strip_wrapped_prefix(key) not in raw_by_bare
        ),
        "skipped_shape": skipped_shape,
    }
    if only_prefixes:
        report["only_prefixes"] = list(only_prefixes)
    model.load_state_dict(matched, strict=False)
    return report


def _wrap_presence_head(base):
    """Cabeça de PRESENÇA imagem-nível sobre o último estado do encoder.

    Âncora: Chen et al., arXiv:2608.11135 (OPCNet, 2026) — segmentadores
    treinados sob "o objeto está sempre presente" tinham 97%+ de FP em
    imagens só-fundo; uma cabeça explícita de existência + treino rico
    em negativos cortou o FP de fundo para 5,0%. Aqui a cabeça lê o
    último hidden state do encoder (NÃO os logits da máscara: sinal
    estruturalmente independente, como manda o protocolo de gate do
    PEOS-Net, Frontiers in AI 2026), faz pooling global e devolve 1
    logit por imagem. A saída do modelo ganha a chave
    ``presence_logit``; ``logits`` segue idêntica.
    """
    import torch

    class PresenceSegformer(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.base = wrapped
            self.config = wrapped.config
            hidden = int(wrapped.config.hidden_sizes[-1])
            self.presence_pool = torch.nn.AdaptiveAvgPool2d(1)
            self.presence_head = torch.nn.Linear(hidden, 1)

        def forward(self, pixel_values):
            from transformers.modeling_outputs import SemanticSegmenterOutput

            seg = self.base.segformer(
                pixel_values=pixel_values, output_hidden_states=True
            )
            hidden_states = seg.hidden_states
            logits = self.base.decode_head(hidden_states)
            pooled = self.presence_pool(hidden_states[-1]).flatten(1)
            out = SemanticSegmenterOutput(logits=logits)
            out["presence_logit"] = self.presence_head(pooled).squeeze(-1)
            return out

    return PresenceSegformer(base)


def presence_targets(masks):
    """1.0 por imagem quando há papel visível (classe 1) ou tinta (3).

    Mão sozinha (2) ou só fundo (0) = ausência. Pixel 255 (ignorado)
    não cria presença nem ausência: a imagem conta pelo que está
    rotulado — uma cena sem-papel do gerador é toda fundo/mão.
    """
    import torch

    present = (masks == 1) | (masks == 3)
    return present.reshape(masks.shape[0], -1).any(dim=1).to(torch.float32)


def presence_loss(
    presence_logit, masks, sample_weights=None, targets=None, pos_weight=None
):
    """BCE imagem-nível da cabeça de presença (termo L_pres do OPCNet).

    ``targets`` explícitos (tensor float, um por imagem) vencem a
    derivação pela máscara; valor < 0 numa posição = derivar da máscara
    naquela posição. É o caminho do pool de presença (Rodada 2): frames
    reais rotulados só com has_paper, cuja máscara é toda IGNORE —
    derivar dela diria "ausente" até para os controles positivos.
    ``pos_weight`` repassa ao BCEWithLogits (peso do termo positivo):
    com a dieta dominada por positivos (SmartDoc é 100% presença), ele
    devolve voz aos negativos sem mudar a amostragem.
    """
    import torch
    import torch.nn.functional as F

    if targets is None:
        targets = presence_targets(masks)
    else:
        targets = torch.as_tensor(
            targets, dtype=presence_logit.dtype, device=presence_logit.device
        )
        if bool((targets < 0).any()):
            derived = presence_targets(masks)
            targets = torch.where(targets >= 0, targets, derived)
    pw = None
    if pos_weight is not None:
        pw = torch.as_tensor(
            [float(pos_weight)],
            dtype=presence_logit.dtype,
            device=presence_logit.device,
        )
    per_sample = F.binary_cross_entropy_with_logits(
        presence_logit, targets, reduction="none", pos_weight=pw
    )
    if sample_weights is None:
        return per_sample.mean()
    weights = torch.as_tensor(
        sample_weights, dtype=per_sample.dtype, device=per_sample.device
    )
    return (per_sample * weights).sum() / weights.sum().clamp(min=1e-12)


class WeightEMA:
    """Média móvel exponencial dos pesos (pacote de eficiência do C).

    Mantém uma cópia-sombra dos tensores float do modelo, atualizada a
    cada passo do otimizador: shadow ← decay·shadow + (1−decay)·peso.
    Avaliar/entregar os pesos-EMA em vez dos crus é ganho de qualidade
    quase grátis (a trajetória ruidosa do SGD é suavizada); o treino em
    si não muda. Buffers não-float (contadores etc.) não entram na
    sombra — na aplicação, só os tensores float são trocados.
    """

    def __init__(self, model, decay):
        import torch

        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        self._torch = torch

    def update(self, model):
        torch = self._torch
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(
                        v.detach(), alpha=1.0 - self.decay
                    )

    def apply(self, model):
        """Troca os pesos do modelo pelos da sombra; devolve o backup."""
        torch = self._torch
        backup = {}
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    backup[k] = v.detach().clone()
                    v.copy_(self.shadow[k])
        return backup

    def restore(self, model, backup):
        torch = self._torch
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in backup:
                    v.copy_(backup[k])


def _make_augmentation():
    """Compose de augmentation da config (geométrica idêntica em imagem
    e máscara; fotométrica só na imagem; máscara sempre nearest)."""
    import albumentations as A

    try:
        return A.Compose(
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


def build_dataset(
    data_dir,
    split,
    cfg,
    augment=False,
    extra_pairs=None,
    extra_sample_weight=1.0,
    return_sample_weights=False,
    cache_mode="none",
    cache_cap_mb=24576,
):
    """Dataset torch. Import lazy: sem torch, esta função não existe.

    ``extra_pairs`` pode receber um peso menor para propostas reais não
    revisadas. O peso é por amostra e não altera validação nem teste.

    ``cache_mode="resized"`` (pacote de eficiência do modelo C): decodifica
    TUDO uma vez na construção, já redimensionado pro tamanho do modelo
    (imagem LINEAR, máscara NEAREST — as interpolações do caminho normal),
    e serve de RAM. Sem isso, cada época repaga decode de PNG/JPG +
    resize em CPU — o gargalo medido nos recibos do A/B (GPU a 20–56%).
    Diferença deliberada: a augmentation passa a operar na resolução
    do modelo (512) em vez da nativa; pro sintético é idêntico (o
    gerador já nasce 512×512), pro SmartDoc é a troca que compra a
    velocidade. Estouro do teto de RAM é erro alto, nunca silencioso.
    """
    import torch
    from torch.utils.data import Dataset

    pairs = pair_images_masks(Path(data_dir) / split)
    sample_weights = [1.0] * len(pairs)
    if extra_pairs:
        if not 0.0 < float(extra_sample_weight) <= 1.0:
            raise ValueError("extra_sample_weight precisa estar em (0, 1]")
        pairs.extend(extra_pairs)
        sample_weights.extend([float(extra_sample_weight)] * len(extra_pairs))
    size = cfg["model"]["input"]["height"]
    cache = None
    if cache_mode == "resized":
        est_bytes = len(pairs) * size * size * 4  # RGB uint8 + máscara uint8
        cap_bytes = int(cache_cap_mb) * 1024 * 1024
        if est_bytes > cap_bytes:
            raise ValueError(
                f"cache resized precisa de ~{est_bytes / 2**30:.1f} GiB para "
                f"{len(pairs)} amostras, acima do teto de {cache_cap_mb} MiB; "
                "suba training.cache_cap_mb ou desligue o cache."
            )
        cache = []
        for img_path, mask_path, _ in pairs:
            im = load_image_rgb(img_path)
            mk = load_mask(mask_path)
            im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
            mk = cv2.resize(
                mk.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST
            )
            cache.append((im, mk))
        print(
            f"cache RAM ({split}): {len(cache)} amostras {size}x{size} "
            f"≈ {est_bytes / 2**30:.1f} GiB"
        )
    elif cache_mode != "none":
        raise ValueError(f"cache_mode desconhecido: {cache_mode!r}")
    aug = _make_augmentation() if augment else None

    class SegDataset(Dataset):
        def __len__(self):
            return len(pairs)

        def __getitem__(self, i):
            img_path, mask_path, _ = pairs[i]
            if cache is not None:
                image = cache[i][0].copy()
                mask = cache[i][1].astype(np.int64)
            else:
                image = load_image_rgb(img_path)
                mask = load_mask(mask_path)
            if image.shape[:2] != mask.shape[:2]:
                raise ValueError(
                    f"imagem/máscara com dimensões diferentes em {img_path}: "
                    f"{image.shape[:2]} x {mask.shape[:2]}"
                )
            if aug is not None:
                out = aug(image=image, mask=mask)
                image, mask = out["image"], out["mask"]
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
            image = image.astype(np.float32) / 255.0
            image = (image - IMAGENET_MEAN) / IMAGENET_STD
            image = np.transpose(image, (2, 0, 1))
            image_tensor = torch.from_numpy(image)
            mask_tensor = torch.from_numpy(mask.astype(np.int64))
            if return_sample_weights:
                return image_tensor, mask_tensor, float(sample_weights[i])
            return image_tensor, mask_tensor

    return SegDataset()


def build_pool_dataset(pool_dir, cfg, augment=False):
    """Dataset do POOL DE PRESENÇA (Rodada 2): frames reais rotulados só
    com has_paper (labels.csv: file, has_paper), SEM máscara de
    segmentação. Cada item devolve (imagem, máscara toda IGNORE, peso
    1.0, alvo de presença explícito): a loss de segmentação se anula
    sozinha (nenhum pixel válido — ver combined_loss/sample_is_valid)
    e só a cabeça de presença aprende com estes frames. É o remédio
    medido da Rodada 1: a cabeça convergiu (presence_loss 0,003) sobre
    negativos SINTÉTICOS e não transferiu pra mesa real (mediana 0,9993
    nas 187 cenas vazias da bancada). Negativos reais e disjuntos do
    holdout de julgamento — absence_v1 NUNCA entra aqui.

    Frames decodificados uma vez, já no tamanho do modelo, e servidos
    da RAM: o pool é pequeno (336) mas o sampler o revisita ~18x/época.
    """
    import csv as _csv

    import torch
    from torch.utils.data import Dataset

    pool_dir = Path(pool_dir)
    labels_path = pool_dir / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels.csv do pool ausente: {labels_path}")
    rows = []
    with labels_path.open(newline="", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            raw = str(row.get("has_paper", "")).strip().lower()
            if raw in ("1", "true"):
                has_paper = 1.0
            elif raw in ("0", "false"):
                has_paper = 0.0
            else:
                raise ValueError(f"has_paper ilegível na linha {row}: {raw!r}")
            img_path = pool_dir / str(row["file"])
            if not img_path.is_file():
                raise FileNotFoundError(f"frame do pool ausente: {img_path}")
            rows.append((img_path, has_paper))
    if not rows:
        raise ValueError(f"pool de presença vazio: {labels_path}")
    size = cfg["model"]["input"]["height"]
    cache = []
    for img_path, _ in rows:
        im = load_image_rgb(img_path)
        cache.append(cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR))
    n_neg = sum(1 for _, hp in rows if hp == 0.0)
    print(
        f"pool de presença: {len(rows)} frames reais "
        f"({n_neg} sem papel, {len(rows) - n_neg} com papel) de {pool_dir}"
    )
    aug = _make_augmentation() if augment else None

    class PoolDataset(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            image = cache[i].copy()
            if aug is not None:
                dummy = np.zeros(image.shape[:2], dtype=np.uint8)
                image = aug(image=image, mask=dummy)["image"]
            image = image.astype(np.float32) / 255.0
            image = (image - IMAGENET_MEAN) / IMAGENET_STD
            image = np.transpose(image, (2, 0, 1))
            mask = np.full((size, size), IGNORE_INDEX, dtype=np.int64)
            return (
                torch.from_numpy(image),
                torch.from_numpy(mask),
                1.0,
                float(rows[i][1]),
            )

    return PoolDataset()


def collate_with_presence(items):
    """Collate do treino com pool: itens de 3 campos (base) ganham alvo
    de presença -1 (derivar da máscara); itens de 4 campos (pool) já
    trazem o alvo explícito. Devolve (imagens, máscaras, pesos, alvos).
    """
    import torch

    images, masks, weights, targets = [], [], [], []
    for item in items:
        if len(item) == 3:
            img, mask, weight = item
            target = -1.0
        elif len(item) == 4:
            img, mask, weight, target = item
        else:
            raise ValueError(f"item de treino com {len(item)} campos (esperado 3 ou 4)")
        images.append(img)
        masks.append(mask)
        weights.append(float(weight))
        targets.append(float(target))
    return (
        torch.stack(images),
        torch.stack(masks),
        torch.tensor(weights, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )


def pair_sequence_files(split_dir):
    """Pares consecutivos das sequências temporais do gerador.

    Layout plano: ``tseq_<split>_<i>_f<k>.png`` + ``_mask`` e, para
    k>=1, ``_change``. Devolve lista de
    ((img_a, mask_a, None), (img_b, mask_b, change_b)). Sequência com
    buraco de frames ou sem ``_change`` é erro de geração: falha alto,
    nunca degrada em silêncio. Sem sequências no split, devolve [].
    """
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        return []
    seqs: dict[str, list[tuple[int, Path]]] = {}
    for img_path in sorted(split_dir.iterdir()):
        stem = img_path.stem
        if not stem.startswith("tseq_") or img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        if stem.endswith(("_mask", "_ink_full", "_change")):
            continue
        seq_id, sep, frame = stem.rpartition("_f")
        if not sep or not frame.isdigit():
            continue
        seqs.setdefault(seq_id, []).append((int(frame), img_path))
    pairs = []
    for seq_id in sorted(seqs):
        frames = sorted(seqs[seq_id])
        for (ka, pa), (kb, pb) in zip(frames, frames[1:]):
            if kb != ka + 1:
                raise ValueError(
                    f"sequência {seq_id} com buraco de frames ({ka} -> {kb})"
                )
            mask_a = split_dir / (pa.stem + "_mask.png")
            mask_b = split_dir / (pb.stem + "_mask.png")
            change_b = split_dir / (pb.stem + "_change.png")
            for needed in (mask_a, mask_b, change_b):
                if not needed.exists():
                    raise FileNotFoundError(
                        f"arquivo de sequência ausente: {needed} "
                        "(gere com --temporal-train/--temporal-val)"
                    )
            pairs.append(((pa, mask_a, None), (pb, mask_b, change_b)))
    return pairs


def build_pair_dataset(data_dir, split, cfg):
    """Dataset de PARES consecutivos da mesma cena (consistência temporal).

    Cada item: (img_a, mask_a, img_b, mask_b, change_b) — change_b é a
    máscara binária (0/1) de "onde o rótulo visível mudou de verdade"
    do gerador. SEM albumentations geométrico: qualquer warp aplicado
    a um frame só quebraria o alinhamento do par e invalidaria o
    change. Normalização e resize idênticos aos do SegDataset.
    """
    import torch
    from torch.utils.data import Dataset

    pairs = pair_sequence_files(Path(data_dir) / split)
    size = cfg["model"]["input"]["height"]

    def load_frame(img_path, mask_path):
        image = load_image_rgb(img_path)
        mask = load_mask(mask_path)
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"imagem/máscara com dimensões diferentes em {img_path}: "
                f"{image.shape[:2]} x {mask.shape[:2]}"
            )
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image), torch.from_numpy(mask.astype(np.int64))

    class PairDataset(Dataset):
        def __len__(self):
            return len(pairs)

        def __getitem__(self, i):
            (pa, ma, _), (pb, mb, cb) = pairs[i]
            img_a, mask_a = load_frame(pa, ma)
            img_b, mask_b = load_frame(pb, mb)
            change = cv2.imread(str(cb), cv2.IMREAD_GRAYSCALE)
            if change is None:
                raise FileNotFoundError(f"change ilegível: {cb}")
            change = cv2.resize(change, (size, size), interpolation=cv2.INTER_NEAREST)
            return (
                img_a,
                mask_a,
                img_b,
                mask_b,
                torch.from_numpy((change > 0).astype(np.uint8)),
            )

    return PairDataset()


def _photometric_jitter(image01, rng):
    """Jitter fotométrico leve num frame RGB float em [0, 1].

    Gamma, ganho e ruído gaussiano — NADA geométrico: os dois frames
    do par real vêm da mesma câmera parada, e inventar geometria
    ensinaria o modelo a tolerar movimento que não houve. Cada frame
    do par recebe um sorteio independente: é essa independência que
    ensina "mesma cena, ruído novo, mesma máscara".
    """
    gamma = float(rng.uniform(0.85, 1.18))
    gain = float(rng.uniform(0.90, 1.10))
    noise = float(rng.uniform(0.0, 0.016))
    out = np.clip(image01, 0.0, 1.0) ** gamma
    out = out * gain
    if noise > 0.0:
        out = out + rng.normal(0.0, noise, size=out.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def build_real_pair_dataset(pairs_dir, cfg, split="train", augment=False):
    """Pares REAIS (t, t+Δ) com teacher congelado (Rodada 3).

    Layout (gerado por pré-computação, nunca à mão): ``pairs.csv``
    com colunas frame_a, frame_b, mask_a, mask_b, change, split —
    caminhos relativos a ``pairs_dir``. As máscaras são do teacher
    (FT-dyn) já traduzidas pras classes do aluno (0 fundo, 1 papel,
    255 onde o teacher discorda de si mesmo no par) e ``change`` é
    exatamente esse mapa de discordância: a loss de consistência lê
    change == 1 como "livre para mudar", a supervisão lê 255 como
    IGNORE — as duas leituras concordam por construção.

    Cada item tem a MESMA forma do PairDataset sintético:
    (img_a, mask_a, img_b, mask_b, change). Sem augmentation
    geométrica (ver build_pair_dataset); com ``augment``, jitter
    fotométrico independente por frame.
    """
    import csv

    import torch
    from torch.utils.data import Dataset

    pairs_dir = Path(pairs_dir)
    manifest = pairs_dir / "pairs.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"manifesto de pares ausente: {manifest}")
    rows = []
    with manifest.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("split", "train") == split:
                rows.append(row)
    if not rows:
        raise ValueError(f"nenhum par real com split={split!r} em {manifest}")
    size = cfg["model"]["input"]["height"]
    rng = np.random.default_rng()

    def load_frame(img_rel, mask_rel):
        image = load_image_rgb(pairs_dir / img_rel)
        mask = load_mask(pairs_dir / mask_rel)
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"imagem/máscara com dimensões diferentes em {img_rel}: "
                f"{image.shape[:2]} x {mask.shape[:2]}"
            )
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
        image = image.astype(np.float32) / 255.0
        if augment:
            image = _photometric_jitter(image, rng)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image), torch.from_numpy(mask.astype(np.int64))

    class RealPairDataset(Dataset):
        orig_wh = None  # (largura, altura) do frame original, pro flicker

        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            row = rows[i]
            img_a, mask_a = load_frame(row["frame_a"], row["mask_a"])
            img_b, mask_b = load_frame(row["frame_b"], row["mask_b"])
            change = cv2.imread(str(pairs_dir / row["change"]), cv2.IMREAD_GRAYSCALE)
            if change is None:
                raise FileNotFoundError(f"change ilegível: {row['change']}")
            change = cv2.resize(change, (size, size), interpolation=cv2.INTER_NEAREST)
            return (
                img_a,
                mask_a,
                img_b,
                mask_b,
                torch.from_numpy((change > 0).astype(np.uint8)),
            )

    first = load_image_rgb(pairs_dir / rows[0]["frame_a"])
    RealPairDataset.orig_wh = (int(first.shape[1]), int(first.shape[0]))
    return RealPairDataset()


def flicker_metrics(model, pair_dataset, device, max_pairs=64):
    """Régua de flicker sobre pares reais: o modelo pisca quanto?

    Por par: passo do centróide da máscara de papel predita (em px do
    frame ORIGINAL, via ``dataset.orig_wh``), tIoU entre as duas
    máscaras preditas e |Δárea| relativo. Devolve medianas/médias —
    os mesmos três números que a bancada cobra do candidato. Par com
    alguma máscara vazia é contado em ``skipped`` e sai das médias.
    """
    import torch

    model.eval()
    orig_w, orig_h = getattr(pair_dataset, "orig_wh", None) or (512, 512)
    size = None
    steps, tious, area_diffs = [], [], []
    skipped = 0
    with torch.no_grad():
        for i in range(min(len(pair_dataset), max_pairs)):
            img_a, _, img_b, _, _ = pair_dataset[i]
            if size is None:
                size = img_a.shape[-1]
            batch = torch.stack([img_a, img_b]).to(device)
            logits = _model_logits(model, batch)
            classes = logits.argmax(dim=1)
            paper = ((classes == 1) | (classes == 3)).cpu().numpy()
            cents = []
            ok = True
            for k in range(2):
                ys, xs = np.nonzero(paper[k])
                if len(xs) == 0:
                    ok = False
                    break
                cents.append((xs.mean(), ys.mean()))
            if not ok:
                skipped += 1
                continue
            dx = (cents[1][0] - cents[0][0]) * (orig_w / size)
            dy = (cents[1][1] - cents[0][1]) * (orig_h / size)
            steps.append(float(np.hypot(dx, dy)))
            inter = float(np.logical_and(paper[0], paper[1]).sum())
            union = float(np.logical_or(paper[0], paper[1]).sum())
            tious.append(inter / union if union else 1.0)
            a_a = float(paper[0].mean())
            a_b = float(paper[1].mean())
            area_diffs.append(abs(a_a - a_b) / max(a_a, a_b, 1e-6))
    model.train()
    if not steps:
        return {"pairs": 0, "skipped": skipped}
    return {
        "pairs": len(steps),
        "skipped": skipped,
        "centroid_step_median": round(float(np.median(steps)), 2),
        "tiou_mean": round(float(np.mean(tious)), 4),
        "area_diff_median": round(float(np.median(area_diffs)), 4),
    }


def temporal_settings(cfg):
    """Bloco opcional ``training.temporal`` da config.

    Ausente = desligado (comportamento histórico, provado por teste).
    Presente, valida os limites: pair_fraction em [0, 1] e
    consistency_weight >= 0.

    Rodada 3: ``mode`` (mse|l1|kl — a forma do termo de consistência;
    default mse = comportamento histórico), ``trim`` (fração dos
    maiores termos por pixel descartada antes da média, TrimMean da
    pesquisa 2026; default 0 = média cheia) e ``area_weight`` (peso do
    termo auxiliar de área; default 0 = desligado, é braço de ablação).
    """
    block = (cfg.get("training") or {}).get("temporal") or {}
    pair_fraction = float(block.get("pair_fraction", 0.25))
    consistency_weight = float(block.get("consistency_weight", 1.0))
    if not 0.0 <= pair_fraction <= 1.0:
        raise ValueError("training.temporal.pair_fraction precisa estar em [0, 1]")
    if consistency_weight < 0.0:
        raise ValueError("training.temporal.consistency_weight precisa ser >= 0")
    mode = str(block.get("mode", "mse"))
    if mode not in ("mse", "l1", "kl"):
        raise ValueError("training.temporal.mode precisa ser mse, l1 ou kl")
    trim = float(block.get("trim", 0.0))
    if not 0.0 <= trim < 0.5:
        raise ValueError("training.temporal.trim precisa estar em [0, 0.5)")
    area_weight = float(block.get("area_weight", 0.0))
    if area_weight < 0.0:
        raise ValueError("training.temporal.area_weight precisa ser >= 0")
    return {
        "enabled": bool(block.get("enabled", False)),
        "pair_fraction": pair_fraction,
        "consistency_weight": consistency_weight,
        "mode": mode,
        "trim": trim,
        "area_weight": area_weight,
    }


def hnbs_loss(logits, target, threshold=0.35, topk_fraction=0.005):
    """Hard-Negative Background Suppression (HNBS), âncorado em Mao et al.,
    Sensors 2026 (DOI 10.3390/s26165185, CGHN-Net).

    Pressão extra SÓ nos pixels de FUNDO (GT classe 0) em que o modelo
    insiste em ver primeiro plano: prob. de não-fundo >= ``threshold``,
    e só os ``topk_fraction`` piores por imagem (pela CE do pixel). O
    top-k é o que impede a supressão excessiva: no ablation do paper,
    penalizar TODOS os pixels de fundo acima do limiar deu a melhor
    precisão mas derrubou o recall para 0,8279. Sem candidato no lote,
    devolve 0 ligado ao grafo (backward válido, gradiente zero).
    """
    import torch
    import torch.nn.functional as F

    prob = torch.softmax(logits, dim=1)
    fg_prob = 1.0 - prob[:, 0]
    ce_map = F.cross_entropy(
        logits, target.clamp(min=0, max=NUM_CLASSES - 1), reduction="none"
    )
    terms = []
    for i in range(target.shape[0]):
        bg = target[i] == 0
        n_bg = int(bg.sum())
        if n_bg == 0:
            continue
        cand = bg & (fg_prob[i] >= threshold)
        n_cand = int(cand.sum())
        if n_cand == 0:
            continue
        k = max(1, min(n_cand, int(topk_fraction * n_bg)))
        terms.append(torch.topk(ce_map[i][cand], k).values.mean())
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()


def combined_loss(
    logits,
    target,
    class_weights,
    ce_weight=0.5,
    dice_weight=0.5,
    sample_weights=None,
    hnbs_weight=0.0,
    hnbs_threshold=0.35,
    hnbs_topk_fraction=0.005,
):
    """CE ponderada + Dice multiclasse; target=255 não supervisiona o pixel.

    Sem ``sample_weights`` preserva a redução original do treino. Com ela,
    calcula CE+Dice por imagem e faz a média ponderada das amostras válidas;
    é o caminho usado para reduzir a influência de pseudo-rótulos reais não
    revisados sem alterar os pixels sintéticos supervisionados.

    ``hnbs_weight`` > 0 soma o termo HNBS (ver ``hnbs_loss``); com 0
    (padrão histórico) o valor é idêntico ao de antes, termo a termo.
    """
    import torch
    import torch.nn.functional as F

    valid = target != IGNORE_INDEX
    if not torch.any(valid):
        # Mantém a loss ligada ao grafo para que backward seja válido, mas
        # produz gradiente zero quando o lote inteiro foi deliberadamente ignorado.
        return logits.sum() * 0.0

    if sample_weights is None:
        ce = F.cross_entropy(
            logits, target, weight=class_weights, ignore_index=IGNORE_INDEX
        )
    else:
        ce_map = F.cross_entropy(
            logits,
            target,
            weight=class_weights,
            ignore_index=IGNORE_INDEX,
            reduction="none",
        )
        safe_target = target.clamp(min=0, max=NUM_CLASSES - 1)
        pixel_weights = class_weights[safe_target] * valid
        ce = (ce_map * valid).reshape(target.shape[0], -1).sum(dim=1) / (
            pixel_weights.reshape(target.shape[0], -1).sum(dim=1).clamp(min=1e-12)
        )
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
    if sample_weights is None:
        # A loss precisa ser escalar inclusive com acumulação de gradiente.
        base = ce_weight * ce + dice_weight * dice[sample_is_valid].mean()
    else:
        per_sample = ce_weight * ce + dice_weight * dice
        weights = torch.as_tensor(
            sample_weights, dtype=logits.dtype, device=logits.device
        )
        if weights.shape != (target.shape[0],):
            raise ValueError("sample_weights precisa ter um valor por imagem do batch")
        if bool((weights <= 0).any()):
            raise ValueError("sample_weights precisa ser estritamente positivo")
        weights = weights * sample_is_valid
        base = (per_sample * weights).sum() / weights.sum().clamp(min=1e-12)
    if hnbs_weight > 0.0:
        base = base + hnbs_weight * hnbs_loss(
            logits, target, threshold=hnbs_threshold, topk_fraction=hnbs_topk_fraction
        )
    return base


def masked_temporal_consistency_loss(logits_a, logits_b, change, mode="mse", trim=0.0):
    """Consistência temporal MASCARADA entre dois frames da mesma cena.

    Penaliza a predição que muda onde o rótulo NÃO mudou:
    ``change == 0``. Onde a fonte do par diz que mudou de verdade
    (change == 1 — tinta nova, mão que andou, borda que derivou,
    ignore em algum dos frames, ou discordância do teacher nos pares
    reais da Rodada 3) a penalidade é ZERO: o modelo é livre — e
    supervisionado pela loss normal — para mudar. É o "reconheceu
    uma vez, basta" virando gradiente: sem máscara, a consistência
    ingênua ensinaria o modelo a NÃO registrar tinta nova. Sem nenhum
    pixel estável no lote, devolve 0 ligado ao grafo (backward
    válido, gradiente zero).

    ``mode``: mse (histórico: quadrado da diferença de probabilidades
    softmax), l1 (a forma com número publicado em 2026 — Time2General)
    ou kl (KL simétrica entre as duas distribuições). ``trim`` > 0
    descarta essa fração dos MAIORES termos por pixel antes da média
    (TrimMean): protege o termo de movimento real que escapou do
    portão. ``mode="mse", trim=0`` reproduz o histórico exatamente.
    """
    import torch

    prob_a = torch.softmax(logits_a, dim=1)
    prob_b = torch.softmax(logits_b, dim=1)
    stable = change == 0
    if not bool(stable.any()):
        return logits_a.sum() * 0.0
    if mode == "mse":
        per_pixel = (prob_a - prob_b).pow(2).sum(dim=1)
    elif mode == "l1":
        per_pixel = (prob_a - prob_b).abs().sum(dim=1)
    elif mode == "kl":
        eps = 1e-6
        pa = prob_a.clamp(min=eps)
        pb = prob_b.clamp(min=eps)
        per_pixel = 0.5 * (
            (pa * (pa.log() - pb.log())).sum(dim=1)
            + (pb * (pb.log() - pa.log())).sum(dim=1)
        )
    else:
        raise ValueError(f"modo de consistência desconhecido: {mode!r}")
    terms = per_pixel[stable]
    if trim > 0.0 and terms.numel() > 10:
        keep = max(1, int(terms.numel() * (1.0 - trim)))
        terms = torch.topk(terms, keep, largest=False).values
    return terms.mean()


def temporal_area_loss(logits_a, logits_b):
    """Termo auxiliar de área (braço de ablação da Rodada 3).

    |A_a − A_b| / max(A_a, A_b) por imagem, com A = fração de papel
    (probabilidade somada de papel+tinta). A literatura de 2026 não
    usa penalidade geométrica como perda — área é métrica/portão;
    este termo existe para o smoke comparar o braço ligado × desligado
    na régua de flicker, nunca como default.
    """
    import torch

    def area(logits):
        prob = torch.softmax(logits, dim=1)
        return (prob[:, 1] + prob[:, 3]).mean(dim=(1, 2))

    a_a = area(logits_a)
    a_b = area(logits_b)
    return (a_a - a_b).abs().div(torch.maximum(a_a, a_b).clamp(min=1e-6)).mean()


def build_optimizer(model, tr):
    """AdamW da corrida. Com ``training.presence_head_lr`` na config, a
    cabeça de presença ganha grupo próprio com esse LR e todo o resto
    fica com ``training.learning_rate`` — o LR DIFERENCIAL da Rodada 2:
    a cabeça enfrenta um sinal novo (negativos reais) e precisa andar
    10-50x mais rápido que o backbone já treinado, que só coopera
    devagar (congelar o backbone está fora da spec: ele precisa
    acompanhar, não travar). Sem a chave, um grupo só com o LR
    histórico: comportamento idêntico ao de antes.
    """
    import torch

    base_lr = float(tr["learning_rate"])
    weight_decay = float(tr["weight_decay"])
    head_lr = tr.get("presence_head_lr")
    if head_lr is None:
        return torch.optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=weight_decay
        )
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head_params if "presence_head" in name else backbone_params).append(param)
    if not head_params:
        print(
            "AVISO: presence_head_lr configurado mas o modelo não tem "
            "parâmetros presence_head; usando um grupo só.",
            file=sys.stderr,
        )
        return torch.optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=weight_decay
        )
    print(
        f"LR diferencial: backbone {base_lr:g} ({len(backbone_params)} tensores) | "
        f"cabeça de presença {float(head_lr):g} ({len(head_params)} tensores)"
    )
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": base_lr},
            {"params": head_params, "lr": float(head_lr)},
        ],
        weight_decay=weight_decay,
    )


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


def cuda_bf16_supported(torch, device):
    """True only when CUDA bf16 has a real engine behind it (Ampere+).

    `torch.cuda.is_bf16_supported()` alone is not enough: on current
    torch it returns True on Turing (e.g. T4) via emulation, but cuDNN
    has no bf16 engine there and the first convolution dies with
    "GET was unable to find an engine to execute this computation"
    (measured on Colab, 2026-11-05). Require capability >= 8 as well.
    """
    if not bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
        return False
    try:
        major, _minor = torch.cuda.get_device_capability(device)
    except Exception:
        return False
    return major >= 8


def configure_mixed_precision(torch, device, training_cfg):
    """Select configured bf16/fp16 on CUDA, fp32 elsewhere."""
    if torch.device(device).type != "cuda":
        return "fp32", None, None
    preferred = training_cfg.get("mixed_precision", "bf16")
    fallback = training_cfg.get("mixed_precision_fallback", "fp16")
    if preferred == "fp32":
        return "fp32", None, None
    supports_bf16 = cuda_bf16_supported(torch, device)
    precision = preferred if preferred == "bf16" and supports_bf16 else fallback
    if precision == "bf16":
        return precision, torch.bfloat16, None
    if precision == "fp16":
        return precision, torch.float16, torch.amp.GradScaler("cuda")
    return "fp32", None, None


def configure_fast_runtime(torch, device):
    """Liga TF32 e o benchmark do cuDNN em CUDA; em CPU não muda nada."""
    if torch.device(device).type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def torch_device_is_accelerated(device):
    return str(device).split(":", 1)[0] != "cpu"


def resolve_compile(cli_value, tr, device):
    """Decide torch.compile: CLI > ``training.compile`` > padrão do dispositivo.

    Padrão ligado em CUDA e desligado em CPU; ``--no-compile`` ou
    ``"compile": false`` na config continuam sendo a saída para eager.
    """
    if cli_value is not None:
        return bool(cli_value)
    return bool(tr.get("compile", torch_device_is_accelerated(device)))


def apply_compile(torch, model, device, backend=None):
    """``torch.compile(model)`` com fallback seguro para eager.

    Devolve ``(modelo, status)``: ``"compiled"`` ou ``"eager_fallback"``.
    ``backend`` só existe para o smoke provar os dois caminhos de fallback.
    Falha na chamada cai para eager na hora; falha tardia (traçado ou
    compilação do backend no 1º passo) cai para eager por frame via
    ``torch._dynamo.config.suppress_errors``, com o motivo no log do dynamo.
    """
    if not torch_device_is_accelerated(device):
        print(
            "AVISO: torch.compile em CPU (prova local); o ganho é medido "
            "no smoke da sessão GPU.",
            file=sys.stderr,
        )
    try:
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True
        compiled = torch.compile(model, **({"backend": backend} if backend else {}))
    except Exception as exc:  # noqa: BLE001 - qualquer falha vira eager
        print(
            f"AVISO: torch.compile falhou ({type(exc).__name__}: {exc}); "
            "seguindo em eager.",
            file=sys.stderr,
        )
        return model, "eager_fallback"
    print(
        "torch.compile: LIGADO (modo default; erro de compilação no 1º passo "
        "cai para eager)"
    )
    return compiled, "compiled"


def enable_channels_last(torch, model):
    """Pesos 4D e ``pixel_values`` em ``channels_last`` (experimento opt-in).

    Os pesos vão com ``model.to(memory_format=...)``; o hook converte o
    lote de entrada em toda chamada do modelo — treino, passos, validação,
    sonda e régua de flicker usam o mesmo caminho. Devolve o handle do
    hook (``disable_channels_last`` o remove antes do export ONNX).
    """

    def _to_channels_last(_module, args, kwargs):
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is not None and pixel_values.dim() == 4:
            kwargs["pixel_values"] = pixel_values.contiguous(
                memory_format=torch.channels_last
            )
        elif args and hasattr(args[0], "dim") and args[0].dim() == 4:
            args = (args[0].contiguous(memory_format=torch.channels_last),) + tuple(
                args[1:]
            )
        return args, kwargs

    model.to(memory_format=torch.channels_last)
    return model.register_forward_pre_hook(_to_channels_last, with_kwargs=True)


def disable_channels_last(torch, model, handle):
    """Desfaz ``enable_channels_last``: remove o hook e volta ao layout contíguo."""
    if handle is not None:
        handle.remove()
    model.to(memory_format=torch.contiguous_format)


def _model_logits(model, images):
    """Accepta SegformerOutput ou Tensor (saída do DataParallel multi-GPU)."""
    output = model(pixel_values=images)
    return output if hasattr(output, "shape") else output.logits


def _model_logits_and_presence(model, images):
    """(logits, presence_logit|None) numa ÚNICA passada do modelo.

    Sem cabeça de presença, devolve (logits, None) — caminho histórico.
    Com a cabeça ligada mas a saída sem o logit (ex.: wrapper
    DataParallel, que descarta saídas auxiliares), devolve None e quem
    chama decide: no treino isso é erro alto, nunca degradação muda.
    """
    output = model(pixel_values=images)
    if hasattr(output, "shape"):
        return output, None
    presence = None
    try:
        presence = output["presence_logit"]
    except Exception:
        presence = getattr(output, "presence_logit", None)
    return output.logits, presence


def _data_parallel(torch, model):
    """Usa todas as GPUs visíveis sem mudar o contrato do modelo base."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return model, model

    class LogitsOnly(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, pixel_values):
            return self.wrapped(pixel_values=pixel_values).logits

    parallel = torch.nn.DataParallel(LogitsOnly(model))
    return parallel, model


def loader_kwargs(torch, device, training_cfg):
    """Kwargs agressivos de DataLoader, gated por config (defaults ligados).

    - num_workers: config manda; sem config, min(núcleos, 8).
    - pin_memory só em CUDA (em CPU é custo sem ganho).
    - persistent_workers/prefetch_factor só com workers > 0 (o PyTorch
      rejeita os dois com num_workers=0).
    Desligar: "fast_loader": false na config, ou num_workers: 0.
    """
    workers = training_cfg.get("num_workers")
    if workers is None:
        workers = min(os.cpu_count() or 1, 8)
    kwargs = {"num_workers": int(workers)}
    if not training_cfg.get("fast_loader", True):
        return kwargs
    if torch.device(device).type == "cuda":
        kwargs["pin_memory"] = True
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(training_cfg.get("prefetch_factor", 2))
    return kwargs


def probe_batch_size(
    torch,
    model,
    dataset,
    class_weights,
    device,
    scaler=None,
    amp_dtype=None,
    start=8,
    target_fraction=0.85,
):
    """Maior batch que cabe em ~target_fraction da VRAM, medido no PASSO REAL.

    Cada candidato roda forward + combined_loss + backward + passo de
    otimizador num lote de verdade do dataset, com o mesmo autocast/scaler
    da corrida — todos os termos da loss que a corrida usa. Sonda parcial
    não é sonda: a v2 agressiva mediu só parte do grafo, escolheu batch
    192 e o passo real estourou OOM com o termo que faltava (2026-11-03).

    Restaura os pesos ao fim (a sonda não treina). Sem CUDA: no-op limpo,
    retorna (start, {}). Retorna (batch, {candidato: GiB ou "oom"}).
    """
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    start = max(int(start), 1)
    if torch.device(device).type != "cuda" or len(dataset) == 0:
        return start, {}
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    budget_gib = target_fraction * total_gib
    saved_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    probe_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    def try_batch(bs):
        loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        if len(batch) == 4:
            images, masks, sample_weights, _ = batch
            sample_weights = sample_weights.to(device)
        elif len(batch) == 3:
            images, masks, sample_weights = batch
            sample_weights = sample_weights.to(device)
        else:
            images, masks = batch
            sample_weights = None
        images, masks = images.to(device), masks.to(device)
        probe_optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                logits = _model_logits(model, images)
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                loss = combined_loss(
                    logits, masks, class_weights, sample_weights=sample_weights
                )
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(probe_optimizer)
                scaler.update()
            else:
                loss.backward()
                probe_optimizer.step()
            torch.cuda.synchronize()
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            return None
        finally:
            probe_optimizer.zero_grad(set_to_none=True)
        return torch.cuda.max_memory_allocated() / 2**30

    report = {}
    chosen = None
    bs = start
    while bs >= 1:  # desce até um candidato respeitar a meta de VRAM
        peak = try_batch(bs)
        report[bs] = round(peak, 2) if peak is not None else "oom"
        if peak is not None and peak <= budget_gib:
            chosen = bs
            break
        bs //= 2
    if chosen is None:  # nem batch 1 respeitou a meta: usa 1 e segue (a corrida decide)
        chosen = 1
    while chosen < len(dataset):  # sobe dobrando enquanto couber na meta
        nxt = min(chosen * 2, len(dataset))
        if nxt == chosen:
            break
        peak = try_batch(nxt)
        report[nxt] = round(peak, 2) if peak is not None else "oom"
        if peak is None or peak > budget_gib:
            break
        chosen = nxt
    model.load_state_dict(saved_state)
    model.to(device)
    torch.cuda.empty_cache()
    print(
        f"sonda de batch (passo real): escolhido {chosen} "
        f"(VRAM total {total_gib:.1f} GiB, meta {budget_gib:.1f} GiB) -> {report}"
    )
    return chosen, report


def utilization_receipt(
    torch, device, epoch_s, steps, batch_size, precision, probe_report=None
):
    """Recibo de utilização da 1ª época (mesmo espírito do receipt do TPU).

    Tempo/passo e amostras/s sempre; em CUDA, pico de VRAM via torch.cuda
    e utilização instantânea da GPU via pynvml SE ele estiver instalado
    (dependência mole: sem pynvml o campo fica null, nada quebra).
    """
    steps = max(int(steps), 1)
    receipt = {
        "epoch": 0,
        "epoch_s": round(float(epoch_s), 1),
        "s_per_step": round(float(epoch_s) / steps, 3),
        "samples_s": round(steps * batch_size / float(epoch_s), 2)
        if epoch_s > 0
        else None,
        "batch_size": int(batch_size),
        "precision": precision,
        "device": str(device),
        "probe_batch": (
            {str(k): v for k, v in probe_report.items()} if probe_report else None
        ),
        "gpu_name": None,
        "vram_peak_gib": None,
        "vram_total_gib": None,
        "gpu_util_percent_sample": None,
    }
    if torch.device(device).type == "cuda":
        receipt["gpu_name"] = torch.cuda.get_device_name(0)
        receipt["vram_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        receipt["vram_total_gib"] = round(
            torch.cuda.get_device_properties(0).total_memory / 2**30, 2
        )
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            receipt["gpu_util_percent_sample"] = int(
                pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            )
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001 - pynvml é opcional; ausência não é erro
            receipt["gpu_util_percent_sample"] = None
    return receipt


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
    pair_loader=None,
    pair_fraction=0.0,
    consistency_weight=0.0,
    consistency_mode="mse",
    consistency_trim=0.0,
    temporal_area_weight=0.0,
    epoch_stats=None,
    presence_weight=0.0,
    presence_pos_weight=None,
    hnbs_weight=0.0,
    hnbs_threshold=0.35,
    hnbs_topk_fraction=0.005,
    ema=None,
    ema_model=None,
):
    """Uma época. Com ``pair_loader`` + ``pair_fraction`` > 0, essa
    fração dos lotes vira lote de PAR (dois frames da mesma cena):
    os dois frames passam pela loss supervisionada normal e, se
    ``consistency_weight`` > 0, pela consistência temporal mascarada.
    Sem pair_loader (ou fração 0), o laço é o histórico — mesmo RNG,
    mesma ordem, mesma loss (provado por teste de identidade).
    ``epoch_stats``, quando passado, recebe {"pair_batches": n}.
    """
    import random as _random

    import torch
    import torch.nn.functional as F

    model.train()
    total, n = 0.0, 0
    presence_total, presence_n = 0.0, 0
    presence_pos_sum, presence_pos_n = 0.0, 0
    presence_neg_sum, presence_neg_n = 0.0, 0
    accumulation_steps = max(int(accumulation_steps), 1)
    num_batches = len(loader)
    pair_iter = None
    pair_rng = None
    pair_batches = 0
    if pair_loader is not None and pair_fraction > 0.0:
        # NUNCA um iterador que guarda os lotes que entrega: ~9 MB por
        # par × 10.000 pares e a RAM do host morre no meio da época
        # (rc 247, sem traceback — visto na noite t8, kernels A v3 e B v1,
        # lote 16 e 32). Reitera o loader quando esgota.
        pair_iter = iter(pair_loader)
        pair_rng = _random.Random(20261105)
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        use_pair = pair_iter is not None and pair_rng.random() < pair_fraction
        change = None
        half = 0
        presence_targets_batch = None
        if use_pair:
            try:
                p_img_a, p_mask_a, p_img_b, p_mask_b, p_change = next(pair_iter)
            except StopIteration:
                # Pares esgotados no meio da época: reitera do zero
                # (ordem nova se o loader embaralha; sem cache acumulado).
                pair_iter = iter(pair_loader)
                p_img_a, p_mask_a, p_img_b, p_mask_b, p_change = next(pair_iter)
            images = torch.cat([p_img_a, p_img_b], dim=0)
            masks = torch.cat([p_mask_a, p_mask_b], dim=0)
            change = p_change.to(device)
            half = p_img_a.shape[0]
            sample_weights = None
            pair_batches += 1
        elif len(batch) == 4:
            images, masks, sample_weights, presence_targets_batch = batch
            sample_weights = sample_weights.to(device)
            presence_targets_batch = presence_targets_batch.to(device)
        elif len(batch) == 3:
            images, masks, sample_weights = batch
            sample_weights = sample_weights.to(device)
        else:
            images, masks = batch
            sample_weights = None
        images, masks = images.to(device), masks.to(device)
        device_type = torch.device(device).type
        with torch.autocast(
            device_type=device_type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            if presence_weight > 0.0:
                logits, presence_logit = _model_logits_and_presence(model, images)
                if presence_logit is None:
                    raise RuntimeError(
                        "presence_weight > 0 mas o modelo não devolveu "
                        "presence_logit (cabeça de presença desligada na "
                        "config ou wrapper multi-GPU descartando a saída)."
                    )
            else:
                logits = _model_logits(model, images)
                presence_logit = None
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
            loss = combined_loss(
                logits,
                masks,
                class_weights,
                sample_weights=sample_weights,
                hnbs_weight=hnbs_weight,
                hnbs_threshold=hnbs_threshold,
                hnbs_topk_fraction=hnbs_topk_fraction,
            )
            if presence_logit is not None:
                p_loss = presence_loss(
                    presence_logit,
                    masks,
                    sample_weights,
                    targets=presence_targets_batch,
                    pos_weight=presence_pos_weight,
                )
                loss = loss + presence_weight * p_loss
                presence_total += float(p_loss.detach().item())
                presence_n += 1
                with torch.no_grad():
                    eff_targets = presence_targets(masks)
                    if presence_targets_batch is not None:
                        eff_targets = torch.where(
                            presence_targets_batch >= 0,
                            presence_targets_batch,
                            eff_targets,
                        )
                    probs = torch.sigmoid(presence_logit.detach())
                    pos_sel = eff_targets >= 0.5
                    if bool(pos_sel.any()):
                        presence_pos_sum += float(probs[pos_sel].sum())
                        presence_pos_n += int(pos_sel.sum())
                    if bool((~pos_sel).any()):
                        presence_neg_sum += float(probs[~pos_sel].sum())
                        presence_neg_n += int((~pos_sel).sum())
            if use_pair and consistency_weight > 0.0:
                loss = loss + consistency_weight * masked_temporal_consistency_loss(
                    logits[:half],
                    logits[half:],
                    change,
                    mode=consistency_mode,
                    trim=consistency_trim,
                )
                if temporal_area_weight > 0.0:
                    loss = loss + temporal_area_weight * temporal_area_loss(
                        logits[:half], logits[half:]
                    )
        group_start = (batch_index // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, num_batches - group_start)
        scaled_loss = loss / group_size
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        should_step = (
            batch_index + 1
        ) % accumulation_steps == 0 or batch_index + 1 == num_batches
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
            if ema is not None:
                ema.update(ema_model if ema_model is not None else model)
        total += float(loss.detach().item())
        n += 1
    if n == 0:
        raise ValueError("época sem lotes de treino")
    if epoch_stats is not None:
        epoch_stats["pair_batches"] = pair_batches
        if presence_n:
            epoch_stats["presence_loss"] = presence_total / presence_n
        if presence_pos_n:
            epoch_stats["presence_mean_pos"] = presence_pos_sum / presence_pos_n
        if presence_neg_n:
            epoch_stats["presence_mean_neg"] = presence_neg_sum / presence_neg_n
    return total / n


def train_steps(
    model,
    replay_buffer,
    pool_buffer,
    pair_buffer,
    *,
    steps,
    batch_replay,
    batch_pool,
    batch_pairs,
    augmenter,
    optimizer,
    class_weights,
    device,
    scaler=None,
    scheduler=None,
    amp_dtype=None,
    consistency_weight=0.0,
    consistency_mode="mse",
    consistency_trim=0.0,
    temporal_area_weight=0.0,
    presence_weight=0.0,
    presence_pos_weight=None,
    hnbs_weight=0.0,
    hnbs_threshold=0.35,
    hnbs_topk_fraction=0.005,
    ema=None,
    ema_model=None,
    step_stats=None,
    stats_every=50,
    profile_window=None,
):
    """Loop guiado por PASSOS sobre buffers em RAM (Fase 2, 2026-11-09).

    Alternativa ao caminho de épocas para FINE-TUNE incremental: em vez
    de varrer ~20,8 mil imagens/época (decode + albumentations em CPU por
    item — passo de parede medido de 0,567 s na A100 contra 0,0445 s de
    passo puro), cada passo amostra índices dos buffers residentes
    (training/ram_buffer.py), transfere uint8 pro device e aumenta/
    normaliza na GPU (draw/gpu_augmenter.py). A matemática de loss é A
    MESMA do train_one_epoch: lote principal supervisionado (replay +
    pool, com pesos por amostra e alvos de presença) + lote de pares
    supervisionado pelas máscaras do teacher + consistência temporal
    mascarada — os pares seguem supervisionados, como na Rodada 3/t14
    (não só o termo temporal). Composição do lote (batch 32, frações da
    config t14): 12 replay + 10 pool + 5 pares (10 frames).

    Divergências deliberadas do caminho de épocas, registradas:
    - augmentation geométrica por passo é só flip horizontal (na GPU);
      affine/perspective/blur/HSV ficam de fora — a diversidade
      geométrica do replay nasce na geração das amostras do buffer.
    - imagens do buffer já estão redimensionadas pro tamanho do modelo
      (a mesma troca do cache_mode="resized").
    - sem acumulação de gradiente: 1 passo = 1 update.
    - estatísticas de presença (pstats) e o acumulado da loss NÃO
      sincronizam a cada passo: cada `.item()` drena a fila da GPU e
      custa o passo inteiro (o smoke A100 mediu 0,375 s/passo contra
      0,0445 s de passo puro — ~9-11 syncs/passo). pstats roda só no
      lote principal a cada `stats_every` passos (passo 0 incluído) e
      o total da loss acumula em tensor no device, com UM `.item()`
      no retorno. As médias de presença em step_stats seguem válidas
      (amostradas); o caminho de épocas tem o mesmo vício e fica para
      depois — aqui só o caminho de passos é o portão.
    - perfil por fase (opcional, `profile_window=(inicio, fim)`): nos
      passos da janela, cada fase (sample/augment/forward/loss/
      temporal/backward/optim_ema) é medida com sync nas fronteiras —
      cada fase paga o próprio custo serializado e a soma fecha o
      tempo do passo. Nasceu da refutação da hipótese dos `.item()`:
      o re-run mediu 0,401 s/passo com os syncs removidos, então o
      recibo seguinte tem que dizer ONDE o tempo mora em vez de
      outro chute. Fora da janela, custo zero.
    - curva por passo: com perfil ativo, `step_stats["step_wall_ms"]`
      guarda a parede de CADA passo da janela (sync nas duas pontas do
      passo inteiro). Nasceu do smoke perfilado de 2026-11-09: as fases
      somaram ~150 ms/passo nos passos 10–29, mas a capa do recibo
      seguiu 0,403 s/passo — a curva diz ONDE a média se concentra
      (warmup nos primeiros passos × custo uniforme) em vez de média
      de fases sobre uma janela só.
    """
    import contextlib
    import time as _time

    import torch
    import torch.nn.functional as F

    model.train()
    total = torch.zeros((), device=device)
    n = 0
    pstats = {
        "total": 0.0,
        "n": 0,
        "pos_sum": 0.0,
        "pos_n": 0,
        "neg_sum": 0.0,
        "neg_n": 0,
    }
    device_type = torch.device(device).type

    phase_acc: dict = {}
    phase_steps = 0
    step_wall: list = []

    @contextlib.contextmanager
    def _phase(name, active):
        if not active:
            yield
            return
        if device_type == "cuda":
            torch.cuda.synchronize()
        _t0 = _time.perf_counter()
        try:
            yield
        finally:
            if device_type == "cuda":
                torch.cuda.synchronize()
            phase_acc[name] = phase_acc.get(name, 0.0) + (_time.perf_counter() - _t0)

    def supervised(
        images_u8,
        masks,
        sample_weights,
        pres_targets,
        flip,
        collect_stats=False,
        prof=False,
    ):
        with _phase("augment", prof):
            images, masks = augmenter(images_u8, masks, jitter=True, flip=flip)
        with _phase("forward", prof):
            if presence_weight > 0.0:
                logits, presence_logit = _model_logits_and_presence(model, images)
                if presence_logit is None:
                    raise RuntimeError(
                        "presence_weight > 0 mas o modelo não devolveu "
                        "presence_logit (cabeça de presença desligada na "
                        "config ou wrapper multi-GPU descartando a saída)."
                    )
            else:
                logits = _model_logits(model, images)
                presence_logit = None
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
        with _phase("loss", prof):
            loss = combined_loss(
                logits,
                masks,
                class_weights,
                sample_weights=sample_weights,
                hnbs_weight=hnbs_weight,
                hnbs_threshold=hnbs_threshold,
                hnbs_topk_fraction=hnbs_topk_fraction,
            )
            if presence_logit is not None:
                p_loss = presence_loss(
                    presence_logit,
                    masks,
                    sample_weights,
                    targets=pres_targets,
                    pos_weight=presence_pos_weight,
                )
                loss = loss + presence_weight * p_loss
        if presence_logit is not None and collect_stats:
            pstats["total"] += float(p_loss.detach().item())
            pstats["n"] += 1
            with torch.no_grad():
                eff_targets = presence_targets(masks)
                if pres_targets is not None:
                    eff_targets = torch.where(
                        pres_targets >= 0, pres_targets, eff_targets
                    )
                probs = torch.sigmoid(presence_logit.detach())
                pos_sel = eff_targets >= 0.5
                if bool(pos_sel.any()):
                    pstats["pos_sum"] += float(probs[pos_sel].sum())
                    pstats["pos_n"] += int(pos_sel.sum())
                if bool((~pos_sel).any()):
                    pstats["neg_sum"] += float(probs[~pos_sel].sum())
                    pstats["neg_n"] += int((~pos_sel).sum())
        return loss, logits

    for step_idx in range(steps):
        prof = (
            profile_window is not None
            and profile_window[0] <= step_idx < profile_window[1]
        )
        if prof and device_type == "cuda":
            torch.cuda.synchronize()
        _step_t0 = _time.perf_counter() if prof else None
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device_type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            with _phase("sample", prof):
                imgs, msks, wts, pres = [], [], [], []
                if batch_replay > 0:
                    i, m, w, p = replay_buffer.sample(batch_replay, device)
                    imgs.append(i)
                    msks.append(m)
                    wts.append(w)
                    pres.append(p)
                if pool_buffer is not None and batch_pool > 0:
                    i, m, w, p = pool_buffer.sample(batch_pool, device)
                    imgs.append(i)
                    msks.append(m)
                    wts.append(w)
                    pres.append(p)
                main_batch = (
                    torch.cat(imgs),
                    torch.cat(msks),
                    torch.cat(wts),
                    torch.cat(pres),
                )
            loss, _ = supervised(
                *main_batch,
                flip=True,
                collect_stats=(step_idx % stats_every == 0),
                prof=prof,
            )
            if pair_buffer is not None and batch_pairs > 0:
                with _phase("sample", prof):
                    ia, ma, ib, mb, change = pair_buffer.sample_pairs(
                        batch_pairs, device
                    )
                    pair_batch = (
                        torch.cat([ia, ib]),
                        torch.cat([ma, mb]),
                    )
                pair_loss, pair_logits = supervised(
                    *pair_batch,
                    None,
                    None,
                    flip=False,
                    prof=prof,
                )
                loss = loss + pair_loss
                if consistency_weight > 0.0:
                    with _phase("temporal", prof):
                        half = ia.shape[0]
                        loss = loss + consistency_weight * (
                            masked_temporal_consistency_loss(
                                pair_logits[:half],
                                pair_logits[half:],
                                change,
                                mode=consistency_mode,
                                trim=consistency_trim,
                            )
                        )
                        if temporal_area_weight > 0.0:
                            loss = loss + (
                                temporal_area_weight
                                * temporal_area_loss(
                                    pair_logits[:half], pair_logits[half:]
                                )
                            )
        with _phase("backward", prof):
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        with _phase("optim_ema", prof):
            if scaler is not None:
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                did_step = scaler.get_scale() >= previous_scale
            else:
                optimizer.step()
                did_step = True
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None and did_step:
                scheduler.step()
            if ema is not None:
                ema.update(ema_model if ema_model is not None else model)
        if prof:
            phase_steps += 1
            if device_type == "cuda":
                torch.cuda.synchronize()
            step_wall.append(round((_time.perf_counter() - _step_t0) * 1000.0, 2))
        total = total + loss.detach()
        n += 1
    if n == 0:
        raise ValueError("train_steps sem passos executados")
    if step_stats is not None:
        step_stats["pair_batches"] = n if pair_buffer is not None else 0
        if phase_steps:
            step_stats["phase_profile_ms"] = {
                k: round(v * 1000.0 / phase_steps, 2)
                for k, v in sorted(phase_acc.items())
            }
            step_stats["phase_profile_steps"] = phase_steps
        if step_wall:
            step_stats["step_wall_ms"] = step_wall
        if pstats["n"]:
            step_stats["presence_loss"] = pstats["total"] / pstats["n"]
        if pstats["pos_n"]:
            step_stats["presence_mean_pos"] = pstats["pos_sum"] / pstats["pos_n"]
        if pstats["neg_n"]:
            step_stats["presence_mean_neg"] = pstats["neg_sum"] / pstats["neg_n"]
    return float((total / n).detach().item())


def evaluate(model, loader, device, target_loader=None):
    """Avalia todo o loader usando as métricas compartilhadas do projeto.

    ``target_loader`` permite rodar a rede numa resolução (ex.: 256) e
    avaliar os logits na resolução do ground truth de referência (ex.: 512).
    Assim, ablações de resolução compartilham o mesmo canvas e a mesma
    tolerância física de tinta. Os dois loaders precisam ter a mesma ordem,
    batch size e quantidade de lotes.
    """
    import torch

    if target_loader is not None and len(loader) != len(target_loader):
        raise ValueError("loader de entrada e target_loader com tamanhos diferentes")
    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    supervised_px, ignored_px = 0, 0
    tolerance_totals = {
        radius: {"pred_ink": 0, "gt_ink": 0, "matched_pred": 0, "matched_gt": 0}
        for radius in (2, 3)
    }
    iterator = zip(loader, target_loader) if target_loader is not None else loader
    with torch.no_grad():
        for batch in iterator:
            if target_loader is not None:
                (images, input_masks), (_, masks) = batch
                if input_masks.shape[0] != masks.shape[0]:
                    raise ValueError("batches de entrada e referência fora de ordem")
            else:
                images, masks = batch
            images = images.to(device)
            logits = _model_logits(model, images)
            if logits.shape[-2:] != masks.shape[-2:]:
                import torch.nn.functional as F

                logits = F.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
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
        raise ValueError(
            "avaliação sem pixels supervisionados: todas as máscaras são 255"
        )
    metrics = shared_metrics.evaluate_confusion_matrix(confusion, ignored_px=ignored_px)
    aggregate_tolerance = {}
    for radius, counts in tolerance_totals.items():
        pred_ink, gt_ink = counts["pred_ink"], counts["gt_ink"]
        precision = (
            counts["matched_pred"] / pred_ink
            if pred_ink
            else (1.0 if not gt_ink else 0.0)
        )
        recall = (
            counts["matched_gt"] / gt_ink if gt_ink else (1.0 if not pred_ink else 0.0)
        )
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
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


def make_onnx_export_model(model, size, include_presence=False):
    """Wrap SegFormer so exported logits match the full input resolution.

    ``include_presence=True`` devolve (logits, presence) — o ONNX ganha
    a segunda saída ``presence`` SEMPRE depois de ``logits``: bancada e
    validação leem a saída [0], então o contrato antigo não muda.
    """
    import torch
    import torch.nn.functional as F

    class ExportModel(torch.nn.Module):
        def __init__(self, wrapped_model, output_size):
            super().__init__()
            self.model = wrapped_model
            self.output_size = output_size

        def forward(self, image):
            output = self.model(pixel_values=image)
            logits = output.logits
            if logits.shape[-2:] != (self.output_size, self.output_size):
                logits = F.interpolate(
                    logits,
                    size=(self.output_size, self.output_size),
                    mode="bilinear",
                    align_corners=False,
                )
            if include_presence:
                return logits, output["presence_logit"]
            return logits

    return ExportModel(model, size).eval()


def export_onnx(model, out_path, cfg):
    """Exporta para ONNX com o contrato do Draw: input 'image', output 'logits'."""
    import torch

    model.eval()
    size = cfg["model"]["input"]["height"]
    with_presence = bool((cfg.get("presence") or {}).get("enabled"))
    export_model = make_onnx_export_model(model, size, include_presence=with_presence)
    dummy = _make_onnx_dummy(export_model, size)
    # dynamo=False: exportador legado. O exportador dynamo padrão do torch
    # 2.11 exige onnxscript, que o ambiente do Kaggle não tem — sem isso,
    # o treino inteiro falhava na exportação, depois das épocas, e perdia
    # até o metrics.json. O exportador legado usa só o pacote onnx.
    output_names = ["logits", "presence"] if with_presence else ["logits"]
    dynamic_axes = {"image": [0], "logits": [0]}
    if with_presence:
        dynamic_axes["presence"] = [0]
    torch.onnx.export(
        export_model,
        (dummy,),
        f=out_path,
        input_names=["image"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=cfg["export"]["opset_version"],
        do_constant_folding=True,
        dynamo=False,
    )
    return out_path


def onnx_delivery_status(report):
    """Status de entrega: somente uma comparação executada e aprovada libera."""
    status = report.get("status") if isinstance(report, dict) else None
    return "validated" if status == "ok" else "blocked_onnx_validation"


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
        print(
            f"AVISO: validação ONNX não executada: {report['reason']}", file=sys.stderr
        )
    return report


# ---------------------------------------------------------------- smoke
def _smoke_compile_and_channels_last(torch):
    """Prova barata (sem inductor) do fallback do compile e do channels_last."""
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(4, 4, 1)
    )
    x = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        eager_out = net(x)

    # Falha na chamada (backend inexistente) -> eager imediato.
    same, status = apply_compile(torch, net, "cpu", backend="draw_smoke_inexistente")
    assert status == "eager_fallback" and same is net, status

    # Falha tardia (backend quebra no 1º passo) -> eager por frame.
    def _broken_backend(_gm, _example_inputs):
        raise RuntimeError("falha simulada do backend (smoke)")

    import torch._dynamo

    previous_suppress = torch._dynamo.config.suppress_errors
    try:
        compiled, status = apply_compile(torch, net, "cpu", backend=_broken_backend)
        assert status == "compiled", status
        out = compiled(x)
        out.sum().backward()
        assert torch.isfinite(out).all(), "fallback tardio do compile não finito"
        assert torch.allclose(out.detach(), eager_out, atol=1e-6), (
            "fallback tardio do compile mudou a saída"
        )
    finally:
        torch._dynamo.config.suppress_errors = previous_suppress
        torch._dynamo.reset()

    handle = enable_channels_last(torch, net)
    assert net[0].weight.is_contiguous(memory_format=torch.channels_last)
    seen = []
    probe = net[0].register_forward_pre_hook(
        lambda _m, a: seen.append(a[0].is_contiguous(memory_format=torch.channels_last))
    )
    with torch.no_grad():
        cl_out = net(x)
    probe.remove()
    assert seen == [True], "lote não chegou em channels_last"
    assert torch.allclose(cl_out, eager_out, atol=1e-5), "channels_last mudou a saída"
    disable_channels_last(torch, net, handle)
    assert net[0].weight.is_contiguous()
    print(
        "SMOKE OK: torch.compile cai para eager na chamada e no 1º passo; "
        "channels_last preserva a saída e é reversível"
    )


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
    cv2.imwrite(
        str(img_dir / "scene_0002.png"), np.full((64, 64, 3), 90, dtype=np.uint8)
    )
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
    pred[pred == IGNORE_INDEX] = (
        0  # predições sempre são classes válidas; só o GT usa 255.
    )
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
        print(
            "AVISO: PyTorch completo ausente; smoke de loss torch com pixel ignorado pulado.",
            file=sys.stderr,
        )
    else:
        # ONNX export receives a CUDA model after Kaggle training. The dummy
        # input must follow that model's device; CPU is still the fallback for
        # parameterless modules and the local smoke path.
        for test_device in ("cpu", "meta"):
            probe = torch.nn.Linear(1, 1).to(test_device)
            export_input = _make_onnx_dummy(probe, 8)
            assert export_input.device == next(probe.parameters()).device, (
                test_device,
                export_input.device,
                next(probe.parameters()).device,
            )
        empty_probe = torch.nn.Identity()
        assert _make_onnx_dummy(empty_probe, 8).device.type == "cpu"
        target_t = torch.from_numpy(ignored_target[None].astype(np.int64))
        logits = torch.zeros(
            (1, NUM_CLASSES, 64, 64), dtype=torch.float32, requires_grad=True
        )
        weights_t = torch.tensor(weights, dtype=torch.float32)
        loss = combined_loss(logits, target_t, weights_t)
        assert torch.isfinite(loss), loss
        loss.backward()
        assert torch.isfinite(logits.grad).all(), "gradiente não finito com pixels 255"
        assert torch.count_nonzero(logits.grad[:, :, 54:60, 54:60]) == 0, (
            "pixel 255 deve ter gradiente zero"
        )
        all_ignored = torch.full_like(target_t, IGNORE_INDEX)
        ignored_logits = torch.zeros_like(logits.detach(), requires_grad=True)
        ignored_loss = combined_loss(ignored_logits, all_ignored, weights_t)
        assert torch.isfinite(ignored_loss) and ignored_loss.item() == 0.0, ignored_loss
        ignored_loss.backward()
        assert torch.count_nonzero(ignored_logits.grad) == 0, (
            "lote ignorado deve gerar gradiente zero"
        )

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
        assert eval_metrics["iou"]["ink"] == 1.0, eval_metrics
        assert eval_metrics["support_px"]["ink"] == 36, eval_metrics
        assert eval_metrics["supervised_px"] == 64 * 64 - ignored, eval_metrics
        assert eval_metrics["ignored_px"] == ignored, eval_metrics
        try:
            evaluate(_FixedModel(), [(images_t, all_ignored)], "cpu")
        except ValueError as exc:
            assert "sem pixels supervisionados" in str(exc), exc
        else:
            raise AssertionError(
                "avaliação totalmente ignorada deveria falhar claramente"
            )
        _smoke_compile_and_channels_last(torch)
    # Contrato ONNX: nomes e shapes esperados, sem rodar modelo.
    contract = {
        "input_name": "image",
        "output_name": "logits",
        "input_shape": [1, 3, 512, 512],
        "output_shape": [1, 4, 512, 512],
    }
    assert contract["output_shape"][1] == NUM_CLASSES
    print(
        f"SMOKE OK: {len(pairs)} pares, pixels supervisionados {counts.tolist()}, ignorados {ignored}, pesos {[round(w, 2) for w in weights]}"
    )
    print(
        f"SMOKE OK: ink IoU estrita {m['ink']['iou']:.3f}, recall com tolerância 2px {tol['radius_2px']['recall']:.3f}"
    )
    print(
        "SMOKE OK: pixels 255 excluídos de métricas e contagens; loss com pixel ignorado é verificada quando PyTorch está disponível"
    )
    print(
        "SMOKE OK: amostra ONNX acompanha o dispositivo do modelo (CPU/meta; CUDA se disponível no runtime)"
    )
    print(f"SMOKE OK: contrato ONNX {contract}")
    return 0


# ---------------------------------------------------------------- main
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Treino SegFormer-B0 do Draw (4 classes)."
    )
    parser.add_argument("--config", default="training/config.json")
    parser.add_argument("--data-dir", default="dataset/synthetic")
    parser.add_argument(
        "--real-train-dir",
        default=None,
        help="Diretório opcional com images/ e masks/ para adicionar ao treino sintético; val/test continuam em --data-dir.",
    )
    parser.add_argument(
        "--presence-pool-dir",
        default=None,
        help="Diretório opcional do pool de presença (labels.csv + frames): "
        "frames reais rotulados só com has_paper; treinam SÓ a cabeça de presença.",
    )
    parser.add_argument(
        "--real-pairs-dir",
        default=None,
        help="Diretório opcional de pares reais (pairs.csv + frames + "
        "máscaras do teacher): pares (t, t+Δ) de cenas reais pro termo de "
        "consistência temporal da Rodada 3; exige training.temporal.enabled.",
    )
    parser.add_argument(
        "--init-presence",
        default=None,
        help="Checkpoint opcional de onde transplantar SÓ a cabeça de "
        "presença (Rodada 3: a cabeça já treinada da Rodada 2) por cima do "
        "warm init principal; também pode vir de training.init_presence_checkpoint.",
    )
    parser.add_argument("--out-dir", default="out")
    parser.add_argument(
        "--allow-unverified-real-train",
        action="store_true",
        help="Autoriza propostas reais não revisadas com peso reduzido; val/test continuam fora delas.",
    )
    parser.add_argument(
        "--real-loss-weight",
        type=float,
        default=None,
        help="Peso por amostra real extra na loss (padrão: loss.unverified_real_sample_weight).",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Smoke test sem GPU, sem treino."
    )
    parser.add_argument("--smoke-dir", default="/tmp/draw-smoke")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--precision",
        choices=("bf16", "fp16", "fp32"),
        default=None,
        help="Substitui a precisão mista da config nesta corrida.",
    )
    parser.add_argument(
        "--probe-batch",
        action="store_true",
        help="Dimensiona o batch pela VRAM rodando o passo real de treino (só CUDA); "
        'equivale a batch_size: "auto" na config.',
    )
    parser.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        default=None,
        help="Liga torch.compile no modelo de treino (sobrescreve a config).",
    )
    parser.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help="Desliga torch.compile (sobrescreve a config).",
    )
    parser.add_argument(
        "--channels-last",
        dest="channels_last",
        action="store_true",
        default=None,
        help="Experimento: pesos e lotes em channels_last (training.channels_last). "
        "Meça s/passo e métricas contra o eager contíguo antes de adotar.",
    )
    parser.add_argument(
        "--no-channels-last",
        dest="channels_last",
        action="store_false",
        help="Desliga channels_last (sobrescreve a config).",
    )
    parser.add_argument(
        "--mode",
        choices=("epochs", "steps"),
        default=None,
        help="training.mode: épocas sobre o catálogo ou passos sobre buffers em RAM.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="training.max_steps do modo steps (padrão da config: 1500).",
    )
    parser.add_argument(
        "--eval-interval-steps",
        type=int,
        default=None,
        help="training.eval_interval_steps: passos por bloco de avaliação (300).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="early_stopping.patience_epochs: épocas (ou blocos, no modo steps) "
        "sem melhora de val_mean_iou antes de parar.",
    )
    args = parser.parse_args(argv)
    for name in ("max_steps", "eval_interval_steps", "patience"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} precisa ser positivo")

    cfg = load_config(args.config)
    if args.smoke:
        return smoke(args.smoke_dir)
    if args.precision is not None:
        cfg["training"]["mixed_precision"] = args.precision
    if args.channels_last is not None:
        cfg["training"]["channels_last"] = args.channels_last
    if args.mode is not None:
        cfg["training"]["mode"] = args.mode
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
    if args.eval_interval_steps is not None:
        cfg["training"]["eval_interval_steps"] = args.eval_interval_steps
    if args.patience is not None:
        cfg["training"]["early_stopping"]["patience_epochs"] = args.patience

    real_manifest = None
    real_labels_unverified = False
    if args.real_train_dir:
        real_manifest_path = Path(args.real_train_dir) / "dataset_manifest.json"
        if real_manifest_path.is_file():
            real_manifest = json.loads(real_manifest_path.read_text(encoding="utf-8"))
            review_text = " ".join(
                str(real_manifest.get(key, ""))
                for key in ("semantic_review_status", "label_status", "warning")
            ).lower()
            real_labels_unverified = (
                "unverified" in review_text or "not ground truth" in review_text
            )
        if real_labels_unverified and not args.allow_unverified_real_train:
            parser.error(
                "o pacote real declara rótulos não revisados; revise as máscaras ou "
                "passe --allow-unverified-real-train para usar peso reduzido"
            )
    real_loss_weight = (
        args.real_loss_weight
        if args.real_loss_weight is not None
        else float(cfg["loss"].get("unverified_real_sample_weight", 0.25))
    )
    if not 0.0 < real_loss_weight <= 1.0:
        parser.error("--real-loss-weight precisa estar em (0, 1]")

    import torch
    from torch.utils.data import DataLoader

    tr = cfg["training"]
    epochs = args.epochs or tr["epochs"]
    raw_batch = args.batch_size or tr.get("batch_size", 8)
    probe_requested = bool(args.probe_batch or raw_batch == "auto")
    batch_size = 8 if raw_batch == "auto" else int(raw_batch)
    probe_report = None
    torch.manual_seed(tr["seed"])
    np.random.seed(tr["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "AVISO: sem GPU; o treino completo em CPU é inviável. Use o Kaggle com GPU.",
            file=sys.stderr,
        )

    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    split_manifest, split_manifest_path, split_validation = load_split_manifest(
        args.data_dir
    )
    if split_manifest_path is not None:
        print(
            f"split manifest: {split_manifest_path} | famílias verificadas: "
            f"{split_validation['checked_source_families']}"
        )
    real_pairs = []
    if args.real_train_dir:
        real_root = Path(args.real_train_dir)
        if (real_root / "train" / "images").is_dir():
            real_root = real_root / "train"
        real_pairs = pair_images_masks(real_root)
    synthetic_train_pairs = pair_images_masks(Path(args.data_dir) / "train")
    cache_mode = str(tr.get("cache_mode", "none"))
    train_ds = build_dataset(
        args.data_dir,
        "train",
        cfg,
        augment=True,
        extra_pairs=real_pairs,
        extra_sample_weight=real_loss_weight,
        return_sample_weights=bool(real_pairs or args.presence_pool_dir),
        cache_mode=cache_mode,
        cache_cap_mb=int(tr.get("cache_cap_mb", 24576)),
    )
    val_ds = build_dataset(args.data_dir, "val", cfg, augment=False)
    test_ds = build_dataset(args.data_dir, "test", cfg, augment=False)
    pool_ds = None
    pool_count = 0
    if args.presence_pool_dir:
        pool_ds = build_pool_dataset(args.presence_pool_dir, cfg, augment=True)
        pool_count = len(pool_ds)
    configure_fast_runtime(torch, device)
    train_pairs = synthetic_train_pairs + real_pairs
    counts = pixel_counts(train_pairs)
    print(
        f"pares de treino: {len(synthetic_train_pairs)} sintéticos + "
        f"{len(real_pairs)} reais = {len(train_pairs)}"
    )
    print(f"pixels de treino por classe: {counts.tolist()}")
    print(f"pesos da config: {cfg['loss']['cross_entropy_class_weights']}")
    print(
        f"pesos calculados (referência): {[round(w, 2) for w in compute_class_weights(counts)]}"
    )
    if real_pairs:
        print(
            f"peso por amostra real na loss: {real_loss_weight} | "
            f"não revisados: {real_labels_unverified}"
        )

    model, base_model = _data_parallel(torch, build_model(cfg).to(device))
    channels_last = bool(tr.get("channels_last", False))
    channels_last_hook = None
    if channels_last:
        channels_last_hook = enable_channels_last(torch, base_model)
        print(
            "channels_last: LIGADO (experimento; meça s/passo e métricas "
            "contra o eager contíguo antes de adotar)"
        )
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"data parallel: usando {torch.cuda.device_count()} GPUs CUDA")
    presence_init_path = args.init_presence or tr.get("init_presence_checkpoint")
    presence_init_report = None
    if presence_init_path:
        presence_init_report = load_init_checkpoint(
            base_model, presence_init_path, only_prefixes=("presence_head.",)
        )
        print(
            f"init presença (transplante da cabeça): {presence_init_report} "
            f"de {presence_init_path}"
        )
        if presence_init_report["loaded"] == 0:
            parser.error(
                "init de presença não carregou nenhum tensor "
                f"de {presence_init_path} — confira o checkpoint da Rodada 2"
            )
    precision, amp_dtype, scaler = configure_mixed_precision(torch, device, tr)
    class_weights = torch.tensor(
        cfg["loss"]["cross_entropy_class_weights"], dtype=torch.float32, device=device
    )
    if probe_requested:
        batch_size, probe_report = probe_batch_size(
            torch,
            model,
            train_ds,
            class_weights,
            device,
            scaler=scaler,
            amp_dtype=amp_dtype,
            start=batch_size,
            target_fraction=float(tr.get("probe_vram_fraction", 0.85)),
        )
    dl_kwargs = loader_kwargs(torch, device, tr)
    print(f"dataloader: {dl_kwargs} | batch_size: {batch_size}")
    # --- Modelo C (t9): presença + HNBS + amostragem balanceada de reais.
    # Tudo desligado por padrão: config sem os blocos = caminho histórico.
    pres_cfg = cfg.get("presence") or {}
    presence_weight = (
        float(pres_cfg.get("loss_weight", 1.0)) if pres_cfg.get("enabled") else 0.0
    )
    hnbs_cfg = (cfg.get("loss") or {}).get("hnbs") or {}
    hnbs_weight = float(hnbs_cfg.get("weight", 0.0))
    hnbs_threshold = float(hnbs_cfg.get("threshold", 0.35))
    hnbs_topk_fraction = float(hnbs_cfg.get("topk_fraction", 0.005))
    presence_pos_weight = pres_cfg.get("pos_weight")
    presence_pos_weight = (
        None if presence_pos_weight is None else float(presence_pos_weight)
    )
    train_sampler = None
    sampler_weights = None
    bal_cfg = tr.get("balanced_real") or {}
    if bal_cfg.get("enabled"):
        if not real_pairs:
            parser.error(
                "training.balanced_real.enabled=true mas não há pares reais "
                "(--real-train-dir); balancear o quê?"
            )
        # Ordem do dataset = pares sintéticos do split + pares reais
        # anexados (build_dataset monta exatamente nessa ordem).
        target_real = float(bal_cfg.get("target_real_fraction", 0.4))
        if not 0.0 < target_real < 1.0:
            parser.error("balanced_real.target_real_fraction precisa estar em (0, 1)")
        n_syn, n_real = len(synthetic_train_pairs), len(real_pairs)
        w_real = (target_real / (1.0 - target_real)) * (n_syn / n_real)
        sampler_weights = [1.0] * n_syn + [w_real] * n_real
        print(
            f"amostragem balanceada: alvo {target_real:.0%} real/lote | "
            f"peso por amostra real {w_real:.2f} (sintética 1,0) | "
            f"{n_syn} sintéticas + {n_real} reais"
        )
    if presence_weight > 0.0 or hnbs_weight > 0.0:
        print(
            f"modelo C: presence_weight {presence_weight} | hnbs_weight "
            f"{hnbs_weight} (limiar {hnbs_threshold}, top {hnbs_topk_fraction:.1%})"
        )
    if pool_ds is not None:
        pool_target = float(tr.get("presence_pool_fraction", 0.30))
        if not 0.0 < pool_target < 1.0:
            parser.error("training.presence_pool_fraction precisa estar em (0, 1)")
        if sampler_weights is None:
            sampler_weights = [1.0] * len(train_ds)
        base_sum = float(sum(sampler_weights))
        w_pool = (pool_target / (1.0 - pool_target)) * (base_sum / pool_count)
        sampler_weights = sampler_weights + [w_pool] * pool_count
        print(
            f"pool de presença na amostragem: alvo {pool_target:.0%} dos lotes | "
            f"peso por frame do pool {w_pool:.2f} | {pool_count} frames"
        )
        from torch.utils.data import ConcatDataset

        train_ds = ConcatDataset([train_ds, pool_ds])
    if sampler_weights is not None:
        from torch.utils.data import WeightedRandomSampler

        train_sampler = WeightedRandomSampler(
            sampler_weights, num_samples=len(sampler_weights), replacement=True
        )
    if train_sampler is not None:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=collate_with_presence if pool_ds is not None else None,
            **dl_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, **dl_kwargs
        )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **dl_kwargs)
    tset = temporal_settings(cfg)
    pair_loader = None
    real_pair_val_ds = None
    if tset["enabled"]:
        from torch.utils.data import ConcatDataset as _ConcatDs

        pair_parts = []
        synth_pair_ds = build_pair_dataset(args.data_dir, "train", cfg)
        if len(synth_pair_ds) > 0:
            pair_parts.append(synth_pair_ds)
        if args.real_pairs_dir:
            real_pair_ds = build_real_pair_dataset(
                args.real_pairs_dir, cfg, split="train", augment=True
            )
            pair_parts.append(real_pair_ds)
            try:
                real_pair_val_ds = build_real_pair_dataset(
                    args.real_pairs_dir, cfg, split="val", augment=False
                )
            except ValueError:
                real_pair_val_ds = None
            print(
                f"temporal real: {len(real_pair_ds)} pares reais de treino"
                + (
                    f" | {len(real_pair_val_ds)} de validação (régua de flicker)"
                    if real_pair_val_ds is not None
                    else " | sem pares de validação"
                )
            )
        if not pair_parts:
            parser.error(
                "training.temporal.enabled=true, mas nenhum par de sequência "
                f"em {Path(args.data_dir) / 'train'} nem pares reais — gere o "
                "dataset com --temporal-train (tools/gen_synthetic.py), passe "
                "--real-pairs-dir ou desligue o bloco"
            )
        pair_ds = pair_parts[0] if len(pair_parts) == 1 else _ConcatDs(pair_parts)
        pair_loader = DataLoader(
            pair_ds, batch_size=batch_size, shuffle=True, **dl_kwargs
        )
        print(
            f"temporal: {len(pair_ds)} pares | fração de lotes "
            f"{tset['pair_fraction']} | peso da consistência "
            f"{tset['consistency_weight']} | modo {tset['mode']} | "
            f"trim {tset['trim']} | peso área {tset['area_weight']}"
        )
    # --- Pacote de eficiência: torch.compile (padrão ligado em CUDA, com
    # fallback para eager) + EMA (gated, padrão off).
    compile_enabled = resolve_compile(args.compile, tr, device)
    compile_status = "off"
    if compile_enabled:
        model, compile_status = apply_compile(torch, model, device)
    else:
        print("torch.compile: DESLIGADO (eager)")
    ema_cfg = tr.get("ema") or {}
    ema = None
    if ema_cfg.get("enabled"):
        ema = WeightEMA(base_model, float(ema_cfg.get("decay", 0.999)))
        print(f"EMA de pesos: LIGADA (decay {ema.decay}) — best/avaliação usam EMA")
    optimizer = build_optimizer(base_model, tr)
    accumulation_steps = max(int(tr.get("gradient_accumulation_steps", 1)), 1)
    # --- Fase 2 (2026-11-09): modo de treino guiado por passos.
    # training.mode="steps" troca a varredura de épocas por max_steps
    # passos sobre buffers em RAM (fine-tune incremental); o caminho de
    # épocas segue intacto como padrão ("epochs").
    steps_mode = str(tr.get("mode", "epochs")).lower() == "steps"
    max_steps = int(tr.get("max_steps", 1500)) if steps_mode else 0
    eval_interval = int(tr.get("eval_interval_steps", 300)) if steps_mode else 0
    _pw = tr.get("profile_window_steps") if steps_mode else None
    profile_window = (int(_pw[0]), int(_pw[1])) if _pw and len(_pw) == 2 else None
    replay_buffer = pool_buffer = pair_buffer = None
    augmenter = None
    batch_replay = batch_pool = batch_pairs = 0
    if steps_mode:
        import random as _random

        from draw.gpu_augmenter import GPUBatchAugmenter
        from training.ram_buffer import RAMImageBuffer, RAMPairBuffer

        size = cfg["model"]["input"]["height"]
        replay_size = int(tr.get("replay_size", 1500))
        rng = _random.Random(int(tr.get("seed", 42)))
        real_items = [(p[0], p[1], real_loss_weight) for p in real_pairs]
        syn_items = [(p[0], p[1], 1.0) for p in synthetic_train_pairs]
        rng.shuffle(syn_items)
        replay_items = real_items + syn_items[: max(0, replay_size - len(real_items))]
        rng.shuffle(replay_items)
        replay_buffer = RAMImageBuffer.from_pairs(
            replay_items, size, "replay", load_image_rgb, load_mask, cv2
        )
        if args.presence_pool_dir:
            pool_buffer = RAMImageBuffer.from_pool_dir(
                args.presence_pool_dir, size, load_image_rgb, cv2
            )
        if args.real_pairs_dir:
            pair_buffer = RAMPairBuffer.from_pairs_dir(
                args.real_pairs_dir,
                size,
                load_image_rgb,
                load_mask,
                cv2,
                split="train",
            )
        norm = cfg["model"]["input"]["normalization"]
        augmenter = GPUBatchAugmenter(mean=norm["mean"], std=norm["std"]).to(device)
        pool_fraction = (
            float(tr.get("presence_pool_fraction", 0.30)) if pool_buffer else 0.0
        )
        pair_frames = (
            int(round(batch_size * tset["pair_fraction"]))
            if pair_buffer is not None and tset["enabled"]
            else 0
        )
        batch_pairs = pair_frames // 2
        batch_pool = int(round(batch_size * pool_fraction)) if pool_buffer else 0
        batch_replay = batch_size - batch_pool - 2 * batch_pairs
        if batch_replay <= 0:
            parser.error(
                "modo steps: a composição do lote estourou o batch "
                f"(replay {batch_replay} + pool {batch_pool} + pares "
                f"{2 * batch_pairs} > {batch_size})"
            )
        print(
            f"modo steps: max_steps {max_steps} | avaliação a cada "
            f"{eval_interval} | lote = {batch_replay} replay + {batch_pool} "
            f"pool + {batch_pairs} pares | buffers em RAM: replay "
            f"{len(replay_buffer)} amostras"
            + (f", pool {len(pool_buffer)}" if pool_buffer else "")
            + (f", pares {len(pair_buffer)}" if pair_buffer else "")
        )
    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    scheduler = warmup_cosine_scheduler(
        optimizer,
        total_steps=max_steps if steps_mode else epochs * max(steps_per_epoch, 1),
        warmup_ratio=tr["warmup_ratio"],
    )
    print(f"precisão de treino: {precision}; acumulação: {accumulation_steps}")

    best_iou, bad_epochs = -1.0, 0
    history = []
    run_started = time.perf_counter()
    for epoch in range(0 if steps_mode else epochs):
        epoch_started = time.perf_counter()
        epoch_stats: dict = {}
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
            pair_loader=pair_loader,
            pair_fraction=tset["pair_fraction"],
            consistency_weight=tset["consistency_weight"],
            consistency_mode=tset["mode"],
            consistency_trim=tset["trim"],
            temporal_area_weight=tset["area_weight"],
            epoch_stats=epoch_stats,
            presence_weight=presence_weight,
            presence_pos_weight=presence_pos_weight,
            hnbs_weight=hnbs_weight,
            hnbs_threshold=hnbs_threshold,
            hnbs_topk_fraction=hnbs_topk_fraction,
            ema=ema,
            ema_model=base_model,
        )
        epoch_s = time.perf_counter() - epoch_started
        metrics = evaluate(model, val_loader, device)
        history_entry = {
            "epoch": epoch,
            "train_loss": loss,
            "val": metrics,
            "epoch_s": round(epoch_s, 1),
        }
        if real_pair_val_ds is not None:
            fl = flicker_metrics(model, real_pair_val_ds, device)
            history_entry["flicker"] = fl
            print(
                f"época {epoch}: flicker real — centróide mediano "
                f"{fl.get('centroid_step_median')} px | tIoU {fl.get('tiou_mean')} "
                f"| Δárea {fl.get('area_diff_median')} ({fl.get('pairs')} pares)"
            )
        if "presence_loss" in epoch_stats:
            history_entry["presence_loss"] = round(epoch_stats["presence_loss"], 4)
        for _key in ("presence_mean_pos", "presence_mean_neg"):
            if _key in epoch_stats:
                history_entry[_key] = round(epoch_stats[_key], 4)
        if tset["enabled"]:
            history_entry["pair_batches"] = epoch_stats.get("pair_batches", 0)
        history.append(history_entry)
        if epoch == 0:
            receipt = utilization_receipt(
                torch,
                device,
                epoch_s,
                len(train_loader),
                batch_size,
                precision,
                probe_report=probe_report,
            )
            if tset["enabled"]:
                # Lotes de par processam 2x frames: samples_s do recibo
                # conta lotes do loader principal; o bloco temporal
                # registra o acréscimo para a projeção ser honesta.
                receipt["temporal"] = {
                    "pair_fraction": tset["pair_fraction"],
                    "consistency_weight": tset["consistency_weight"],
                    "pair_batches_epoch0": epoch_stats.get("pair_batches", 0),
                    "pairs_total": len(pair_loader.dataset),
                }
            receipt["efficiency"] = {
                "cache_mode": cache_mode,
                "compile": compile_enabled,
                "compile_status": compile_status,
                "channels_last": channels_last,
                "ema_decay": ema.decay if ema is not None else None,
            }
            receipt["projected_total_min"] = round(epoch_s * epochs / 60, 1)
            receipt["elapsed_s"] = round(time.perf_counter() - run_started, 1)
            (out_dir / "receipt.json").write_text(
                json.dumps(receipt, indent=2), encoding="utf-8"
            )
            print(f"RECEIPT {json.dumps(receipt)}")
        print(f"época {epoch}: loss {loss:.4f}, val_mean_iou {metrics['mean_iou']:.4f}")
        torch.save(base_model.state_dict(), out_dir / "checkpoints" / "last.pt")
        monitor = metrics
        ema_backup = None
        if ema is not None:
            # Avalia e (se for o melhor) salva com os pesos-EMA aplicados;
            # depois restaura os crus — o treino segue da trajetória real.
            ema_backup = ema.apply(base_model)
            ema_metrics = evaluate(model, val_loader, device)
            history_entry["val_ema"] = ema_metrics
            monitor = ema_metrics
            print(f"época {epoch}: val_mean_iou EMA {ema_metrics['mean_iou']:.4f}")
        if monitor["mean_iou"] > best_iou:
            best_iou = monitor["mean_iou"]
            bad_epochs = 0
            torch.save(base_model.state_dict(), out_dir / "checkpoints" / "best.pt")
        else:
            bad_epochs += 1
        if ema_backup is not None:
            ema.restore(base_model, ema_backup)
        if bad_epochs >= tr["early_stopping"]["patience_epochs"]:
            print(
                f"early stopping na época {epoch} (val_mean_iou sem melhora há {bad_epochs} épocas)."
            )
            break

    steps_steady_s_per_step = None
    if steps_mode:
        # Blocos de passos com a MESMA contabilidade do caminho de épocas:
        # avaliar, régua de flicker, recibo no 1º bloco, best por
        # val_mean_iou (EMA quando ligada) e early stopping por bloco.
        n_blocks = math.ceil(max_steps / eval_interval)
        done_steps = 0
        for block in range(n_blocks):
            block_steps = min(eval_interval, max_steps - done_steps)
            block_started = time.perf_counter()
            block_stats: dict = {}
            loss = train_steps(
                model,
                replay_buffer,
                pool_buffer,
                pair_buffer,
                steps=block_steps,
                batch_replay=batch_replay,
                batch_pool=batch_pool,
                batch_pairs=batch_pairs,
                augmenter=augmenter,
                optimizer=optimizer,
                class_weights=class_weights,
                device=device,
                scaler=scaler,
                scheduler=scheduler,
                amp_dtype=amp_dtype,
                consistency_weight=tset["consistency_weight"],
                consistency_mode=tset["mode"],
                consistency_trim=tset["trim"],
                temporal_area_weight=tset["area_weight"],
                presence_weight=presence_weight,
                presence_pos_weight=presence_pos_weight,
                hnbs_weight=hnbs_weight,
                hnbs_threshold=hnbs_threshold,
                hnbs_topk_fraction=hnbs_topk_fraction,
                ema=ema,
                ema_model=base_model,
                step_stats=block_stats,
                profile_window=profile_window,
            )
            done_steps += block_steps
            block_s = time.perf_counter() - block_started
            metrics = evaluate(model, val_loader, device)
            history_entry = {
                "epoch": block,
                "step": done_steps,
                "train_loss": loss,
                "val": metrics,
                "epoch_s": round(block_s, 1),
            }
            if real_pair_val_ds is not None:
                fl = flicker_metrics(model, real_pair_val_ds, device)
                history_entry["flicker"] = fl
                print(
                    f"passo {done_steps}: flicker real — centróide mediano "
                    f"{fl.get('centroid_step_median')} px | tIoU "
                    f"{fl.get('tiou_mean')} | Δárea {fl.get('area_diff_median')} "
                    f"({fl.get('pairs')} pares)"
                )
            if "presence_loss" in block_stats:
                history_entry["presence_loss"] = round(block_stats["presence_loss"], 4)
            for _key in ("presence_mean_pos", "presence_mean_neg"):
                if _key in block_stats:
                    history_entry[_key] = round(block_stats[_key], 4)
            if tset["enabled"]:
                history_entry["pair_batches"] = block_stats.get("pair_batches", 0)
            history.append(history_entry)
            if block == 0:
                receipt = utilization_receipt(
                    torch,
                    device,
                    block_s,
                    block_steps,
                    batch_size,
                    precision,
                    probe_report=probe_report,
                )
                if tset["enabled"] and pair_buffer is not None:
                    receipt["temporal"] = {
                        "pair_fraction": tset["pair_fraction"],
                        "consistency_weight": tset["consistency_weight"],
                        "pair_batches_block0": block_stats.get("pair_batches", 0),
                        "pairs_total": len(pair_buffer),
                    }
                receipt["efficiency"] = {
                    "cache_mode": "ram_buffer",
                    "compile": compile_enabled,
                    "compile_status": compile_status,
                    "channels_last": channels_last,
                    "ema_decay": ema.decay if ema is not None else None,
                }
                receipt["steps_mode"] = {
                    "max_steps": max_steps,
                    "eval_interval": eval_interval,
                    "batch_replay": batch_replay,
                    "batch_pool": batch_pool,
                    "batch_pairs": batch_pairs,
                    "replay_size": len(replay_buffer),
                }
                if "phase_profile_ms" in block_stats:
                    receipt["phase_profile_ms"] = block_stats["phase_profile_ms"]
                    receipt["phase_profile_steps"] = block_stats.get(
                        "phase_profile_steps"
                    )
                if "step_wall_ms" in block_stats:
                    receipt["step_wall_ms"] = block_stats["step_wall_ms"]
                receipt["projected_total_min"] = round(block_s * n_blocks / 60, 1)
                receipt["elapsed_s"] = round(time.perf_counter() - run_started, 1)
                (out_dir / "receipt.json").write_text(
                    json.dumps(receipt, indent=2), encoding="utf-8"
                )
                print(f"RECEIPT {json.dumps(receipt)}")
            print(
                f"passos {done_steps}/{max_steps} (bloco {block}): loss "
                f"{loss:.4f}, val_mean_iou {metrics['mean_iou']:.4f}"
            )
            torch.save(base_model.state_dict(), out_dir / "checkpoints" / "last.pt")
            monitor = metrics
            ema_backup = None
            if ema is not None:
                ema_backup = ema.apply(base_model)
                ema_metrics = evaluate(model, val_loader, device)
                history_entry["val_ema"] = ema_metrics
                monitor = ema_metrics
                print(f"bloco {block}: val_mean_iou EMA {ema_metrics['mean_iou']:.4f}")
            if monitor["mean_iou"] > best_iou:
                best_iou = monitor["mean_iou"]
                bad_epochs = 0
                torch.save(base_model.state_dict(), out_dir / "checkpoints" / "best.pt")
            else:
                bad_epochs += 1
            if ema_backup is not None:
                ema.restore(base_model, ema_backup)
            if bad_epochs >= tr["early_stopping"]["patience_epochs"]:
                print(
                    f"early stopping no bloco {block} (val_mean_iou sem "
                    f"melhora há {bad_epochs} blocos)."
                )
                break

    if steps_mode and len(history) > 1:
        # s/passo em estado estacionário: blocos depois do primeiro (o
        # bloco 0 carrega o warmup único — autotune/alocador — que no
        # smoke de 100 passos inflou a média pra 0,403 com passo real
        # de ~0,155; medido 2026-11-09). O recibo do bloco 0 segue
        # como portão de lançamento; este número é o custo por passo
        # de uma corrida longa, e vai pras métricas finais.
        _s, _n, _prev = 0.0, 0, int(history[0]["step"])
        for _e in history[1:]:
            _s += float(_e["epoch_s"])
            _n += int(_e["step"]) - _prev
            _prev = int(_e["step"])
        if _n:
            steps_steady_s_per_step = round(_s / _n, 4)

    # Exportação ONNX a partir do melhor checkpoint.
    # Não-fatal: se a exportação falhar, as métricas e os demais artefatos
    # ainda são escritos, e a falha fica registrada em onnx_validation —
    # antes, uma falha aqui perdia o metrics.json inteiro após o treino.
    if channels_last:
        disable_channels_last(torch, base_model, channels_last_hook)
    base_model.load_state_dict(
        torch.load(out_dir / "checkpoints" / "best.pt", map_location=device)
    )
    onnx_path = out_dir / cfg["export"]["onnx"]
    try:
        export_onnx(base_model, str(onnx_path), cfg)
        onnx_check = validate_onnx(base_model, onnx_path, val_loader, device, cfg)
    except Exception as exc:  # noqa: BLE001 - registrar, não perder o treino
        print(
            f"AVISO: exportação ONNX falhou ({exc}); métricas serão escritas sem ONNX.",
            file=sys.stderr,
        )
        onnx_check = {"status": "failed", "error": str(exc)}
    final_metrics = evaluate(base_model, val_loader, device)
    final_metrics["test"] = evaluate(base_model, test_loader, device)
    final_metrics["onnx_validation"] = onnx_check
    final_metrics["delivery_status"] = onnx_delivery_status(onnx_check)
    final_metrics["history"] = history
    if steps_steady_s_per_step is not None:
        final_metrics["steps_steady_s_per_step"] = steps_steady_s_per_step
    final_metrics["train_pixel_counts"] = counts.tolist()
    final_metrics["train_source_counts"] = {
        "synthetic": len(synthetic_train_pairs),
        "real": len(real_pairs),
        "presence_pool": pool_count,
    }
    cfg["data_sources"] = {
        "synthetic_root": str(Path(args.data_dir)),
        "real_train_root": str(Path(args.real_train_dir))
        if args.real_train_dir
        else None,
        "presence_pool_root": str(Path(args.presence_pool_dir))
        if args.presence_pool_dir
        else None,
        "real_pairs_root": str(Path(args.real_pairs_dir))
        if args.real_pairs_dir
        else None,
        "init_presence": (
            {"path": str(presence_init_path), "report": presence_init_report}
            if presence_init_path
            else None
        ),
        "train_pair_counts": final_metrics["train_source_counts"],
        "validation_source": "synthetic --data-dir/val only",
        "test_source": "synthetic --data-dir/test only; evaluated once after selecting best checkpoint",
        "mixed_precision_used": precision,
        "gradient_accumulation_steps": accumulation_steps,
        "real_sample_loss_weight": real_loss_weight if real_pairs else None,
        "real_labels_unverified": real_labels_unverified,
        "real_dataset_manifest": real_manifest,
        "split_manifest_validation": split_validation,
    }

    (out_dir / "classes.json").write_text(
        json.dumps(CLASSES, indent=2), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(final_metrics, indent=2), encoding="utf-8"
    )
    (out_dir / "training_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )
    # split_manifest: cópia já validada no início, se o gerador a forneceu.
    if split_manifest is None:
        split_manifest = {
            "note": "manifesto não fornecido pelo gerador; splits lidos das pastas train/val/test sem certificação de fontes.",
            "splits": ["train", "val", "test"],
        }
    split_manifest["validation"] = split_validation
    split_manifest["training_sources"] = cfg["data_sources"]
    (out_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )
    print(
        f"ONNX: {onnx_path} | métricas: {out_dir / 'metrics.json'} | val_mean_iou: {final_metrics['mean_iou']:.4f}"
    )
    if final_metrics["delivery_status"] != "validated":
        print(
            f"ENTREGA BLOQUEADA: validação ONNX={onnx_check.get('status')}; "
            "checkpoints e métricas foram preservados.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
