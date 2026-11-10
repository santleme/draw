"""Buffers de treino residentes em RAM (Fase 2 — loop guiado por passos).

Auditoria de 2026-11-09: o caminho de épocas re-decodifica e re-aumenta
~20,8 mil imagens por época em workers de CPU enquanto a GPU (passo puro
de 0,0445 s na G4) espera — passo de parede medido de 0,567 s na A100.
Aqui as fontes do fine-tune incremental são materializadas UMA vez, já
redimensionadas pro tamanho do modelo (mesma troca deliberada do
``cache_mode="resized"`` do build_dataset: interpolações LINEAR na imagem
e NEAREST na máscara), guardadas como uint8 e amostradas por índice —
durante os passos NÃO há leitura de disco, decode nem resize.

Fontes (as mesmas do caminho de épocas, mesmos carregadores):
- Replay âncora: pares (imagem, máscara) sintéticos/SmartDoc + reais
  supervisionados, com o peso de loss por amostra.
- Pool de presença: frames reais com ``has_paper`` explícito
  (labels.csv) — a máscara é IGNORE em tudo e nasce no device.
- Pares temporais reais: (frame_a, mask_a, frame_b, mask_b, change) do
  manifesto pairs.csv, máscaras do teacher já traduzidas (255 = IGNORE
  onde o teacher discorda; change == 1 = livre pra mudar).

Memória (512x512): ~1,05 MB por frame com máscara — replay de 1.500
amostras ~1,6 GB, pool de 501 ~0,4 GB (sem máscara), 593 pares ~1,9 GB.
Estouro do teto configurado é erro alto, nunca silencioso.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _load_resized(image_path, mask_path, size, load_image_rgb, load_mask, cv2):
    im = load_image_rgb(image_path)
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
    if mask_path is None:
        return im, None
    mk = load_mask(mask_path)
    mk = cv2.resize(mk.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return im, mk


def _stack_images(images):
    import torch

    arr = np.stack(images)  # [N, H, W, 3] uint8
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2)))


def _stack_masks(masks):
    import torch

    return torch.from_numpy(np.stack(masks).astype(np.uint8, copy=False))


class RAMImageBuffer:
    """Amostras (imagem, máscara, peso, presença) residentes em RAM.

    ``presence``: alvo explícito por amostra; -1 = derivar da máscara
    (mesma convenção do collate_with_presence). Pool entra com máscara
    None: a amostra nasce com máscara toda IGNORE no device.
    """

    def __init__(self, images, masks, weights, presence, label):
        self.images = images
        self.masks = masks  # tensor uint8 [N, H, W] ou None (tudo IGNORE)
        self.weights = weights
        self.presence = presence
        self.label = label
        self.size = int(images.shape[0])

    def __len__(self):
        return self.size

    @classmethod
    def from_pairs(cls, pairs, size, label, load_image_rgb, load_mask, cv2,
                   ignore_index=255):
        """pairs: lista de (img_path, mask_path, peso) ou (img, mask, peso, pres)."""
        images, masks, weights, presence = [], [], [], []
        for item in pairs:
            img_path, mask_path, weight = item[0], item[1], float(item[2])
            pres = float(item[3]) if len(item) > 3 else -1.0
            im, mk = _load_resized(img_path, mask_path, size,
                                   load_image_rgb, load_mask, cv2)
            images.append(im)
            masks.append(mk)
            weights.append(weight)
            presence.append(pres)
        return cls(_stack_images(images), _stack_masks(masks),
                   _floats(weights), _floats(presence), label)

    @classmethod
    def from_pool_dir(cls, pool_dir, size, load_image_rgb, cv2):
        """Pool de presença: labels.csv (file, has_paper); sem máscaras."""
        pool_dir = Path(pool_dir)
        labels_path = pool_dir / "labels.csv"
        if not labels_path.is_file():
            raise FileNotFoundError(f"labels.csv do pool ausente: {labels_path}")
        images, presence = [], []
        with labels_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
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
                im = load_image_rgb(img_path)
                images.append(cv2.resize(im, (size, size),
                                         interpolation=cv2.INTER_LINEAR))
                presence.append(has_paper)
        if not images:
            raise ValueError(f"pool de presença vazio: {labels_path}")
        n = len(images)
        return cls(_stack_images(images), None,
                   _floats([1.0] * n), _floats(presence), "pool")

    def sample(self, n, device, ignore_index=255):
        """(imagens uint8, máscaras int64, pesos, presenças) no device."""
        import torch

        idx = torch.randint(0, self.size, (n,))
        images = self.images[idx].to(device, non_blocking=True)
        if self.masks is None:
            h, w = images.shape[-2:]
            masks = torch.full((n, h, w), ignore_index, dtype=torch.long,
                               device=device)
        else:
            masks = self.masks[idx].to(device, non_blocking=True).long()
        weights = self.weights[idx].to(device, non_blocking=True)
        presence = self.presence[idx].to(device, non_blocking=True)
        return images, masks, weights, presence


class RAMPairBuffer:
    """Pares temporais reais residentes: (img_a, mask_a, img_b, mask_b, change)."""

    def __init__(self, img_a, mask_a, img_b, mask_b, change):
        self.img_a, self.mask_a = img_a, mask_a
        self.img_b, self.mask_b = img_b, mask_b
        self.change = change
        self.size = int(img_a.shape[0])

    def __len__(self):
        return self.size

    @classmethod
    def from_pairs_dir(cls, pairs_dir, size, load_image_rgb, load_mask, cv2,
                       split="train"):
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
        ia, ma, ib, mb, ch = [], [], [], [], []
        for row in rows:
            a, m_a = _load_resized(pairs_dir / row["frame_a"],
                                   pairs_dir / row["mask_a"], size,
                                   load_image_rgb, load_mask, cv2)
            b, m_b = _load_resized(pairs_dir / row["frame_b"],
                                   pairs_dir / row["mask_b"], size,
                                   load_image_rgb, load_mask, cv2)
            change = cv2.imread(str(pairs_dir / row["change"]),
                                cv2.IMREAD_GRAYSCALE)
            if change is None:
                raise FileNotFoundError(f"change ilegível: {row['change']}")
            change = cv2.resize(change, (size, size),
                                interpolation=cv2.INTER_NEAREST)
            ia.append(a)
            ma.append(m_a)
            ib.append(b)
            mb.append(m_b)
            ch.append((change > 0).astype(np.uint8))
        return cls(_stack_images(ia), _stack_masks(ma), _stack_images(ib),
                   _stack_masks(mb), _stack_masks(ch))

    def sample_pairs(self, n, device):
        """(img_a, mask_a, img_b, mask_b, change) no device; máscaras int64."""
        import torch

        idx = torch.randint(0, self.size, (n,))
        return (
            self.img_a[idx].to(device, non_blocking=True),
            self.mask_a[idx].to(device, non_blocking=True).long(),
            self.img_b[idx].to(device, non_blocking=True),
            self.mask_b[idx].to(device, non_blocking=True).long(),
            self.change[idx].to(device, non_blocking=True),
        )


def _floats(values):
    import torch

    return torch.tensor(values, dtype=torch.float32)
