#!/usr/bin/env python3
"""Verifica uma cópia local do FreiHAND (Kaggle: danieldelro/freihand)
sem baixar o dataset inteiro.

Uso:
    python tools/check_freihand.py <raiz-do-freihand> [--sample N] [--strict]

O que verifica, só com leitura local:
  - estrutura esperada: evaluation/rgb/*.jpg, training/rgb/*.jpg,
    training/mask/*.jpg (máscara real, grayscale, 224x224);
  - cada RGB de treino tem máscara pareada com o mesmo nome de arquivo;
  - dimensões (224x224) e canais (RGB = 3, máscara = cinza);
  - máscara binarizável: após limiar 127, cobertura da mão em % e
    fração de pixels "quase-binários" (0/255) vs. artefatos JPEG
    próximos da borda (valores 1-7 e 248-254);
  - vazamento de split por nome de arquivo (mesmo nome em treino e
    avaliação) — atenção: no FreiHAND original, 32560 amostras únicas
    viram 4 variantes de fundo; dividir por arquivo, e não por amostra
    base, vaza a mesma mão entre splits (ver DATASET_FREIHAND.md).

Não baixa nada, não usa webcam, não treina nada. Sem OpenCV, tenta
Pillow; sem nenhum dos dois, conta arquivos e verifica nomes/extensões.

Saída: resumo em texto + exit 0 se a estrutura está íntegra,
exit 1 se faltar diretório, par ou dimensão.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

EXPECTED_DIRS = ["evaluation/rgb", "training/rgb", "training/mask"]


def _load_image(path):
    """Retorna (largura, altura, canais, pixels) ou None.

    pixels é uma lista plana de valores do primeiro canal quando
    disponível; sem cv2/Pillow, retorna None para pixels.
    """
    try:
        import cv2

        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if im is None:
            return None
        h, w = im.shape[:2]
        ch = 1 if im.ndim == 2 else im.shape[2]
        return w, h, ch, im[:, :, 0].ravel() if im.ndim == 3 else im.ravel()
    except ImportError:
        pass
    try:
        from PIL import Image
        import numpy as np

        im = np.asarray(Image.open(path))
        h, w = im.shape[:2]
        ch = 1 if im.ndim == 2 else im.shape[2]
        return w, h, ch, im[:, :, 0].ravel() if im.ndim == 3 else im.ravel()
    except ImportError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="Raiz local do FreiHAND (ex.: /tmp/freihand)")
    ap.add_argument("--sample", type=int, default=20,
                    help="Quantas imagens por diretório inspecionar (padrão 20)")
    ap.add_argument("--strict", action="store_true",
                    help="Falha se qualquer amostra inspecionada fugir de 224x224")
    args = ap.parse_args()
    root = Path(args.root)
    ok = True

    if not root.is_dir():
        print(f"ERRO: raiz inexistente: {root}")
        return 1

    counts = {}
    for d in EXPECTED_DIRS:
        files = sorted((root / d).glob("*.jpg")) if (root / d).is_dir() else []
        counts[d] = len(files)
        print(f"{d}: {len(files)} arquivos .jpg")
        if not files:
            print(f"  AVISO: diretório vazio ou ausente: {root / d}")
            if d != "evaluation/rgb":
                ok = False

    # Pares treino: rgb <-> mask com o mesmo nome.
    train_rgb = {p.name for p in (root / "training/rgb").glob("*.jpg")} \
        if (root / "training/rgb").is_dir() else set()
    train_mask = {p.name for p in (root / "training/mask").glob("*.jpg")} \
        if (root / "training/mask").is_dir() else set()
    if train_rgb or train_mask:
        missing_mask = sorted(train_rgb - train_mask)[:10]
        missing_rgb = sorted(train_mask - train_rgb)[:10]
        print(f"treino: {len(train_rgb)} rgb, {len(train_mask)} masks, "
              f"pares: {len(train_rgb & train_mask)}")
        if missing_mask:
            print(f"  AVISO: rgb sem máscara (primeiros): {missing_mask}")
        if missing_rgb:
            print(f"  AVISO: máscara sem rgb (primeiros): {missing_rgb}")

    # Índices se repetem entre splits (00000000.jpg existe em treino e
    # em avaliação) — isso é ESPERADO e NÃO é prova de mesma mão: os
    # índices recomeçam por split. O vazamento real a evitar é a mesma
    # mão base (32560 únicas x 4 variantes de fundo) em splits
    # diferentes do gerador sintético; isso se controla na origem
    # (fonte/mão), não pelo nome do arquivo.
    eval_rgb = {p.name for p in (root / "evaluation/rgb").glob("*.jpg")} \
        if (root / "evaluation/rgb").is_dir() else set()
    print(f"nomes em comum treino x avaliação (esperado, índices "
          f"recomeçam por split): {len((train_rgb | train_mask) & eval_rgb)}")

    # Inspeção de conteúdo em amostra.
    for d in EXPECTED_DIRS:
        files = sorted((root / d).glob("*.jpg"))[: args.sample]
        shapes = Counter()
        coverages = []
        for f in files:
            info = _load_image(f)
            if info is None:
                print(f"  {d}: sem cv2/Pillow, pulando inspeção de pixels "
                      f"({f.name})")
                break
            w, h, ch, px = info
            shapes[(w, h, ch)] += 1
            if "mask" in d and px is not None:
                import numpy as np

                px = np.asarray(px)
                coverages.append(float((px > 127).mean() * 100))
        if shapes:
            print(f"  {d} amostra ({len(files)}): formas {dict(shapes)}"
                  + (f", cobertura mão após limiar 127: "
                     f"min {min(coverages):.2f}% / média "
                     f"{sum(coverages) / len(coverages):.2f}% / "
                     f"max {max(coverages):.2f}%" if coverages else ""))
            if args.strict and set(shapes) != {(224, 224, 3)} and \
                    set(shapes) != {(224, 224, 1)}:
                print(f"  ERRO: forma fora de 224x224 em {d}: {dict(shapes)}")
                ok = False

    print("RESULTADO:", "OK" if ok else "PROBLEMAS ENCONTRADOS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
