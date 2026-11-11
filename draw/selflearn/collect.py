"""Coleta (U1): grava exemplos de sessao no formato de dataset.

Cada registro vira um arquivo em input/ (o warp BGR), um em prob/
(o mapa de probabilidade em uint8) e uma linha em manifest.jsonl
com o sha256 dos bytes da imagem gravada, sessao, instante, modelo
e origem. A linhagem por linha e o que permite expurgar depois tudo
que uma origem ruim tocou. Privado por padrao: isto escreve em disco
local, nunca publica nada.
"""

import hashlib
import json
import time
import uuid

import cv2
import numpy as np

from ..safe_io import ensure_private_dir
from . import split as split_module


class ExampleRecorder:
    """Gravador de exemplos. Uma chamada de record() por passada."""

    def __init__(self, out_dir):
        self.out_dir = ensure_private_dir(out_dir)
        self.input_dir = self.out_dir / "input"
        self.prob_dir = self.out_dir / "prob"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.prob_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.out_dir / "manifest.jsonl"
        self._count = 0

    def record(
        self, session_id, warped_bgr, prob, corners, model_sha, origin, split="train"
    ):
        """Grava um exemplo e retorna a linha do manifesto (dict).

        O nome combina sessao, instante, contador e um sufixo
        aleatorio, e os dois arquivos sao criados com criacao
        exclusiva ("xb"): dois gravadores no mesmo diretorio nunca
        produzem o mesmo nome nem sobrescrevem amostra existente —
        colisao gera um nome novo. A linha do manifesto so e
        publicada depois que os dois arquivos estao gravados.
        """
        # Linhagem obrigatória na escrita, não só na leitura: sem
        # session_id/origem/split válidos a linha não é gravada.
        split_module._require_lineage({"session_id": session_id, "origin": origin})
        if split not in (split_module.TRAIN, split_module.HOLDOUT):
            raise split_module.LeakageError(f"split desconhecido: {split!r}")
        stamp = time.time()

        ok, buf = cv2.imencode(".png", warped_bgr)
        if not ok:
            raise RuntimeError("falha ao codificar o warp em PNG")
        image_bytes = buf.tobytes()

        prob_u8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
        ok, prob_buf = cv2.imencode(".png", prob_u8)
        if not ok:
            raise RuntimeError("falha ao codificar o mapa de probabilidade")
        prob_bytes = prob_buf.tobytes()

        name = None
        for _ in range(100):
            candidate = (
                f"{session_id}_{int(stamp * 1000)}_{self._count:06d}"
                f"_{uuid.uuid4().hex[:8]}"
            )
            self._count += 1
            input_path = self.input_dir / f"{candidate}.png"
            prob_path = self.prob_dir / f"{candidate}.png"
            try:
                with open(input_path, "xb") as f:
                    f.write(image_bytes)
            except FileExistsError:
                continue
            try:
                with open(prob_path, "xb") as f:
                    f.write(prob_bytes)
            except FileExistsError:
                input_path.unlink(missing_ok=True)
                continue
            except BaseException:
                input_path.unlink(missing_ok=True)
                raise
            name = candidate
            break
        if name is None:
            raise RuntimeError("sem nome unico disponivel para o exemplo")

        row = {
            "file": name,
            "session_id": session_id,
            "ts": stamp,
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
            "model_sha": model_sha,
            "origin": origin,
            "split": split,
            "corners": np.asarray(corners, dtype=float).tolist(),
        }
        with open(self.manifest_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row
