"""Integridade dos modelos ONNX contra o manifesto do repositório.

O repositório não publica binário de modelo: o `draw_seg_v3_dyn.onnx`
de produção vive fora dele. O efeito colateral é que qualquer arquivo
com o nome certo é aceito na carga, sem nenhuma verificação — um
modelo trocado, truncado ou de outra corrida entra no app em
silêncio.

Este módulo é o mecanismo que fecha esse buraco sem obrigar o
binário a morar no git: `models/MANIFEST.json` registra, por nome de
arquivo, o sha256, o tamanho em bytes, a régua que aprovou o modelo
e a data da aprovação. Toda carga de modelo (detector neural,
página->limpa, máscara de mão) passa por verify_model antes de abrir
a sessão.

Política (estrita por padrão):

- entrada conferindo: a carga segue;
- hash ou tamanho divergente: RuntimeError;
- arquivo de modelo sem entrada no manifesto (ou manifesto ausente):
  RuntimeError — um modelo não registrado não é carregado em silêncio.

O rebaixamento para aviso é opt-in explícito: DRAW_MODEL_STRICT=0
(ou false/no/off), ou a flag --allow-unverified do app, que define a
mesma variável. Qualquer outro valor, inclusive vazio, é estrito.

Formato:

.. code-block:: json

    {
      "schema": 1,
      "models": {
        "draw_seg_v3_dyn.onnx": {
          "sha256": "751e74da...",
          "bytes": 15191109,
          "approved_by": "tools/eval_real_frames.py: 43/43, 0 FP, IoU 0,962",
          "date": "2026-10-29"
        }
      }
    }

O manifesto é lido com teto de tamanho e cada entrada é validada
(sha256 hexadecimal de 64 caracteres, bytes inteiro não negativo):
um manifesto corrompido nunca vira "sem verificação".
"""

import hashlib
import os
import re
from pathlib import Path

from .safe_io import InputLimitError, read_json

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "models" / "MANIFEST.json"
_ENV_STRICT_VAR = "DRAW_MODEL_STRICT"
_RELAXED_VALUES = ("0", "false", "no", "off")
_SCHEMA = 1
_CHUNK = 1 << 20
_MAX_MANIFEST_BYTES = 1 << 20
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MODEL_SUFFIXES = (".onnx", ".task")


def sha256_of(path):
    """sha256 do arquivo, em blocos de 1 MiB."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path=None):
    """Lê o manifesto. Devolve (caminho, dicionário de modelos).

    Manifesto ausente devolve (caminho, {}) — e não é erro: o
    repositório publica o esquema, não os binários. Manifesto
    ilegível (JSON inválido, schema desconhecido) é erro explícito:
    um manifesto corrompido não pode virar "sem verificação" em
    silêncio.
    """
    manifest_path = Path(path) if path is not None else MANIFEST_PATH
    if not manifest_path.exists():
        return manifest_path, {}
    try:
        data = read_json(manifest_path, max_bytes=_MAX_MANIFEST_BYTES)
    except InputLimitError as error:
        raise RuntimeError(
            f"manifesto de modelos ilegível em {manifest_path}: {error}"
        ) from error
    if not isinstance(data, dict) or "models" not in data:
        raise RuntimeError(
            f"manifesto de modelos sem a chave 'models': {manifest_path}"
        )
    schema = data.get("schema", _SCHEMA)
    if schema != _SCHEMA:
        raise RuntimeError(
            f"manifesto de modelos com schema {schema!r}; este código "
            f"entende o schema {_SCHEMA}"
        )
    models = data["models"]
    if not isinstance(models, dict):
        raise RuntimeError(
            f"'models' do manifesto precisa ser um objeto: {manifest_path}"
        )
    for name, entry in models.items():
        _validate_entry(manifest_path, name, entry)
    return manifest_path, models


def _validate_entry(manifest_path, name, entry):
    if not isinstance(entry, dict):
        raise RuntimeError(f"entrada {name!r} do manifesto não é um objeto")
    sha = str(entry.get("sha256", "")).strip().lower()
    if not sha:
        raise RuntimeError(
            f"entrada {name!r} do manifesto não tem sha256: {manifest_path}"
        )
    if not _SHA256_RE.match(sha):
        raise RuntimeError(
            f"entrada {name!r} do manifesto com sha256 malformado: {manifest_path}"
        )
    size = entry.get("bytes")
    if size is not None and (
        isinstance(size, bool) or not isinstance(size, int) or size < 0
    ):
        raise RuntimeError(
            f"entrada {name!r} do manifesto com 'bytes' inválido: {size!r}"
        )


def strict_enabled():
    """True salvo opt-out explícito (DRAW_MODEL_STRICT=0/false/no/off)."""
    value = os.environ.get(_ENV_STRICT_VAR, "").strip().lower()
    return value not in _RELAXED_VALUES


def allow_unverified():
    """Rebaixa a verificação para aviso neste processo (--allow-unverified)."""
    os.environ[_ENV_STRICT_VAR] = "0"


def _refuse_or_warn(message, strict):
    if strict:
        raise RuntimeError(
            message + f" Para carregar mesmo assim, defina {_ENV_STRICT_VAR}=0 "
            "ou use --allow-unverified."
        )
    print(f"AVISO: {message}")


def find_entry(model_path, manifest_path, models):
    """Entrada do manifesto para o arquivo (por nome ou caminho relativo)."""
    path = Path(model_path)
    entry = models.get(path.name)
    if entry is None:
        try:
            relative = (
                path.resolve()
                .relative_to(Path(manifest_path).parent.resolve())
                .as_posix()
            )
        except ValueError:
            relative = None
        if relative is not None:
            entry = models.get(relative)
    return entry


def unregistered_model_files(models_dir=None, manifest_path=None):
    """Arquivos .onnx/.task da pasta de modelos sem entrada no manifesto."""
    resolved_manifest, models = load_manifest(manifest_path)
    folder = Path(models_dir) if models_dir is not None else resolved_manifest.parent
    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix in MODEL_SUFFIXES
        and find_entry(path, resolved_manifest, models) is None
    )


def verify_model(model_path, manifest_path=None, strict=None):
    """Confere o modelo contra o manifesto. Devolve a entrada ou None.

    Arquivo inexistente devolve None (o chamador trata a ausência).
    Entrada conferindo devolve a entrada. Divergência ou arquivo sem
    entrada levantam RuntimeError no modo estrito (o padrão); com
    DRAW_MODEL_STRICT=0 (ou strict=False) imprimem o aviso e devolvem
    a entrada (divergência) ou None (não registrado).
    """
    if strict is None:
        strict = strict_enabled()
    path = Path(model_path)
    if not path.exists():
        return None
    resolved_manifest, models = load_manifest(manifest_path)
    # A chave é o nome do arquivo (model.onnx) ou o caminho relativo
    # à pasta do manifesto (pasta/model.onnx), para o registro não
    # depender de dois modelos homônimos.
    entry = find_entry(path, resolved_manifest, models)
    if entry is None:
        where = (
            f"no manifesto {resolved_manifest}"
            if resolved_manifest.exists()
            else f"(manifesto ausente: {resolved_manifest})"
        )
        _refuse_or_warn(
            f"MODELO NAO REGISTRADO {where}: {path}. Nenhum sha256 aprovado "
            "cobre este arquivo.",
            strict,
        )
        return None
    expected = str(entry["sha256"]).strip().lower()
    size_expected = entry.get("bytes")
    size_actual = path.stat().st_size
    problems = []
    if isinstance(size_expected, int) and size_expected != size_actual:
        problems.append(f"{size_actual} bytes != {size_expected} esperados")
    else:
        actual = sha256_of(path)
        if actual != expected:
            problems.append(f"sha256 {actual} != {expected}")
    if not problems:
        return entry
    message = (
        f"MODELO DIVERGENTE do manifesto {resolved_manifest.name}: "
        f"{path.name} — "
        + "; ".join(problems)
        + ". O arquivo em disco não é o modelo que a régua aprovou."
    )
    _refuse_or_warn(message, strict)
    return entry
