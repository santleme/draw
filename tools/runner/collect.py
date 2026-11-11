"""Coleta de artefatos com resume, hash e manifesto.

Licao que gerou esta peca (2026-10-31): o downloader simples baixou
o ONNX da tinta v4 truncado (0 bytes) e sobrescreveu o arquivo bom;
a recuperacao exigiu curl -C - com resume e conferencia de hash.
Nunca mais: toda coleta da ponte passa por aqui.

Garantias:
- Download via curl com resume (-C -): retomar do byte onde parou
  e a unica via; baixar do zero por cima de arquivo existente, nao.
- Integridade: sha256 calculado ao fim de cada arquivo e gravado
  num manifesto (manifest.json no diretorio de destino) com tamanho
  em bytes. Arquivo de 0 bytes e tratado como falha, nunca sucesso.
- Idempotencia: rodar de novo sobre o mesmo destino retoma o que
  falta e reconfere o que ja esta la.
- Rede (mesmos controles do runner_kaggle): so https (curl --proto
  =https), host na allowlist (padrao DEFAULT_ALLOWED_HOSTS, ou
  DRAW_COLLECT_ALLOWED_HOSTS separado por virgula), redirect
  recusado (sem -L, --max-redirs 0, e status diferente de 200/206 e
  falha). Nenhuma credencial vai no pedido: a URL assinada basta.

A lista de arquivos vem do runner (saida do kernel no Kaggle); esta
peca cuida so de trazer bytes do URL assinado para o disco local
com as garantias acima. Testada em tests/test_collect.py com um
servidor HTTP local que corta a conexao no meio.
"""

import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from draw.net_policy import check_url  # noqa: E402

# Hosts das URLs assinadas da saida de kernels do Kaggle.
DEFAULT_ALLOWED_HOSTS = ("www.kaggle.com", "storage.googleapis.com")
_ENV_ALLOWED_HOSTS = "DRAW_COLLECT_ALLOWED_HOSTS"


def allowed_hosts():
    raw = os.environ.get(_ENV_ALLOWED_HOSTS, "")
    hosts = tuple(h.strip().lower() for h in raw.split(",") if h.strip())
    return hosts or DEFAULT_ALLOWED_HOSTS


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_resume(url, target, attempts=5, hosts=None, allow_loopback_http=False):
    """Baixa url para target com resume. Devolve (bytes, sha256).

    Cada tentativa retoma do byte onde a anterior parou (curl -C -).
    Falha de transferencia com arquivo parcial NAO apaga o parcial:
    ele e o ponto de retomada da proxima tentativa. URL fora da
    politica (esquema, host) levanta antes de qualquer conexao, e
    redirect conta como falha."""
    check_url(url, hosts or allowed_hosts(), allow_loopback_http)
    protocols = "=http,https" if allow_loopback_http else "=https"
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    last_error = None
    for _ in range(attempts):
        result = subprocess.run(
            [
                "curl",
                "-fsS",
                "--proto",
                protocols,
                "--max-redirs",
                "0",
                "-w",
                "%{http_code}",
                "-C",
                "-",
                "-o",
                target,
                url,
            ],
            capture_output=True,
            text=True,
        )
        status = (result.stdout or "").strip()[-3:]
        size = os.path.getsize(target) if os.path.exists(target) else 0
        if result.returncode == 0 and status in ("200", "206") and size > 0:
            return size, sha256_of(target)
        if status.startswith("3"):
            raise IOError(f"redirect HTTP {status} recusado para {url}")
        last_error = f"curl rc={result.returncode} http={status}, {size} B em disco"
    raise IOError(f"download nao completou apos {attempts} tentativas: {last_error}")


def collect(files, dest, hosts=None, allow_loopback_http=False):
    """files: lista de dicts {name, url} (formato da saida do
    kernel). Baixa todos para dest e grava manifest.json.
    Devolve o manifesto."""
    os.makedirs(dest, exist_ok=True)
    manifest = {}
    for entry in files:
        name = entry.get("fileName") or entry.get("name")
        url = entry.get("url")
        if not name or not url:
            continue
        target = os.path.join(dest, os.path.basename(name))
        size, sha = download_resume(
            url, target, hosts=hosts, allow_loopback_http=allow_loopback_http
        )
        manifest[os.path.basename(name)] = {"bytes": size, "sha256": sha}
    with open(os.path.join(dest, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
