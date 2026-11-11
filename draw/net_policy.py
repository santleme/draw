"""Política de rede dos downloads do Draw: HTTPS e allowlist de host.

Os caminhos que buscam bytes fora da máquina (modelos da Release,
artefatos de kernels, pesos de partida do fine-tune) seguem a mesma
regra do runner do Kaggle (tools/runner/runner_kaggle.py):

- só https://, salvo http:// em loopback quando o chamador pede
  explicitamente (servidores de teste locais);
- o host precisa estar na allowlist do chamador (igualdade exata,
  sem sufixo curinga);
- redirect só é seguido se o destino passar pela mesma regra; com
  allow_redirects=False, qualquer redirect é recusado.
"""

import ipaddress
import urllib.parse
import urllib.request

GITHUB_RELEASE_HOSTS = (
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
)


class UrlPolicyError(ValueError):
    """URL fora da política (esquema, host ou redirect recusado)."""


def _is_loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def host_of(url):
    return (urllib.parse.urlsplit(url).hostname or "").lower()


def check_url(url, allowed_hosts, allow_loopback_http=False):
    """Devolve url se ela cumpre a política; senão UrlPolicyError."""
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.username or parts.password:
        raise UrlPolicyError(f"URL com credencial embutida recusada: {host}")
    if parts.scheme == "http" and allow_loopback_http and _is_loopback(host):
        return url
    if parts.scheme != "https":
        raise UrlPolicyError(f"esquema {parts.scheme!r} recusado (so https): {url}")
    allowed = {h.lower() for h in allowed_hosts}
    if host not in allowed:
        raise UrlPolicyError(
            f"host {host!r} fora da allowlist ({', '.join(sorted(allowed))})"
        )
    return url


class _PolicyRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_hosts, allow_redirects, allow_loopback_http):
        super().__init__()
        self._allowed = tuple(allowed_hosts)
        self._follow = allow_redirects
        self._loopback = allow_loopback_http

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not self._follow:
            raise UrlPolicyError(f"redirect HTTP {code} recusado para {newurl}")
        check_url(newurl, self._allowed, self._loopback)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_opener(allowed_hosts, allow_redirects=True, allow_loopback_http=False):
    """Opener urllib que aplica a política a cada salto de redirect."""
    return urllib.request.build_opener(
        _PolicyRedirect(allowed_hosts, allow_redirects, allow_loopback_http)
    )


def open_url(
    url,
    allowed_hosts,
    allow_redirects=True,
    allow_loopback_http=False,
    timeout=60,
    headers=None,
):
    """urlopen com a política aplicada na URL inicial e nos redirects."""
    check_url(url, allowed_hosts, allow_loopback_http)
    opener = build_opener(allowed_hosts, allow_redirects, allow_loopback_http)
    request = urllib.request.Request(url, headers=dict(headers or {}))
    return opener.open(request, timeout=timeout)
