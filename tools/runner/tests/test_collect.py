"""Testes da coleta com resume (tools/runner/collect.py).

O teste principal sobe um servidor HTTP local que entrega metade
do corpo e corta a conexao na primeira tentativa; a coleta precisa
retomar do byte certo (Range) e entregar o arquivo integro com o
sha256 esperado. Sem esse comportamento, a peca nao cumpre a licao
da v4 (download truncado de 0 bytes).
"""

import hashlib
import http.server
import os
import sys
import threading

import pytest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from tools.runner.collect import collect, sha256_of

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB deterministico
EXPECTED_SHA = hashlib.sha256(PAYLOAD).hexdigest()
STATE = {"cut_done": False}


class FlakyHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        start = 0
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            start = int(range_header[6:].split("-")[0])
        body = PAYLOAD[start:]
        if not STATE["cut_done"]:
            STATE["cut_done"] = True
            half = len(body) // 2
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body[:half])
            self.connection.close()
            return
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        if start:
            self.send_header(
                "Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}"
            )
        self.end_headers()
        self.wfile.write(body)


@pytest.mark.lento
def test_resume_recovers_full_file(tmp_path):
    server = http.server.HTTPServer(("127.0.0.1", 0), FlakyHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{port}/model.onnx"
        manifest = collect(
            [{"name": "model.onnx", "url": url}],
            str(tmp_path),
            allow_loopback_http=True,
        )
    finally:
        server.shutdown()
    target = tmp_path / "model.onnx"
    assert target.read_bytes() == PAYLOAD
    assert manifest["model.onnx"]["bytes"] == len(PAYLOAD)
    assert manifest["model.onnx"]["sha256"] == EXPECTED_SHA
    assert sha256_of(str(target)) == EXPECTED_SHA


class RedirectHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(302)
        self.send_header("Location", "https://evil.example/model.onnx")
        self.send_header("Content-Length", "0")
        self.end_headers()


def test_redirect_is_refused(tmp_path):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/model.onnx"
        with pytest.raises(IOError, match="redirect"):
            collect(
                [{"name": "model.onnx", "url": url}],
                str(tmp_path),
                allow_loopback_http=True,
            )
    finally:
        server.shutdown()


@pytest.mark.parametrize(
    "url",
    [
        "http://storage.googleapis.com/x/model.onnx",
        "https://evil.example/model.onnx",
        "https://storage.googleapis.com.evil.example/model.onnx",
        "https://user:pw@storage.googleapis.com/model.onnx",
        "file:///etc/passwd",
        "http://127.0.0.1:1/model.onnx",
    ],
)
def test_url_outside_policy_is_refused_before_connecting(tmp_path, url):
    with pytest.raises(ValueError):
        collect([{"name": "model.onnx", "url": url}], str(tmp_path))
    assert not (tmp_path / "model.onnx").exists()
