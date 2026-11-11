"""Configuração comum da suíte.

A verificação de modelos é estrita por padrão (draw/manifest.py): um
ONNX fora de models/MANIFEST.json não carrega. As fixtures ONNX da
suíte (tests/onnx_fixtures.py) são sintéticas e nunca serão
registradas, então a suíte roda com o opt-out explícito
DRAW_MODEL_STRICT=0. Os testes que fixam o padrão estrito
(tests/test_manifest.py) removem a variável com monkeypatch.
"""

import pytest


@pytest.fixture(autouse=True)
def _synthetic_models_are_unverified(monkeypatch):
    monkeypatch.setenv("DRAW_MODEL_STRICT", "0")
