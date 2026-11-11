"""Contrato do manifesto de modelos (draw/manifest.py).

O manifesto é o mecanismo do R1 da auditoria de 2026-10-31: o
binário de produção não mora no repositório, então a carga confere
sha256 e tamanho contra um registro versionado. Estes testes fixam
o comportamento nos casos que importam. O padrão é estrito:
divergência e arquivo não registrado (ou manifesto ausente) levantam
RuntimeError. Só o opt-out explícito DRAW_MODEL_STRICT=0 (ou
--allow-unverified no app) rebaixa os dois para aviso.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import manifest  # noqa: E402


@pytest.fixture
def default_policy(monkeypatch):
    """Sem DRAW_MODEL_STRICT: o padrão de produção (estrito)."""
    monkeypatch.delenv("DRAW_MODEL_STRICT", raising=False)


def _model(tmp_path, name="m.onnx", payload=b"pesos"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_manifest(tmp_path, models):
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps({"schema": 1, "models": models}))
    return path


def _entry(path, **extra):
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        **extra,
    }


def test_without_manifest_load_is_refused_by_default(tmp_path, default_policy):
    model = _model(tmp_path)
    missing = tmp_path / "nao-existe.json"
    with pytest.raises(RuntimeError, match="NAO REGISTRADO"):
        manifest.verify_model(model, missing)


def test_without_manifest_opt_out_only_warns(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DRAW_MODEL_STRICT", "0")
    model = _model(tmp_path)
    assert manifest.verify_model(model, tmp_path / "nao-existe.json") is None
    assert "NAO REGISTRADO" in capsys.readouterr().out


def test_unregistered_model_is_refused_by_default(tmp_path, default_policy):
    registered = _model(tmp_path, "ok.onnx")
    stranger = _model(tmp_path, "intruso.onnx", b"outros pesos")
    path = _write_manifest(tmp_path, {"ok.onnx": _entry(registered)})
    with pytest.raises(RuntimeError, match="NAO REGISTRADO"):
        manifest.verify_model(stranger, path)


def test_missing_file_is_left_to_the_caller(tmp_path, default_policy):
    path = _write_manifest(tmp_path, {})
    assert manifest.verify_model(tmp_path / "nao.onnx", path) is None


@pytest.mark.parametrize("value", ["", "1", "true", "on", "qualquer"])
def test_strict_unless_explicit_opt_out(monkeypatch, value):
    monkeypatch.setenv("DRAW_MODEL_STRICT", value)
    assert manifest.strict_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", " OFF "])
def test_explicit_opt_out_values(monkeypatch, value):
    monkeypatch.setenv("DRAW_MODEL_STRICT", value)
    assert manifest.strict_enabled() is False


def test_allow_unverified_sets_opt_out(default_policy):
    manifest.allow_unverified()
    assert manifest.strict_enabled() is False


def test_strict_argument_overrides_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("DRAW_MODEL_STRICT", "0")
    model = _model(tmp_path)
    with pytest.raises(RuntimeError):
        manifest.verify_model(model, tmp_path / "nao.json", strict=True)


def test_unregistered_model_files_lists_extras(tmp_path):
    registered = _model(tmp_path, "ok.onnx")
    _model(tmp_path, "extra.onnx", b"x")
    _model(tmp_path, "extra.task", b"y")
    _model(tmp_path, "leia.txt", b"z")
    path = _write_manifest(tmp_path, {"ok.onnx": _entry(registered)})
    names = [p.name for p in manifest.unregistered_model_files(tmp_path, path)]
    assert names == ["extra.onnx", "extra.task"]


@pytest.mark.parametrize(
    "entry",
    [
        "texto",
        {"bytes": 1},
        {"sha256": "abc"},
        {"sha256": "g" * 64},
        {"sha256": "0" * 64, "bytes": -1},
        {"sha256": "0" * 64, "bytes": "10"},
        {"sha256": "0" * 64, "bytes": True},
    ],
)
def test_malformed_entries_are_refused(tmp_path, entry):
    path = _write_manifest(tmp_path, {"m.onnx": entry})
    with pytest.raises(RuntimeError, match="entrada"):
        manifest.load_manifest(path)


def test_oversized_manifest_is_refused(tmp_path):
    path = tmp_path / "MANIFEST.json"
    path.write_text('{"schema": 1, "models": {}, "pad": "' + "x" * (2 << 20) + '"}')
    with pytest.raises(RuntimeError, match="ilegível"):
        manifest.load_manifest(path)


@pytest.mark.parametrize("cut", [1, 10, 40])
def test_truncated_manifest_is_refused(tmp_path, cut):
    model = _model(tmp_path)
    full = json.dumps({"schema": 1, "models": {"m.onnx": _entry(model)}})
    path = tmp_path / "MANIFEST.json"
    path.write_text(full[:-cut])
    with pytest.raises(RuntimeError, match="ilegível"):
        manifest.load_manifest(path)


def test_matching_entry_passes(tmp_path):
    model = _model(tmp_path)
    path = _write_manifest(tmp_path, {"m.onnx": _entry(model)})
    assert manifest.verify_model(model, path) is not None


def test_entry_by_relative_path_also_matches(tmp_path):
    model = _model(tmp_path, "sub/m.onnx")
    path = _write_manifest(tmp_path, {"sub/m.onnx": _entry(model)})
    assert manifest.verify_model(model, path) is not None


def test_divergence_is_fatal_by_default(tmp_path, default_policy):
    model = _model(tmp_path)
    path = _write_manifest(tmp_path, {"m.onnx": {"sha256": "0" * 64}})
    with pytest.raises(RuntimeError, match="DIVERGENTE"):
        manifest.verify_model(model, path)


def test_divergence_only_warns_with_explicit_opt_out(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DRAW_MODEL_STRICT", "0")
    model = _model(tmp_path)
    path = _write_manifest(tmp_path, {"m.onnx": {"sha256": "0" * 64, "bytes": 1}})
    assert manifest.verify_model(model, path) is not None
    printed = capsys.readouterr().out
    assert "DIVERGENTE" in printed


def test_strict_mode_turns_divergence_into_an_error(tmp_path, monkeypatch):
    model = _model(tmp_path)
    path = _write_manifest(tmp_path, {"m.onnx": {"sha256": "0" * 64, "bytes": 1}})
    monkeypatch.setenv("DRAW_MODEL_STRICT", "1")
    with pytest.raises(RuntimeError, match="DIVERGENTE"):
        manifest.verify_model(model, path)


def test_size_mismatch_is_reported_even_with_the_right_hash(tmp_path, monkeypatch):
    model = _model(tmp_path)
    entry = _entry(model)
    entry["bytes"] = 999
    path = _write_manifest(tmp_path, {"m.onnx": entry})
    monkeypatch.setenv("DRAW_MODEL_STRICT", "1")
    with pytest.raises(RuntimeError, match="bytes"):
        manifest.verify_model(model, path)


def test_corrupted_manifest_is_an_explicit_error(tmp_path):
    path = tmp_path / "MANIFEST.json"
    path.write_text("{ nao e json")
    with pytest.raises(RuntimeError, match="ilegível"):
        manifest.load_manifest(path)


def test_unknown_schema_is_refused(tmp_path):
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps({"schema": 99, "models": {}}))
    with pytest.raises(RuntimeError, match="schema"):
        manifest.load_manifest(path)


def test_repository_manifest_is_valid():
    """O MANIFEST.json versionado precisa carregar sem erro."""
    path, models = manifest.load_manifest()
    if not path.exists():
        pytest.skip("repositório sem models/MANIFEST.json")
    assert isinstance(models, dict)


def test_repository_manifest_covers_every_default_model():
    """Todo modelo que o app abre por padrão tem entrada registrada."""
    _path, models = manifest.load_manifest()
    for name in (
        "draw_seg_v3_dyn.onnx",
        "page_clean_unet.onnx",
        "hand_landmarker.task",
    ):
        assert name in models, name


def test_neural_loader_refuses_unregistered_model_by_default(tmp_path, default_policy):
    pytest.importorskip("onnxruntime")
    from draw import neural

    model = _model(tmp_path, "sem_registro.onnx", b"nao e onnx")
    with pytest.raises(RuntimeError, match="NAO REGISTRADO"):
        neural._load_session(model)


def test_page_clean_loader_refuses_unregistered_model_by_default(
    tmp_path, default_policy
):
    pytest.importorskip("onnxruntime")
    from draw import page_clean

    model = _model(tmp_path, "sem_registro.onnx", b"nao e onnx")
    with pytest.raises(RuntimeError, match="NAO REGISTRADO"):
        page_clean._load_session(model)


def test_hand_masker_refuses_unregistered_model_by_default(tmp_path, default_policy):
    pytest.importorskip("cv2")
    from draw import hand

    model = _model(tmp_path, "mao_qualquer.task", b"stub")
    with pytest.raises(RuntimeError, match="NAO REGISTRADO"):
        hand.HandMasker(str(model))


def test_check_models_fails_on_unregistered_file(tmp_path, monkeypatch, capsys):
    pytest.importorskip("cv2")
    from draw import app

    monkeypatch.delenv("DRAW_MODEL_STRICT", raising=False)
    extra = _model(tmp_path, "intruso.onnx", b"x")
    monkeypatch.setattr(
        app.manifest_module, "unregistered_model_files", lambda *a, **k: [extra]
    )
    assert app._check_models() is False
    assert "FAIL: intruso.onnx" in capsys.readouterr().out


def test_check_models_fails_on_unregistered_file_without_manifest_entries(
    tmp_path, monkeypatch, capsys
):
    pytest.importorskip("cv2")
    from draw import app

    monkeypatch.delenv("DRAW_MODEL_STRICT", raising=False)
    extra = _model(tmp_path, "intruso.onnx", b"x")
    monkeypatch.setattr(
        app.manifest_module,
        "load_manifest",
        lambda path=None: (tmp_path / "MANIFEST.json", {}),
    )
    monkeypatch.setattr(
        app.manifest_module, "unregistered_model_files", lambda *a, **k: [extra]
    )
    assert app._check_models() is False
    assert "FAIL: intruso.onnx" in capsys.readouterr().out
