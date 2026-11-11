"""Entradas externas corrompidas/truncadas não derrubam nem estouram memória.

Cobre os leitores endurecidos de SECURITY_MODEL.md ("Entry points"):
imagens (draw/safe_io), JSON/JSONL de sessões e datasets, manifesto
do pool, receitas do runner, política de rede dos downloads, gravação
de exemplos com linhagem e a carga de checkpoints só com tensores.
"""

import ast
import importlib.util
import io
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from draw import net_policy, safe_io  # noqa: E402
from draw.selflearn import process, select  # noqa: E402
from draw.selflearn import session as session_module  # noqa: E402
from draw.selflearn.collect import ExampleRecorder  # noqa: E402
from draw.selflearn.split import LeakageError  # noqa: E402
from tools.runner.recipe import Recipe  # noqa: E402

import fetch_models  # noqa: E402


def _png(width=32, height=24, channels=3):
    shape = (height, width, channels) if channels > 1 else (height, width)
    ok, buf = cv2.imencode(".png", np.full(shape, 200, dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _jpg(width=32, height=24):
    ok, buf = cv2.imencode(".jpg", np.full((height, width, 3), 90, np.uint8))
    assert ok
    return buf.tobytes()


# --- imagens -------------------------------------------------------------


def test_image_dimensions_from_headers():
    assert safe_io.image_dimensions(_png(40, 30)) == (40, 30)
    assert safe_io.image_dimensions(_jpg(40, 30)) == (40, 30)
    assert safe_io.image_dimensions(b"GIF89a....") is None


@pytest.mark.parametrize("make", [_png, _jpg])
@pytest.mark.parametrize("cut", [0, 4, 12, 23])
def test_truncated_headers_are_rejected(make, cut):
    data = make()[:cut]
    with pytest.raises(safe_io.InputLimitError):
        safe_io.check_image_bytes(data)
    assert safe_io.decode_image(data) is None


@pytest.mark.parametrize("make", [_png, _jpg])
def test_truncated_body_never_raises(make):
    data = make(256, 256)
    for cut in (len(data) // 3, len(data) // 2, len(data) - 3):
        result = safe_io.decode_image(data[:cut])
        assert result is None or result.shape[:2] == (256, 256)


def test_png_bomb_dimensions_rejected_before_decode():
    data = bytearray(_png())
    struct.pack_into(">II", data, 16, 100_000, 100_000)
    with pytest.raises(safe_io.InputLimitError, match="dimensoes"):
        safe_io.check_image_bytes(bytes(data))
    assert safe_io.decode_image(bytes(data)) is None


def test_zero_dimension_rejected():
    data = bytearray(_png())
    struct.pack_into(">II", data, 16, 0, 10)
    assert safe_io.decode_image(bytes(data)) is None


def test_oversized_file_not_read(tmp_path):
    path = tmp_path / "grande.png"
    path.write_bytes(_png() + b"\x00" * 2048)
    assert safe_io.read_image(path, max_bytes=1024) is None
    assert safe_io.read_image(path) is not None


def test_read_image_missing_and_garbage(tmp_path):
    assert safe_io.read_image(tmp_path / "nao.png") is None
    junk = tmp_path / "lixo.png"
    junk.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\xff" * 64)
    assert safe_io.read_image(junk) is None


def test_read_image_grayscale_flag(tmp_path):
    path = tmp_path / "m.png"
    path.write_bytes(_png(16, 8, channels=1))
    assert safe_io.read_image(path, cv2.IMREAD_GRAYSCALE).shape == (8, 16)


# --- JSON / JSONL ----------------------------------------------------------


def test_read_json_limits_and_syntax(tmp_path):
    path = tmp_path / "a.json"
    path.write_text('{"a": 1}')
    assert safe_io.read_json(path) == {"a": 1}
    with pytest.raises(safe_io.InputLimitError, match="limite"):
        safe_io.read_json(path, max_bytes=4)
    path.write_text('{"a": ')
    with pytest.raises(safe_io.InputLimitError, match="JSON invalido"):
        safe_io.read_json(path)
    path.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(safe_io.InputLimitError):
        safe_io.read_json(path)


def test_read_json_deep_nesting_is_controlled(tmp_path):
    path = tmp_path / "fundo.json"
    path.write_text("[" * 200_000 + "]" * 200_000)
    with pytest.raises(safe_io.InputLimitError):
        safe_io.read_json(path)


def test_iter_jsonl_reports_line_and_limits(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text('{"a": 1}\n\n{"b": 2}\n{"c": \n')
    rows = []
    with pytest.raises(safe_io.InputLimitError, match=":4:"):
        for row in safe_io.iter_jsonl(path):
            rows.append(row)
    assert rows == [(1, {"a": 1}), (3, {"b": 2})]
    path.write_text("[1, 2]\n")
    with pytest.raises(safe_io.InputLimitError, match="objeto"):
        list(safe_io.iter_jsonl(path))
    path.write_text('{"a": 1}\n' * 5)
    with pytest.raises(safe_io.InputLimitError, match="linhas"):
        list(safe_io.iter_jsonl(path, max_lines=3))
    path.write_text('{"a": "' + "x" * 100 + '"}\n')
    with pytest.raises(safe_io.InputLimitError, match="bytes"):
        list(safe_io.iter_jsonl(path, max_line_bytes=50))


# --- pool de sessões (draw/selflearn/process.py) ---------------------------


def _session(root, sid, manifest_text=None, rel="gen_0001/{kind}.png"):
    sdir = root / sid
    (sdir / "gen_0001").mkdir(parents=True)
    entry = {"generation": 1, "files": {}}
    for kind in ("reference_warp", "final_warp", "final_canvas"):
        name = rel.format(kind=kind)
        target = (sdir / name).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_png(64, 64))
        entry["files"][kind] = name
        entry[f"{kind}_sha256"] = session_module.sha256_file(target)
    text = manifest_text
    if text is None:
        text = json.dumps({"generations": [entry], "models": {}})
    (sdir / "manifest.json").write_text(text)
    return sdir


@pytest.mark.parametrize(
    "text",
    ["", '{"generations": [', "[]", '{"generations": "x"}', '{"generations": [1]}'],
    ids=["vazio", "truncado", "lista", "tipo_errado", "geracao_nao_objeto"],
)
def test_scan_sessions_survives_corrupt_manifest(tmp_path, text):
    _session(tmp_path, "boa")
    _session(tmp_path, "ruim", manifest_text=text)
    sessions, problems = process.scan_sessions(tmp_path)
    by_id = {s["session_id"]: s for s in sessions}
    assert len(by_id["boa"]["generations"]) == 1
    assert "ruim" not in by_id or by_id["ruim"]["generations"] == []
    assert any("ruim" in p or "malformad" in p or "ilegivel" in p for p in problems)


def test_scan_sessions_refuses_paths_outside_the_session(tmp_path):
    root = tmp_path / "sessions"
    root.mkdir()
    _session(root, "fuga", rel="../../fora_{kind}.png")
    sessions, problems = process.scan_sessions(root)
    assert sessions[0]["generations"] == []
    assert any("fora da sessao" in p for p in problems)


def test_process_pool_marks_output_private(tmp_path):
    _session(tmp_path / "sessions", "s1")
    process.process_pool(tmp_path / "sessions", tmp_path / "pool", 1)
    assert (tmp_path / "pool" / ".gitignore").read_text().strip().endswith("*")


# --- gravação privada e linhagem ------------------------------------------


def test_example_recorder_marks_dir_private_and_requires_lineage(tmp_path):
    recorder = ExampleRecorder(tmp_path / "ex")
    assert (tmp_path / "ex" / ".gitignore").exists()
    warped = np.zeros((8, 8, 3), np.uint8)
    prob = np.zeros((8, 8), np.float32)
    corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
    with pytest.raises(LeakageError):
        recorder.record("", warped, prob, corners, "sha", "live")
    with pytest.raises(LeakageError):
        recorder.record("s1", warped, prob, corners, "sha", "")
    with pytest.raises(LeakageError, match="split"):
        recorder.record("s1", warped, prob, corners, "sha", "live", split="val")
    row = recorder.record("s1", warped, prob, corners, "sha", "live")
    assert row["session_id"] == "s1" and row["split"] == "train"


def test_session_recorder_folder_is_private(tmp_path):
    recorder = session_module.SessionRecorder(tmp_path, session_id="s1")
    recorder._write_manifest({})
    marker = tmp_path / "sessions" / "s1" / ".gitignore"
    assert marker.exists() and "*" in marker.read_text()


def test_ensure_private_dir_keeps_existing_gitignore(tmp_path):
    (tmp_path / ".gitignore").write_text("custom\n")
    safe_io.ensure_private_dir(tmp_path)
    assert (tmp_path / ".gitignore").read_text() == "custom\n"


def test_repository_gitignore_covers_private_data():
    lines = set((ROOT / ".gitignore").read_text().splitlines())
    for pattern in ("captures/", "sessions/", "dataset/", "*.drawcap"):
        assert pattern in lines, pattern


def test_selection_requires_explicit_holdout_and_lineage():
    items = [{"session_id": "a", "origin": "live", "tta_disagreement": 0.5}]
    with pytest.raises(TypeError):
        select.rank_examples(items)
    with pytest.raises(LeakageError):
        select.rank_examples(items, holdout_session_ids=["a"])
    with pytest.raises(LeakageError, match="linhagem"):
        select.rank_examples([{"session_id": "b"}], holdout_session_ids=[])


# --- train_round: manifesto do dataset e ledger -----------------------------


def _train_round():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "nn"):
        pytest.skip("torch real nao disponivel")
    sys.modules["torch"] = torch
    spec = importlib.util.spec_from_file_location(
        "train_round_hardening", ROOT / "tools" / "selflearn" / "train_round.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "text,match",
    [
        ('{"file": "a.png"}\n{"file": ', "JSON invalido"),
        ('{"file": 3}\n', "texto"),
        ("[1]\n", "objeto"),
    ],
)
def test_dataset_manifest_corruption_is_a_clear_error(tmp_path, text, match):
    tr = _train_round()
    (tmp_path / "manifest.jsonl").write_text(text)
    with pytest.raises(ValueError, match=match):
        tr._manifest_rows(tmp_path)


def test_dataset_corrupt_image_is_a_clear_error(tmp_path):
    tr = _train_round()
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "p.png").write_bytes(_png()[:30])
    row = {"file": "p.png", "session_id": "s", "origin": "live", "split": "train"}
    (tmp_path / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="ilegível"):
        tr.load_pages(tmp_path, with_mask=False)


def test_sealed_holdout_ids_from_ledger(tmp_path):
    tr = _train_round()
    ledger = tmp_path / "split.json"
    ledger.write_text(json.dumps({"assignments": {"a": "holdout", "b": "train"}}))
    assert tr.sealed_holdout_ids(ledger) == {"a"}
    ledger.write_text('{"assignments": ')
    with pytest.raises(ValueError):
        tr.sealed_holdout_ids(ledger)


# --- checkpoints -------------------------------------------------------------

# Corpos gerados/vendados byte a byte (pyproject.toml): fora do escopo.
_FROZEN = {"training/train_ink3.py", "training/train_body_tri.py"}


def _torch_load_calls():
    for folder in ("draw", "tools", "training"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if rel in _FROZEN:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "load"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch"
                ):
                    yield rel, node


def test_every_checkpoint_load_is_weights_only():
    offenders = []
    for rel, node in _torch_load_calls():
        flags = {kw.arg: kw.value for kw in node.keywords}
        value = flags.get("weights_only")
        if not (isinstance(value, ast.Constant) and value.value is True):
            offenders.append(f"{rel}:{node.lineno}")
    assert offenders == []


def test_npz_stamps_refuse_pickled_objects(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "vectorize_hardening", ROOT / "tools" / "vectorize.py"
    )
    vectorize = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vectorize)
    path = tmp_path / "stamps.npz"
    np.savez(path, first_seen_ms=np.array([{"x": 1}], dtype=object))
    with pytest.raises(ValueError):
        vectorize.load_stamps(path)


# --- receitas do runner --------------------------------------------------


def _recipe(**overrides):
    data = {
        "name": "r",
        "train_script": "s.py",
        "kernel_slug": "draw-ink-v1",
        "datasets": ["owner/data-v1"],
    }
    data.update(overrides)
    return Recipe.from_dict(data)


@pytest.mark.parametrize(
    "overrides",
    [
        {"kernel_slug": "x&userName=outro"},
        {"kernel_slug": "../kernels"},
        {"kernel_slug": "Maiuscula"},
        {"datasets": "owner/data"},
        {"datasets": ["sem-dono"]},
        {"datasets": ["owner/data?x=1"]},
        {"datasets": [3]},
    ],
)
def test_recipe_rejects_injection_in_slugs(overrides):
    with pytest.raises(ValueError):
        _recipe(**overrides).validate()


def test_recipe_accepts_repository_candidates():
    for line in (
        (ROOT / "tools" / "runner" / "candidates.jsonl").read_text().splitlines()
    ):
        if line.strip():
            Recipe.from_dict(json.loads(line)["recipe"]).validate()


def test_recipe_from_non_object_is_refused():
    with pytest.raises(ValueError):
        Recipe.from_dict(["nao", "objeto"])


# --- política de rede -----------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/x",
        "ftp://github.com/x",
        "file:///etc/passwd",
        "https://github.com.evil.example/x",
        "https://evil.example/github.com",
        "https://user:pw@github.com/x",
        "http://127.0.0.1/x",
    ],
)
def test_check_url_refuses(url):
    with pytest.raises(net_policy.UrlPolicyError):
        net_policy.check_url(url, net_policy.GITHUB_RELEASE_HOSTS)


def test_check_url_accepts_allowlisted_and_explicit_loopback():
    net_policy.check_url("https://github.com/a", net_policy.GITHUB_RELEASE_HOSTS)
    net_policy.check_url("http://127.0.0.1:9/a", (), allow_loopback_http=True)


def test_redirect_handler_validates_each_hop():
    handler = net_policy._PolicyRedirect(("github.com",), True, False)
    request = __import__("urllib.request").request.Request("https://github.com/a")
    with pytest.raises(net_policy.UrlPolicyError):
        handler.redirect_request(request, None, 302, "", {}, "https://evil.example/")
    strict = net_policy._PolicyRedirect(("github.com",), False, False)
    with pytest.raises(net_policy.UrlPolicyError, match="redirect"):
        strict.redirect_request(request, None, 302, "", {}, "https://github.com/b")


def test_fetch_models_refuses_plain_http_base(tmp_path):
    with pytest.raises(net_policy.UrlPolicyError):
        fetch_models.ensure_model(
            "m.onnx",
            {"sha256": "0" * 64, "bytes": 1},
            models_dir=tmp_path,
            base_url="http://mirror.example/rel",
            opener=lambda url: io.BytesIO(b"x"),
        )


def test_fetch_models_refuses_path_in_name(tmp_path):
    with pytest.raises(RuntimeError, match="caminho"):
        fetch_models.ensure_model(
            "../m.onnx",
            {"sha256": "0" * 64, "bytes": 1},
            models_dir=tmp_path,
            base_url="https://example.invalid/rel",
            opener=lambda url: io.BytesIO(b"x"),
        )


def test_fetch_models_stops_at_registered_size(tmp_path):
    with pytest.raises(RuntimeError, match="passou"):
        fetch_models.ensure_model(
            "m.onnx",
            {"sha256": "0" * 64, "bytes": 4},
            models_dir=tmp_path,
            base_url="https://example.invalid/rel",
            opener=lambda url: io.BytesIO(b"x" * (3 << 20)),
        )
    assert not (tmp_path / "m.onnx.part").exists()
    assert not (tmp_path / "m.onnx").exists()


def test_finetune_download_allowlist():
    spec = importlib.util.spec_from_file_location(
        "finetune_real_hardening", ROOT / "training" / "finetune_real.py"
    )
    source = spec.loader.get_source("finetune_real_hardening")
    assert "urlretrieve" not in source
    tree = ast.parse(source)
    namespace = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "_DOWNLOAD_HOSTS" for t in node.targets
        ):
            exec(compile(ast.Module([node], []), "fr", "exec"), namespace)
        if isinstance(node, ast.FunctionDef) and node.name == "_check_download_url":
            exec(compile(ast.Module([node], []), "fr", "exec"), namespace)
    check = namespace["_check_download_url"]
    check("https://github.com/santleme/draw/releases/download/x/y.onnx")
    for bad in ("http://github.com/x", "https://evil.example/x"):
        with pytest.raises(RuntimeError):
            check(bad)
