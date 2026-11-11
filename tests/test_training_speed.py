"""Pacote de tempo por experimento: compile padrão em CUDA com fallback
para eager, channels_last opt-in e o modo steps exposto pelo CLI."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from training import train as T  # noqa: E402


def _load_dispatcher():
    spec = importlib.util.spec_from_file_location(
        "run_dispatcher_speed", ROOT / "training/run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _net():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(4, 4, 1)
    )


@pytest.mark.parametrize(
    ("cli", "tr", "device", "expected"),
    [
        (None, {}, "cuda", True),
        (None, {}, "cuda:0", True),
        (None, {}, "cpu", False),
        (None, {"compile": False}, "cuda", False),
        (None, {"compile": True}, "cpu", True),
        (False, {"compile": True}, "cuda", False),
        (True, {}, "cpu", True),
    ],
)
def test_resolve_compile_padrao_ligado_so_em_cuda(cli, tr, device, expected):
    assert T.resolve_compile(cli, tr, device) is expected


def test_configs_que_fixam_compile_false_seguem_em_eager():
    cfg = T.load_config("training/configs/t15-steps.json")
    assert T.resolve_compile(None, cfg["training"], "cuda") is False
    base = T.load_config("training/configs/base.json")
    assert "compile" not in base["training"]
    assert T.resolve_compile(None, base["training"], "cuda") is True


def test_apply_compile_falha_na_chamada_cai_para_eager(monkeypatch, capsys):
    def _boom(*_a, **_k):
        raise RuntimeError("E1001 simulado")

    monkeypatch.setattr(torch, "compile", _boom)
    net = _net()
    out, status = T.apply_compile(torch, net, "cuda")
    assert status == "eager_fallback" and out is net
    assert "E1001 simulado" in capsys.readouterr().err


def test_apply_compile_falha_no_primeiro_passo_cai_para_eager(monkeypatch):
    import torch._dynamo

    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", False)

    def _broken(_gm, _inputs):
        raise RuntimeError("backend quebrado")

    net = _net()
    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        eager = net(x)
    compiled, status = T.apply_compile(torch, net, "cpu", backend=_broken)
    try:
        assert status == "compiled"
        out = compiled(x)
        out.sum().backward()
        status, reason = T.compile_runtime_status(status)
        assert status == "eager_fallback_runtime"
        assert "backend quebrado" in reason or "CONVERT" in reason, reason
        assert torch.allclose(out.detach(), eager, atol=1e-6)
        assert all(torch.isfinite(p.grad).all() for p in net.parameters())
    finally:
        torch._dynamo.reset()


def test_compile_runtime_status_sem_falha_segue_compiled():
    net = _net()
    compiled, status = T.apply_compile(torch, net, "cpu", backend="eager")
    try:
        compiled(torch.randn(1, 3, 8, 8))
        assert T.compile_runtime_status(status) == ("compiled", None)
        assert T.compile_runtime_status("off") == ("off", None)
    finally:
        torch._dynamo.reset()


def test_channels_last_converte_pesos_e_lote_e_reverte():
    net = _net()
    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        eager = net(x)
    handle = T.enable_channels_last(torch, net)
    assert net[0].weight.is_contiguous(memory_format=torch.channels_last)
    seen = []
    probe = net[0].register_forward_pre_hook(
        lambda _m, a: seen.append(a[0].is_contiguous(memory_format=torch.channels_last))
    )
    with torch.no_grad():
        out = net(x)
    probe.remove()
    assert seen == [True]
    assert torch.allclose(out, eager, atol=1e-5)
    T.disable_channels_last(torch, net, handle)
    assert net[0].weight.is_contiguous()
    assert not net._forward_pre_hooks


def test_channels_last_converte_pixel_values_por_kwarg():
    class _Seg(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 1)
            self.seen = None

        def forward(self, pixel_values):
            self.seen = pixel_values.is_contiguous(memory_format=torch.channels_last)
            return self.conv(pixel_values)

    model = _Seg()
    T.enable_channels_last(torch, model)
    T._model_logits(model, torch.randn(1, 3, 8, 8))
    assert model.seen is True


def test_dispatcher_repassa_flags_de_velocidade(capsys):
    run = _load_dispatcher()
    rc = run.main(
        [
            "--backend",
            "cpu",
            "--dry-run",
            "--mode",
            "steps",
            "--max-steps",
            "200",
            "--eval-interval-steps",
            "50",
            "--patience",
            "2",
            "--channels-last",
            "--no-compile",
        ]
    )
    assert rc == 0
    command = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("COMMAND")
    )
    for piece in (
        "--mode steps",
        "--max-steps 200",
        "--eval-interval-steps 50",
        "--patience 2",
        "--channels-last",
        "--no-compile",
    ):
        assert piece in command, (piece, command)


def test_train_cli_rejeita_passos_nao_positivos():
    with pytest.raises(SystemExit):
        T.main(["--max-steps", "0", "--smoke"])
