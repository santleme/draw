"""Contrato da bateria de avaliação: perfis, geração e cache."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import eval_battery  # noqa: E402
import eval_scenario  # noqa: E402


def test_default_profile_pins_hardened_flags_off():
    """O perfil default zera toda flag que o hardened liga.

    Sem o pin explícito, uma flag exportada no ambiente vazava
    para a rodada "default" e a linha de base media o
    comportamento endurecido — a comparação entre perfis
    deixava de medir o efeito das proteções.
    """
    hardened = eval_battery.PROFILES["hardened"]
    default = eval_battery.PROFILES["default"]
    assert hardened, "perfil hardened vazio — o teste não mede nada"
    for key, value in hardened.items():
        assert value == "1"
        assert default.get(key) == "0", key


def test_sampled_erase_rect_always_inside_the_page():
    """O retângulo de borracha amostrado cabe na folha gerada.

    O sim_source fixa a folha em 360x510 (size_idx=0). Amostrar
    x e w independentes passava da borda e o _resolve_erase_rect
    rejeitava o cenário inteiro na validação.
    """
    rng = np.random.default_rng(20260707)
    rects = 0
    for _ in range(400):
        argv = eval_battery._sample_scenario_args(rng, [])
        if "--erase-rect" not in argv:
            continue
        rects += 1
        x, y, w, h = (int(v) for v in argv[argv.index("--erase-rect") + 1].split(","))
        assert 0 < x < eval_battery._GEN_PAGE_W
        assert 0 < y < eval_battery._GEN_PAGE_H
        assert 0 < w <= eval_battery._GEN_PAGE_W - x
        assert 0 < h <= eval_battery._GEN_PAGE_H - y
    assert rects > 0, "nenhum cenário com borracha — o teste não mede nada"


def _write_bundle(target):
    """O que o sim_source deixa no disco quando termina são."""
    target = Path(target)
    target.write_bytes(b"mp4")
    np.savez(target.with_suffix(".gt.npz"), version=1)
    np.save(target.with_name(target.stem + "_ink_full.npy"), np.zeros(4))
    target.with_suffix(".json").write_text("{}", encoding="utf-8")


def _fake_run_factory(calls, fail=False):
    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if not fail:
            _write_bundle(Path(cmd[cmd.index("--out") + 1]))
        return subprocess.CompletedProcess(
            cmd, 1 if fail else 0, "", "boom" if fail else ""
        )

    return fake_run


def test_failed_generation_aborts_instead_of_reporting_empty(tmp_path, monkeypatch):
    """Cenário pedido que não se materializa não vira relatório
    verde: --generate saía 0 com results vazio."""
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(calls, fail=True))
    with pytest.raises(SystemExit):
        eval_battery._generate_captures(1, 0, tmp_path)
    assert calls


def test_incomplete_bundle_is_regenerated(tmp_path, monkeypatch):
    """MP4 sem o resto do pacote (geração morta no meio) é
    descartado e gerado de novo — não aborta o np.load."""
    target = tmp_path / "gen-000.mp4"
    target.write_bytes(b"mp4")  # sem .gt.npz/_ink_full/.json
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(calls))
    generated = eval_battery._generate_captures(1, 0, tmp_path)
    assert generated == {"gen-000": {}}
    assert len(calls) == 1
    assert eval_battery._capture_bundle_complete(target)


def test_validated_cache_reuses_and_photo_change_invalidates(tmp_path, monkeypatch):
    """O cache vale só com a identidade conferida: mesmos argv,
    mesmas fotos (tamanho+mtime), pacote completo."""
    photo = tmp_path / "desk.jpg"
    photo.write_bytes(b"foto-v1")
    monkeypatch.setattr(
        eval_battery,
        "_sample_scenario_args",
        lambda rng, photos: ["--desk-photo", str(photo)],
    )
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(calls))

    eval_battery._generate_captures(1, 0, tmp_path)
    assert len(calls) == 1

    # Mesma identidade: segunda execução reusa sem regerar.
    eval_battery._generate_captures(1, 0, tmp_path)
    assert len(calls) == 1

    # A foto da mesa mudou: o cenário em cache já não é a mesma
    # cena — regera em cima do pacote velho.
    photo.write_bytes(b"foto-v2-maior")
    eval_battery._generate_captures(1, 0, tmp_path)
    assert len(calls) == 2

    manifest = json.loads(
        (tmp_path / eval_battery._GENERATION_MANIFEST).read_text(encoding="utf-8")
    )
    assert manifest["scenarios"]["gen-000"]["argv"] == [
        "--desk-photo",
        str(photo),
    ]


def test_final_checkpoint_healthy_reports_ok():
    """O veredito do evento final precisa de uma palavra — sem
    'status' a tabela lia None e imprimia '-' numa rodada válida."""
    events = [{"kind": "draw", "start": 0, "end": 10}]
    checkpoints = [
        {
            "frame": 9,
            "phase": "draw",
            "iou": 0.9,
            "recall": 0.8,
            "confirmed_px": 100,
        }
    ]
    final = eval_scenario._final_checkpoint(checkpoints, events)
    assert final["status"] == "ok"
    assert final["iou"] == 0.9
    # Os problemas continuam com a palavra deles.
    assert "checkpoint" in eval_scenario._final_checkpoint([], events)["status"]
