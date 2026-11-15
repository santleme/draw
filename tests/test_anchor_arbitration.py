"""Arbitragem de ancora por conteudo (v3, frente 19 de 2026-11-10).

O defeito (FALHAS M2): no pouso sustentado, migrar o estado pela
inversa do registro (v1) PIORA a duplicacao (0,69-1,14 contra
0,44-0,77 da base), porque a inversa assume a tinta parada na
camera; no modo espurio, trocar so a ancora erra o contrario
(0,74). As hipoteses sao indistinguiveis pelo quad. A v3 arbitra
pelo conteudo (Sp = massa da prob sobre o canvas confirmado na
colocacao identidade; Sc = na colocacao registrada) num streak de
assentamento, e — o mecanismo que faltava na v2 — a passada de
evento aberto NAO vota: a copia deslocada confirmava pelo voto
antes de qualquer veredito.

Contratos:
1. Pouso (Sp >> Sc): exatamente 1 troca de ancora, estado
   confirmado byte-identico ao pre-evento.
2. Espurio (Sc >> Sp): nenhuma acao, ancora intacta, canvas = base.
3. Segmentacao: passadas de meio-movimento (inconsistentes) nao
   fecham o streak; a troca so vem apos a pose assentar.
4. Abstencao: nenhuma confirmacao nasce de passada de evento.
5. Sem massa para julgar: troca por default (custo zero nas duas
   hipoteses).
6. Flag desligada: paridade byte a byte com o comportamento atual.
"""

import importlib.util
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from draw import ink_layers as ink_layers_module  # noqa: E402


def _load_replay_tool():
    spec = importlib.util.spec_from_file_location(
        "replay_reregister", REPO / "tools" / "replay_reregister.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rr = _load_replay_tool()
SHAPE = (120, 85)
BASE = rr.base_quad(SHAPE)
FLAGS = (
    "DRAW_ANCHOR_ARBITRATION",
    "DRAW_ANCHOR_MIGRATION",
    "DRAW_REG_ANCHOR_MEDIAN3",
    "DRAW_REG_DEADBAND",
    "DRAW_ECC_REFINE",
)
STEP = BASE + np.float32([120.0, 0.0])


class _ProbIsWarp:
    """page_clean falso: o "warp" submetido ja e o mapa."""

    threshold = 0.7

    def ink_prob(self, warped):
        return warped


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 0.005)
    for name in FLAGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DRAW_ANCHOR_ARBITRATION", "1")
    return monkeypatch


def _content():
    _drawing, settled = rr.synthetic_pages(SHAPE, stages=3, seed=3)
    return settled[0].astype(np.float32)


def _drive(worker, generation, sequence):
    for prob, corners in sequence:
        before = worker.stats()["passes_total"]
        worker.submit(prob, generation, corners)
        deadline = time.monotonic() + 5.0
        while worker.stats()["passes_total"] <= before:
            assert time.monotonic() < deadline, "a passada nao executou"
            time.sleep(0.001)


def _run(sequence):
    worker = ink_layers_module._PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, sequence)
        _version, confirmed, provisional = worker.poll_display(generation, 0)
        return worker, confirmed, provisional
    finally:
        worker.stop()


def _settled(content, passes=4):
    return [(content, BASE.copy()) for _ in range(passes)]


@pytest.mark.lento
def test_landing_swaps_anchor_without_touching_state(fast):
    content = _content()
    sequence = _settled(content) + [(content, STEP.copy()) for _ in range(3)]
    worker, confirmed, _provisional = _run(sequence)
    snapshot = worker.telemetry_snapshot()
    assert snapshot["anchor_arbitration_actions"] == 1
    events = snapshot["anchor_arbitration_events"]
    assert [e["verdict"] for e in events] == ["landing"]
    assert events[0]["score_page"] > 0.5
    # O contrato e a separacao (na bancada real, Sc <= 0,012); aqui a
    # pagina miniatura tem 85 px de largura e o degrau de 120 px
    # degenera a homografia, entao Sc mede 0,06 — a margem segue
    # larga.
    assert events[0]["score_page"] - events[0]["score_camera"] > 0.4
    np.testing.assert_allclose(worker._anchor_corners, STEP)
    # O estado nao se moveu: o canvas confirmado e o mesmo que as
    # passadas assentadas construiram (a tinta esta na grade da
    # pagina; quem anda e a ancora).
    _w2, truth, _p2 = _run(_settled(content))
    assert np.array_equal(confirmed, truth)


@pytest.mark.lento
def test_spurious_mode_keeps_anchor_and_canvas(fast):
    content = _content()
    # O quad mente: a prob chega distorcida pela homografia inversa
    # do registro; registrada contra a ancora, ela volta ao lugar.
    matrix = ink_layers_module._registration_matrix(BASE, STEP, SHAPE)
    distorted = cv2.warpPerspective(
        content, np.linalg.inv(matrix), (SHAPE[1], SHAPE[0])
    ).astype(np.float32)
    sequence = _settled(content) + [(distorted, STEP.copy()) for _ in range(3)]
    worker, confirmed, _provisional = _run(sequence)
    snapshot = worker.telemetry_snapshot()
    assert snapshot.get("anchor_arbitration_actions", 0) == 0
    events = snapshot["anchor_arbitration_events"]
    assert [e["verdict"] for e in events] == ["spurious"]
    assert events[0]["score_camera"] > 0.5
    assert events[0]["score_page"] < 0.05
    np.testing.assert_allclose(worker._anchor_corners, BASE)
    _w2, truth, _p2 = _run(_settled(content))
    assert np.array_equal(confirmed, truth)


@pytest.mark.lento
def test_midmovement_passes_do_not_close_the_streak(fast):
    content = _content()
    ramp = [
        (content, (BASE + np.float32([dx, 0.0])).astype(np.float32))
        for dx in (30.0, 60.0, 90.0)
    ]
    settled_step = [(content, STEP.copy()) for _ in range(3)]
    worker, _confirmed, _provisional = _run(_settled(content) + ramp + settled_step)
    snapshot = worker.telemetry_snapshot()
    # As passadas da rampa abstiveram e nenhuma acao nasceu delas;
    # a unica troca vem do streak assentado no fim.
    assert snapshot["anchor_arbitration_actions"] == 1
    events = snapshot["anchor_arbitration_events"]
    assert len(events) == 1 and events[0]["verdict"] == "landing"
    records = snapshot["records"]
    abstained = [bool(r.get("arb_abstained", False)) for r in records]
    assert abstained[4:7] == [True, True, True]


@pytest.mark.lento
def test_open_event_births_no_confirmation(fast):
    content = _content()
    other = np.zeros(SHAPE, np.float32)
    other[10:14, 10:40] = 1.0  # tinta nova, fora do canvas confirmado
    sequence = _settled(content) + [(other, STEP.copy()) for _ in range(2)]
    worker = ink_layers_module._PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, sequence)
        _version, confirmed, _provisional = worker.poll_display(generation, 0)
    finally:
        worker.stop()
    _w2, truth, _p2 = _run(_settled(content))
    assert np.array_equal(confirmed, truth), (
        "passada de evento aberto nao pode confirmar tinta nova"
    )


def _permute(img, k):
    # A construcao FISICA da frente 23 (independente do codigo sob
    # teste): a prob que o warp posicional entrega quando a folha
    # assentou rodada de k x 90 graus.
    k %= 4
    if k == 0:
        return img
    if k == 2:
        return np.rot90(img, 2).copy()
    rotated = np.rot90(img, 1 if k == 1 else -1)
    return cv2.resize(
        rotated, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)


@pytest.mark.lento
@pytest.mark.parametrize("k", [1, 2])
def test_rotated_landing_relabels_anchor_by_k(fast, k):
    # Frente 23/24: sob rotacao, a correspondencia posicional nao
    # distingue "folha rodada" de "quad errado" — o conteudo sim,
    # pontuado como permutacao da imagem entregue. Geometria
    # dedicada onde a pagina ENCHE o quadro entregue (como no
    # pipeline real, em que a prob e o warp da pagina): o veredito
    # tem que ser landing_rotated com o k fisico, a ancora carrega
    # os rotulos re-rotulados, o estado nao se move e as passadas
    # seguintes (ainda entregues rodadas) confirmam sobre o MESMO
    # canvas — e esse ultimo ponto que fixa a direcao do roll.
    full = np.array(
        [[0.0, 0.0], [199.0, 0.0], [199.0, 99.0], [0.0, 99.0]], dtype=np.float32
    )
    moved = full + np.array([30.0, 10.0], dtype=np.float32)
    content = np.zeros(SHAPE, dtype=np.float32)
    content[15:22, 30:110] = 1.0
    content[45:70, 55:63] = 1.0
    content[75:82, 120:180] = 1.0
    rotated = _permute(content, k)
    sequence = (
        [(content, full.copy()) for _ in range(3)]
        + [(rotated, moved.copy()) for _ in range(3)]
        + [(rotated, moved.copy()) for _ in range(3)]
    )
    worker, confirmed, _provisional = _run(sequence)
    snapshot = worker.telemetry_snapshot()
    assert snapshot["anchor_arbitration_actions"] == 1
    events = snapshot["anchor_arbitration_events"]
    assert [e["verdict"] for e in events] == ["landing_rotated"]
    assert events[0]["rotation_k"] == k
    np.testing.assert_allclose(worker._anchor_corners, np.roll(moved, k, axis=0))
    _tw, truth, _p2 = _run([(content, full.copy()) for _ in range(3)])
    # Retencao exata (nada do canvas se perde) e IoU alto; a
    # igualdade byte a byte dos outros contratos nao esta
    # disponivel aqui: a miniatura reamostra a prob de forma
    # anisotropica e a permutacao acontece antes dessa reamostragem
    # num braco e nao no outro — a borda de ~1 px e artefato da
    # miniatura; quem julga o numero real e a bancada da frente 23.
    assert np.count_nonzero(truth & ~confirmed) == 0
    inter = np.count_nonzero(confirmed & truth)
    union = np.count_nonzero(confirmed | truth)
    assert inter / union >= 0.85


@pytest.mark.lento
def test_straight_landing_after_rotated_landing(fast):
    # Frente 27 (pousos_3_rot): a era 3 — pouso RETO sobre a grade
    # rolada — julga-se pela hipotese-pagina compensada (a prob
    # des-rotada pelo roll parado: 0,193 medidos na captura contra
    # 0,002 do Sc) e a ancora nova e a mediana do streak rolada
    # pelo roll parado, que se preserva. Gemeo dos contratos de
    # pouso rodado, mesma construcao fisica.
    full = np.array(
        [[0.0, 0.0], [199.0, 0.0], [199.0, 99.0], [0.0, 99.0]], dtype=np.float32
    )
    moved = full + np.array([30.0, 10.0], dtype=np.float32)
    moved2 = full + np.array([70.0, 25.0], dtype=np.float32)
    content = np.zeros(SHAPE, dtype=np.float32)
    content[15:22, 30:110] = 1.0
    content[45:70, 55:63] = 1.0
    content[75:82, 120:180] = 1.0
    rotated = _permute(content, 2)
    sequence = (
        [(content, full.copy()) for _ in range(3)]
        + [(rotated, moved.copy()) for _ in range(6)]
        + [(rotated, moved2.copy()) for _ in range(6)]
    )
    worker, confirmed, _provisional = _run(sequence)
    snapshot = worker.telemetry_snapshot()
    assert snapshot["anchor_arbitration_actions"] == 2
    events = snapshot["anchor_arbitration_events"]
    assert [e["verdict"] for e in events] == ["landing_rotated", "landing"]
    assert events[1]["score_page_roll_compensated"] is True
    assert worker._anchor_roll == 2
    np.testing.assert_allclose(worker._anchor_corners, np.roll(moved2, 2, axis=0))
    _tw, truth, _p2 = _run([(content, full.copy()) for _ in range(3)])
    assert np.count_nonzero(truth & ~confirmed) == 0
    inter = np.count_nonzero(confirmed & truth)
    union = np.count_nonzero(confirmed | truth)
    assert inter / union >= 0.85


@pytest.mark.lento
def test_same_page_verdict_keeps_roll_armed_mid_generation(fast):
    # Armamento de identidade no meio da geracao (vigia, pose_jump):
    # new_generation grava (cantos, roll) juntos; o armamento
    # gravava so os cantos e o roll herdado (0) era restaurado pelo
    # veredito same_page sobre a ancora rolada — o pareamento
    # posicional passava a ler rotulo como deslocamento e o motion
    # gate travava a pagina parada.
    full = np.array(
        [[0.0, 0.0], [199.0, 0.0], [199.0, 99.0], [0.0, 99.0]], dtype=np.float32
    )
    moved = full + np.array([30.0, 10.0], dtype=np.float32)
    content = np.zeros(SHAPE, dtype=np.float32)
    content[15:22, 30:110] = 1.0
    content[45:70, 55:63] = 1.0
    content[75:82, 120:180] = 1.0
    rotated = _permute(content, 2)
    worker = ink_layers_module._PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, [(content, full.copy()) for _ in range(3)])
        _drive(worker, generation, [(rotated, moved.copy()) for _ in range(6)])
        events = worker.telemetry_snapshot()["anchor_arbitration_events"]
        assert [e["verdict"] for e in events] == ["landing_rotated"]
        assert worker._anchor_roll == 2
        np.testing.assert_allclose(worker._anchor_corners, np.roll(moved, 2, axis=0))
        # A vigia fica desligada durante o pouso (nao arma sozinha
        # no deslocamento) e liga so para o armamento; o piso de
        # canvas da miniatura e menor que o da bancada.
        fast.setenv("DRAW_IDENTITY_WATCH", "1")
        fast.setattr(ink_layers_module, "_IDENTITY_WATCH_MIN_CANVAS_PX", 100)
        assert worker.arm_identity_check("pose_jump")
        _drive(worker, generation, [(rotated, moved.copy())])
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
        # O par ancora/roll sai coerente: a ancora rolada do pouso
        # com o roll que a descreve — nao o 0 herdado.
        assert worker._anchor_roll == 2
        np.testing.assert_allclose(worker._anchor_corners, np.roll(moved, 2, axis=0))
        positional = float(
            np.max(
                np.linalg.norm(
                    moved
                    - ink_layers_module._positional_anchor(
                        worker._anchor_corners, worker._anchor_roll
                    ),
                    axis=1,
                )
            )
        )
        assert positional < 1.0
    finally:
        worker.stop()


@pytest.mark.lento
def test_no_mass_defaults_to_anchor_swap(fast):
    content = _content()
    empty = np.zeros(SHAPE, np.float32)
    # A geracao assenta numa pagina vazia (ancora nasce, nada
    # confirma) e a folha anda antes de qualquer tinta: sem massa
    # para julgar, a troca e por default.
    sequence = [(empty, BASE.copy()) for _ in range(3)] + [
        (content, STEP.copy()) for _ in range(3)
    ]
    worker, _confirmed, _provisional = _run(sequence)
    snapshot = worker.telemetry_snapshot()
    assert snapshot["anchor_arbitration_actions"] == 1
    events = snapshot["anchor_arbitration_events"]
    assert [e["verdict"] for e in events] == ["landing_default"]
    np.testing.assert_allclose(worker._anchor_corners, STEP)


@pytest.mark.lento
def test_flag_off_is_byte_identical(fast):
    fast.delenv("DRAW_ANCHOR_ARBITRATION", raising=False)
    content = _content()
    sequence = _settled(content) + [(content, STEP.copy()) for _ in range(3)]
    _worker, confirmed, _provisional = _run(sequence)
    snapshot = _worker.telemetry_snapshot()
    assert "anchor_arbitration_actions" not in snapshot
    # Sem a flag, o pouso segue o comportamento de hoje (a tinta
    # nova entra registrada na ancora velha): canvas diferente do
    # assentado puro, e nenhuma chave de arbitragem na telemetria.
    _w2, truth, _p2 = _run(_settled(content))
    assert not np.array_equal(confirmed, truth)
    assert all("arb_verdict" not in r for r in snapshot["records"])
