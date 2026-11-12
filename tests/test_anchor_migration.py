"""Migracao de ancora intra-geracao (DRAW_ANCHOR_MIGRATION).

Decisao A do Lucas (2026-11-10): no pouso sustentado (a folha anda e
assenta em pose nova no meio da geracao), o estado do acumulador e
re-projetado para a grade nova por substituicao. Contratos:

(a) deslocamento > _ANCHOR_MIGRATION_DISPLACEMENT_PX por
    _ANCHOR_MIGRATION_PASSES passadas => exatamente 1 migracao, com a
    ancora na mediana dos quads do streak;
(b) tremor <= 3 px e spikes de 1 passada => 0 migracoes;
(c) mascaras migram por vizinho mais proximo (nenhum valor novo);
(d) a janela do voto migra e o voto segue funcionando;
(e) regressao de publicacao: poll() devolve o canvas migrado na
    propria passada da migracao;
(f) identidade pendente e presenca perdida travam a migracao;
(g) flag desligada: canvas byte-identico ao baseline.
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from draw import ink_layers as ink  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rr = _load("replay_reregister", REPO / "tools" / "replay_reregister.py")
discipline = _load("test_reg_discipline_golden", REPO / "tests/test_reg_discipline.py")

SHAPE = (120, 85)
BASE = rr.base_quad(SHAPE)
# Modo espurio sustentado do detector: ~60 px de camera (~15 px da
# grade 120x85), folha parada.
OFFSET = np.float32([(60, 6), (64, -4), (58, 8), (61, 2)])
FLAGS = (
    "DRAW_ANCHOR_MIGRATION",
    "DRAW_REG_ANCHOR_MEDIAN3",
    "DRAW_REG_DEADBAND",
    "DRAW_ECC_REFINE",
)


class _ProbIsWarp:
    threshold = 0.7

    def ink_prob(self, warped):
        return warped


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(ink, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(ink, "_IDLE_PROBE_SECONDS", 0.005)
    for name in FLAGS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _page():
    _drawing, settled = rr.synthetic_pages(SHAPE, stages=3, seed=3)
    return settled[0]


def _step(worker, generation, prob, corners, visible=None):
    before = worker.stats()["passes_total"]
    worker.submit(prob, generation, corners, visible)
    deadline = time.monotonic() + 5.0
    while worker.stats()["passes_total"] <= before:
        assert time.monotonic() < deadline, "a passada nao executou"
        time.sleep(0.001)


def _pass(page, corners):
    corners = np.asarray(corners, np.float32)
    return rr.render_pass(page, BASE, corners), corners


def _settled_then(worker, generation, page, rng, quads, settle=6):
    for _ in range(settle):
        _step(worker, generation, *_pass(page, rr._noisy(BASE, rng)))
    for quad in quads:
        _step(worker, generation, *_pass(page, quad))


def test_flag_is_explicit_opt_in(monkeypatch):
    for value, expected in ((None, False), ("0", False), ("1", True), ("on", True)):
        monkeypatch.delenv("DRAW_ANCHOR_MIGRATION", raising=False)
        if value is not None:
            monkeypatch.setenv("DRAW_ANCHOR_MIGRATION", value)
        assert ink._anchor_migration_enabled() is expected


def test_constants_are_the_specified_starting_point():
    assert ink._ANCHOR_MIGRATION_DISPLACEMENT_PX == 8.0
    assert ink._ANCHOR_MIGRATION_PASSES == 3


def test_grid_displacement_of_a_translation():
    matrix = np.float64([[1, 0, 6], [0, 1, 8], [0, 0, 1]])
    assert ink._matrix_grid_displacement(matrix, SHAPE) == pytest.approx(10.0)
    assert ink._matrix_grid_displacement(np.eye(3), SHAPE) == 0.0


# --- (a) pouso sustentado: 1 migracao, ancora = mediana ----------------


@pytest.mark.lento
def test_sustained_displacement_migrates_once_to_median_anchor(fast):
    fast.setenv("DRAW_ANCHOR_MIGRATION", "1")
    page = _page()
    rng = np.random.default_rng(0)
    streak = [rr._noisy(BASE + OFFSET, rng, 1.0) for _ in range(3)]
    after = [rr._noisy(BASE + OFFSET, rng, 1.0) for _ in range(4)]
    worker = ink._PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = worker.new_generation()
        _settled_then(worker, generation, page, rng, streak + after)
        snapshot = worker.telemetry_snapshot()
        anchor = worker._anchor_corners.copy()
    finally:
        worker.stop()
    assert snapshot["anchor_migrations"] == 1
    np.testing.assert_allclose(anchor, np.median(np.stack(streak), axis=0))
    (event,) = snapshot["anchor_migration_events"]
    assert event["generation"] == generation
    assert event["pass_index"] == 6 + 2
    assert event["displacement_px"] > ink._ANCHOR_MIGRATION_DISPLACEMENT_PX
    assert event["anchor_delta_px"] > 50
    migrated = [r["anchor_migrated"] for r in snapshot["records"]]
    assert migrated == [False] * 8 + [True] + [False] * 4
    assert [r["anchor_migration_streak"] for r in snapshot["records"]][6:9] == [
        1,
        2,
        0,
    ]


# --- (b) tremor e spikes nao migram -----------------------------------


@pytest.mark.lento
def test_jitter_and_single_pass_spikes_never_migrate(fast):
    fast.setenv("DRAW_ANCHOR_MIGRATION", "1")
    page = _page()
    rng = np.random.default_rng(1)
    quads = []
    for i in range(16):
        quad = BASE + rng.uniform(-8, 8, size=(4, 2)).astype(np.float32)
        if i % 3 == 1:
            quad = BASE + np.float32([200.0, 0.0])
        quads.append(quad)
    for quad in quads:
        if not np.allclose(quad, BASE + np.float32([200.0, 0.0])):
            matrix = ink._registration_matrix(BASE, quad, SHAPE)
            assert ink._matrix_grid_displacement(matrix, SHAPE) <= 3.0
    worker = ink._PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = worker.new_generation()
        _settled_then(worker, generation, page, rng, quads)
        snapshot = worker.telemetry_snapshot()
    finally:
        worker.stop()
    assert snapshot["anchor_migrations"] == 0
    assert snapshot["anchor_migration_events"] == []
    assert max(r["anchor_migration_streak"] for r in snapshot["records"]) == 1


# --- (c) mascaras: vizinho mais proximo --------------------------------


def test_masks_migrate_without_synthetic_shoulder():
    acc = ink._LayeredInkAccumulator(0.7)
    prob = np.zeros(SHAPE, np.float32)
    prob[30:60, 20:50] = 0.95
    prob[80:82, 10:70] = 0.8
    for _ in range(3):
        acc.add(prob)
    acc._provisional[90:95, 40:45] = 255
    before = int(np.count_nonzero(acc._confirmed))
    matrix = np.float64([[1, 0, 3.5], [0, 1, -2.25], [0, 0, 1]])
    acc.migrate(matrix)
    for layer in (acc._confirmed, acc._provisional):
        assert layer.dtype == np.uint8
        assert set(np.unique(layer)) <= {0, 255}
    assert abs(int(np.count_nonzero(acc._confirmed)) - before) <= 0.05 * before
    assert acc._score_buf is None and acc._hits_buf is None and acc._seen_buf is None
    assert acc._low_overlap_streak == 0


def test_migrate_before_any_pass_is_a_no_op():
    acc = ink._LayeredInkAccumulator(0.7)
    acc.migrate(np.eye(3))
    assert acc._confirmed is None


# --- (d) janela migrada, voto segue -------------------------------------


def test_vote_window_migrates_and_vote_keeps_working():
    # Lapis (0,6): so confirma por 3 hits na janela. 2 passadas antes
    # da migracao + 1 depois so fecham o voto se a janela migrou.
    def pencil(dx):
        prob = np.zeros(SHAPE, np.float32)
        prob[40:50, 20 + dx : 40 + dx] = 0.6
        return prob

    acc = ink._LayeredInkAccumulator(0.7)
    for _ in range(2):
        assert not acc.add(pencil(0))["confirmed_changed"]
    acc.migrate(np.float64([[1, 0, 10], [0, 1, 0], [0, 0, 1]]))
    assert all(item.dtype == np.float32 for item in acc._window)
    np.testing.assert_allclose(acc._window[0], pencil(10), atol=1e-6)
    result = acc.add(pencil(10))
    assert result["confirmed_changed"]
    assert np.array_equal(result["confirmed"] > 0, pencil(10) > 0)

    control = ink._LayeredInkAccumulator(0.7)
    for _ in range(2):
        control.add(pencil(0))
    unmigrated = control.add(pencil(10))["confirmed"]
    assert not np.any(unmigrated[:, 40:50]), "sem migrar, a parte nova nao fecha"


# --- (e) publicacao forcada na passada da migracao ----------------------


@pytest.mark.lento
def test_poll_returns_migrated_canvas_on_the_migration_pass(fast):
    fast.setenv("DRAW_ANCHOR_MIGRATION", "1")
    page = _page()
    rng = np.random.default_rng(2)
    worker = ink._PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = worker.new_generation()
        _settled_then(worker, generation, page, rng, [], settle=8)
        version, before = worker.poll(generation, 0)
        for k in range(3):
            assert worker.poll(generation, version) is None or k == 0
            _step(worker, generation, *_pass(page, BASE + OFFSET))
            if k < 2:
                got = worker.poll(generation, version)
                if got is not None:
                    version, before = got
        got = worker.poll(generation, version)
        record = worker.telemetry_snapshot()["records"][-1]
        internal = worker._accumulator._confirmed.copy()
    finally:
        worker.stop()
    assert record["anchor_migrated"]
    assert got is not None, "poll() devolveu a copia pre-migracao"
    _version, canvas = got
    assert np.array_equal(canvas, internal)
    assert not np.array_equal(canvas, before)


# --- (f) travas: identidade pendente e presenca perdida -----------------


@pytest.mark.lento
def test_pending_identity_blocks_migration(fast):
    fast.setenv("DRAW_ANCHOR_MIGRATION", "1")
    page = _page()
    rng = np.random.default_rng(3)
    worker = ink._PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = worker.new_generation()
        _settled_then(worker, generation, page, rng, [], settle=6)
        generation = worker.new_generation(preserve_identity=True)
        hidden = np.zeros(SHAPE, bool)
        for _ in range(4):
            prob, corners = _pass(page, BASE + OFFSET)
            _step(worker, generation, prob, corners, hidden)
        snapshot = worker.telemetry_snapshot()
        pending = worker._identity_recovery_requested
    finally:
        worker.stop()
    assert pending, "a quarentena por oclusao seguia em curso"
    assert snapshot["anchor_migrations"] == 0
    assert all(
        r["anchor_migration_streak"] == 0
        for r in snapshot["records"]
        if r["generation"] == generation
    )


@pytest.mark.lento
def test_lost_presence_resets_the_streak(fast):
    fast.setenv("DRAW_ANCHOR_MIGRATION", "1")
    page = _page()
    rng = np.random.default_rng(4)
    worker = ink._PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = worker.new_generation()
        _settled_then(worker, generation, page, rng, [], settle=6)
        moved, corners = _pass(page, BASE + OFFSET)
        for quad in (corners, corners, None, corners, corners, None, corners):
            _step(worker, generation, moved, quad)
        snapshot = worker.telemetry_snapshot()
    finally:
        worker.stop()
    assert snapshot["anchor_migrations"] == 0
    streaks = [r["anchor_migration_streak"] for r in snapshot["records"]][6:]
    assert streaks == [1, 2, 0, 1, 2, 0, 1]


# --- (g) paridade com a flag desligada ----------------------------------


@pytest.mark.lento
@pytest.mark.parametrize("flag_value", [None, "0"])
@pytest.mark.parametrize("scenario", sorted(discipline.GOLDEN))
def test_flag_off_canvas_is_byte_identical_to_baseline(fast, scenario, flag_value):
    if flag_value is not None:
        fast.setenv("DRAW_ANCHOR_MIGRATION", flag_value)
    digest, snapshot = discipline._parity_canvas(scenario)
    assert digest == discipline.GOLDEN[scenario]
    assert "anchor_migrations" not in snapshot
    assert not any(
        k.startswith("anchor_") and k != "anchor_arbitration_enabled"
        for r in snapshot["records"]
        for k in r
    )
    assert all(r["anchor_arbitration_enabled"] is False for r in snapshot["records"])


@pytest.mark.lento
@pytest.mark.parametrize("scenario", sorted(discipline.GOLDEN))
def test_flag_on_never_triggers_on_settled_jitter_replay(fast, scenario):
    fast.setenv("DRAW_ANCHOR_MIGRATION", "1")
    digest, snapshot = discipline._parity_canvas(scenario)
    assert snapshot["anchor_migrations"] == 0
    assert digest == discipline.GOLDEN[scenario]


@pytest.mark.lento
def test_migration_pass_keeps_ecc_against_the_migrated_canvas(fast):
    fast.setenv("DRAW_ANCHOR_MIGRATION", "1")
    fast.setenv("DRAW_ECC_REFINE", "1")
    calls = []
    original = ink._register_ink_probability

    def spy(prob, anchor, corners, reference, visible=None):
        calls.append(None if reference is None else reference.copy())
        return original(prob, anchor, corners, reference, visible)

    fast.setattr(ink, "_register_ink_probability", spy)
    page = _page()
    rng = np.random.default_rng(5)
    worker = ink._PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = worker.new_generation()
        _settled_then(worker, generation, page, rng, [BASE + OFFSET] * 3)
        record = worker.telemetry_snapshot()["records"][-1]
    finally:
        worker.stop()
    assert record["anchor_migrated"]
    pre_migration, migrated = calls[-2], calls[-1]
    assert migrated is not None, "a passada da migracao pulou o ECC"
    assert not np.array_equal(migrated, pre_migration)
