"""Estado de sessão do app: folha nova, falha da tinta e epílogo.

Três comportamentos que só aparecem no laço de `run()` e que não
tinham teste: o canvas da folha anterior vazando para a folha nova,
a trabalhadora de tinta morrendo calada e o epílogo (liberar a
captura, parar as threads, drenar os PNGs) rodando só no caminho
feliz.

O harness é o mesmo de tests/test_capture_thread.py: captura falsa,
`cv2.imshow`/`waitKey` "espias" e a GUI nunca abre janela.
"""

import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from draw import app as app_module  # noqa: E402
from draw import ink_layers as ink_layers_module  # noqa: E402
from selftest import build_markerless_scene  # noqa: E402

INK_RECT = (slice(300, 500), slice(300, 500))


class _SpyGui:
    def __init__(self):
        self.shown = []

    def imshow(self, _title, frame):
        self.shown.append(frame.copy())

    def waitKey(self, _delay):
        return -1

    def destroyAllWindows(self):
        pass


class _ScriptedCapture:
    """Quadros roteirizados: folha, mesa vazia, folha de novo."""

    def __init__(self, sheet, blank, plan, period_s=0.006):
        self._sheet = sheet
        self._blank = blank
        self._plan = plan
        self._index = 0
        self._period = period_s
        self.released = False

    def read(self):
        if self._index >= len(self._plan):
            return False, None
        kind = self._plan[self._index]
        self._index += 1
        time.sleep(self._period)
        frame = self._sheet if kind == "sheet" else self._blank
        return True, frame.copy()

    def release(self):
        self.released = True


def _run_headless(plan, fake_ink_factory, period_s=0.006):
    """Roda run() com GUI espia. Devolve (spy, capture, erro do run)."""
    sheet, _gt, _dst = build_markerless_scene()
    sheet = cv2.resize(sheet, (1280, 720), interpolation=cv2.INTER_AREA)
    blank = np.full((720, 1280, 3), 40, dtype=np.uint8)
    capture = _ScriptedCapture(sheet, blank, plan, period_s)
    spy = _SpyGui()

    real_open = app_module._open_capture
    real_page_clean = app_module.page_clean_module.PageCleanInk
    real_neural_detector = app_module.neural_module.NeuralSheetDetector
    real_imshow, real_wait = cv2.imshow, cv2.waitKey
    real_destroy = cv2.destroyAllWindows
    errors = []

    def target():
        try:
            app_module.run(source=0, save_dir=tempfile.mkdtemp())
        except Exception as error:  # noqa: BLE001 - o teste le o erro
            errors.append(error)

    app_module._open_capture = lambda _source: capture
    app_module.page_clean_module.PageCleanInk = fake_ink_factory

    class _NoNeuralDetector:
        last_presence_logit = None

        def __init__(self, _path):
            pass

        def load(self):
            return True

        def detect_sheet_with_confidence(self, _frame):
            return None, None

    app_module.neural_module.NeuralSheetDetector = _NoNeuralDetector
    cv2.imshow, cv2.waitKey = spy.imshow, spy.waitKey
    cv2.destroyAllWindows = spy.destroyAllWindows
    try:
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=120.0)
        finished = not thread.is_alive()
    finally:
        app_module._open_capture = real_open
        app_module.page_clean_module.PageCleanInk = real_page_clean
        app_module.neural_module.NeuralSheetDetector = real_neural_detector
        cv2.imshow, cv2.waitKey = real_imshow, real_wait
        cv2.destroyAllWindows = real_destroy
    return spy, capture, errors, finished


def _panels(view):
    """(esquerda = câmera, direita = canvas) da vista composta."""
    width = view.shape[1]
    return view[:, : int(width * 0.69)], view[:, int(width * 0.72) :]


def _marked(panel):
    """Pixels do contorno verde (a folha detectada) na câmera."""
    blue, green, red = panel[:, :, 0], panel[:, :, 1], panel[:, :, 2]
    return int(np.count_nonzero((green > 180) & (red < 120) & (blue < 120)))


def _ink(panel):
    """Pixels escuros no painel do canvas."""
    return int(np.count_nonzero(np.all(panel < 60, axis=2)))


def _fake_ink_class(generations, blob_only_first=True, fail=False):
    """Classe falsa de tinta; conta gerações para saber qual folha é."""

    class _FakeInk:
        model_path = "fake.onnx"
        threshold = 0.7

        def __init__(self, _path=None):
            self.calls = 0

        def load(self):
            return True

        def ink_mask(self, _warped):
            if fail:
                raise RuntimeError("modelo explodiu de proposito")
            self.calls += 1
            mask = np.zeros(
                (
                    app_module.ink_module.config.PAPER_H,
                    app_module.ink_module.config.PAPER_W,
                ),
                dtype=np.uint8,
            )
            show = (generations[0] <= 1) if blob_only_first else True
            if show:
                mask[INK_RECT] = 255
            return mask

    return _FakeInk


def _patched_generation_counter(monkeypatch, generations, preserve_flags=None):
    original = app_module._PageCleanWorker.new_generation

    def counted(self, **kwargs):
        generations[0] += 1
        if preserve_flags is not None:
            preserve_flags.append(kwargs.get("preserve_identity", False))
        return original(self, **kwargs)

    monkeypatch.setattr(app_module._PageCleanWorker, "new_generation", counted)


@pytest.mark.lento
def test_new_sheet_does_not_inherit_the_previous_canvas(monkeypatch):
    """A tinta anterior só persiste enquanto a troca aguarda consenso.

    Depois de confirmar a folha nova, o painel não pode exibir
    o `accumulated_canvas` esquecido da sessão anterior.
    """
    monkeypatch.setenv("DRAW_ECC_REFINE", "0")
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 0.01)
    generations = [0]
    _patched_generation_counter(monkeypatch, generations)
    decision = [None]
    shown_decisions = []
    original_poll_display = app_module._PageCleanWorker.poll_display
    original_imshow = _SpyGui.imshow

    def capture_display(self, generation, after_version=0):
        with self._condition:
            result = original_poll_display(self, generation, after_version)
            if result is not None:
                events = [
                    e for e in self._identity_events if e["generation"] == generation
                ]
                decision[0] = events[-1]["decision"] if events else None
            return result

    def capture_imshow(self, title, frame):
        shown_decisions.append(decision[0])
        original_imshow(self, title, frame)

    monkeypatch.setattr(app_module._PageCleanWorker, "poll_display", capture_display)
    monkeypatch.setattr(_SpyGui, "imshow", capture_imshow)
    plan = ["sheet"] * 50 + ["blank"] * 130 + ["sheet"] * 70
    spy, capture, errors, finished = _run_headless(plan, _fake_ink_class(generations))

    assert finished and not errors, errors
    assert capture.released, "o epílogo precisa liberar a captura"
    assert spy.shown, "nenhum quadro foi exibido"

    # Divisor: o contorno verde diz quando o app voltou a ver folha.
    marked = [_marked(_panels(view)[0]) for view in spy.shown]
    assert max(marked) > 0, "a folha nunca foi marcada"

    # Primeiro quadro marcado depois do trecho sem marcação.
    gap = [i for i, count in enumerate(marked) if count == 0]
    assert gap, "não houve trecho sem folha"
    last_gap = max(gap)
    after_return = [i for i, count in enumerate(marked) if count > 0 and i > last_gap]
    assert after_return, "a folha não voltou a ser detectada"

    with_ink = [
        index for index in after_return if _ink(_panels(spy.shown[index])[1]) > 0
    ]
    assert all(shown_decisions[i] == "deferred_switch_consensus" for i in with_ink)
    committed = [i for i in after_return if shown_decisions[i] == "new_page"]
    assert committed, "a troca não completou os três vereditos limpos"
    assert not any(_ink(_panels(spy.shown[i])[1]) for i in committed)


@pytest.mark.lento
def test_same_page_recovery_preserves_canvas_through_app_loop(monkeypatch):
    """Uma queda breve e retorno da mesma página não abre canvas duplicado."""
    generations = [0]
    preserve_flags = []
    _patched_generation_counter(monkeypatch, generations, preserve_flags)
    original_poll_display = app_module._PageCleanWorker.poll_display
    canvases_by_generation = {}

    def capture_display(self, generation, after_version=0):
        result = original_poll_display(self, generation, after_version)
        if result is not None:
            version, confirmed, provisional = result
            canvases_by_generation.setdefault(generation, []).append(
                (version, confirmed.copy(), provisional.copy())
            )
        return result

    monkeypatch.setattr(app_module._PageCleanWorker, "poll_display", capture_display)
    plan = ["sheet"] * 50 + ["blank"] * 130 + ["sheet"] * 70
    spy, capture, errors, finished = _run_headless(
        plan, _fake_ink_class(generations, blob_only_first=False)
    )

    assert finished and not errors, errors
    assert capture.released
    assert generations[0] >= 2, "o replay não causou drop e recuperação"
    assert preserve_flags[:2] == [False, True], (
        "só a retomada após drop deve pedir comparação de identidade"
    )

    initial = canvases_by_generation.get(1, [])
    recovered = canvases_by_generation.get(2, [])
    assert initial and recovered, (
        "o app não publicou canvas nas duas gerações do replay"
    )
    initial_confirmed = initial[-1][1]
    recovered_confirmed = recovered[-1][1]
    assert np.count_nonzero(initial_confirmed) > 0
    assert np.array_equal(recovered_confirmed, initial_confirmed), (
        "retornar à mesma página deve reusar o canvas confirmado "
        "pixel a pixel, sem duplicá-lo ou zerá-lo"
    )


@pytest.mark.lento
def test_page_clean_failure_is_reported_instead_of_freezing(monkeypatch):
    """Exceção na trabalhadora de tinta não pode morrer calada."""
    generations = [0]
    _patched_generation_counter(monkeypatch, generations)
    plan = ["sheet"] * 40
    spy, capture, errors, finished = _run_headless(
        plan, _fake_ink_class(generations, fail=True)
    )

    assert finished and not errors, errors
    assert capture.released
    # O aviso de falha aparece na janela e o laço termina em vez de
    # congelar no último quadro.
    assert len(spy.shown) >= 1
    assert len(spy.shown) < len(plan), (
        "o laço deveria encerrar no aviso de falha, não rodar até o fim"
    )


@pytest.mark.lento
def test_epilogue_runs_when_the_loop_raises(monkeypatch):
    """Exceção no laço não pode vazar a câmera aberta."""
    plan = ["sheet"] * 30

    def boom(*_args, **_kwargs):
        raise RuntimeError("falha de propósito no meio do laço")

    monkeypatch.setattr(app_module, "_compose_view", boom)
    spy, capture, errors, finished = _run_headless(
        plan, _fake_ink_class([0], blob_only_first=False)
    )

    assert finished
    assert errors and isinstance(errors[0], RuntimeError)
    assert capture.released, "o finally precisa liberar a captura"


def test_run_drives_the_shared_loop_object(monkeypatch):
    """run() passa todo quadro sem marcadores por draw.loop.SheetInkLoop.

    É o mesmo objeto que tools/replay_reregister.py dirige com
    dependências simuladas: se o laço de run() voltasse a ter lógica
    própria de folha/geração/tinta, os replays deixariam de medi-lo.
    """
    from draw import loop as loop_module

    steps = []
    original = loop_module.SheetInkLoop.step

    def counted(self, frame, frame_capture_ms, ink_passes_total=None):
        steps.append(self.frame_number)
        return original(self, frame, frame_capture_ms, ink_passes_total)

    monkeypatch.setattr(loop_module.SheetInkLoop, "step", counted)
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    plan = ["sheet"] * 12
    _spy, _capture, errors, finished = _run_headless(plan, _fake_ink_class([0]))
    assert finished and not errors
    # Latest-wins na captura: nem todo quadro do plano chega ao laco.
    assert steps and steps == list(range(len(steps)))
