"""Contratos ONNX do detector, com sessões falsas e com ONNX de verdade.

A primeira metade usa sessões falsas em Python (sem onnxruntime, sem
webcam) e fixa o contrato do adaptador: qual canal é papel, o que o
veredito de presença significa, o que acontece com saídas
inesperadas. A segunda metade gera arquivos ONNX minúsculos e passa
pelo ONNX Runtime de verdade — é a única forma de exercitar o que a
sessão falsa não vê: o nome da entrada no arquivo, a forma declarada
e o lote simbólico. Sem o pacote `onnx`, esses testes saem como skip.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw.neural import (  # noqa: E402
    NeuralSheetDetector,
    _fit_plausible,
)


class _Output:
    def __init__(self, name):
        self.name = name


class _Input:
    def __init__(self, name="image", shape=None):
        self.name = name
        self.shape = shape or [1, 3, 8, 8]


class _Session:
    """Sessão falsa que confere o feed contra o próprio contrato.

    O nome da entrada é conferido de propósito: o adaptador antigo
    mandava o tensor sempre como "image" e uma sessão que aceitasse
    qualquer chave nunca pegaria o erro.
    """

    def __init__(self, outputs, input_name="image", input_shape=None):
        self._outputs = outputs
        self._input = _Input(input_name, input_shape)

    def get_outputs(self):
        return [_Output(name) for name in self._outputs]

    def get_inputs(self):
        return [self._input]

    def run(self, output_names, feed):
        assert list(feed) == [self._input.name], (
            f"feed com {list(feed)} numa entrada {self._input.name!r}"
        )
        assert output_names == list(self._outputs)
        return [self._outputs[name] for name in output_names]


def _logits_for_classes(classes, class_count=4):
    logits = np.full((1, class_count, *classes.shape), -4.0, dtype=np.float32)
    for class_id in range(class_count):
        logits[0, class_id][classes == class_id] = 4.0
    return logits


def test_four_class_logits_use_paper_and_ink_but_exclude_hand():
    classes = np.zeros((8, 8), dtype=np.uint8)
    classes[1:7, 1:7] = 1
    classes[2:4, 2:4] = 2
    classes[4:6, 4:6] = 3
    session = _Session({"logits": _logits_for_classes(classes)})
    detector = NeuralSheetDetector("unused.onnx")
    detector._session = session

    presence, mask = detector.predict(np.zeros((32, 48, 3), dtype=np.uint8))

    assert presence == 1.0
    assert mask.shape == (32, 48)
    assert np.all(mask[8:16, 12:24] == 0)  # mão não vira papel
    assert np.all(mask[16:24, 24:36] == 255)  # tinta integra a folha
    assert mask[0, 0] == 0


def test_four_class_logits_without_page_are_a_strong_negative():
    classes = np.zeros((8, 8), dtype=np.uint8)
    classes[1:3, 1:3] = 2
    session = _Session({"logits": _logits_for_classes(classes)})
    detector = NeuralSheetDetector("unused.onnx")
    detector._session = session

    presence, mask = detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))

    assert presence == -10.0
    assert np.count_nonzero(mask) == 0


def test_legacy_mask_and_presence_outputs_keep_their_contract():
    logits = np.full((1, 2, 8, 8), -4.0, dtype=np.float32)
    logits[0, 1, 1:7, 1:7] = 4.0
    logits[0, 0, 1:7, 1:7] = -4.0
    session = _Session(
        {
            "mask_logits": logits,
            "presence_logit": np.array([2.5], dtype=np.float32),
        }
    )
    detector = NeuralSheetDetector("unused.onnx")
    detector._session = session

    presence, mask = detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))

    assert presence == 2.5
    assert mask[4, 4] == 255
    assert mask[0, 0] == 0


def test_four_class_logits_expose_hand_mask():
    """A classe mão sai em last_hand_mask (segunda fonte de oclusão)."""
    classes = np.zeros((8, 8), dtype=np.uint8)
    classes[1:7, 1:7] = 1
    classes[2:4, 2:4] = 2
    session = _Session({"logits": _logits_for_classes(classes)})
    detector = NeuralSheetDetector("unused.onnx")
    detector._session = session

    detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))

    hand = detector.last_hand_mask
    assert hand is not None and hand.shape == (8, 8)
    assert np.all(hand[2:4, 2:4] == 255)
    assert hand.sum() / 255 == 4
    assert np.all(hand[1:7, 1:7][classes[1:7, 1:7] != 2] == 0)


def test_two_channel_contract_has_no_hand_mask():
    """O contrato legado (2 canais) não sabe o que é mão: None."""
    logits = np.full((1, 2, 8, 8), -4.0, dtype=np.float32)
    logits[0, 1, 1:7, 1:7] = 4.0
    session = _Session(
        {
            "mask_logits": logits,
            "presence_logit": np.array([2.5], dtype=np.float32),
        }
    )
    detector = NeuralSheetDetector("unused.onnx")
    detector._session = session

    detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))

    assert detector.last_hand_mask is None


def test_input_name_comes_from_the_file_not_from_a_literal():
    """A entrada pode se chamar "input": o feed tem de seguir o arquivo."""
    logits = np.zeros((1, 2, 8, 8), dtype=np.float32)
    session = _Session(
        {"mask_logits": logits, "presence_logit": np.array([1.0], np.float32)},
        input_name="input",
    )
    detector = NeuralSheetDetector("unused.onnx")
    detector._session = session

    detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))
    assert detector.contract().input_name == "input"


def test_rectangular_input_is_refused_instead_of_squeezed():
    session = _Session(
        {"logits": _logits_for_classes(np.zeros((8, 8), np.uint8))},
        input_shape=[1, 3, 384, 640],
    )
    detector = NeuralSheetDetector("unused.onnx")
    detector._session = session

    with pytest.raises(RuntimeError, match="retangular"):
        detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))


def test_dynamic_input_falls_back_to_the_measured_resolution():
    session = _Session(
        {"logits": _logits_for_classes(np.zeros((8, 8), np.uint8))},
        input_shape=["N", 3, "H", "W"],
    )
    detector = NeuralSheetDetector("unused.onnx")
    detector._session = session

    assert detector.contract().size == (384, 384)


def test_four_channel_mask_logits_with_presence_use_paper_and_ink():
    """Export híbrido: 4 canais + presença treinada (não ler como fundo x 1)."""
    classes = np.zeros((8, 8), dtype=np.uint8)
    classes[1:7, 1:7] = 1
    classes[2:4, 2:4] = 2  # mão
    session = _Session(
        {
            "mask_logits": _logits_for_classes(classes),
            "presence_logit": np.array([6.0], np.float32),
        }
    )
    detector = NeuralSheetDetector("unused.onnx")
    detector._session = session

    presence, mask = detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))

    assert presence == 6.0
    # a máscara de 8x8 sobe para 16x16 por vizinho mais próximo:
    # cada célula de classe vira 2x2 px
    assert np.all(mask[10:14, 10:14] == 255)  # papel
    assert np.all(mask[4:7, 4:7] == 0)  # região de mão fica fora


def test_mask_logits_with_three_channels_is_a_named_error():
    session = _Session(
        {
            "mask_logits": np.zeros((1, 3, 8, 8), np.float32),
            "presence_logit": np.array([1.0], np.float32),
        }
    )
    detector = NeuralSheetDetector("unused.onnx")
    detector._session = session

    with pytest.raises(ValueError, match="2 canais"):
        detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))


def test_extra_auxiliary_output_does_not_break_the_four_class_branch():
    classes = np.zeros((8, 8), dtype=np.uint8)
    classes[2:6, 2:6] = 1
    logits = _logits_for_classes(classes)
    session = _Session({"logits": logits, "aux": logits})
    detector = NeuralSheetDetector("unused.onnx")
    detector._session = session

    presence, mask = detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))

    assert presence == 1.0
    assert np.count_nonzero(mask) > 0


def test_incompatible_outputs_are_a_named_error():
    session = _Session({"mystery": np.zeros((1, 2, 8, 8), np.float32)})
    detector = NeuralSheetDetector("unused.onnx")
    detector._session = session

    with pytest.raises(ValueError, match="saídas ONNX incompatíveis"):
        detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))


# --------------------------------------------------------------- guarda
def test_plausibility_guard_refuses_non_finite_quads():
    """NaN atravessava a guarda: toda comparação com NaN é falsa."""
    nan_quad = np.array(
        [[np.nan, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        dtype=np.float32,
    )
    inf_quad = np.array(
        [[np.inf, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        dtype=np.float32,
    )
    assert _fit_plausible(nan_quad, (720, 1280)) is False
    assert _fit_plausible(inf_quad, (720, 1280)) is False


def test_plausibility_guard_still_accepts_a_normal_sheet():
    quad = np.array(
        [[300.0, 200.0], [1000.0, 180.0], [980.0, 650.0], [320.0, 660.0]],
        dtype=np.float32,
    )
    assert _fit_plausible(quad, (720, 1280)) is True


# ------------------------------------------------------- ONNX de verdade
onnx_fixtures = pytest.importorskip("onnx_fixtures")


def _detector_for(path):
    detector = NeuralSheetDetector(path)
    detector.load()
    return detector


def test_real_onnx_with_input_named_input(tmp_path):
    path = onnx_fixtures.sheet_model(
        tmp_path / "legacy.onnx", input_name="input", size=64, channels=2
    )
    detector = _detector_for(path)

    assert detector.contract().input_name == "input"
    presence, mask = detector.predict(np.full((120, 160, 3), 200, dtype=np.uint8))

    assert presence > 0
    assert mask.shape == (120, 160)


def test_real_onnx_four_class_with_auxiliary_output(tmp_path):
    path = onnx_fixtures.four_class_model(
        tmp_path / "four.onnx", size=64, extra_output=True
    )
    detector = _detector_for(path)

    assert detector.contract().outputs == ("logits", "aux")
    presence, mask = detector.predict(np.full((64, 64, 3), 230, dtype=np.uint8))
    assert presence in (1.0, -10.0)
    assert mask.shape == (64, 64)
