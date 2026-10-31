"""Contratos ONNX do detector, sem depender de onnxruntime ou webcam."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draw.neural import NeuralSheetDetector  # noqa: E402


class _Output:
    def __init__(self, name):
        self.name = name


class _Input:
    shape = [1, 3, 8, 8]


class _Session:
    def __init__(self, outputs):
        self._outputs = outputs

    def get_outputs(self):
        return [_Output(name) for name in self._outputs]

    def get_inputs(self):
        return [_Input()]

    def run(self, output_names, _feed):
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
    session = _Session({
        "mask_logits": logits,
        "presence_logit": np.array([2.5], dtype=np.float32),
    })
    detector = NeuralSheetDetector("unused.onnx")
    detector._session = session

    presence, mask = detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))

    assert presence == 2.5
    assert mask[4, 4] == 255
    assert mask[0, 0] == 0
