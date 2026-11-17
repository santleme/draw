"""Roda o detector neural (modelo piloto do repo) e grava a saida crua.

O mesmo script roda na arvore antiga e na nova; a comparacao e feita
depois, byte a byte, por outro processo.
"""
import sys, json
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "tests"))
import numpy as np, cv2
from draw.neural import NeuralSheetDetector
import selftest

out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

frames = {}
scene, _gt, _dst = selftest.build_markerless_scene()
frames["positivo"] = cv2.resize(scene, (1280, 720), interpolation=cv2.INTER_AREA)
neg = np.full((1200, 1600, 3), 60, np.uint8)
cv2.rectangle(neg, (320, 160), (1300, 980), (150, 150, 150), -1)
frames["negativo"] = cv2.resize(neg, (1280, 720), interpolation=cv2.INTER_AREA)
frames["rotacionado"] = cv2.rotate(frames["positivo"], cv2.ROTATE_90_CLOCKWISE)
frames["vazio"] = np.zeros((720, 1280, 3), np.uint8)
rng = np.random.default_rng(7)
noise = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
frames["ruido"] = noise

det = NeuralSheetDetector(str(root / "models" / "kaggle-real-pilot-v1" / "model.onnx"))
det.load()
if hasattr(det, "contract"):
    info = {"input": det.contract().input_name, "size": list(det.contract().size),
            "outputs": list(det.contract().outputs)}
else:
    sess = det._get_session()
    info = {"input": sess.get_inputs()[0].name,
            "size": [sess.get_inputs()[0].shape[2]] * 2,
            "outputs": [o.name for o in sess.get_outputs()]}
(out / "contract.json").write_text(json.dumps(info))

for name, frame in frames.items():
    corners, verdict, mask = det(frame)
    out.joinpath(f"{name}_mask.npy").write_bytes(np.asarray(mask, np.uint8).tobytes())
    out.joinpath(f"{name}_mask_shape.txt").write_text(str(mask.shape))
    out.joinpath(f"{name}_meta.json").write_text(json.dumps({
        "verdict": verdict,
        "corners": None if corners is None else np.asarray(corners).round(6).tolist(),
        "presence": det.last_presence_logit,
    }))
    print(name, verdict, None if corners is None else np.asarray(corners).round(1).tolist(), det.last_presence_logit)
