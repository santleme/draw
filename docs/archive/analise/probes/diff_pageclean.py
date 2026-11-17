"""Diferencial do page_clean: antigo (1 bloco/run) x novo (lote)."""
import sys, json, tempfile, os
from pathlib import Path
tree = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
sys.path.insert(0, str(tree)); sys.path.insert(0, "/home/user/draw/tests")
import numpy as np
import onnx_fixtures as fx
from draw import page_clean

tmp = Path(tempfile.mkdtemp())
path = fx.page_clean_model(tmp / "pc.onnx", patch=256)
ink = page_clean.PageCleanInk(path, threshold=0.55)
ink.load()

rng = np.random.default_rng(5)
page = rng.integers(90, 250, (1200, 850, 3), dtype=np.uint8)
page[400:520, 150:400] = 25
mask = ink.ink_mask(page)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(mask.tobytes())
print(sys.argv[1], "->", mask.shape, "tinta", int((mask > 0).sum()))
