import sys
from pathlib import Path

# Rodam da raiz do repositório: este arquivo mora em
# docs/analise/probes/, então a raiz é parents[3].
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tests"))
"""Custo do mosaico: 20 passadas de 256 vs 1 passada em lote (modelo ONNX fake)."""
import sys, time, statistics, tempfile, os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

def build_model(path):
    def w(name, *shape):
        rng = np.random.default_rng(0)
        return numpy_helper.from_array(
            (rng.standard_normal(shape) * 0.05).astype(np.float32), name)
    nodes, inits = [], []
    # encoder 256 -> 128 -> 64
    inits += [w("c1", 16, 3, 3, 3), w("b1", 16)]
    nodes.append(helper.make_node("Conv", ["image", "c1", "b1"], ["h1"], pads=[1,1,1,1]))
    nodes.append(helper.make_node("Relu", ["h1"], ["r1"]))
    nodes.append(helper.make_node("MaxPool", ["r1"], ["p1"], kernel_shape=[2,2], strides=[2,2]))
    inits += [w("c2", 32, 16, 3, 3), w("b2", 32)]
    nodes.append(helper.make_node("Conv", ["p1", "c2", "b2"], ["h2"], pads=[1,1,1,1]))
    nodes.append(helper.make_node("Relu", ["h2"], ["r2"]))
    nodes.append(helper.make_node("MaxPool", ["r2"], ["p2"], kernel_shape=[2,2], strides=[2,2]))
    inits += [w("c3", 64, 32, 3, 3), w("b3", 64)]
    nodes.append(helper.make_node("Conv", ["p2", "c3", "b3"], ["h3"], pads=[1,1,1,1]))
    nodes.append(helper.make_node("Relu", ["h3"], ["r3"]))
    # decoder (simples upsample + conv)
    nodes.append(helper.make_node("Resize", ["r3", "", "scales", ""], ["u1"], mode="nearest"))
    inits.append(numpy_helper.from_array(np.array([1,1,2,2], np.float32), "scales"))
    inits += [w("c4", 32, 64, 3, 3), w("b4", 32)]
    nodes.append(helper.make_node("Conv", ["u1", "c4", "b4"], ["h4"], pads=[1,1,1,1]))
    nodes.append(helper.make_node("Relu", ["h4"], ["r4"]))
    nodes.append(helper.make_node("Resize", ["r4", "", "scales", ""], ["u2"], mode="nearest"))
    inits += [w("c5", 1, 32, 3, 3), w("b5", 1)]
    nodes.append(helper.make_node("Conv", ["u2", "c5", "b5"], ["logits"], pads=[1,1,1,1]))
    graph = helper.make_graph(nodes, "fake_unet",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["N", 3, 256, 256])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", 1, 256, 256])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path

path = os.path.join(tempfile.mkdtemp(), "fake_unet.onnx")
build_model(path)
print("modelo fake:", os.path.getsize(path), "bytes")

opt = ort.SessionOptions(); opt.intra_op_num_threads = 4
sess = ort.InferenceSession(path, sess_options=opt, providers=["CPUExecutionProvider"])
print("entrada:", sess.get_inputs()[0].name, sess.get_inputs()[0].shape)

tiles = np.random.rand(20, 3, 256, 256).astype(np.float32)
def seq():
    return [sess.run(None, {"image": tiles[i:i+1]})[0] for i in range(20)]
def batched():
    return sess.run(None, {"image": tiles})[0]

for name, fn, n in (("20 x run() 1 tile", seq, 5), ("1 x run() lote 20", batched, 5)):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1000)
    print(f"{name:22s}: mediana {statistics.median(ts):7.2f} ms")

a = np.concatenate(seq(), axis=0); b = batched()
print("saidas iguais:", np.array_equal(a, b),
      "max diff", float(np.abs(a-b).max()))
# Saida de referencia (lote de 20) para conferir cada tamanho de lote.
b_out = b

print("-" * 50)
for b in (2, 4, 5, 8, 10, 20):
    def run_b():
        outs = []
        for i in range(0, 20, b):
            outs.append(sess.run(None, {"image": tiles[i:i+b]})[0])
        return np.concatenate(outs, axis=0)
    run_b()
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); run_b(); ts.append((time.perf_counter()-t0)*1000)
    print(f"lote {b:2d}: mediana {statistics.median(ts):7.2f} ms  (identico: {np.array_equal(run_b(), b_out)})")
