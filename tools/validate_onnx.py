"""Validador ONNX para o modelo de segmentação do Draw.

Uso pretendido (pelo agente de treinamento, no Kaggle ou local):

    1. rodar a versão original do modelo (PyTorch);
    2. rodar a versão ONNX na mesma entrada;
    3. comparar as saídas;
    4. registrar se a conversão degradou a segmentação.

Este módulo NÃO instala nem exige PyTorch/ONNX Runtime na importação:
os imports são preguiçosos, dentro das funções que os usam. Isso é
deliberado — o venv do Draw (e o CI) tem só NumPy + OpenCV, e a
comparação em si é NumPy puro. Quando as libs faltam, ``validate_onnx``
devolve um relatório com ``status: "skipped"`` e o motivo, em vez de
fingir que validou. Nunca se deve entregar um ONNX como "validado"
com base num relatório skipped.

Critério de degradação (parâmetros, não constantes escondidas):

- ``max_diff_tol`` (padrão 1e-3): diferença absoluta máxima tolerada
  entre os logits/probabilidades PyTorch e ONNX;
- ``min_argmax_iou`` (padrão 0.99): IoU mínima, por classe, entre as
  máscaras argmax dos dois modelos.

``degraded`` é True se qualquer um dos dois critérios falhar.
Os limites são apertados de propósito: conversão ONNX bem feita, em
float32, costuma diferir em ~1e-5 a 1e-4. Se o relatório disser
``degraded: True``, a entrega do ONNX deve ser bloqueada até
investigar (op, precisão, normalização ou shape de entrada).

Formato de saída esperado dos modelos: tensor/array
``(N, C, H, W)`` ou ``(C, H, W)`` com C = 4, na ordem
background, paper, hand, ink (o mesmo contrato de
``tools/seg_metrics.py``). A máscara argmax é o argmax sobre o eixo
de classe. ``compare_model_outputs`` aceita também arrays 2D já em
forma de máscara? Não — para evitar comparar coisa errada em
silêncio, ela exige logits/probabilidades com eixo de classe; para
comparar máscaras prontas, use ``tools.seg_metrics.evaluate``.
"""

import numpy as np

from tools.seg_metrics import CLASSES, NUM_CLASSES, iou_per_class

DEFAULT_MAX_DIFF_TOL = 1e-3
DEFAULT_MIN_ARGMAX_IOU = 0.99


def _as_logits_array(output):
    """Converte saída de modelo (torch, numpy ou lista) para ndarray float."""
    if hasattr(output, "detach"):  # tensor PyTorch
        output = output.detach().cpu().numpy()
    arr = np.asarray(output, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 4:
        raise ValueError(
            f"saída do modelo deve ser (C, H, W) ou (N, C, H, W); veio {arr.shape}"
        )
    if arr.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"saída deve ter {NUM_CLASSES} classes (ordem {list(CLASSES)}); "
            f"veio {arr.shape[1]}"
        )
    return arr


def compare_model_outputs(torch_output, onnx_output):
    """Compara saídas PyTorch e ONNX da mesma entrada.

    Retorna dict com ``max_abs_diff``, ``mean_abs_diff``,
    ``argmax_agreement`` (fração de pixels com a mesma classe argmax),
    ``argmax_iou`` (IoU por classe entre as máscaras argmax) e
    ``min_argmax_iou``. Não decide degradação sozinho — isso é papel
    de ``validate_onnx``, com os limites documentados no módulo.
    """
    torch_arr = _as_logits_array(torch_output)
    onnx_arr = _as_logits_array(onnx_output)
    if torch_arr.shape != onnx_arr.shape:
        raise ValueError(
            f"shapes diferentes: PyTorch {torch_arr.shape} != ONNX {onnx_arr.shape}"
        )
    diff = np.abs(torch_arr - onnx_arr)
    torch_mask = np.argmax(torch_arr, axis=1)
    onnx_mask = np.argmax(onnx_arr, axis=1)
    # IoU argmax por imagem, agregada por classe sobre o batch.
    iou_acc = {c: [] for c in CLASSES}
    for i in range(torch_mask.shape[0]):
        per_class = iou_per_class(onnx_mask[i], torch_mask[i])
        for c in CLASSES:
            iou_acc[c].append(per_class[c])
    argmax_iou = {c: float(np.mean(iou_acc[c])) for c in CLASSES}
    return {
        "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(diff)) if diff.size else 0.0,
        "argmax_agreement": float(np.mean(torch_mask == onnx_mask)),
        "argmax_iou": argmax_iou,
        "min_argmax_iou": float(min(argmax_iou.values())),
    }


def run_torch_model(model, dummy_input):
    """Roda um modelo PyTorch em eval/no_grad e devolve ndarray.

    Levanta RuntimeError com mensagem clara se torch não estiver
    instalado, em vez de falhar com ImportError cru no meio do
    relatório de validação.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch não instalado neste ambiente; instale torch para "
            "rodar o modelo original na validação ONNX"
        ) from exc
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.no_grad():
            output = model(dummy_input)
    finally:
        if was_training:
            model.train()
    if isinstance(output, (tuple, list)):
        output = output[0]
    return _as_logits_array(output)


def run_onnx_model(onnx_path, dummy_input):
    """Roda um modelo ONNX via ONNX Runtime e devolve ndarray.

    ``dummy_input`` é um ndarray NumPy; o nome do input é lido do
    próprio modelo (primeiro input da sessão). Levanta RuntimeError
    claro se onnxruntime não estiver instalado.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime não instalado neste ambiente; instale "
            "onnxruntime para rodar o modelo ONNX na validação"
        ) from exc
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: np.asarray(dummy_input)})
    return _as_logits_array(outputs[0])


def validate_onnx(torch_model, onnx_path, dummy_input,
                  max_diff_tol=DEFAULT_MAX_DIFF_TOL,
                  min_argmax_iou=DEFAULT_MIN_ARGMAX_IOU,
                  expected_shape=None):
    """Valida um ONNX contra o modelo PyTorch original.

    Retorna um relatório dict, sempre com ``status``:

    - ``"ok"``: comparação feita e dentro dos limites;
    - ``"degraded"``: comparação feita e fora dos limites — a
      entrega do ONNX deve ser bloqueada;
    - ``"skipped"``: torch ou onnxruntime ausente (motivo em
      ``reason``). Skipped NÃO é validação: não entregar o ONNX
      como validado com base nele.

    ``expected_shape`` pode fixar o shape de saída esperado; use ``None``
    em dimensões dinâmicas. Uma divergência de shape também retorna
    ``status: "degraded"``. Os campos de comparação só existem quando a
    execução PyTorch e ONNX foi possível.
    """
    try:
        torch_out = run_torch_model(torch_model, dummy_input)
    except RuntimeError as exc:
        return {"status": "skipped", "reason": str(exc), "degraded": None}
    try:
        onnx_out = run_onnx_model(onnx_path, dummy_input)
    except RuntimeError as exc:
        return {"status": "skipped", "reason": str(exc), "degraded": None}
    actual_shape = tuple(torch_out.shape)
    onnx_shape = tuple(onnx_out.shape)
    def matches_expected(shape):
        return expected_shape is None or (
            len(shape) == len(expected_shape)
            and all(expected is None or actual == expected
                    for actual, expected in zip(shape, expected_shape))
        )
    shape_matches = (
        actual_shape == onnx_shape
        and matches_expected(actual_shape)
        and matches_expected(onnx_shape)
    )
    if not shape_matches:
        return {
            "status": "degraded",
            "degraded": True,
            "comparison": None,
            "expected_shape": list(expected_shape) if expected_shape is not None else None,
            "actual_shape": list(actual_shape),
            "onnx_shape": list(onnx_shape),
            "shape_matches": False,
            "thresholds": {
                "max_diff_tol": max_diff_tol,
                "min_argmax_iou": min_argmax_iou,
            },
        }
    comparison = compare_model_outputs(torch_out, onnx_out)
    degraded = bool(
        comparison["max_abs_diff"] > max_diff_tol
        or comparison["min_argmax_iou"] < min_argmax_iou
        or not shape_matches
    )
    return {
        "status": "degraded" if degraded else "ok",
        "degraded": degraded,
        "comparison": comparison,
        "expected_shape": list(expected_shape) if expected_shape is not None else None,
        "actual_shape": list(actual_shape),
        "onnx_shape": list(onnx_shape),
        "shape_matches": shape_matches,
        "thresholds": {
            "max_diff_tol": max_diff_tol,
            "min_argmax_iou": min_argmax_iou,
        },
    }


def main(argv=None):
    """CLI mínima: valida um ONNX contra um checkpoint PyTorch.

    Como carregar um checkpoint depende da arquitetura do modelo
    (que vive no agente de treinamento, não aqui), esta CLI aceita
    um arquivo .npy com logits de referência? Não — ela não tenta
    adivinhar a arquitetura. Ela existe para documentar o contrato
    de uso programático (``validate_onnx``) e falha explicitamente
    pedindo o uso via Python, em vez de aceitar um checkpoint e
    produzir um relatório vazio. O uso real é:

        from tools.validate_onnx import validate_onnx
        report = validate_onnx(model, "draw_segmentation_v1.onnx", x)
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Validador ONNX do Draw (uso programático; ver docstring)."
    )
    parser.add_argument("--onnx", help="caminho do modelo ONNX a validar")
    args = parser.parse_args(argv)
    print(
        "Uso programático: from tools.validate_onnx import validate_onnx; "
        "validate_onnx(modelo_torch, caminho_onnx, entrada). "
        "Esta CLI não carrega checkpoints sozinha, para não adivinhar "
        "a arquitetura do modelo.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
