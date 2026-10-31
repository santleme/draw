"""Métricas de segmentação semântica para o modelo neural do Draw.

Classes (ordem fixa, igual ao contrato de entrega do modelo):

    0 = background
    1 = paper
    2 = hand
    3 = ink

Este módulo é puro NumPy: não depende de PyTorch, ONNX Runtime nem de
webcam, para poder rodar no CI e ser reutilizado no notebook do Kaggle
sem instalar nada além do que o Draw já usa.

Regras de rótulo que as métricas pressupõem (definidas pelo projeto):

- A máscara representa o que a câmera realmente vê. Se a mão cobre o
  traço, o pixel visível é ``hand``, nunca ``ink``.
- A máscara ``ink_full`` (desenho completo antes da oclusão) NÃO é
  classe de treinamento e NÃO entra nestas métricas. Ela serve só para
  avaliar, no futuro, o sistema de memória do Draw.

Métricas obrigatórias (nunca só accuracy global):

- IoU por classe (background, paper, hand, ink);
- Dice por classe (paper, hand, ink; background também é calculado);
- Precision e Recall por classe;
- Accuracy global, reportada apenas como contexto — um modelo que
  acerta todo o fundo e erra toda a tinta pode ter accuracy alta e
  ser inútil, por isso ela nunca é o critério de aceite.

Convenção para classe ausente nos dois lados: IoU, Dice, Precision e
Recall valem 1.0 (nada para acertar, nada errado). Classe presente no
ground truth e ausente na predição vale 0.0 em todas as métricas —
inclusive Precision, que seria indefinida (0/0) nesse caso: uma
predição vazia não ganha Precision 1.0 quando há tinta real para
achar. Isso é documentado aqui porque afeta a média: ``mean_iou`` é
a média sobre as 4 classes com essa convenção, e
``mean_iou_present`` é a média só sobre as classes presentes no
ground truth.

Métrica específica de ink com tolerância espacial
--------------------------------------------------
Traços finos são poucos pixels. Um traço previsto 1 ou 2 pixels ao
lado do traço real está visualmente correto, mas a IoU exata cai
muito, porque a interseção de linhas finas deslocadas é quase zero.
Por isso, além da IoU exata, medimos Precision e Recall de ink com
tolerância espacial (padrão: 3 px, dentro da faixa 2-3 px do projeto):

- ``tolerance_precision``: fração dos pixels de ink previstos que
  estão a no máximo ``tolerance_px`` de algum pixel de ink real.
  Implementação: dilata-se a máscara de ink real por um disco de
  raio ``tolerance_px`` e conta-se quantos pixels previstos caem
  dentro dela.
- ``tolerance_recall``: fração dos pixels de ink reais que estão a
  no máximo ``tolerance_px`` de algum pixel de ink previsto.
  Implementação simétrica: dilata-se a máscara prevista e conta-se
  quantos pixels reais caem dentro dela.

A dilatação usa um disco euclidiano (deslocamentos com
``dx**2 + dy**2 <= tolerance_px**2``), implementada em NumPy puro,
sem depender de SciPy. Ela mede proximidade espacial direta, não
forma: duas linhas paralelas a 2 px contam como acerto com
tolerância, mesmo sem sobreposição exata. Alternativas consideradas
e não usadas como métrica principal: comparação de esqueletos
(sensível a pequenas ramificações do thinning) e distance transform
(equivalente à dilatação aqui, mas exigiria SciPy ou OpenCV no
núcleo das métricas). A IoU exata continua sendo reportada ao lado,
para que a tolerância nunca esconda um deslocamento sistemático:
um modelo bom tem IoU de ink razoável E tolerância alta; um modelo
só com tolerância alta e IoU próxima de zero está deslocando tudo.

Bateria fixa de validação
--------------------------
``SCENARIOS`` tem exatamente os 15 cenários fixos do projeto. A
função ``save_predictions`` grava, por cenário, a predição
visualizada em PNG e um manifesto JSON, para que predições de
versões diferentes do modelo sejam comparáveis caso a caso, sem
depender de re-rodar o modelo antigo.
"""

import json
from pathlib import Path

import numpy as np

# Ordem fixa das classes. Não reordenar: ela é o contrato do ONNX.
CLASSES = ("background", "paper", "hand", "ink")
BACKGROUND, PAPER, HAND, INK = 0, 1, 2, 3
NUM_CLASSES = len(CLASSES)
IGNORE_INDEX = 255

# Paleta de visualização (BGR, para cv2.imwrite): fundo preto, folha
# cinza-clara, mão avermelhada, tinta azul. Cores distintas por
# classe, para a PNG ser legível sem legenda.
PALETTE_BGR = {
    BACKGROUND: (0, 0, 0),
    PAPER: (200, 200, 200),
    HAND: (80, 80, 220),
    INK: (220, 120, 40),
}

# Bateria fixa de 15 cenários de validação, na ordem do projeto.
# "id" é estável e vira nome de arquivo; "name" é legível.
SCENARIOS = (
    {"id": "01_clean_sheet", "name": "folha limpa"},
    {"id": "02_sheet_with_drawing", "name": "folha com desenho"},
    {"id": "03_heavily_drawn_sheet", "name": "folha muito desenhada"},
    {"id": "04_hand_covering_corner", "name": "mão cobrindo canto"},
    {"id": "05_hand_covering_center", "name": "mão cobrindo centro"},
    {"id": "06_hand_covering_stroke", "name": "mão cobrindo traço"},
    {"id": "07_hand_partly_outside", "name": "mão parcialmente fora da imagem"},
    {"id": "08_light_background", "name": "fundo claro"},
    {"id": "09_dark_background", "name": "fundo escuro"},
    {"id": "10_strong_shadow", "name": "sombra forte"},
    {"id": "11_sheet_partly_outside", "name": "folha parcialmente fora da imagem"},
    {"id": "12_hand_without_sheet", "name": "mão sem folha"},
    {"id": "13_white_object_no_sheet", "name": "objeto branco sem folha"},
    {"id": "14_thin_pencil_stroke", "name": "traço fino de lápis"},
    {"id": "15_thick_pen_stroke", "name": "traço grosso de caneta"},
)
SCENARIO_IDS = tuple(s["id"] for s in SCENARIOS)


def _validate_masks(pred, gt, require_supervised=True):
    """Valida predição e ground truth; pixels 255 do ground truth são ignorados."""
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred shape {pred.shape} != gt shape {gt.shape}; "
            "predição e ground truth precisam ter o mesmo formato"
        )
    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError("máscaras devem ser 2D (H, W) com valores de classe")
    if pred.size == 0 or pred.min() < 0 or pred.max() >= NUM_CLASSES:
        raise ValueError(
            f"pred contém classe fora de 0..{NUM_CLASSES - 1}; "
            f"classes válidas: {list(CLASSES)}"
        )
    valid_gt = (gt >= 0) & (gt < NUM_CLASSES)
    ignored_gt = gt == IGNORE_INDEX
    if np.any(~(valid_gt | ignored_gt)):
        bad = np.unique(gt[~(valid_gt | ignored_gt)]).tolist()
        raise ValueError(
            f"gt contém valores inválidos: {bad}; permitidos: "
            f"0..{NUM_CLASSES - 1} e {IGNORE_INDEX}=ignorado"
        )
    if require_supervised and not np.any(valid_gt):
        raise ValueError("gt sem pixels supervisionados; todas as classes são 255")
    return pred, gt


def _tp_fp_fn(pred, gt):
    """Conta TP/FP/FN por classe. Retorna arrays de tamanho NUM_CLASSES."""
    tp = np.zeros(NUM_CLASSES, dtype=np.int64)
    fp = np.zeros(NUM_CLASSES, dtype=np.int64)
    fn = np.zeros(NUM_CLASSES, dtype=np.int64)
    valid = gt != IGNORE_INDEX
    for cls in range(NUM_CLASSES):
        p = (pred == cls) & valid
        g = gt == cls
        tp[cls] = int(np.count_nonzero(p & g))
        fp[cls] = int(np.count_nonzero(p & ~g))
        fn[cls] = int(np.count_nonzero(~p & g))
    return tp, fp, fn


def _safe_div(num, den):
    """Divisão com convenção de classe ausente: 0/0 vale 1.0."""
    return 1.0 if den == 0 else float(num) / float(den)


def iou_per_class(pred, gt):
    """IoU por classe. Retorna dict {nome_da_classe: valor}."""
    pred, gt = _validate_masks(pred, gt)
    tp, fp, fn = _tp_fp_fn(pred, gt)
    return {
        CLASSES[cls]: _safe_div(tp[cls], tp[cls] + fp[cls] + fn[cls])
        for cls in range(NUM_CLASSES)
    }


def dice_per_class(pred, gt):
    """Dice (F1) por classe. Retorna dict {nome_da_classe: valor}."""
    pred, gt = _validate_masks(pred, gt)
    tp, fp, fn = _tp_fp_fn(pred, gt)
    return {
        CLASSES[cls]: _safe_div(2 * tp[cls], 2 * tp[cls] + fp[cls] + fn[cls])
        for cls in range(NUM_CLASSES)
    }


def precision_recall_per_class(pred, gt):
    """Precision e Recall por classe.

    Retorna {"precision": {classe: valor}, "recall": {classe: valor}}.
    """
    pred, gt = _validate_masks(pred, gt)
    tp, fp, fn = _tp_fp_fn(pred, gt)
    precision, recall = {}, {}
    for cls in range(NUM_CLASSES):
        p_den, g_den = tp[cls] + fp[cls], tp[cls] + fn[cls]
        # 0/0 só vale 1.0 quando a classe está ausente dos dois lados;
        # predição vazia com ground truth presente vale 0.0 (ver módulo).
        precision[CLASSES[cls]] = (
            1.0 if (p_den == 0 and g_den == 0)
            else (float(tp[cls]) / p_den if p_den > 0 else 0.0)
        )
        recall[CLASSES[cls]] = (
            1.0 if (g_den == 0 and p_den == 0)
            else (float(tp[cls]) / g_den if g_den > 0 else 0.0)
        )
    return {"precision": precision, "recall": recall}


def confusion_matrix(pred, gt):
    """Matriz real × predito; pixels 255 do ground truth não entram."""
    pred, gt = _validate_masks(pred, gt, require_supervised=False)
    valid = gt != IGNORE_INDEX
    encoded = gt[valid].astype(np.int64) * NUM_CLASSES + pred[valid].astype(np.int64)
    return np.bincount(encoded, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def evaluate_confusion_matrix(matrix, ignored_px=0):
    """Reduz uma matriz real × predito às métricas agregadas do projeto."""
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (NUM_CLASSES, NUM_CLASSES) or np.any(matrix < 0):
        raise ValueError(f"matriz de confusão inválida: shape={matrix.shape}")
    supervised_px = int(matrix.sum())
    if supervised_px == 0:
        raise ValueError("sem pixels supervisionados para calcular métricas")
    tp = np.diag(matrix)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    iou = {}
    dice = {}
    precision = {}
    recall = {}
    support_px = {}
    for cls, name in enumerate(CLASSES):
        iou[name] = _safe_div(tp[cls], tp[cls] + fp[cls] + fn[cls])
        dice[name] = _safe_div(2 * tp[cls], 2 * tp[cls] + fp[cls] + fn[cls])
        precision[name] = 1.0 if (tp[cls] + fp[cls] == 0 and tp[cls] + fn[cls] == 0) else (
            float(tp[cls]) / (tp[cls] + fp[cls]) if tp[cls] + fp[cls] else 0.0
        )
        recall[name] = 1.0 if (tp[cls] + fn[cls] == 0 and tp[cls] + fp[cls] == 0) else (
            float(tp[cls]) / (tp[cls] + fn[cls]) if tp[cls] + fn[cls] else 0.0
        )
        support_px[name] = int(matrix[cls].sum())
    present = [name for cls, name in enumerate(CLASSES) if matrix[cls].sum() > 0]
    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "mean_iou": float(np.mean([iou[name] for name in CLASSES])),
        "mean_iou_present": float(np.mean([iou[name] for name in present])),
        "mean_dice": float(np.mean([dice[name] for name in CLASSES])),
        "accuracy": float(tp.sum() / supervised_px),
        "support_px": support_px,
        "supervised_px": supervised_px,
        "ignored_px": int(ignored_px),
    }


def evaluate(pred, gt):
    """Avaliação completa; pixels 255 do ground truth são ignorados.

    Retorna dict com ``iou``, ``dice``, ``precision``, ``recall``
    (todos por classe), ``mean_iou``, ``mean_iou_present`` (média só
    sobre classes presentes no ground truth), ``mean_dice`` e
    ``accuracy`` (contexto apenas, nunca critério único).
    """
    pred, gt = _validate_masks(pred, gt)
    return evaluate_confusion_matrix(
        confusion_matrix(pred, gt), ignored_px=int(np.count_nonzero(gt == IGNORE_INDEX))
    )


def _dilate_disk(mask, radius):
    """Dilata uma máscara binária por um disco euclidiano de raio dado.

    Implementação NumPy pura (sem SciPy): para cada deslocamento
    (dx, dy) com dx**2 + dy**2 <= radius**2, desloca a máscara e
    acumula com OR. Raio 0 devolve a máscara original.
    """
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask.copy()
    dilated = np.zeros_like(mask)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            shifted = np.zeros_like(mask)
            y_src = slice(max(0, -dy), mask.shape[0] - max(0, dy))
            y_dst = slice(max(0, dy), mask.shape[0] - max(0, -dy))
            x_src = slice(max(0, -dx), mask.shape[1] - max(0, dx))
            x_dst = slice(max(0, dx), mask.shape[1] - max(0, -dx))
            shifted[y_dst, x_dst] = mask[y_src, x_src]
            dilated |= shifted
    return dilated


def ink_tolerance_metrics(pred, gt, tolerance_px=3):
    """Precision/Recall de ink com tolerância espacial.

    ``tolerance_px`` em pixels (padrão 3, dentro da faixa 2-3 px do
    projeto). Um pixel de ink previsto conta como acerto se estiver
    a no máximo ``tolerance_px`` de um pixel de ink real, e vice-versa
    para o recall. Ver a metodologia no docstring do módulo.

    Retorna dict com ``tolerance_px``, ``precision``, ``recall`` e
    ``f1``, mais as contagens ``pred_ink``, ``gt_ink`` e
    ``matched`` (pixels previstos dentro da tolerância).
    """
    pred, gt = _validate_masks(pred, gt)
    valid = gt != IGNORE_INDEX
    pred_ink = (pred == INK) & valid
    gt_ink = (gt == INK) & valid
    n_pred = int(np.count_nonzero(pred_ink))
    n_gt = int(np.count_nonzero(gt_ink))
    if n_pred == 0 and n_gt == 0:
        return {
            "tolerance_px": tolerance_px,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "pred_ink": 0,
            "gt_ink": 0,
            "matched": 0,
            "matched_pred": 0,
            "matched_gt": 0,
        }
    matched_pred = int(np.count_nonzero(pred_ink & _dilate_disk(gt_ink, tolerance_px) & valid))
    matched_gt = int(np.count_nonzero(gt_ink & _dilate_disk(pred_ink, tolerance_px) & valid))
    # Mesma convenção do módulo: 0/0 só vale 1.0 com ink ausente dos
    # dois lados; predição vazia com ink real vale 0.0.
    precision = 0.0 if (n_pred == 0 and n_gt > 0) else _safe_div(matched_pred, n_pred)
    recall = 0.0 if (n_gt == 0 and n_pred > 0) else _safe_div(matched_gt, n_gt)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {
        "tolerance_px": tolerance_px,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_ink": n_pred,
        "gt_ink": n_gt,
        "matched": matched_pred,
        "matched_pred": matched_pred,
        "matched_gt": matched_gt,
    }


def visualize_mask(mask):
    """Converte uma máscara de classes em imagem BGR (H, W, 3) colorida."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("máscara para visualizar deve ser 2D (H, W)")
    if mask.min() < 0 or mask.max() >= NUM_CLASSES:
        raise ValueError(f"classe fora de 0..{NUM_CLASSES - 1}")
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls in range(NUM_CLASSES):
        out[mask == cls] = PALETTE_BGR[cls]
    return out


def evaluate_scenarios(predictions, ground_truths):
    """Avalia predições da bateria fixa, por cenário e agregada.

    ``predictions`` e ``ground_truths`` são dicts {scenario_id: máscara}.
    Cenários sem ground truth são ignorados no agregado e reportados
    com ``metrics: None``. Retorna {"per_scenario": {...}, "aggregate":
    {"mean_iou": ..., "mean_ink_tolerance_f1": ...}}.
    """
    per_scenario = {}
    ious, ink_f1s = [], []
    for scenario in SCENARIOS:
        sid = scenario["id"]
        if sid not in predictions:
            continue
        entry = {"name": scenario["name"]}
        if sid in ground_truths:
            metrics = evaluate(predictions[sid], ground_truths[sid])
            tol = ink_tolerance_metrics(predictions[sid], ground_truths[sid])
            entry["metrics"] = metrics
            entry["ink_tolerance"] = tol
            ious.append(metrics["mean_iou_present"])
            ink_f1s.append(tol["f1"])
        else:
            entry["metrics"] = None
            entry["ink_tolerance"] = None
        per_scenario[sid] = entry
    return {
        "per_scenario": per_scenario,
        "aggregate": {
            "mean_iou": float(np.mean(ious)) if ious else None,
            "mean_ink_tolerance_f1": float(np.mean(ink_f1s)) if ink_f1s else None,
            "n_evaluated": len(ious),
        },
    }


def save_predictions(output_dir, predictions, model_name=""):
    """Salva predições da bateria fixa para comparação entre versões.

    Para cada cenário presente em ``predictions``, grava
    ``<scenario_id>_pred.png`` (máscara visualizada com a paleta do
    módulo) e um manifesto ``predictions.json`` com modelo, cenários,
    formatos e, se o manifesto for gerado junto com ground truths, as
    métricas — neste módulo, o manifesto registra o que foi salvo;
    as métricas vivem em ``evaluate_scenarios`` e devem ser salvas
    pelo chamador junto ao manifesto quando houver ground truth.

    Retorna o caminho do manifesto. Exige OpenCV apenas para gravar
    PNG (dependência que o Draw já tem); sem cv2, levanta
    RuntimeError com mensagem clara, em vez de gravar arquivo mudo.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - ambiente sem cv2
        raise RuntimeError(
            "save_predictions precisa de OpenCV para gravar PNG; "
            "instale opencv-contrib-python (já é dependência do Draw)"
        ) from exc
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for scenario in SCENARIOS:
        sid = scenario["id"]
        if sid not in predictions:
            continue
        mask = np.asarray(predictions[sid])
        if mask.ndim != 2:
            raise ValueError(f"predição do cenário {sid} deve ser 2D (H, W)")
        filename = f"{sid}_pred.png"
        ok = cv2.imwrite(str(out_dir / filename), visualize_mask(mask))
        if not ok:
            raise RuntimeError(f"falha ao gravar {out_dir / filename}")
        files[sid] = {"name": scenario["name"], "file": filename, "shape": list(mask.shape)}
    manifest = {"model_name": model_name, "scenarios": files}
    manifest_path = out_dir / "predictions.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path
