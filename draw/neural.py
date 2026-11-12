"""Detector neural da folha: segmentação por ONNX atrás do contrato
de draw.paper.detect_sheet_with_confidence.

O modelo da fase 2 (SegFormer-B0 exportado para ONNX) recebe o frame
esmagado para a resolução de entrada, normalizado com média e desvio
do ImageNet, e devolve logits de máscara em dois canais e um logit
de presença. A resolução segue o arquivo: um ONNX de formato fixo
impõe o tamanho dele (512x512 no export original); um ONNX dinâmico
(eixos H/W simbólicos) roda a 384x384, o ponto medido no Ryzen 7
7700X do Lucas em 2026-10-29 — session.run de 122 para 58 ms, IoU de
máscara de 0,984 contra o 512, nenhum veredito de presença trocado
nos frames dele fora as três frames da caixa branca, que a 384
deixam de ser falso positivo. A sessão roda com 4 threads intra-op,
também medido: o padrão do ONNX Runtime marcava 248 ms no mesmo
tensor, 4 threads marcam 122 ms e 16 threads pioram para 210 ms
(draw-lab/perf-ryzen-2026-10-29/REPORT.md). A
presença decide se há papel. A máscara decide onde: o quadrilátero da
folha sai do maior contorno da máscara limiarizada, com a mesma escada
de epsilon do detector clássico, um reajuste de cada aresta como reta
sobre os pontos do contorno e refino subpixel nos cantos dentro do
quadro.

Quando a escada de epsilon não fecha sobre o contorno, o
quadrilátero é o retângulo de área mínima do contorno, sem reajuste
nem refino: medido nos frames reais em que a escada falha, os dois
passos degradam essa reserva (os números ficam em _min_area_quad). A
reserva só existe onde antes o retorno era nenhum quadrilátero;
onde a escada fecha, o caminho é o mesmo de sempre.

O reajuste de arestas existe para o caso comum de uso: a página
cortada pela borda do quadro. Ali o vértice da aproximação mora sobre
a borda e não é o canto da página. A reta ajustada sobre os pontos
visíveis da aresta projeta o canto para fora do quadro, que é onde a
página continua. Pontos do contorno colados na borda do frame ficam
fora do ajuste: ali quem cortou a máscara foi o quadro, não a página.
Cada canto reajustado só é aceito se ficar a no máximo
_FIT_MAX_DISPLACE_PX do vértice da aproximação; sem essa trava, uma
aresta curta ou contaminada projeta cantos para longe demais.

Aresta cuja cobertura de pontos é curta (a mão cobre a maior parte
dela) não ganha reta ajustada: dentro de um quadro só, os pontos
visíveis dela são a borda do braço, não a aresta, e a reta do
fragmento projeta os cantos a centenas de pixels (o teto acima veta
a correção e o canto congela no vértice da aproximação, que nesses
quadros mora sobre a borda do quadro). Quem resolve é a memória de
arestas do detector: a aresta coberta herda a última reta confiável
dela mesma, e só sem herança vale a reta travada na direção do
segmento. Os cantos vizinhos de uma aresta travada ou herdada não
passam pelo teto: a projeção já nasce limitada.

A sessão ONNX é preguiçosa e o onnxruntime só é importado no primeiro
uso. Importar este módulo nunca quebra em ambiente sem a dependência;
usar o detector sem ela levanta um erro explícito.

O arquivo do modelo fica fora do repositório (binário de 15 MB). O
caminho chega por argumento ou pela variável de ambiente
DRAW_NEURAL_MODEL.
"""

import os
import time
from pathlib import Path
from typing import NamedTuple, Tuple

import cv2
import numpy as np

from . import paper as _paper
from .manifest import verify_model

_INPUT_SIZE = 384
_ORT_INTRA_THREADS = 3
_ENV_THREADS_VAR = "DRAW_ORT_THREADS"
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_NORM_SCALE = (1.0 / (255.0 * _STD)).astype(np.float32)
_NORM_BIAS = (-_MEAN / _STD).astype(np.float32)


def _frame_to_tensor(frame, size):
    """Pré-processo ÚNICO frame→tensor do detector de folha.

    Contrato: resize INTER_LINEAR para (largura, altura) de `size`
    (par (altura, largura) do contrato do modelo), BGR→RGB, /255 e
    normalização ImageNet (média/desvio acima). Treino
    (training/train.py) e produção leem o MESMO contrato; qualquer
    ferramenta externa (bancada de julgamento, sondas) deve importar
    esta função em vez de reimplementar o pré-processo — a divergência
    silenciosa BGR-sem-normalização da bancada (2026-11-08) é o motivo
    desta extração. Reduzir ANTES de converter: o resize é linear por
    canal e a conversão é uma permutação exata de canais; as operações
    comutam e a conversão roda sobre a imagem já reduzida.
    """
    input_h, input_w = size
    small = cv2.resize(
        frame,
        (input_w, input_h),
        interpolation=cv2.INTER_LINEAR,
    )
    resized = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = (resized * _NORM_SCALE + _NORM_BIAS).transpose(2, 0, 1)
    return tensor[None].astype(np.float32)


_MASK_THRESHOLD = 0.5
_LOGIT_MARGIN = float(np.log(_MASK_THRESHOLD / (1.0 - _MASK_THRESHOLD)))
_MIN_CONTOUR_AREA_RATIO = 0.01
_FIT_MAX_DISPLACE_PX = 100.0
_FIT_MIN_EDGE_COVERAGE = 0.5
_FIT_MAX_INLIER_RESIDUAL_PX = 6.0
_FIT_MIN_INLIER_RATIO = 0.50
_FIT_MAX_INLIER_RMS_PX = 4.0
_MEMORY_MIN_MASK_IOU = 0.7
_MEMORY_MASK_STRIDE = 4
_MEMORY_STATIONARY_ANGLE_DEG = 3.0
_MEMORY_STATIONARY_SHIFT_PX = 8.0
_MEMORY_SINGLE_OUTLIER_ANGLE_DEG = 6.0
_MEMORY_SINGLE_OUTLIER_SHIFT_PX = 18.0
_ENV_MODEL_VAR = "DRAW_NEURAL_MODEL"
_PAPER_CLASS = 1
_INK_CLASS = 3
_SEGMENTATION_PRESENCE_MIN_RATIO = 0.01
_SEGMENTATION_ABSENT_LOGIT = -10.0
_SEGMENTATION_PRESENT_LOGIT = 1.0
_TIMING_PHASES = ("preprocess", "session_run", "postprocess")
_TIMING_LOG_EVERY = 100

# Portão do caminho clássico (medido em 2026-10-30 no laboratório,
# draw-lab/fp-gate-2026-10-30/): com um modelo neural carregado, um
# logit de presença até este valor é um "não há papel" forte o bastante
# para o clássico não poder declarar folha por cima dele (o incidente
# da caixa branca: a v2 media logit ≈ −10,9 na caixa e o banner vinha
# do clássico). Nos frames reais medidos (44 positivos por modelo: os
# 43 do frames-dataset e o frame_real), o menor logit de positivo foi
# +7,38 (v3 dinâmico); nos 6 negativos do dataset, o maior logit foi
# −6,98 (v3). O limiar fica 10,4 abaixo do positivo mais fraco e 4,0
# acima do negativo mais forte: veta o clássico em 18/18 corridas
# (3 modelos × 6 negativos) sobre os negativos do dataset e não custa
# nenhum positivo medido. Ele NÃO cobre a caixa branca na v3 a 512
# (logit +0,41 a +0,98: a própria neural afirma papel, fraco, e o
# quadrilátero fecha sobre ~3,5% de máscara) nem os logits −0,006 a
# −0,382 da v3 dinâmica na caixa: nenhum dos dois é "fortemente
# negativo", e baixar o limiar para perto de zero desligaria a reserva
# clássica justamente no cenário de generalização, a folha real que a
# neural subdispara num setup novo.
PRESENCE_VETO_LOGIT = -3.0

# Portão de PRIMEIRA confirmação anti-fantasma (frente 85, medido em
# 2026-11-12 no laboratório; implementação na frente 88). Na sessão
# viva de 2026-11-11, a máscara deste detector fechou um blob único e
# liso sobre a mancha desgastada do mousepad (presença piscando em
# +0,07…+0,51) e o rastreador confirmou a "folha" com UM quad só:
# 2 gerações nasceram sobre a mesa vazia. Toda folha real daquela
# sessão entrou com logit ≥ +3,81. O portão — ativo só com a variável
# DRAW_FIRST_CONFIRM_GATE=1 e só enquanto o rastreador NÃO tem folha
# (a lógica mora em draw.loop) — exige, para a detecção neural que
# fecharia a primeira folha: logit de presença ≥ 2,0 E fração de
# arestas internas da máscara ≥ 0,005. Na bancada da frente 85 ele
# zera as entradas fantasma da sessão sem atrasar a folha real,
# corta as cenas sustentadas da bancada de ausência de 67 para 40 e
# mantém 43/43 controles positivos (folga: logit mín 8,3, arestas
# mín 0,019). NÃO é o portão de presença do modelo C (a cabeça
# `presence` de outro modelo, que segue desligada): isto é regra de
# aquisição sobre o logit DESTE detector + textura da máscara dele.
FIRST_CONFIRM_MIN_LOGIT = 2.0
FIRST_CONFIRM_MIN_EDGE_IN = 0.005
_FIRST_CONFIRM_GRID = 384
_FIRST_CONFIRM_CANNY_LOW = 60
_FIRST_CONFIRM_CANNY_HIGH = 180
_ENV_FIRST_CONFIRM_GATE_VAR = "DRAW_FIRST_CONFIRM_GATE"


def first_confirm_gate_enabled():
    """True só com DRAW_FIRST_CONFIRM_GATE=1 (default: desligado)."""
    return os.environ.get(_ENV_FIRST_CONFIRM_GATE_VAR, "").strip() == "1"


def _fill_interior_holes(mask):
    """Fecha os buracos internos de uma máscara binária 0/1.

    Fundo encerrado por folha — mão, objeto ou traço que o modelo
    não reconheceu dentro da página — volta para a cobertura; fundo
    ligado à borda da cena (a mesa de verdade) segue fora. É a
    diferença entre "o detector não viu papel ali" e "não é folha".
    """
    outside = (mask == 0).astype(np.uint8)
    # A margem de 1 px garante um fundo contíguo para a semente.
    flood = np.pad(outside, 1, constant_values=1)
    cv2.floodFill(
        flood, np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8), (0, 0), 0
    )
    holes = flood[1:-1, 1:-1]
    return mask | (holes != 0).astype(np.uint8)


def mask_edge_fraction(frame, mask):
    """Fração de píxeis de aresta DENTRO da máscara, na grade 384².

    A receita medida da frente 85, sem desvio: frame e máscara
    reduzidos a 384x384 (máscara com vizinho-mais-próximo), Canny
    60/180 sobre o cinza, e a fração dos píxeis da máscara que são
    aresta. O blob fantasma do mousepad é liso (0,0003–0,004); papel
    real tem pauta/textura (0,019+ nos controles dele). Máscara vazia
    devolve 0,0.
    """
    small = cv2.resize(frame, (_FIRST_CONFIRM_GRID, _FIRST_CONFIRM_GRID))
    small_mask = (
        cv2.resize(
            mask,
            (_FIRST_CONFIRM_GRID, _FIRST_CONFIRM_GRID),
            interpolation=cv2.INTER_NEAREST,
        )
        > 0
    )
    if not small_mask.any():
        return 0.0
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, _FIRST_CONFIRM_CANNY_LOW, _FIRST_CONFIRM_CANNY_HIGH)
    return float((canny[small_mask] > 0).mean())


def first_confirm_passes(presence_logit, edge_in):
    """O veredito do portão: logit E arestas, os dois com folga.

    Sinal ausente (None) reprova: sem logit ou sem máscara não há
    como distinguir a folha real do blob liso, e a detecção fraca é
    rebaixada só naquela ronda — a folha entra no primeiro frame
    forte, não fica bloqueada.
    """
    if presence_logit is None or edge_in is None:
        return False
    return (
        float(presence_logit) >= FIRST_CONFIRM_MIN_LOGIT
        and float(edge_in) >= FIRST_CONFIRM_MIN_EDGE_IN
    )


# Guarda de plausibilidade do ajuste (medida em 2026-10-30 no
# laboratório, draw-lab/noite-2026-10-30/block6/): o ajuste neural cru
# é biestável sob oclusão. Nas 1.132 detecções gravadas dos dois
# vídeos de replay, uma família de ajustes colapsa uma aresta (a
# direita com 213–269 px contra a esquerda com 980–1.025 px: razão
# entre arestas opostas de 3,81 a 4,67), enquanto tudo o que o app
# exibiu em trechos quietos fica em 2,42 ou menos e o máximo do outro
# vídeo é 3,00. A guarda recusa o quadrilátero cuja maior razão entre
# arestas opostas passa de _FIT_OPPOSITE_RATIO_MAX. Recusa também o
# que extrapola um canto para além de _FIT_EXCURSION_DIAG_FRAC da
# diagonal do quadro: nas gravações a folha sai do recorte de tela de
# verdade até 9,9% da diagonal, mas o modo fora do quadro medido no
# frame_real.png (1289x710) chegou a 21,4%. Ajuste recusado vale
# "sem quadrilátero" e não alimenta a memória de arestas: quem chamou
# decide (no app, o rastreador segura o quadrilátero anterior). Nas
# gravações, a guarda recusa 26 de 1.132 ajustes (25 da família
# colapsada de um vídeo, 1 do outro). Ressalva: as gravações são de
# tela, com overlay queimado; a guarda precisa de reconfirmação em
# captura crua.
_FIT_OPPOSITE_RATIO_MAX = 3.0
_FIT_EXCURSION_DIAG_FRAC = 0.15


def _fit_plausible(corners, frame_shape):
    """True se o quadrilátero passa na guarda de plausibilidade.

    A primeira regra é finitude. Toda comparação com NaN é falsa, e
    sem esta linha um quadrilátero com NaN atravessava as outras
    regras sem disparar nenhuma: a guarda aprovava justamente o
    ajuste mais inválido possível. Infinito já era recusado pela
    comparação com a margem do quadro, mas a regra explícita cobre
    os dois casos com a mesma mensagem.
    """
    quad = np.asarray(corners, dtype=np.float64)
    if quad.shape != (4, 2):
        return False
    if not np.all(np.isfinite(quad)):
        return False
    edges = np.array([np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)])
    if edges.min() <= 0:
        return False
    for a, b in ((0, 2), (3, 1)):
        ratio = max(edges[a], edges[b]) / min(edges[a], edges[b])
        if ratio > _FIT_OPPOSITE_RATIO_MAX:
            return False
    height, width = frame_shape[:2]
    margin = _FIT_EXCURSION_DIAG_FRAC * float(np.hypot(width, height))
    return not (
        quad[:, 0].min() < -margin
        or quad[:, 0].max() > width + margin
        or quad[:, 1].min() < -margin
        or quad[:, 1].max() > height + margin
    )


def _load_session(model_path):
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime nao esta instalado; o detector neural precisa "
            "dele. Instale com: pip install onnxruntime"
        ) from error
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"modelo ONNX nao encontrado: {path}. Passe o caminho do "
            f"arquivo .onnx ou defina a variavel {_ENV_MODEL_VAR}."
        )
    # Integridade opcional: com models/MANIFEST.json no lugar, o
    # hash do arquivo é conferido antes de a sessão abrir; sem
    # manifesto, nada muda (ver draw/manifest.py).
    verify_model(path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = _ort_intra_threads()
    _disable_spin(options)
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _disable_spin(options):
    """Desliga o busy-wait (spin) dos pools desta sessao.

    O app roda duas sessoes ORT + MediaPipe + OpenCV no mesmo
    processo; com spin ligado (o padrao do ORT), as threads de uma
    sessao giram em espera ativa e roubam CPU da sessao que esta
    trabalhando — em pipelines multi-modelo medidos por terceiros,
    detectores rodaram 2-3x mais lentos que isolados com spin
    ligado. O estudo de performance de 2026-11-03 apontou o spin
    como suspeito do piso de ~2,5 nucleos sem explicacao no regime
    sem folha. DRAW_ORT_SPIN=1 religa, para o A/B na mesa.
    """
    if os.environ.get("DRAW_ORT_SPIN", "").strip() == "1":
        return
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")


def _ort_intra_threads():
    """Threads intra-op da sessão: 3, do teto combinado de pools.

    Os pools nativos do app somados disputavam os nucleos fisicos
    (oversubscription medida na mesa do Lucas, 2026-11-02: CPU
    desenhando 550,8% com detector 4 + tinta 8 + OpenCV 16; com
    detector 2 + tinta 4 caiu para 289,8%, passada igual). O
    padrao do detector fica em 3: a soma dos pools ORT (3 + 4)
    cabe nos nucleos com folga, e 2 mediram +31% por chamada de
    deteccao — 3 e o ponto medio. A variável DRAW_ORT_THREADS
    troca o valor sem tocar no código, para medir em outra
    máquina; valor inválido ou não positivo volta ao medido.
    """
    raw = os.environ.get(_ENV_THREADS_VAR)
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
    return _ORT_INTRA_THREADS


class _SessionContract(NamedTuple):
    """O que o arquivo ONNX diz sobre si mesmo.

    input_name é o nome que o feed do session.run precisa usar —
    nunca um literal no código. size é a resolução quadrada de
    entrada, já resolvida (o dinâmico cai em _INPUT_SIZE). outputs
    são os nomes das saídas, na ordem do arquivo, para não
    reconstruir a lista a cada frame.
    """

    input_name: str
    size: Tuple[int, int]
    outputs: Tuple[str, ...]


def _session_contract(session):
    """Lê e valida o contrato de entrada/saída de uma sessão ONNX.

    O adaptador antigo mandava o tensor no feed com o nome literal
    "image" e lia só shape[2], presumindo quadrado. Um export com
    entrada "input" quebrava no primeiro frame (não na carga), e um
    export retangular era esmagado para quadrado sem aviso. Aqui o
    nome vem do arquivo e a resolução é validada na primeira
    chamada, com erro explícito.
    """
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise RuntimeError(
            "o modelo ONNX precisa de exatamente uma entrada de imagem; "
            f"este arquivo declara {[spec.name for spec in inputs]}"
        )
    spec = inputs[0]
    shape = list(spec.shape)
    if len(shape) != 4:
        raise RuntimeError(
            "a entrada do modelo precisa ser [N, C, H, W]; o arquivo "
            f"declara {spec.name} com forma {shape}"
        )
    height, width = shape[2], shape[3]
    fixed = [
        dimension
        for dimension in (height, width)
        if isinstance(dimension, int) and dimension > 0
    ]
    if len(fixed) == 2:
        if fixed[0] != fixed[1]:
            raise RuntimeError(
                "entrada retangular não é suportada: o pré-processamento "
                "esmaga o frame para uma resolução quadrada e deformaria "
                f"a imagem em silêncio (o arquivo declara {height}x{width})"
            )
        size = (fixed[0], fixed[1])
    elif len(fixed) == 0:
        size = (_INPUT_SIZE, _INPUT_SIZE)
    else:
        raise RuntimeError(
            "entrada com altura fixa e largura dinâmica (ou o inverso) "
            f"não é suportada; o arquivo declara {shape}"
        )
    return _SessionContract(
        input_name=spec.name,
        size=size,
        outputs=tuple(output.name for output in session.get_outputs()),
    )


def _order_corners(points):
    return _paper._order_corners(points)


def _largest_contour(mask_bin):
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _approx_quad(contour):
    perimeter = cv2.arcLength(contour, True)
    for epsilon in _paper._SHEET_EPSILON_LADDER:
        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return _order_corners(approx.reshape(4, 2))
    return None


def _min_area_quad(contour):
    """Reserva do ajuste: o retângulo de área mínima do contorno.

    Só é chamado quando a escada de epsilon não fecha. Os cantos
    saem na mesma convenção TL/TR/BR/BL do caminho principal, e o
    retângulo volta cru, sem o reajuste de arestas e sem o refino
    subpixel. Medido: nos frames 24 e 26 das capturas reais, os
    únicos em que a escada falha com o modelo v2, o IoU do
    quadrilátero preenchido do frame 24 contra a máscara anotada
    dele é 0,895 com o retângulo cru, 0,850 com o reajuste, 0,843
    com reajuste e refino e 0,887 só com o refino. O reajuste foi
    desenhado para os vértices da aproximação, que moram perto dos
    cantos da folha; os vértices do retângulo mínimo moram fora
    dela, e as retas ajustadas passam a seguir a borda visível da
    máscara, que nesses frames é oclusão, não a aresta da página.
    """
    return _order_corners(cv2.boxPoints(cv2.minAreaRect(contour)))


def _fit_line(points):
    vx, vy, x0, y0 = cv2.fitLine(
        points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
    ).ravel()
    direction = np.array([vx, vy], dtype=np.float64)
    point = np.array([x0, y0], dtype=np.float64)
    return direction, point


def _intersect(line_a, line_b):
    dir_a, point_a = line_a
    dir_b, point_b = line_b
    matrix = np.column_stack([dir_a, -dir_b])
    if abs(np.linalg.det(matrix)) < 1e-6:
        return None
    t = np.linalg.solve(matrix, point_b - point_a)[0]
    return point_a + t * dir_a


def _segment_distances(points, start, end):
    segment = end - start
    t = np.clip(
        ((points - start) @ segment) / max(float(segment @ segment), 1e-9),
        0.0,
        1.0,
    )
    return np.linalg.norm(points - (start + np.outer(t, segment)), axis=1)


def _line_residuals(points, line):
    dir_fit, point_fit = line
    normal = np.array([-dir_fit[1], dir_fit[0]], dtype=np.float64)
    return np.abs((points - point_fit) @ normal)


def _line_angle_diff_deg(line_a, line_b):
    dir_a = line_a[0] / max(float(np.linalg.norm(line_a[0])), 1e-9)
    dir_b = line_b[0] / max(float(np.linalg.norm(line_b[0])), 1e-9)
    cos_ang = min(1.0, abs(float(dir_a @ dir_b)))
    return float(np.degrees(np.arccos(cos_ang)))


def _line_normal_shift_at(point, line_ref):
    dir_ref, pt_ref = line_ref
    normal = np.array([-dir_ref[1], dir_ref[0]], dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-9)
    return abs(float((point - pt_ref) @ normal))


def _best_inlier_seed_line(side, along, pixel_scale=1.0):
    """Escolhe semente linear robusta a entalhe de braço no meio da aresta."""
    candidates = [_fit_line(side)]
    order = np.argsort(along)
    n = len(order)
    if n >= 16:
        for q0, q1 in ((0.06, 0.94), (0.06, 0.38), (0.62, 0.94), (0.18, 0.82)):
            p0 = side[order[int(q0 * (n - 1))]]
            p1 = side[order[int(q1 * (n - 1))]]
            chord = p1 - p0
            chord_len = float(np.linalg.norm(chord))
            if chord_len >= 12.0 * pixel_scale:
                candidates.append((chord / chord_len, p0))

    best_line = candidates[0]
    best_score = (-1, -1e9)
    for cand in candidates:
        res = _line_residuals(side, cand)
        inliers = res <= _FIT_MAX_INLIER_RESIDUAL_PX * pixel_scale
        count = int(inliers.sum())
        neg_med = -float(np.median(res[inliers])) if count > 0 else -1e9
        score = (count, neg_med)
        if score > best_score:
            best_score = score
            best_line = cand
    return best_line


def _edge_line(side, start, end, centroid, fallback=None, pixel_scale=1.0):
    """A reta de uma aresta: ajustada, herdada ou travada na direção.

    Com cobertura boa E colinearidade real dentro de resíduo absoluto
    (_FIT_MAX_INLIER_RESIDUAL_PX), vale o ajuste de Huber podado aos
    inliers retos e a reta é confiável. Quando um braço/mão curvo invade
    a máscara mantendo cobertura >= 0,5, o portão absoluto de retidão
    reprova a aresta: ela não contamina a memória (_edge_lines) e herda
    `fallback` quando disponível.
    """
    segment = end - start
    length = float(np.linalg.norm(segment))
    if length < 1e-6:
        return _fit_line(side), False, False
    direction = segment / length
    along = (side - start) @ direction
    coverage = float(along.max() - along.min()) / length
    if coverage >= _FIT_MIN_EDGE_COVERAGE:
        seed = _best_inlier_seed_line(side, along, pixel_scale)
        residuals = _line_residuals(side, seed)
        tol = min(
            _FIT_MAX_INLIER_RESIDUAL_PX * pixel_scale,
            max(3.0 * pixel_scale, float(np.percentile(residuals, 70))),
        )
        keep = residuals <= tol
        if int(keep.sum()) >= 8:
            line = _fit_line(side[keep])
            residuals = _line_residuals(side, line)
            inliers = residuals <= _FIT_MAX_INLIER_RESIDUAL_PX * pixel_scale
            if int(inliers.sum()) >= 8:
                line = _fit_line(side[inliers])
                residuals = _line_residuals(side, line)
                inliers = residuals <= _FIT_MAX_INLIER_RESIDUAL_PX * pixel_scale
        else:
            line = seed
            inliers = residuals <= _FIT_MAX_INLIER_RESIDUAL_PX * pixel_scale

        n_inliers = int(inliers.sum())
        if n_inliers >= 8:
            inlier_along = along[inliers]
            inlier_cov = float(inlier_along.max() - inlier_along.min()) / length
            inlier_ratio = n_inliers / float(len(side))
            inlier_rms = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
            # Aceita também entalhe localizado no meio quando sobram >= 24
            # inliers colineares cobrindo ambos os lados da aresta.
            ratio_ok = (inlier_ratio >= _FIT_MIN_INLIER_RATIO) or (
                n_inliers >= 24 and inlier_cov >= 0.70
            )
            if (
                inlier_cov >= _FIT_MIN_EDGE_COVERAGE
                and ratio_ok
                and inlier_rms <= _FIT_MAX_INLIER_RMS_PX * pixel_scale
            ):
                return line, False, True

    if fallback is not None:
        return fallback, True, False
    normal = np.array([-direction[1], direction[0]])
    if float((centroid - start) @ normal) < 0:
        normal = -normal
    offset = float(np.median((side - start) @ normal))
    return (direction, start + offset * normal), True, False


def _refit_corners(contour, quad, frame_shape, fallback_lines=None, scale=(1.0, 1.0)):
    """Reajusta os cantos como interseções das retas das arestas.

    Cada ponto do contorno é atribuído à aresta mais próxima. Pontos
    colados nos vértices da aproximação ficam fora (são a virada de
    uma aresta para outra). Pontos sobre a borda do frame também
    ficam fora quando sobra ponto bastante sem eles: a máscara ali
    foi cortada pelo quadro. A reta de cada aresta sai de _edge_line
    (ajustada; sem cobertura, herdada de fallback_lines ou travada
    na direção do segmento). Um canto reajustado só substitui o
    vértice se ficar a no máximo _FIT_MAX_DISPLACE_PX dele, exceto
    quando uma das arestas do canto é travada: ali a projeção já
    nasce limitada pela trava e o vértice de referência é justamente
    o contaminado. O quadrilátero final precisa continuar convexo.

    Retorna (quadrilatero, retas usadas, confiavel por aresta); as
    retas voltam None quando alguma aresta tem ponto de menos e o
    reajuste inteiro é abandonado.
    """
    frame_h, frame_w = frame_shape
    # Resíduos usam limites conservadores; deslocamentos dos quatro cantos
    # usam a métrica exata do frame, inclusive quando scale_x != scale_y.
    pixel_scale = 1.0 / max(scale)
    points = contour.reshape(-1, 2).astype(np.float64)
    distances = np.stack(
        [_segment_distances(points, quad[i], quad[(i + 1) % 4]) for i in range(4)],
        axis=1,
    )
    assignment = np.argmin(distances, axis=1)
    centroid = quad.astype(np.float64).mean(axis=0)
    lines = []
    locked = []
    trusted = []
    for i in range(4):
        start = quad[i].astype(np.float64)
        end = quad[(i + 1) % 4].astype(np.float64)
        side = points[assignment == i]
        if len(side) < 8:
            return quad, None, None
        near_vertex = (np.linalg.norm(side - start, axis=1) < 25 * pixel_scale) | (
            np.linalg.norm(side - end, axis=1) < 25 * pixel_scale
        )
        side = side[~near_vertex]
        on_border = (
            (side[:, 0] <= 0.5)
            | (side[:, 0] >= frame_w - 1.5)
            | (side[:, 1] <= 0.5)
            | (side[:, 1] >= frame_h - 1.5)
        )
        off_border = side[~on_border]
        if len(off_border) >= 8:
            side = off_border
        if len(side) < 8:
            return quad, None, None
        fallback = None
        if fallback_lines is not None:
            fallback = fallback_lines[i]
        line, is_locked, is_trusted = _edge_line(
            side, start, end, centroid, fallback, pixel_scale
        )
        lines.append(line)
        locked.append(is_locked)
        trusted.append(is_trusted)

    if fallback_lines is not None:
        # Guarda de corpo rígido: se >= 2 outras arestas confirmam que a
        # folha continua parada, uma única aresta que salte em ângulo ou
        # posição (ex.: manga/antebraço reto cobrindo metade de um lado)
        # não pode girar sozinha nem contaminar a memória.
        steady = []
        jumped = []
        for i in range(4):
            prev = fallback_lines[i]
            if prev is None or not trusted[i]:
                continue
            mid = 0.5 * (
                quad[i].astype(np.float64) + quad[(i + 1) % 4].astype(np.float64)
            )
            # Mede deslocamento no ponto médio da aresta nova contra a reta anterior.
            dir_cur, pt_cur = lines[i]
            proj_mid = pt_cur + float((mid - pt_cur) @ dir_cur) * dir_cur
            ang = _line_angle_diff_deg(lines[i], prev)
            shf = _line_normal_shift_at(proj_mid, prev)
            if (
                ang <= _MEMORY_STATIONARY_ANGLE_DEG
                and shf <= _MEMORY_STATIONARY_SHIFT_PX * pixel_scale
            ):
                steady.append(i)
            elif (
                ang > _MEMORY_SINGLE_OUTLIER_ANGLE_DEG
                or shf > _MEMORY_SINGLE_OUTLIER_SHIFT_PX * pixel_scale
            ):
                jumped.append(i)
        if len(steady) >= 2 and len(jumped) == 1:
            bad = jumped[0]
            lines[bad] = fallback_lines[bad]
            locked[bad] = True
            trusted[bad] = False

    refined = quad.copy()
    for i in range(4):
        corner = _intersect(lines[i - 1], lines[i])
        if corner is None:
            continue
        if locked[i - 1] or locked[i]:
            refined[i] = corner
        elif np.linalg.norm((corner - quad[i]) * scale) <= _FIT_MAX_DISPLACE_PX:
            refined[i] = corner
    refined = _order_corners(refined)
    if not cv2.isContourConvex(refined.reshape(4, 1, 2).astype(np.float32)):
        return quad, lines, trusted
    return refined, lines, trusted


def _refine_inside(frame, corners):
    """Refino subpixel só nos cantos que estão dentro do quadro.

    Canto projetado para fora do quadro (página cortada) não tem
    vizinhança de imagem para refinar; ele fica como a reta o pôs.
    Cantos dentro usam o mesmo acolchoamento do detector clássico.
    """
    frame_h, frame_w = frame.shape[:2]
    inside = [
        i
        for i, (x, y) in enumerate(corners)
        if 0.0 <= x <= frame_w - 1 and 0.0 <= y <= frame_h - 1
    ]
    if not inside:
        return corners
    refined = corners.copy()
    refined[inside] = _paper._refine_corners(frame, corners[inside])
    return refined.astype(np.float32)


def _fit_quad_impl(mask_bin, frame, fallback_lines, scale_x=1.0, scale_y=1.0):
    """O ajuste completo sobre a máscara, com herança opcional.

    Retorna (quadrilatero ou None, retas usadas ou None, confiavel
    por aresta ou None). fallback_lines são as retas confiáveis da
    chamada anterior, oferecidas às arestas sem cobertura.
    """
    contour = _largest_contour(mask_bin)
    if contour is None:
        return None, None, None
    frame_area = float(mask_bin.shape[0] * mask_bin.shape[1])
    if cv2.contourArea(contour) < frame_area * _MIN_CONTOUR_AREA_RATIO:
        return None, None, None
    scale = np.array([scale_x, scale_y], dtype=np.float32)
    quad = _approx_quad(contour)
    if quad is None:
        return _min_area_quad(contour) * scale, None, None
    quad, lines, trusted = _refit_corners(
        contour, quad, mask_bin.shape[:2], fallback_lines, scale
    )
    quad = quad * scale
    if frame is not None:
        quad = _refine_inside(frame, quad)
    return quad.astype(np.float32), lines, trusted


def fit_quad(mask_bin, frame=None, *, scale_x=1.0, scale_y=1.0):
    """Ajusta o quadrilátero da folha sobre a máscara binária.

    Contornos e retas ficam na grade da máscara; scale_x/scale_y
    levam apenas os quatro cantos ao espaço do frame antes do refino.
    Retorna float32[4,2] TL/TR/BR/BL ou None quando a máscara não
    tem contorno ou é pequena demais para sustentar um quadrilátero.
    Se a escada de epsilon não fecha, vale a reserva de
    _min_area_quad, sem reajuste nem refino. Com o frame, os cantos
    do caminho principal que estão dentro do quadro ganham refino
    subpixel sobre a imagem. Esta função não tem estado: a memória
    de arestas entre chamadas mora no NeuralSheetDetector.
    """
    quad, _lines, _trusted = _fit_quad_impl(mask_bin, frame, None, scale_x, scale_y)
    return quad


class NeuralSheetDetector:
    """Detector neural com sessão própria.

    __call__(frame) devolve (cantos, veredito, máscara): o mesmo trio
    da interface dos detectores do harness de avaliação. O veredito é
    "paper" quando o logit de presença é positivo e o quadrilátero
    fecha; fora isso, (None, None, máscara). A máscara é uint8
    0/255 no tamanho do frame, limiarizada em 0,5 como na avaliação
    da fase 2. return_mask=False evita materializar essa máscara;
    o ajuste e a memória de arestas usam sempre a grade nativa.

    O detector guarda as retas confiáveis das arestas entre chamadas
    (a memória de arestas): quando uma aresta perde cobertura numa
    chamada (a mão sobre ela), a reta herdada estima essa aresta até
    ela voltar a ser visível. A herança só vale em cena contínua: a
    máscara da chamada precisa ter IoU de pelo menos
    _MEMORY_MIN_MASK_IOU com a da chamada anterior, medido nas
    máscaras reduzidas por _MEMORY_MASK_STRIDE. Cena nova recomeça
    sem herança.

    O atributo last_presence_logit guarda o sinal de presença da
    última chamada a predict (None antes da primeira e após um frame
    vazio): nos modelos legados é o logit treinado; no SegFormer de
    quatro classes é um sinal derivado da área segmentada. O app usa-o
    para decidir se a reserva clássica pode falar.
    """

    def __init__(self, model_path):
        self.model_path = str(model_path)
        self._session = None
        self._contract = None
        self._contract_session = None
        self._edge_lines = None
        self._memory_mask = None
        self._memory_geometry = None
        self.last_presence_logit = None
        # Cobertura de folha da última previsão (0/255, resolução da
        # máscara devolvida): classes de folha sob qualquer ocupante
        # — papel, tinta E mão. A mão fica dentro para a cobertura
        # não virar buraco permanente onde a mão paira; quem segura
        # o voto sob a mão é a visibilidade, não a cobertura.
        self.last_coverage_mask = None
        self._timing_enabled = os.environ.get("DRAW_TIMING") == "1"
        self._timing_stats = {
            phase: {"count": 0, "sum_ms": 0.0, "max_ms": 0.0}
            for phase in _TIMING_PHASES
        }

    def get_timing_stats(self):
        """Snapshot em ms das previsões completas com DRAW_TIMING=1.

        Ative a variável antes de construir o detector. Carga da sessão,
        leitura do contrato e fit_quad não entram nas três fases.
        """
        return {
            phase: {
                **stats,
                "mean_ms": stats["sum_ms"] / stats["count"] if stats["count"] else 0.0,
            }
            for phase, stats in self._timing_stats.items()
        }

    def _record_timing(self, durations):
        for phase, seconds in zip(_TIMING_PHASES, durations):
            stats = self._timing_stats[phase]
            milliseconds = seconds * 1000.0
            stats["count"] += 1
            stats["sum_ms"] += milliseconds
            stats["max_ms"] = max(stats["max_ms"], milliseconds)
        count = self._timing_stats["session_run"]["count"]
        if count % _TIMING_LOG_EVERY == 0:
            summary = " | ".join(
                f"{phase}: mean={stats['mean_ms']:.3f} ms max={stats['max_ms']:.3f} ms"
                for phase, stats in self.get_timing_stats().items()
            )
            print(f"DRAW_TIMING frames={count} | {summary}")

    def _get_session(self):
        if self._session is None:
            self._session = _load_session(self.model_path)
        return self._session

    def contract(self):
        """O contrato do arquivo carregado (nome da entrada, resolução).

        É lido uma vez por sessão. Um teste que injeta a sessão em
        _session (sem passar pelo disco) continua funcionando: o
        contrato é resolvido da sessão em uso, não do caminho.
        """
        session = self._get_session()
        if self._contract is None or self._contract_session is not session:
            self._contract = _session_contract(session)
            self._contract_session = session
        return self._contract

    def load(self):
        """Carrega a sessão agora, para erro de modelo falhar cedo."""
        self._get_session()

    def _fallback_lines(self, mask):
        """As retas herdáveis para esta máscara, ou None sem continuidade."""
        if self._edge_lines is None or self._memory_mask is None:
            return None
        small = mask[::_MEMORY_MASK_STRIDE, ::_MEMORY_MASK_STRIDE] > 0
        if small.shape != self._memory_mask.shape:
            return None
        inter = int((small & self._memory_mask).sum())
        union = int((small | self._memory_mask).sum())
        if union == 0 or inter / union < _MEMORY_MIN_MASK_IOU:
            return None
        return self._edge_lines

    def _fit(self, mask, frame=None, *, scale_x=1.0, scale_y=1.0):
        """fit_quad com a memória de arestas do detector.

        A guarda de plausibilidade (_fit_plausible) fecha o caminho:
        quadrilátero recusado devolve None e não alimenta a memória
        de arestas, para a geometria selvagem não virar herança.
        """
        geometry = (mask.shape, scale_x, scale_y)
        if geometry != self._memory_geometry:
            self._edge_lines = None
            self._memory_mask = None
            self._memory_geometry = geometry
        fallback = self._fallback_lines(mask)
        corners, lines, trusted = _fit_quad_impl(
            mask, frame, fallback, scale_x, scale_y
        )
        ref_shape = (
            frame.shape
            if frame is not None
            else (mask.shape[0] * scale_y, mask.shape[1] * scale_x)
        )
        if corners is not None and not _fit_plausible(corners, ref_shape):
            return None
        if lines is not None:
            stored = (
                list(self._edge_lines)
                if self._edge_lines is not None
                else [None, None, None, None]
            )
            for i in range(4):
                if trusted[i]:
                    stored[i] = lines[i]
            self._edge_lines = stored
        self._memory_mask = mask[::_MEMORY_MASK_STRIDE, ::_MEMORY_MASK_STRIDE] > 0
        return corners

    def predict(self, frame, *, native=False):
        """Roda o modelo. Retorna (sinal de presença, máscara 0/255).

        native=True mantém a resolução de saída do modelo, sem resize.
        O padrão preserva a máscara full-res para diagnóstico/avaliação.
        O contrato legado tem uma máscara binária e um logit de presença.
        O SegFormer de quatro classes exporta apenas logits semânticos;
        nesse caso, papel e tinta formam a região da folha (a mão continua
        excluída) e a própria área prevista fornece um sinal de presença.

        O nome da entrada e a lista de saídas vêm do arquivo
        (contract()), nunca de literais: um export com entrada
        "input" ou com uma saída auxiliar a mais deixa de quebrar no
        primeiro frame. Um mask_logits de quatro canais é lido pelo
        argmax (papel + tinta), não mais como fundo x classe 1.
        """
        session = self._get_session()
        contract = self.contract()
        preprocess_start = time.perf_counter() if self._timing_enabled else None
        frame_h, frame_w = frame.shape[:2]
        tensor = _frame_to_tensor(frame, contract.size)
        output_names = list(contract.outputs)
        feed = {contract.input_name: tensor}
        if self._timing_enabled:
            preprocess_end = time.perf_counter()
        outputs = session.run(output_names, feed)
        if self._timing_enabled:
            inference_end = time.perf_counter()
        by_name = dict(zip(output_names, outputs))

        if "mask_logits" in by_name and "presence_logit" in by_name:
            logits = by_name["mask_logits"]
            presence = by_name["presence_logit"]
            if logits.ndim != 4:
                raise ValueError(
                    "saída 'mask_logits' precisa ter forma [N, C, H, W]; "
                    f"veio {logits.shape}"
                )
            channels = int(logits.shape[1])
            if channels == 2:
                # canais do contrato legado: 0 = fundo, 1 = papel.
                # prob(papel) > 0,5 <=> exp_papel > exp_fundo <=>
                # logit_papel > logit_fundo: o softmax de dois canais
                # que existia aqui dava a mesma máscara e custava um
                # exp sobre 2 x lado x lado por frame.
                mask_small = (logits[0, 1] > logits[0, 0]).astype(np.uint8)
                coverage_small = mask_small
            elif channels == 4:
                # Export híbrido: máscara semântica de quatro classes
                # com cabeça de presença treinada. Antes, esta forma
                # era lida em silêncio como fundo x classe 1 (máscara
                # errada, sem erro nenhum).
                classes = np.argmax(logits[0], axis=0)
                mask_small = (
                    (classes == _PAPER_CLASS) | (classes == _INK_CLASS)
                ).astype(np.uint8)
                # Cobertura de voto: tudo que não é fundo — papel,
                # tinta e mão sobre a folha entram, mesa não entra.
                coverage_small = (classes != 0).astype(np.uint8)
            else:
                raise ValueError(
                    "saída 'mask_logits' precisa ter 2 canais "
                    "(fundo, papel) ou 4 (fundo, papel, mão, tinta); "
                    f"veio {channels}"
                )
            self.last_presence_logit = float(presence[0])
        elif "logits" in by_name:
            logits = by_name["logits"]
            if logits.ndim != 4 or logits.shape[1] != 4:
                raise ValueError(
                    "saída 'logits' precisa ter forma [N, 4, H, W] "
                    "(fundo, papel, mão, tinta)"
                )
            classes = np.argmax(logits[0], axis=0)
            page = (classes == _PAPER_CLASS) | (classes == _INK_CLASS)
            page_ratio = float(np.mean(page))
            mask_small = page.astype(np.uint8)
            coverage_small = (classes != 0).astype(np.uint8)
            self.last_presence_logit = (
                _SEGMENTATION_PRESENT_LOGIT
                if page_ratio >= _SEGMENTATION_PRESENCE_MIN_RATIO
                else _SEGMENTATION_ABSENT_LOGIT
            )
        else:
            raise ValueError(
                "saídas ONNX incompatíveis: esperava 'mask_logits' e "
                "'presence_logit' ou 'logits' com quatro classes; "
                f"recebi {output_names}"
            )
        mask = mask_small * 255
        # Cobertura para o voto (DRAW_SHEET_COVERAGE): o que o
        # modelo diz não ser fundo, com buracos internos fechados —
        # uma mão que ele não reconheceu pairando sobre a folha não
        # pode secar a tinta debaixo dela. Fica na resolução nativa:
        # quem consome (draw/app.py) faz o upsample no mesmo ponto
        # em que a máscara de chamada ganha a resolução do frame.
        self.last_coverage_mask = _fill_interior_holes(coverage_small) * 255
        if not native:
            mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        if self._timing_enabled:
            postprocess_end = time.perf_counter()
            self._record_timing(
                (
                    preprocess_end - preprocess_start,
                    inference_end - preprocess_end,
                    postprocess_end - inference_end,
                )
            )
        return self.last_presence_logit, mask

    def __call__(self, frame, *, return_mask=True):
        if frame is None or frame.size == 0:
            self.last_presence_logit = None
            return None, None, None
        presence, mask = self.predict(frame, native=True)
        frame_h, frame_w = frame.shape[:2]
        mask_h, mask_w = mask.shape
        corners = None
        if presence > 0:
            corners = self._fit(
                mask, frame, scale_x=frame_w / mask_w, scale_y=frame_h / mask_h
            )
        verdict = "paper" if corners is not None else None
        if return_mask:
            mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        else:
            mask = None
        return corners, verdict, mask

    def detect_sheet_with_confidence(self, frame):
        """O par (cantos, veredito) do contrato de draw.paper."""
        corners, verdict, _ = self(frame, return_mask=False)
        return corners, verdict


_DEFAULT_DETECTOR = None


def _default_detector(model_path=None):
    global _DEFAULT_DETECTOR
    path = model_path or os.environ.get(_ENV_MODEL_VAR)
    if not path:
        raise RuntimeError(
            "caminho do modelo nao informado: passe model_path ou "
            f"defina a variavel {_ENV_MODEL_VAR} com o caminho do .onnx."
        )
    if _DEFAULT_DETECTOR is None or _DEFAULT_DETECTOR.model_path != str(path):
        _DEFAULT_DETECTOR = NeuralSheetDetector(path)
    return _DEFAULT_DETECTOR


def detect_sheet_with_confidence(frame, model_path=None):
    """Encontra os cantos da folha com o modelo neural.

    Mesmo contrato de draw.paper.detect_sheet_with_confidence:
    (cantos, "paper") quando o modelo afirma papel e o quadrilátero
    fecha; (None, None) quando não. O caminho do modelo vem do
    argumento ou da variável DRAW_NEURAL_MODEL.
    """
    return _default_detector(model_path).detect_sheet_with_confidence(frame)
