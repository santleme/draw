"""Máscara de mão: onde a mão está no frame da câmera, via MediaPipe.

O canvas acumulativo (draw.ink.InkAccumulator) precisa saber quais
pixels da folha estão cobertos pela mão em cada frame: pixel coberto
é desconhecido e não ganha nem perde tinta. Quem produz essa
informação é o Hand Landmarker do MediaPipe (API tasks), rodando no
frame da câmera; a máscara sai no espaço do frame e o chamador a leva
ao espaço do papel pela mesma homografia da tinta.

A máscara de uma mão é o casco convexo dos 21 marcos, escalado de
1,30 em torno do centróide para cobrir a carne entre os dedos, unido
a um trapézio de antebraço que sai do pulso na direção oposta ao
dedo médio (o antebraço também cobre a folha e também projeta a
sombra que o detector de tinta enxerga), tudo dilatado de 15 px para
cobrir a caneta segurada e a borda difusa da mão. Medido no replay do
vídeo de desenho com limiares de confiança em 0,3: a mão de desenho é
detectada em cerca de dois terços das amostras em que aparece, e a
fração média da folha coberta fica perto de 10%.

Integração preguiçosa e opcional, no espírito do detector neural: o
mediapipe só é importado no primeiro uso, e qualquer ausência (o
pacote, o arquivo do modelo, uma falha de criação) degrada para
"sem máscara" com um aviso único, nunca para erro. O acumulador
funciona sem a máscara; ela só melhora a separação.

O arquivo do modelo (.task, binário de ~8 MB) fica fora do
repositório. O caminho chega por argumento (--hand-model), pela
variável de ambiente DRAW_HAND_MODEL ou pelo local convencional
models/hand_landmarker.task. A fonte oficial do arquivo é o
armazenamento do Google para modelos do MediaPipe.
"""

import os
from pathlib import Path

import cv2
import numpy as np

from .manifest import verify_model

_ENV_MODEL_VAR = "DRAW_HAND_MODEL"
# O caminho convencional e resolvido a partir da RAIZ DO REPO, como
# o do detector neural e o do pagina->limpa; era relativo ao CWD e
# um app aberto de outro diretorio perdia a mascara de mao sem
# aviso (o modelo existia, so nao era achado). O CWD continua como
# segunda tentativa, para quem roda a partir de uma copia local.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CANDIDATES = (
    str(_REPO_ROOT / "models" / "hand_landmarker.task"),
    "models/hand_landmarker.task",
)
_MIN_CONFIDENCE = 0.3
_HULL_SCALE = 1.30
_DILATE_PX = 15
# Largura maxima do frame na deteccao de PRESENCA (assinatura):
# presenca nao precisa dos pixels, so da geometria dos marcos.
_PRESENCE_MAX_WIDTH = 640


def _resolve_model_path(model_path):
    """Caminho do .task: argumento, ambiente ou local convencional."""
    if model_path:
        return model_path
    env = os.environ.get(_ENV_MODEL_VAR)
    if env:
        return env
    for candidate in _DEFAULT_CANDIDATES:
        if Path(candidate).exists():
            return str(candidate)
    return None


def hand_model_available(model_path=None):
    """True se um arquivo de modelo resolve e existe neste disco."""
    path = _resolve_model_path(model_path)
    return path is not None and Path(path).exists()


def _hand_polygons(points, frame_shape):
    """Os polígonos de uma mão: casco escalado e trapézio do antebraço.

    points são os 21 marcos em pixels do frame. O trapézio sai do
    pulso (marco 0) na direção oposta ao marco 9 (base do dedo
    médio), com meia-largura de 0,75 da distância entre os marcos 5
    e 17 (a largura da palma) e comprimento de 1,5 diagonal do
    frame: o antebraço sai do quadro, e a máscara precisa ir junto.
    """
    hull = cv2.convexHull(points)
    centroid = hull.reshape(-1, 2).mean(axis=0)
    scaled = centroid + (hull.reshape(-1, 2) - centroid) * _HULL_SCALE
    polygons = [scaled.astype(np.int32)]
    wrist = points[0].astype(np.float64)
    middle_base = points[9].astype(np.float64)
    direction = wrist - middle_base
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return polygons
    direction /= norm
    normal = np.array([-direction[1], direction[0]])
    half_width = max(0.75 * float(np.linalg.norm(points[5] - points[17])), 20.0)
    frame_h, frame_w = frame_shape[:2]
    length = 1.5 * float(np.hypot(frame_w, frame_h))
    end = wrist + direction * length
    base_a = wrist + direction * (0.3 * norm) + normal * half_width * 0.8
    base_b = wrist + direction * (0.3 * norm) - normal * half_width * 0.8
    end_a = end + normal * half_width
    end_b = end - normal * half_width
    polygons.append(np.array([base_a, base_b, end_b, end_a]).astype(np.int32))
    return polygons


class HandMasker:
    """Produtor preguiçoso da máscara de mão.

    mask(frame) retorna a máscara uint8 no espaço do frame (255 =
    mão), uma máscara vazia quando nenhuma mão é detectada, ou None
    quando o recurso está indisponível (sem mediapipe, sem modelo ou
    com falha na criação). A indisponibilidade avisa uma vez e é
    permanente na sessão: não adianta tentar de novo a cada frame.
    """

    def __init__(self, model_path=None):
        self._requested_path = model_path
        resolved = _resolve_model_path(model_path)
        if resolved is not None:
            # Estrito por padrao: divergente ou nao registrado levanta
            # aqui, antes de o MediaPipe abrir o arquivo.
            verify_model(resolved)
        self._landmarker = None
        self._video_landmarker = None
        self._unavailable = False
        self._video_unavailable = False
        self._last_video_ts = -1
        self._warned = False

    def _warn_once(self):
        if not self._warned:
            self._warned = True
            print(
                "Aviso: mascara de mao indisponivel (pacote mediapipe "
                "ou modelo hand_landmarker.task ausente). O canvas "
                "acumulativo roda sem ela."
            )

    def _create_landmarker(self, running_mode):
        path = _resolve_model_path(self._requested_path)
        if path is None or not Path(path).exists():
            return None
        try:
            import mediapipe as mp

            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
                running_mode=running_mode,
                num_hands=2,
                min_hand_detection_confidence=_MIN_CONFIDENCE,
                min_hand_presence_confidence=_MIN_CONFIDENCE,
                min_tracking_confidence=_MIN_CONFIDENCE,
            )
            return mp.tasks.vision.HandLandmarker.create_from_options(options)
        except Exception:
            return None

    def _ensure(self):
        if self._landmarker is not None:
            return True
        if self._unavailable:
            return False
        try:
            import mediapipe as mp
        except Exception:
            self._unavailable = True
            self._warn_once()
            return False
        created = self._create_landmarker(mp.tasks.vision.RunningMode.IMAGE)
        if created is None:
            self._unavailable = True
            self._warn_once()
            return False
        self._landmarker = created
        return True

    def _ensure_video(self):
        """Landmarker em modo VIDEO, criado sob demanda.

        O modo VIDEO do MediaPipe rastreia a mao entre quadros e
        pula a deteccao de palma enquanto ela esta localizada —
        no modo IMAGE cada chamada recomeca do zero (pesquisa de
        performance de 2026-11-03, T4). Falha ao criar ou ao
        detectar marca o modo como indisponivel e a presenca cai
        para o IMAGE permanentemente: degradacao, nunca erro.
        """
        if self._video_landmarker is not None:
            return True
        if self._video_unavailable:
            return False
        try:
            import mediapipe as mp
        except Exception:
            self._video_unavailable = True
            return False
        created = self._create_landmarker(mp.tasks.vision.RunningMode.VIDEO)
        if created is None:
            self._video_unavailable = True
            return False
        self._video_landmarker = created
        return True

    def _landmarks(self, frame, max_width=None, timestamp_ms=None):
        """Marcos das maos no frame: lista de arrays (21, 2) em px
        do frame original, [] sem mao, None se indisponivel.

        Com max_width, o frame e reduzido antes da deteccao e os
        marcos (normalizados pelo MediaPipe) sao reescalados de
        volta — a geometria e a mesma, o custo de pre-processamento
        cai com os pixels.

        Com timestamp_ms (o carimbo de captura do frame), a
        deteccao roda no landmarker VIDEO com detect_for_video; o
        carimbo e forcado estritamente crescente, como a API
        exige. Sem carimbo, ou se o modo VIDEO falhar, cai no
        landmarker IMAGE de sempre.
        """
        use_video = timestamp_ms is not None and self._ensure_video()
        if not use_video and not self._ensure():
            return None
        import mediapipe as mp

        work = frame
        if max_width and frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            work = cv2.resize(
                frame,
                (max_width, int(round(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = None
        if use_video:
            timestamp = max(int(timestamp_ms), self._last_video_ts + 1)
            try:
                result = self._video_landmarker.detect_for_video(image, timestamp)
                self._last_video_ts = timestamp
            except Exception:
                self._video_unavailable = True
                self._video_landmarker = None
                result = None
        if result is None:
            if not self._ensure():
                return None
            result = self._landmarker.detect(image)
        frame_h, frame_w = frame.shape[:2]
        hands = []
        for hand in result.hand_landmarks or []:
            hands.append(
                np.array(
                    [[lm.x * frame_w, lm.y * frame_h] for lm in hand],
                    dtype=np.float32,
                )
            )
        return hands

    def mask(self, frame):
        """A máscara de mão do frame, ou None se indisponível."""
        hands = self._landmarks(frame)
        if hands is None:
            return None
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for points in hands:
            for polygon in _hand_polygons(points, frame.shape):
                cv2.fillPoly(mask, [polygon], 255)
        if not hands:
            return mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * _DILATE_PX + 1, 2 * _DILATE_PX + 1)
        )
        return cv2.dilate(mask, kernel)

    def presence_signature(self, frame, timestamp_ms=None):
        """Assinatura de presenca sem construir mascara nenhuma.

        Devolve None (indisponivel), (0.0, 0.0, 0.0) (sem mao) ou
        (area, centroide_x, centroide_y) em px do frame: a area e a
        do casco convexo dos 21 marcos e o centroide e a media dos
        marcos. E tudo o que o adaptador de presenca do app consome
        (draw.app._MediaPipeHandPresence) — o caminho antigo pagava
        cvtColor + fillPoly + dilatacao eliptica de 31x31 sobre o
        frame 720p inteiro so para contar pixels dessa mascara
        (pesquisa de performance de 2026-11-03, item T3). A deteccao
        roda sobre o frame reduzido a _PRESENCE_MAX_WIDTH: o
        landmarker trabalha internamente em ~192-224 px, e os
        marcos normalizados reescalam sem perda de geometria.

        A mascara cheia (mask()) continua existindo para o caminho
        legado de visibilidade da tinta, que precisa dos pixels.
        """
        hands = self._landmarks(
            frame,
            max_width=_PRESENCE_MAX_WIDTH,
            timestamp_ms=timestamp_ms,
        )
        if hands is None:
            return None
        if not hands:
            return (0.0, 0.0, 0.0)
        area = 0.0
        centroids = []
        for points in hands:
            hull = cv2.convexHull(points)
            area += float(cv2.contourArea(hull))
            centroids.append(points.mean(axis=0))
        centroid = np.mean(centroids, axis=0)
        return (area, float(centroid[0]), float(centroid[1]))
