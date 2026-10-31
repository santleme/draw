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

_ENV_MODEL_VAR = "DRAW_HAND_MODEL"
_DEFAULT_CANDIDATES = ("models/hand_landmarker.task",)
_MIN_CONFIDENCE = 0.3
_HULL_SCALE = 1.30
_DILATE_PX = 15
INDEX_FINGER_TIP = 8
_DILATE_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (2 * _DILATE_PX + 1, 2 * _DILATE_PX + 1)
)


def _resolve_model_path(model_path):
    """Caminho do .task: argumento, ambiente ou local convencional."""
    if model_path:
        return model_path
    env = os.environ.get(_ENV_MODEL_VAR)
    if env:
        return env
    for candidate in _DEFAULT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


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
    half_width = max(
        0.75 * float(np.linalg.norm(points[5] - points[17])), 20.0
    )
    frame_h, frame_w = frame_shape[:2]
    length = 1.5 * float(np.hypot(frame_w, frame_h))
    end = wrist + direction * length
    base_a = wrist + direction * (0.3 * norm) + normal * half_width * 0.8
    base_b = wrist + direction * (0.3 * norm) - normal * half_width * 0.8
    end_a = end + normal * half_width
    end_b = end - normal * half_width
    polygons.append(
        np.array([base_a, base_b, end_b, end_a]).astype(np.int32)
    )
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
        self._landmarker = None
        self._unavailable = False
        self._warned = False

    def _warn_once(self):
        if not self._warned:
            self._warned = True
            print(
                "Aviso: mascara de mao indisponivel (pacote mediapipe "
                "ou modelo hand_landmarker.task ausente). O canvas "
                "acumulativo roda sem ela."
            )

    def _ensure(self):
        if self._landmarker is not None:
            return True
        if self._unavailable:
            return False
        path = _resolve_model_path(self._requested_path)
        created = None
        if path is not None and Path(path).exists():
            try:
                import mediapipe as mp

                options = mp.tasks.vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(
                        model_asset_path=str(path)
                    ),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=_MIN_CONFIDENCE,
                    min_hand_presence_confidence=_MIN_CONFIDENCE,
                    min_tracking_confidence=_MIN_CONFIDENCE,
                )
                created = (
                    mp.tasks.vision.HandLandmarker.create_from_options(
                        options
                    )
                )
            except Exception:
                created = None
        if created is None:
            self._unavailable = True
            self._warn_once()
            return False
        self._landmarker = created
        return True

    def detect(self, frame):
        """Retorna (mascara_uint8_ou_None, lista_de_marcos_21x2_float32).

        Expõe os 21 marcos de cada mão detectada no espaço do frame junto
        com a máscara dilatada de oclusão, permitindo alimentar
        draw.fusion.PenTipFusion (marco 8) sem rodar o MediaPipe duas vezes.
        """
        if not self._ensure():
            return None, []
        import mediapipe as mp

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if not result.hand_landmarks:
            return mask, []
        frame_h, frame_w = frame.shape[:2]
        hands_points = []
        for hand in result.hand_landmarks:
            points = np.array(
                [[lm.x * frame_w, lm.y * frame_h] for lm in hand],
                dtype=np.float32,
            )
            hands_points.append(points)
            for polygon in _hand_polygons(points, frame.shape):
                cv2.fillPoly(mask, [polygon], 255)
        return cv2.dilate(mask, _DILATE_KERNEL), hands_points

    def mask(self, frame):
        """A máscara de mão do frame, ou None se indisponível."""
        mask_img, _landmarks = self.detect(frame)
        return mask_img
