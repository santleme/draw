"""Normalização de luz experimental, atrás de flag, sem mudar o padrão.

Problema medido (régua sintética do Senado, não chute): sombra e luz
baixa falham no portão de textura, não na geometria. A fração de branco
cai de 0,98 (base) para 0,25 (sombra) e 0,00 (luz baixa) porque o portão
usa um limiar absoluto de valor (V > 180 no HSV). Nenhum limiar do
portão muda aqui: a imagem é normalizada antes de chegar a ele.

Métodos, sem dependência nova (só numpy + OpenCV, que já existem):

- "background": estimativa de fundo por blur gaussiano largo, dividida
  pela imagem (imagem / fundo * 255). Remove gradiente de luz e sombra
  larga, porque papel e tinta são medidos contra o fundo local, não
  contra um valor absoluto. O fundo é estimado por canal, com sigma
  proporcional ao maior lado da imagem, para valer tanto no frame da
  câmera quanto na folha retificada.
- "clahe": equalização adaptativa no canal L do LAB, com limite de
  contraste. Melhora contraste local, mas sozinha NÃO resgata sombra
  nem luz baixa (medido: continuam sem detecção), porque o valor
  absoluto continua baixo. Serve como complemento, não como solução.
- "both": fundo + divisão, depois CLAHE. Foi o melhor medido: sombra
  96,43% e luz baixa 96,30% de recall, contra sem detecção com a flag
  desligada; base 96,28% e brilho 93,79%, sem queda relevante.
- "off": devolve a imagem sem tocar nela. É o padrão, e o comportamento
  atual é byte a byte idêntico com a flag desligada.

Retinex multi-escala (log(I) − log(G∗I)) foi prototipado e descartado:
resgatava a detecção, mas derrubava o recall para ~88% e transformava
brilho em blank_maybe. Não está implementado, de propósito.

Fiação atual do app: somente "background" está exposto por
--light-normalize, e a detecção passa por draw/light_guard.py. A guarda
tenta primeiro o frame original e só conserva "background" se o
candidato normalizado passar nas verificações de papel colorido e
quadro inteiro. A tinta usa o modo vencedor na folha retificada do
frame original. "clahe" e "both" continuam disponíveis nesta
biblioteca para medições experimentais; não fazem parte da fiação do
app nem da flag CLI. Os resultados desses modos nos testes são
medições da biblioteca, não validação da fiação de produção.
"""

import cv2
import numpy as np

LIGHT_MODES = ("off", "background", "clahe", "both")

# Sigma do blur de fundo como fração do maior lado da imagem: um fundo
# tem que ser largo o bastante para não seguir a tinta, e proporcional
# para valer no frame da câmera (1600 px) e na folha (1200 px).
# O fundo é baixa frequência, então ele é estimado em resolução
# reduzida (lado maior até este teto) e reampliado: o resultado é o
# mesmo para efeito prático, e o blur largo deixa de custar segundos
# por frame em float32.
_BACKGROUND_SIGMA_FRACTION = 20.0
_BACKGROUND_MAX_SIDE = 800
# Piso do denominador da divisão, para fundo quase preto não explodir.
_BACKGROUND_EPSILON = 1.0
# CLAHE: limite de contraste e grade, os valores usuais para documento.
_CLAHE_CLIP_LIMIT = 2.0
_CLAHE_TILE_GRID = (8, 8)


def background_division(image, sigma_fraction=_BACKGROUND_SIGMA_FRACTION):
    """Remove gradiente de luz dividindo a imagem pelo seu fundo.

    O fundo é um blur gaussiano largo por canal. Papel sob sombra e
    papel claro viram ambos ~255, e a tinta continua escura contra o
    fundo local, então o portão de textura volta a enxergar branco.
    Regiões uniformes grandes (mesa, tapete) também achatam, por isso
    o portão continua decidindo por textura, não por brilho: negativos
    uniformes continuam sem veredito de papel (blank_maybe, no máximo).
    """
    if image is None or image.size == 0:
        raise ValueError("background_division precisa de uma imagem")
    height, width = image.shape[:2]
    scale = min(1.0, _BACKGROUND_MAX_SIDE / max(height, width))
    if scale < 1.0:
        small = cv2.resize(
            image, (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = image
    sigma = max(small.shape[:2]) / float(sigma_fraction)
    background = cv2.GaussianBlur(
        small.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma
    )
    if scale < 1.0:
        background = cv2.resize(
            background, (width, height), interpolation=cv2.INTER_LINEAR
        )
    normalized = (
        image.astype(np.float32)
        / np.maximum(background, _BACKGROUND_EPSILON) * 255.0
    )
    return np.clip(normalized, 0, 255).astype(np.uint8)


def clahe_light(image, clip_limit=_CLAHE_CLIP_LIMIT,
                tile_grid=_CLAHE_TILE_GRID):
    """Aplica CLAHE só no canal L (luminosidade) do LAB.

    Cor e saturação não mudam: só o contraste local da luminosidade.
    Sozinho não resgata sombra nem luz baixa (ver docstring do módulo),
    mas depois da divisão por fundo ele recupera contraste da tinta.
    """
    if image is None or image.size == 0:
        raise ValueError("clahe_light precisa de uma imagem")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    equalizer = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=tile_grid
    )
    merged = cv2.merge([equalizer.apply(lightness), a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def normalize_light(image, mode="off"):
    """Normaliza a luz de uma imagem BGR segundo o modo pedido.

    "off" devolve a imagem sem tocar nela. "background", "clahe" e
    "both" aplicam os métodos acima, nessa ordem para "both".
    Modo desconhecido levanta ValueError, para erro de flag falhar alto
    em vez de rodar silenciosamente sem normalizar.
    """
    if mode not in LIGHT_MODES:
        raise ValueError(
            f"modo de luz desconhecido: {mode!r} "
            f"(esperado um de {LIGHT_MODES})"
        )
    if mode == "off":
        return image
    if mode == "background":
        return background_division(image)
    if mode == "clahe":
        return clahe_light(image)
    return clahe_light(background_division(image))
