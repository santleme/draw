"""Escrita vetorial por texto; complementa o simulador procedural.

Cada spec tem o contrato de gen_synthetic._plan_strokes:
(kind, cor BGR, espessura, geometria resolvida). Uma spec é um movimento
contínuo da caneta; pontos, acentos e ligaduras são traços separados.
"""

from __future__ import annotations

import unicodedata

import numpy as np

# Pontos de controle em unidades de altura de letra, y cresce para baixo.
# '/' levanta a caneta. Laços retornam ao início sem fechar outras hastes.
_PATHS = {
    "a": "0.65,0.25 0.4,0.1 0.05,0.4 0.1,0.95 0.55,0.9 0.65,0.25 0.6,1 0.85,0.85",
    "b": "0.1,1 0.35,-0.5 0.1,-0.6 0.1,0.9 0.6,0.85 0.65,0.35 0.35,0.2 0.1,0.5",
    "c": "0.65,0.3 0.4,0.15 0.05,0.45 0.1,0.9 0.4,1 0.7,0.8",
    "d": "0.65,0.3 0.3,0.15 0.05,0.55 0.2,1 0.6,0.8 0.8,-0.6 0.6,-0.5 0.6,1 0.9,0.85",
    "e": "0.05,0.6 0.6,0.5 0.55,0.2 0.2,0.25 0.05,0.7 0.3,1 0.7,0.8",
    "f": "0.05,1.5 0.25,0.3 0.5,-0.6 0.25,-0.7 0.1,0 0.3,1 0.1,1.4/0,0.3 0.65,0.3",
    "g": "0.65,0.3 0.3,0.15 0.05,0.5 0.15,0.95 0.6,0.8 0.65,0.25 0.5,1.5 0.1,1.7 0,1.4 0.8,1",
    "h": "0.05,1 0.35,-0.6 0.1,-0.6 0.05,1 0.3,0.25 0.55,0.3 0.6,1 0.8,0.85",
    "i": "0.1,0.3 0.05,0.9 0.2,1 0.4,0.85/0.18,-0.1 0.2,-0.08",
    "j": "0.35,0.3 0.25,1.4 0.05,1.6 -0.1,1.35/0.4,-0.1 0.42,-0.08",
    "k": "0.05,1 0.3,-0.6 0.1,-0.6 0.05,1/0.65,0.2 0.1,0.65 0.4,0.6 0.65,1",
    "l": "0.05,0.9 0.4,-0.4 0.25,-0.65 0.05,-0.2 0.05,0.8 0.25,1 0.5,0.85",
    "m": "0,0.3 0.05,1 0.2,0.25 0.4,0.3 0.4,1 0.6,0.25 0.8,0.3 0.8,1 1,0.85",
    "n": "0,0.3 0.05,1 0.3,0.25 0.55,0.3 0.6,1 0.8,0.85",
    "o": "0.6,0.3 0.3,0.15 0.05,0.5 0.15,0.95 0.5,0.95 0.65,0.55 0.6,0.3 0.8,0.45",
    "p": "0.15,0.3 0,1.65 0.2,0.45 0.5,0.2 0.7,0.5 0.55,0.9 0.15,0.9",
    "q": "0.65,0.25 0.3,0.15 0.05,0.5 0.15,1 0.6,0.8 0.65,0.25 0.5,1.6 0.85,1.35",
    "r": "0,0.4 0.05,1 0.25,0.2 0.4,0.15 0.6,0.4",
    "s": "0.65,0.3 0.4,0.1 0.1,0.3 0.3,0.6 0.6,0.8 0.35,1 0.05,0.85",
    "t": "0.3,-0.3 0.15,0.85 0.3,1 0.6,0.85/0,0.3 0.65,0.3",
    "u": "0.1,0.25 0.05,0.85 0.2,1 0.6,0.25 0.55,0.95 0.8,0.85",
    "v": "0.05,0.25 0.25,1 0.7,0.25 0.8,0.4",
    "w": "0.05,0.25 0.15,1 0.45,0.35 0.55,1 0.95,0.25 1.05,0.4",
    "x": "0.05,0.25 0.65,1/0.7,0.25 0.05,1",
    "y": "0.1,0.25 0.05,0.9 0.25,1 0.65,0.25 0.45,1.5 0.15,1.7 0,1.4 0.75,1",
    "z": "0.05,0.25 0.7,0.25 0.05,1 0.7,1",
    "0": "0.4,0 0.05,0.25 0.05,0.8 0.4,1 0.7,0.75 0.7,0.2 0.4,0",
    "1": "0.1,0.25 0.4,0 0.4,1/0.1,1 0.7,1",
    "2": "0.05,0.2 0.35,0 0.7,0.25 0.05,1 0.75,1",
    "3": "0.05,0.1 0.65,0.05 0.4,0.5 0.7,0.75 0.35,1 0.05,0.9",
    "4": "0.55,0 0,0.65 0.75,0.65/0.55,0 0.55,1",
    "5": "0.7,0 0.1,0 0.05,0.5 0.5,0.45 0.7,0.75 0.35,1 0.05,0.9",
    "6": "0.65,0.05 0.25,0.15 0.05,0.7 0.25,1 0.65,0.85 0.65,0.5 0.1,0.5",
    "7": "0,0 0.75,0 0.25,1",
    "8": "0.4,0.5 0.05,0.2 0.35,0 0.7,0.2 0.4,0.5 0.05,0.8 0.35,1 0.7,0.8 0.4,0.5",
    "9": "0.65,0.5 0.15,0.5 0.05,0.2 0.4,0 0.7,0.25 0.55,0.85 0.1,1",
    ".": "0.15,0.95 0.18,1",
    ",": "0.2,0.9 0.05,1.25",
    ":": "0.15,0.35 0.18,0.4/0.15,0.95 0.18,1",
    ";": "0.15,0.35 0.18,0.4/0.2,0.9 0.05,1.25",
    "!": "0.2,0 0.15,0.65/0.15,0.95 0.18,1",
    "?": "0.05,0.2 0.35,0 0.65,0.25 0.3,0.6 0.3,0.7/0.3,0.95 0.33,1",
    "-": "0.05,0.55 0.65,0.55",
    "_": "0,1.15 0.7,1.15",
    "+": "0,0.5 0.7,0.5/0.35,0.15 0.35,0.85",
    "=": "0,0.35 0.7,0.35/0,0.7 0.7,0.7",
    "(": "0.4,-0.1 0.1,0.5 0.4,1.1",
    ")": "0.1,-0.1 0.4,0.5 0.1,1.1",
    "'": "0.2,-0.2 0.1,0.1",
    '"': "0.1,-0.2 0.05,0.1/0.4,-0.2 0.35,0.1",
    "/": "0,1.1 0.7,-0.1",
}
_ACCENTS = {
    "\u0301": "0.25,-0.15 0.5,-0.4",  # agudo
    "\u0300": "0.5,-0.15 0.25,-0.4",
    "\u0302": "0.1,-0.15 0.35,-0.4 0.6,-0.15",
    "\u0303": "0.05,-0.2 0.25,-0.35 0.45,-0.15 0.65,-0.3",
    "\u0308": "0.15,-0.2 0.17,-0.18/0.5,-0.2 0.52,-0.18",
    "\u0327": "0.35,1 0.2,1.2 0.45,1.35 0.15,1.45",
}


def _paths(encoded):
    return [
        np.array([[float(v) for v in p.split(",")] for p in path.split()])
        for path in encoded.split("/")
    ]


def _smooth(points, rng, jitter):
    """Catmull–Rom, com perturbação lenta que preserva as extremidades."""
    padded = np.vstack([points[0], points, points[-1]])
    segments = []
    for i in range(1, len(padded) - 2):
        a, b, c, d = padded[i - 1 : i + 3]
        t = np.linspace(0, 1, max(4, int(np.linalg.norm(c - b))), endpoint=False)
        t = t[:, None]
        segments.append(
            0.5
            * (
                2 * b
                + (-a + c) * t
                + (2 * a - 5 * b + 4 * c - d) * t**2
                + (-a + 3 * b - 3 * c + d) * t**3
            )
        )
    curve = np.vstack([*segments, points[-1:]])
    u = np.linspace(0, 1, len(curve))
    noise = rng.normal(0, jitter, (5, 2))
    curve += (
        np.column_stack(
            [np.interp(u, np.linspace(0, 1, 5), noise[:, axis]) for axis in range(2)]
        )
        * np.sin(np.pi * u)[:, None]
    )
    return np.rint(curve).astype(np.int32)


def plan_text_strokes(
    text: str,
    rng: np.random.Generator,
    w: int,
    h: int,
    *,
    height: float = 22,
    margin: int = 25,
    color: tuple[int, int, int] = (25, 25, 30),
    thick: int = 2,
    jitter: float = 0.35,
    ligatures: bool = True,
) -> list:
    """Resolve texto latino em specs ``curve`` aceitas por ``_apply_stroke``.

    Maiúsculas usam a forma cursiva minúscula ampliada. Suporta dígitos,
    pontuação da tabela e acentos portugueses (NFD/NFC). Quebra na margem
    por caractere e respeita novas linhas; texto que não cabe ou glifo
    desconhecido falha explicitamente, sem truncar nem substituir letras.
    """
    if (
        not np.isfinite(height)
        or height < 6
        or margin < 0
        or thick < 1
        or not np.isfinite(jitter)
        or not 0 <= jitter <= 2
        or w <= 2 * margin + 2 * height
        or h <= 2 * margin + 3 * height
    ):
        raise ValueError("dimensões, espessura ou jitter inválidos para a escrita")
    text = unicodedata.normalize("NFC", text)
    specs = []
    x, y = float(margin), margin + height
    previous = None

    def emit(points):
        geo = _smooth(points, rng, jitter)
        if (
            np.any(geo < thick)
            or np.any(geo[:, 0] >= w - thick)
            or np.any(geo[:, 1] >= h - thick)
        ):
            raise ValueError("texto excede a área útil da folha")
        specs.append(("curve", color, thick, geo))
        return geo

    for char in text:
        if char == "\r":
            continue
        if char == "\n":
            x, y, previous = float(margin), y + height * 2.8, None
            continue
        if char in (" ", "\t"):
            x += height * (2.4 if char == "\t" else 0.8)
            previous = None
            continue
        decomposed = unicodedata.normalize("NFD", char)
        base, marks = decomposed[0].lower(), decomposed[1:]
        if base not in _PATHS or any(mark not in _ACCENTS for mark in marks):
            raise ValueError(f"caractere não suportado: {char!r}")
        paths = _paths(_PATHS[base])
        size = height * (1.2 if char.isupper() else 1.0)
        advance = (max(float(p[:, 0].max()) for p in paths) + 0.3) * size
        if x + advance + thick >= w - margin:
            x, y, previous = float(margin), y + height * 2.8, None
        if y + 1.8 * size + thick >= h - margin:
            raise ValueError("texto excede a área útil da folha")
        # Inclinação caligráfica comum; mantém a identidade dos glifos.
        transformed = []
        for path in paths:
            points = path.copy()
            points[:, 0] += (1 - points[:, 1]) * 0.12
            transformed.append(points * size + (x, y))
        if ligatures and previous is not None and base.isalpha():
            start, end = previous.astype(float), transformed[0][0]
            emit(np.array([start, (start + end) / 2 + (0, size * 0.12), end]))
        body = [emit(points) for points in transformed]
        previous = body[0][-1] if base.isalpha() else None
        for mark in marks:
            for path in _paths(_ACCENTS[mark]):
                emit(path * size + (x, y))
        x += advance
    return specs
