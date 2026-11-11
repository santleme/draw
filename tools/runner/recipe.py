"""Receita de treino como dado.

Um candidato do runner do Draw e uma receita declarativa. A receita diz
O QUE rodar (script de treino, datasets, hardware, hiperparametros,
aumentos, semente); o runner diz COMO rodar. Mudar um candidato e
mudar dados, e dois candidatos se comparam campo a campo.

Campos minimos para uma receita ser executavel:
- name: slug estavel do candidato (ex.: "ink-pairs-v5-flagship").
- train_script: caminho do script de treino (o texto do kernel).
- kernel_slug: slug do kernel no Kaggle (sem o usuario).
- datasets: fontes de dados do kernel, no formato do Kaggle.
- hardware: "gpu" ou "tpu".

O resto e livro-razao da corrida: fica registrado no arquivo junto
com as metricas, para a comparacao ser honesta meses depois.
"""

import re
from dataclasses import asdict, dataclass, field

# Slugs viram caminho e query da API do Kaggle: só o alfabeto do
# Kaggle, sem barra, espaço, "?" ou "&".
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,49}$")
_DATASET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,49}/[a-z0-9][a-z0-9-]{1,49}$")
_MAX_DATASETS = 50


@dataclass
class Recipe:
    name: str
    train_script: str
    kernel_slug: str
    datasets: list
    hardware: str = "gpu"
    epochs: int = 0
    augmentations: dict = field(default_factory=dict)
    hyperparams: dict = field(default_factory=dict)
    seed: int = 0
    notes: str = ""

    def validate(self):
        if self.hardware not in ("gpu", "tpu"):
            raise ValueError(f"hardware invalido: {self.hardware!r}")
        for name in ("name", "train_script", "kernel_slug"):
            if not getattr(self, name):
                raise ValueError(f"receita sem {name}")
        if not isinstance(self.kernel_slug, str) or not _SLUG_RE.match(
            self.kernel_slug
        ):
            raise ValueError(f"kernel_slug invalido: {self.kernel_slug!r}")
        if not isinstance(self.datasets, list) or len(self.datasets) > _MAX_DATASETS:
            raise ValueError(f"datasets precisa ser lista de ate {_MAX_DATASETS}")
        for source in self.datasets:
            if not isinstance(source, str) or not _DATASET_RE.match(source):
                raise ValueError(f"dataset invalido (esperado dono/slug): {source!r}")
        return True

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("receita precisa ser um objeto JSON")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})
