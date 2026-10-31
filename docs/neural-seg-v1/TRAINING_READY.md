# Estado operacional do treinamento

> **Código preparado para revisão; treino completo e qualidade do modelo ainda
> não validados.** Este PR entrega ferramentas e contrato de execução, não um
> checkpoint nem métricas de desempenho.

## Pré-requisitos

1. Python 3.11 ou 3.12 com os pacotes fixados em `training/requirements.txt`.
2. Dataset sintético em `train/`, `val/` e `test/`, cada split com pares de
   imagem e máscara. O split test é mantido fora da seleção de checkpoint.
3. Se usar dados reais, passe a pasta de máscaras candidatas com
   `--real-train-dir`; o diretório contém somente pares destinados ao treino.
4. Pesos `nvidia/mit-b0` acessíveis via cache ou internet na primeira carga.
5. Para treino completo, uma GPU compatível; o notebook imprime dispositivo e
   versões antes de iniciar.

## Comandos

Smoke de estrutura, máscaras 255, loss e contrato de saída:

```bash
python training/train.py --config training/config.json --smoke
```

Treino completo com avaliação sintética separada:

```bash
python training/train.py \
  --config training/config.json \
  --data-dir dataset/synthetic \
  --real-train-dir captures/triage/pseudo_training_candidate \
  --out-dir out
```

O diretório real é opcional. Sem ele, o treino usa apenas dados sintéticos.

## Artefatos e leitura dos resultados

O pipeline grava `metrics.json`, `training_config.json`, `split_manifest.json`,
`classes.json`, checkpoints e, quando exportável, `draw_segmentation_v1.onnx`.
`metrics.json` contém métricas de validação no nível superior e métricas do
split test no campo `test`. O ONNX tem status de validação explícito; `skipped`
ou `failed` não pode ser descrito como validado.

Antes de alegar desempenho em dados reais, revise as máscaras pixel a pixel,
execute os testes e o preflight de todos os pares, treine e reserve uma nova
sessão real como holdout. Consulte `REAL_DATA_STATUS.md` para o estado da
tranche de captura atual.
