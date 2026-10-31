# Ambiente de treino e compatibilidade

Este documento descreve o contrato atual do pipeline. O treino completo ainda
não foi executado nesta proposta; as dependências e os checks locais não
substituem uma execução com GPU e dados reais.

## Ambiente reproduzível

Use Python 3.11 ou 3.12 e as versões diretas fixadas em
`training/requirements.txt`. O notebook não instala pacotes: falha no preflight
se uma dependência obrigatória faltar ou tiver versão diferente. Para `torch`,
instale a wheel de versão `2.11.0` compatível com o runtime CUDA da máquina.

| Pacote | Versão | Uso |
|---|---:|---|
| torch | 2.11.0 | Modelo, treino, AMP e exportação |
| transformers | 4.57.1 | SegFormer-B0 pré-treinado |
| albumentations | 2.0.8 | Augmentations pareadas de imagem e máscara |
| numpy | 1.26.4 | Dados e métricas |
| opencv-contrib-python | 4.11.0.86 | Imagens, máscaras e pré-processamento |
| onnx | 1.23.0 | Exportação |
| onnxruntime | 1.31.1 | Validação do artefato exportado |

O notebook recebe `training/` e os datasets como inputs do Kaggle; não clona o
repositório nem baixa dependências. O backbone `nvidia/mit-b0` precisa estar em
cache ou acessível na primeira chamada de `from_pretrained`.

## O que o código executa

- Augmentation geométrica aplica os mesmos parâmetros à imagem e à máscara;
  máscaras usam nearest-neighbor. Fotometria altera somente a imagem.
- O treino usa bf16 em CUDA quando disponível, fp16 escalado como fallback e
  fp32 em CPU. `gradient_accumulation_steps` controla os passos do otimizador.
- O scheduler faz warmup linear seguido de decaimento cosseno, em passos do
  otimizador.
- O split `val` dirige early stopping e seleção de checkpoint. O split `test`
  é avaliado uma vez depois de carregar o melhor checkpoint e não participa da
  seleção.
- O wrapper de exportação redimensiona logits do SegFormer para a resolução de
  entrada. Para entrada 512×512, o contrato é `image` `[N,3,512,512]` e
  `logits` `[N,4,512,512]`.
- A validação ONNX compara shape, diferença máxima absoluta dos logits e IoU
  mínima de argmax. `ok` passou os limites, `degraded` falhou, `skipped` não
  conseguiu comparar. `skipped` nunca significa que o ONNX está validado.
- Métricas do treino usam `tools/seg_metrics.py`, inclusive pixels 255 e
  tolerância de tinta agregada sobre todas as imagens avaliadas.

## Limites conhecidos

- `last.pt` e `best.pt` salvam pesos do modelo, mas o pipeline ainda não retoma
  optimizer, scheduler ou estado de época após interrupção.
- O dataset real atual tem 719 frames de duas sessões, usados apenas no treino.
  As máscaras são pseudo-rótulos com revisão em miniaturas; bordas finas ainda
  não foram revisadas pixel a pixel. Não existe holdout real independente.
- Não foi executado treino completo nem produzido um modelo nesta proposta.
  Resultados de treino com os pseudo-rótulos precisam ser marcados como
  exploratórios; use uma nova sessão real para avaliação independente.
- GPU, cota do Kaggle e cache dos pesos precisam ser confirmados no runtime da
  execução. Um smoke CPU não comprova uso de GPU nem qualidade do modelo.
