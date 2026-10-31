# Execução do treinamento no Kaggle

Este roteiro descreve a execução do pipeline em `training/train.py`. O código e
`training/config.json` são a fonte do contrato; este documento não afirma que um
treino completo ou um modelo já tenham sido produzidos.

## Preparar o ambiente

Use Python 3.11 ou 3.12 e as dependências de `training/requirements.txt`, com a
wheel de PyTorch compatível com a GPU CUDA do runtime. O notebook faz preflight
das versões e falha se alguma dependência direta estiver ausente ou divergente.
O backbone `nvidia/mit-b0` precisa estar em cache ou acessível na primeira
execução.

Anexe ao notebook:

- o dataset sintético com `train/`, `val/`, `test/` e, se disponível,
  `split_manifest.json`;
- o pacote privado de pseudo-rótulos reais, quando disponível. As duas sessões
  reais atuais entram apenas no treino por `--real-train-dir`.

Não use essas duas sessões como holdout. As máscaras reais atuais são propostas
automáticas e ainda precisam de revisão pixel a pixel; resultados com elas são
exploratórios. Reserve uma nova sessão real para avaliação independente.

## Executar

Confirme que a GPU aparece no runtime, que os caminhos dos datasets existem e
que `training/config.json` está disponível. O comando abaixo usa os
hiperparâmetros da config:

```bash
python training/train.py \
  --config training/config.json \
  --data-dir /kaggle/input/<dataset>/dataset/synthetic \
  --real-train-dir /kaggle/input/<real-data>/pseudo_training_candidate \
  --out-dir /kaggle/working/out
```

O argumento `--real-train-dir` é opcional. Para verificar pareamento, máscaras
com `255`, loss e estrutura sem treinar, rode:

```bash
python training/train.py --config training/config.json --smoke
```

O treino usa augmentation geométrica pareada, nearest-neighbor nas máscaras,
warmup linear e decaimento cosseno por atualização do otimizador, acumulação de
gradiente conforme a config e bf16/fp16 em GPU ou fp32 em CPU. Validação dirige
early stopping e seleção do melhor checkpoint. O split test é calculado uma
única vez após a seleção; nunca use test para ajustar pesos ou limiares.

## Artefatos e critérios

O diretório de saída contém checkpoints, `classes.json`, `metrics.json`,
`training_config.json`, `split_manifest.json` e, se a exportação funcionar,
`draw_segmentation_v1.onnx`. O contrato ONNX é entrada `image` `[N,3,512,512]`
e logits `logits` `[N,4,512,512]`; o wrapper redimensiona a saída SegFormer
para a resolução da entrada.

`metrics.json` registra métricas por classe, pixels ignorados, ink com
tolerância espacial agregada sobre todas as imagens, métricas do split test e
estado da validação ONNX. A validação compara shape, diferença máxima dos
logits e IoU mínima de argmax contra PyTorch. `skipped`, `failed` ou `degraded`
não significa que o ONNX foi validado.

Não publique alegações de qualidade real sem revisão dos rótulos e avaliação
em holdout real independente. O pipeline e os testes locais não substituem um
treino completo em GPU, a conferência do artefato ONNX nem a avaliação visual
dos 15 cenários definidos pelo projeto.
