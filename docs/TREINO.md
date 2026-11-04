# Treino, avaliação e exportação

Este guia cobre os scripts versionados. Eles não são uma reprodução dos
binários históricos `draw_seg_v3_dyn.onnx` e `page_clean_unet.onnx`: o
primeiro treina o SegFormer de quatro classes; o segundo é uma rodada de
candidato U-Net. Nenhum candidato entra em produção sem os gates reais.

## SegFormer: quatro classes

```bash
python training/train.py \
  --config training/config.json \
  --data-dir dataset/synthetic \
  --out-dir out
```

Para acrescentar o pacote real de **propostas não revisadas** é necessário
assumir isso explicitamente:

```bash
python training/train.py \
  --config training/config.json \
  --data-dir dataset/synthetic \
  --real-train-dir captures/triage/pseudo_training_candidate \
  --allow-unverified-real-train \
  --out-dir out
```

Essas amostras têm peso `0.25` por padrão
(`loss.unverified_real_sample_weight`), contra `1.0` das sintéticas. Use
`--real-loss-weight` para registrar outra escolha antes da rodada. Esse
peso não transforma proposta em ground truth e não substitui holdout real.
Validação e teste continuam exclusivamente sintéticos.

O treinador agora falha diante de máscara ausente, imagem/máscara com
dimensões diferentes ou fontes repetidas entre train/val/test no manifesto.
`split_manifest.json` e o `manifest.json` nativo do gerador são aceitos.

### Entrega ONNX é um gate

`checkpoints/best.pt`, `last.pt` e `metrics.json` são preservados mesmo se
a exportação falhar. Porém o comando só retorna `0` quando PyTorch e ONNX
foram efetivamente executados e aprovados. `skipped`, `failed` ou
`degraded` grava `delivery_status=blocked_onnx_validation` e retorna `3`.
O tensor do PyTorch em CUDA é copiado para CPU antes do ONNX Runtime.

### Ablation 384/256 sem retreinar

Carregue o mesmo checkpoint e escolha a resolução na **validação**:

```bash
python tools/export_segformer.py \
  --checkpoint out/checkpoints/best.pt \
  --data-dir dataset/synthetic \
  --size 256 --out-dir out/export_256
```

Repita para 384. O modelo roda na resolução pedida, mas as predições são
projetadas para o canvas original da config (512) antes de IoU e tolerância
de tinta. Assim, `2 px` tem o mesmo significado nas comparações. Depois da
escolha, rode a versão escolhida mais uma vez com `--include-test`.

O script não modifica pesos nem retreina. Exportar FP32 não é quantização;
bf16/fp16 no treino em CUDA não muda isso. O detector histórico dinâmico de
produção é outro artefato e continua exigindo `tools/eval_real_frames.py`.

## FreiHAND e geração sintética

`tools/gen_synthetic.py` binariza máscaras com `pixel > 127`, evitando que
ringing JPEG vire mão. Os arquivos das quatro variantes de fundo são
agrupados pelo id da mão base (`indice % 32560`) antes do split. O manifesto
grava `hand_base_splits`, `hand_mask_threshold` e `hand_base_source` por
cena; o treinador rejeita sobreposição entre splits.

## U-Net: rodada self-supervised com holdout selado

Cada diretório labeled/unlabeled/holdout usa:

```
input/<nome>.png
mask/<nome>.png       # labeled e holdout
manifest.jsonl
```

Cada linha do manifesto contém `file`, `session_id`, `origin` e `split`.
No holdout, o rótulo precisa ter:

```json
{
  "origin": "independent_label",
  "split": "holdout",
  "label_source": "final_minus_reference",
  "label_reviewed": true
}
```

As imagens são lidas em **BGR / 255**, exatamente como `PageCleanInk`.
Arquivo sem manifesto, máscara divergente ou sessão de holdout no treino
interrompe a rodada.

```bash
python tools/selflearn/train_round.py \
  --labeled-dir dados/labeled \
  --unlabeled-dir dados/unlabeled \
  --holdout-dir dados/holdout \
  --init-checkpoint rodada_anterior/round_checkpoint.pt \
  --out rodada_nova
```

Para a primeira rodada do zero, `--allow-random-init` precisa ser explícito.
O checkpoint carrega também as sessões seladas anteriormente, impedindo que
um antigo holdout reapareça como treino. A EMA usa rampa inicial e teto
`0.999`.

O holdout é avaliado **somente depois do treino**, em páginas inteiras pelo
mosaico de 256 e threshold fixo `> 0.7`. Não escolhe checkpoint nem limiar.
`metrics.json` inclui:

- loss supervisionada e de consistência, cobertura e fração de tinta dos
  pseudo-rótulos;
- IoU/Dice/precision/recall micro e por página;
- IoU macro por página e pior sessão;
- recall de traço fino e falsos positivos em páginas vazias;
- contagens do holdout e suficiência do mínimo de 5 sessões/25 páginas;
- validação numérica e de batch dinâmico do ONNX.

`candidate_real.json` pode ser comparado ao incumbente:

```bash
python tools/selflearn/judge_gate.py \
  --candidate rodada_nova/candidate_real.json \
  --baseline incumbente/candidate_real.json
```

Avaliação não é promoção: o veredito continua sujeito ao n mínimo, ao pior
grupo e à decisão humana. Cobertura de pseudo-rótulo também não prova ganho
causal; para medir o efeito do ramo não supervisionado, compare com uma
rodada equivalente sem `--unlabeled-dir`.

Smoke completo (otimização, EMA, guards, holdout, checkpoint e ONNX):

```bash
python tools/selflearn/train_round.py --smoke
```
