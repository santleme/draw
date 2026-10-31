# QA — Segmentação neural v1 do Draw

Papel deste documento: checklist de QA independente. Este agente NÃO
corrige arquivos de outros agentes; divergências são apontadas aqui
para o agente dono corrigir. Escopo verificado: só dados, geração
sintética, treino, validação, ONNX e documentação. Nada de webcam,
integração no app, tracking, memória, oclusão em tempo real,
interface, CAD ou simuladores — isso é do Codex, depois.

## Contrato fixo (a régua)

- Classes, nesta ordem, sem reordenar: 0 = background, 1 = paper,
  2 = hand, 3 = ink.
- Entrada ONNX: RGB, 512×512, normalização ImageNet
  (mean 0.485/0.456/0.406, std 0.229/0.224/0.225), igual a `draw/neural.py`.
- Saída ONNX v1: logits `[1, 4, 512, 512]`; máscaras por argmax.
- Regra de rótulo visível: o que a câmera vê. Mão sobre traço = hand,
  nunca ink. `ink_full_mask` NÃO é classe de treino, é só para avaliar
  a memória futura.
- `paper_presence`: decisão documentada — derivado da máscara de paper,
  não cabeça auxiliar (ver INTEGRATION.md).
- Splits por fonte original (mão/folha/fundo), 80/10/10, nunca split
  aleatório depois de gerar.
- Métricas: IoU/Dice/Precision/Recall por classe, nunca só accuracy
  global; ink com tolerância espacial 2–3 px, sempre ao lado da IoU
  exata, nunca substituindo-a.
- Bateria fixa: 15 cenários (folha limpa, com desenho, muito desenhada,
  mão cobrindo canto/centro/traço, mão parcialmente fora, fundo claro,
  fundo escuro, sombra forte, folha parcialmente fora, mão sem folha,
  objeto branco sem folha, traço fino de lápis, traço grosso de caneta).

## Checklist

- [ ] Classes 0–3 idênticas em gerador, treino, métricas, ONNX e docs
- [ ] Máscaras de treino só com valores 0..3, um canal, PNG
- [ ] Entrada 512×512 + ImageNet em treino, ONNX e docs de integração
- [ ] Mão sobre traço rotula hand em gerador, docs e testes
- [ ] `ink_full` separado, nunca como classe nem entrada de treino
- [ ] Split por fonte, com seeds e IDs registrados, sem vazamento
- [ ] Negativos presentes (mesa vazia, mão sem folha, objeto branco,
      retângulo claro, livro, caixa, monitor, folha pequena/fora)
- [ ] Métricas por classe + ink com tolerância 2–3 px documentada
- [ ] Bateria de 15 cenários com predições salvas por versão
- [ ] Validação ONNX vs PyTorch (diferença máxima + IoU) registrada
- [ ] Holdout real separado, nunca no treino, métricas separadas
- [ ] Nenhum arquivo fora do escopo tocado (app, neural.py, webcam, CAD)
- [ ] Testes rodando sem GPU e sem webcam

## Resultados — passada final

Verificado em 2026-10-29, branch `feature/neural-seg-v1`, arquivos ainda
não commitados pelos agentes (trabalho em andamento, sem commit do QA).
Primeira passada: gerador e testes ainda ausentes; segunda passada, após
o gerador aparecer, todos os testes abaixo foram rodados.

### Arquivos encontrados (passada final)

- `tools/seg_metrics.py` — presente
- `tools/validate_onnx.py` — presente
- `tools/gen_synthetic.py` — presente (segunda passada)
- `tools/check_freihand.py` — presente
- `training/train.py` — presente
- `training/config.json` — presente
- `notebooks/train_segformer_kaggle.ipynb` — presente (14 células)
- `tests/test_gen_synthetic.py` — presente (10 testes)
- `tests/test_seg_metrics.py` — presente (13 testes)
- `docs/neural-seg-v1/INTEGRATION.md` — presente
- `docs/neural-seg-v1/README.md` — presente
- `docs/neural-seg-v1/HAND_SOURCES_ALT.md` — presente (agente de dataset)
- `docs/neural-seg-v1/KAGGLE_RUNBOOK.md` — presente
- `docs/neural-seg-v1/DATASET_FREIHAND.md` — AUSENTE até o fim desta
  passada (agente de dataset ainda trabalhando; `check_freihand.py`
  já existe e referencia esse documento)

### Consistência verificada por leitura

- Classes: `seg_metrics.py` (`CLASSES = background, paper, hand, ink`,
  0–3), `train.py` (`CLASSES = {0: background, 1: paper, 2: hand,
  3: ink}`), `config.json` (mesma ordem), INTEGRATION.md e README.md
  (mesma ordem) — CONSISTENTE.
- Shape/normalização: INTEGRATION.md (entrada `[1, 3, 512, 512]`,
  ImageNet mean/std idênticos a `draw/neural.py`, saída
  `[1, 4, 512, 512]`), `config.json` (512×512, mesma normalização),
  `train.py` (mesma normalização) — CONSISTENTE.
- Mão sobre traço: `seg_metrics.py` (docstring: pixel visível é hand,
  nunca ink), `train.py` (mesma regra), `config.json` (mesma regra),
  INTEGRATION.md (mesma regra) — CONSISTENTE.
- `paper_presence`: INTEGRATION.md e `config.json` concordam — derivado
  da máscara de paper, não cabeça auxiliar; modelo de cabeça única.
- Ink com tolerância: `seg_metrics.py` implementa dilatação por disco
  (padrão 3 px, dentro da faixa 2–3 px); `train.py` implementa dilatação
  por cv2 nos raios 2 e 3; `config.json` descreve a mesma metodologia
  (dilatação, raios 2 e 3). Metodologias equivalentes, implementações
  diferentes (NumPy puro vs cv2) — aceitável, mas os valores devem ser
  comparados no mesmo caso de teste antes de misturar números dos dois
  módulos. Ver seção de divergências.
- Bateria de 15 cenários: `seg_metrics.py` tem exatamente os 15 IDs do
  projeto, na ordem — CONSISTENTE.
- Classe ausente nos dois lados: `seg_metrics.py` e `train.py` usam a
  convenção 0/0 = 1.0 — CONSISTENTE (afeta a média; `mean_iou_present`
  existe em `seg_metrics.py` para compensar).

## Testes rodados

Sem GPU, sem webcam, no venv do Draw:

```
.venv/bin/python -m pytest tests/test_gen_synthetic.py tests/test_seg_metrics.py -q
23 passed in 2.01s
```

São 10 testes do gerador (determinismo com mesma seed, máscara só com
classes 0–3, mão sobre traço vira hand e não ink, `ink_full` contém
visível + ocluído, negativos sem paper, split sem vazamento de fonte,
geração em diretório temporário, mão placeholder marcada, cena com duas
mãos, diretório de mãos vazio) e 13 de métricas/ONNX (predição perfeita
= 1.0, predição vazia = 0 para classes presentes, ink deslocado 2 px
passa na tolerância mas não na IoU exata, casos vazios, máscaras
inválidas, bateria com exatamente 15 cenários, salvar predições,
agregado, comparação de saídas idênticas, shape errado, ONNX dummy
idêntico ok, degradado sinalizado, skipped sem bibliotecas).

Verificação direta adicional, fora dos testes dos agentes:

```
seg_metrics: classes ('background', 'paper', 'hand', 'ink'), 15 cenários
perfeito mean_iou = 1.0
vazio vs gt com ink: iou ink = 0.0
ink deslocado 2 px: iou ink = 0.0, tolerância 3 px: precision = recall = f1 = 1.0
train.py: CLASSES {0: background, 1: paper, 2: hand, 3: ink},
  ImageNet mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
  ink com tolerância raios 2 e 3: precision = recall = 1.0 no mesmo caso
```

Gerador rodado pelo QA em `/tmp/draw-qa-gen` (não no repo):

```
.venv/bin/python tools/gen_synthetic.py --out /tmp/draw-qa-gen --train 4 --val 2 --test 2
8 cenas, máscaras 512×512 com valores só em [0, 1, 2, 3]
arquivos por cena: scene_*.png, scene_*_mask.png, scene_*_ink_full.png
```

## Divergências encontradas

1. **BLOQUEANTE — layout do dataset gerado ≠ layout esperado pelo
   treino.** O gerador escreve, por split, arquivos planos:
   `<split>/scene_*.png`, `<split>/scene_*_mask.png`,
   `<split>/scene_*_ink_full.png`. Já `training/train.py`
   (`pair_images_masks`) e `training/config.json` (seção `layout`) e o
   notebook (que conta `DATA / split / "images"`) esperam
   `<split>/images/<nome>.png`, `<split>/masks/<mesmo-nome>.png` e
   `<split>/ink_full/<nome>.png`. Sem uma etapa de conversão ou um
   ajuste em um dos lados, `pair_images_masks(DATA / "train")` falha
   com "split sem images/ ou masks/". Os testes do gerador passam
   porque testam o layout do próprio gerador, não o do treino.
   Dono da correção: agente de geração ou de treino, em acordo — não
   corrigido por este QA.
2. **Resolvido na verificação — duas implementações da métrica de ink
   com tolerância.** `seg_metrics.py` (NumPy, disco euclidiano, padrão
   3 px) e `train.py` (cv2, elipse, raios 2 e 3) deram o mesmo
   resultado no caso testado (ink deslocado 2 px: precision e recall
   1.0 em ambas). Manter a observação: comparar novamente se qualquer
   uma mudar, antes de misturar números dos dois módulos.
3. **Pendente — `DATASET_FREIHAND.md` ausente.** O agente de dataset
   já entregou `tools/check_freihand.py` (que espera
   `training/rgb`, `training/mask` e `evaluation/rgb`, máscara 224×224,
   e alerta para vazamento por variante de fundo) e
   `HAND_SOURCES_ALT.md` (FreiHAND como 1ª escolha se `training/mask/`
   existir, EgoHands como reserva nº 1, RHD como reserva nº 2, resto
   não usar até confirmar licença), mas o relatório de auditoria
   principal ainda não estava no repo ao fim desta passada. Sem ele,
   a decisão final "FreiHAND direto vs alternativa" não está fechada
   neste QA.
4. **Observar — `save_predictions` vs métricas.** `seg_metrics.py`
   documenta que o manifesto registra o que foi salvo e que as
   métricas devem ser salvas pelo chamador junto ao manifesto; se
   nenhum chamador fizer isso, a comparação entre versões perde as
   métricas. O notebook gera `metrics.json` no treino, mas a ligação
   entre `predictions.json` da bateria e esse `metrics.json` deve ser
   confirmada na revisão do treino completo.
5. **Observar — split com poucas fontes.** O gerador documenta que,
   com poucas fontes, tudo vai para train e nenhum outro split é
   gerado a partir delas; isso é honesto, mas significa que um dataset
   pequeno de demonstração não prova o split 80/10/10. Exigir, no
   dataset real, fontes suficientes por split antes de aceitar o
   manifesto.

## Fora do escopo — verificado

Nenhum dos arquivos lidos nesta passada toca em webcam, `draw/app.py`,
`draw/neural.py`, tracking, memória, interface, CAD ou simuladores.
Os arquivos de outros agentes são apenas lidos aqui, nunca editados.
