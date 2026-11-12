# Simulador fechado de escrita

O simulador planeja texto como movimentos de caneta, escreve num papel
sintético em perspectiva e entrega imagens BGR ao mesmo laço de captura,
detecção, retificação e tinta do Draw. O ground truth só entra na avaliação
depois da captura. Nenhum estágio recebe cantos ou máscaras verdadeiros
para facilitar a detecção.

## Gerar e observar no app

Na raiz do repositório, com as dependências de `requirements.txt`:

```sh
python tools/sim_source.py --text "Olá, Draw!" --out captures/sim.mp4
python main.py --source captures/sim.mp4
```

O exportador grava `sim.mp4` (512×512, codec `mp4v`), `sim.gt.npz`,
`sim_ink_full.npy` e `sim.json`. O bundle NPZ contém o desenho completo no
espaço da câmera (`ink_full`), no espaço original do papel
(`page_ink_full`), a homografia papel→câmera e as polylines individuais.
Não usa pickle. O manifesto identifica seed, FPS, número de frames,
traços e se há mão placeholder.

No app, **s** salva o canvas confirmado em `captures/canvas_*.png`.
O app tem uma bomba de captura que pode consumir arquivos mais rápido
que a inferência e descartar frames intermediários; o arquivo permite
observar a cena, mas esse modo não é um benchmark temporal determinístico.
Use o runner headless abaixo para processar todos os frames em ordem.

O planejamento suporta letras latinas, dígitos, pontuação básica, acentos
portugueses, espaços e novas linhas. Maiúsculas usam formas cursivas
minúsculas ampliadas. Uma tabela vetorial de glifos, interpolada por
Catmull–Rom com jitter suave, gera laços e ligaduras entre letras;
pontos, acentos e outras levantadas de caneta são specs separadas.
O texto quebra na margem por caractere. Caracteres desconhecidos ou
texto que excede uma página geram erro explícito. Não é um modelo de
caligrafia humana nem uma fonte tipográfica completa.

```sh
python tools/sim_source.py --text "ação 123" --seed 7 --fps 15 \
  --stroke-seconds 0.3 --warmup-seconds 1 --settle-seconds 4 \
  --height 22 --hand --out captures/sim.mp4
```

Há um intervalo inicial sem tinta, crescimento progressivo dentro de
cada traço e um intervalo final sem mão para o acumulador confirmar o
desenho. `--hand` usa a mão sintética placeholder existente, acompanhando
a ponta da caneta; a máscara visível respeita mão sobre tinta e o
`ink_full` continua completo. FPS/tempos controlam a dinâmica. A pose
da folha é fixa, com textura, fundo e ruído de sensor do gerador original.
Movimento de papel e comportamento real de mão/modelos exigem outros
cenários; os resultados aqui são exclusivamente sintéticos.

## Cenário escrever → apagar → redesenhar → mover

A fonte também encadeia fases além da escrita: uma borracha visível
varre uma região da folha e devolve o papel original, um segundo texto
cai sobre a área limpa e a folha desliza/gira no fim. As fases são
declaradas no GT (`events`, `stroke_phase` 0 para o texto inicial e 1
para o redesenho, `stroke_erased`, `erase_rect`, `page_ink_final` com o
estado real da folha no fim e as poses inicial/final). Sem os flags, a
linha do tempo é exatamente warmup → escrita → settle, como sempre foi.

```sh
python tools/sim_source.py --text "Draw ok" --seed 7 --fps 15 \
  --stroke-seconds 0.3 --warmup-seconds 2 --settle-seconds 8 \
  --height 26 --hand --erase-stroke -1 --redraw-text "ui" \
  --erase-seconds 1.2 --hold-seconds 6 \
  --move 35,22,2.5 --move-seconds 0.8 --out captures/sim-e3.mp4
```

`--erase-stroke N` apaga a caixa do traço N (índice negativo conta do
fim); `--erase-rect x,y,w,h` apaga um retângulo explícito em pixels de
página. As duas opções são alternativas. `--redraw-text` exige uma
região apagada e centraliza o novo texto nela. `--hold-seconds` é a
parada de cena entre fases — dê passadas suficientes para a confirmação
amadurecer antes do próximo evento. `--move dx,dy,graus` desloca e gira
a folha em `move-seconds`; sem ele, não há fase de movimento.

A auditoria por fase roda o mesmo laço real do `eval_capture` e
fotografa a máscara confirmada e a provisória no último frame de cada
fase, pontuando cada uma contra o GT esperado daquele instante (traços
já aplicados, menos o retângulo já apagado, na âncora que a worker
publicava ali). O que se quer ver: recall confirmado crescendo na
escrita, resíduo confirmado na região apagada caindo depois da
borracha, recall confirmado dos traços de redesenho subindo depois da
segunda escrita — o contrato do E3 é que pixel apagado não fica
provisório para sempre — e o canvas sobrevivendo ao movimento.

```sh
python tools/eval_scenario.py --video captures/sim-e3.mp4 \
  --gt captures/sim-e3.gt.npz --out captures/eval-e3.json \
  --save-masks captures/e3-masks
```

O relatório lista os checkpoints com IoU/precisão/recall por fase,
`confirmed_in_erased_px` (resíduo na região apagada),
`redraw_confirmed_recall` e `redraw_provisional_px`. Bundles sem
`events` são recusados: cenário simples é trabalho do `eval_capture`.

Para o app nativo, o arquivo consumido direto esgota em segundos — a
bomba latest-wins descarta quase todos os frames. `tools/mjpeg_server.py`
serve o MP4 como câmera em tempo real (`--fps` nominal), e o laço
inteiro — captura assíncrona, cadência da trabalhadora, coleta de
sessão — opera como na mesa:

```sh
python tools/mjpeg_server.py --video captures/sim-e3.mp4 --port 8554 &
python main.py --source http://127.0.0.1:8554/video
```

## Cenário lift & return (folha sai de cena e volta)

`--lift dx,dy` encadeia três fases depois das anteriores: `lift_out`
(a folha desliza até sair do frame), `absent` (só a mesa na cena por
`--absent-seconds` — a tinta vai junto, fica na folha) e `return` (a
folha desliza de volta). `--return-pose dx,dy,graus` define a pose do
retorno relativa à pose de saída — rotação de verdade, como quem pega
a folha, vira e repousa. É o gesto mais comum da mesa real e o que
mais estressa a vigia de identidade: mesma página ou página nova?

```sh
python tools/sim_source.py --text "Draw v6" --fps 15 \
  --warmup-seconds 1 --settle-seconds 8 --hold-seconds 4 \
  --lift 400,250 --return-pose -50,40,20 \
  --absent-seconds 2.5 --out captures/sim-lift.mp4
```

### Limite medido do detector (2026-11-12)

A primeira auditoria deste cenário expôs um gap de dados, não de
código: a folha retorna rodada 20° e **nenhum dos seis modelos de
segmentação instalados a reconhece** — logit de presença ~-6,5 por
8 s de settle com a folha inteira visível. Girando um frame que o
modelo detecta, o envelope sai: logit positivo até ~12°, cruza zero
perto de 15°, fundo confiante a partir de 20°. O gerador de corpus
amostra rotação em `uniform(-18°, 18°)` (`_sample_normal_quad`,
tools/gen_synthetic.py) — a folha nunca existiu no treino além desse
teto, e o modelo falha logo na borda dele. Rotacionar a folha é gesto
rotineiro na mesa real; a correção é augment de rotação no corpus do
próximo treino, e este cenário vira a bancada de regressão dele.

Efeito colateral medido na janela de perda do tracker: enquanto a
folha some mas `tracker.corners` ainda está na graça
(`_SHEET_LOST_SECONDS`), as passadas seguem no quad defasado sobre
mesa — o provisório mediu 3,7k→346k px em ~1 s antes da queda fechar
a sessão. Nada confirmou (as guardas de geração seguraram), mas é
trabalho morto conhecido.

## Cenário mão em massa (palma, sweep, pegada)

A mão de instrumento (`--hand`) só existe colada na caneta/borracha —
o que o humano faz com a mão SEM instrumento nunca estressava o laço.
Três fases novas cobrem o resto:

- `--rest-hand-seconds S` — fase `rest_hand`: palma apoiada na folha
  depois de cada `hold` (e cria um hold se o cenário não tem nenhum).
  A palma pousa no centro da tinta escrita — cobre o canvas confirmado.
- `--sweep-seconds S` — fase `sweep` antes das fases de geometria: a
  mão atravessa a faixa de texto de ponta a ponta. Estressa a fronteira
  de abstenção do juiz de virada (`visible_px >= 50%` do canvas para
  decidir) — a mão cobre/descobre tinta a cada quadro.
- `--grab-hand` — a mão segura a folha (`grab_at`, borda inferior em
  espaço de página) durante `move`, `lift_out` e `return` — acompanha
  o quad: sai de cena com a folha e volta com ela. Em `absent` não há
  mão (saiu de cena junto). `--hand-scale` dimensiona o sprite de
  palma (default 1.0; o sprite de instrumento segue em 0.6).

```sh
python tools/sim_source.py --text "draw hand" --fps 15 \
  --erase-rect 60,60,300,90 --redraw-text "go" \
  --rest-hand-seconds 2.0 --sweep-seconds 1.5 \
  --move 120,60,8 --grab-hand --hand --out captures/sim-hand.mp4
```

### O que a bancada mediu (2026-11-12)

- **Palma e sweep são benignos por projeto confirmado**: o canvas
  confirmado ficou estável (14,5k→14,5k px) em três `rest_hand` e um
  `sweep` — zero eventos de virada/split. O juiz de virada abstém com
  <50% de tinta visível, como foi escrito para fazer.
- **Falha séria — detector cego a folha clipada**: `move 120,60`
  levou a folha ~10% para fora do frame 512². A partir daí: drop do
  tracker e `no_canvas` permanente — a folha estacionou 90% visível
  na mesa por 87 quadros e nunca foi re-adquirida (logit ~-4). O
  envelope medido deslocando uma cena detectável: clipe na borda
  **direita ou esquerda cega já com ~7-9% fora**; no **topo**, ~7%;
  em **baixo** tolera ~17% — assimetria consistente com viés do
  corpus (folha pendurada na borda frontal da mesa). É o mesmo tipo
  de gap do envelope de rotação: dados, não código — a bancada para
  o augment de clipe é este cenário.
- **Pegada durante o move degrada a detecção**: a palma sobre ~40%
  da folha produziu quad parcial e logit 1,17 no meio do gesto — o
  que acompanhou a queda do tracking. Todo move real tem mão.
- **Tinta fantasma congela sob palma estacionada**: o confirmado na
  região apagada ficou travado (~5,1k px) enquanto a palma cobria
  100% do retângulo — `retained` exige `visible`, e pixel sem visão
  não retrai. Semântica correta (não julga o que não vê), consequência
  real: a tinta apagada segue "viva" na tela enquanto a palma estiver
  parada sobre ela.

A auditoria por fase trata `return` como o `move`: checkpoints após o
retorno pontuam contra `page_to_scene_final`. Checkpoints de `absent`
sem âncora/canvas registram status em vez de métrica — e o veredito
final ancora no último evento declarado (sem IoU emprestado de fases
anteriores).

## Capturar e avaliar sem interface

Com os ONNX oficiais instalados (veja `tools/fetch_models.py`):

```sh
python tools/eval_capture.py --video captures/sim.mp4 \
  --gt captures/sim.gt.npz --out captures/eval.json \
  --save-canvas captures/canvas.npy
```

O runner usa `draw.loop.SheetInkLoop`, o detector combinado de `draw.app`
(`draw.neural` + reserva clássica), `draw.paper.warp_sheet`,
`draw.page_clean.PageCleanInk` e `draw.ink_layers._PageCleanWorker` com
seu acumulador real. Modelos podem ser indicados por `--neural-model` e
`--page-clean-model`; os caminhos padrão e as variáveis de ambiente são
os mesmos do app. Sem ONNX de detecção, usa o detector clássico.
Para a folha, a precedência é `--neural-model`, modelo instalado em
`models/draw_seg_v3_dyn.onnx` e, por último, `DRAW_NEURAL_MODEL`.
Sem ONNX de tinta, falha explicitamente; para exercitar o legado:

```sh
python tools/eval_capture.py --video captures/sim.mp4 \
  --gt captures/sim.gt.npz --legacy-ink --out captures/eval-legacy.json
```

O legado passa pelo mesmo laço e usa `draw.ink.extract_ink` e
`InkAccumulator`; o relatório identifica qual caminho foi usado.
Nenhum modelo é baixado automaticamente. A detecção é síncrona e a worker
real avança com `pump()` no relógio do vídeo; sua cadência e regras de
confirmação permanecem ativas. Isso avalia a qualidade funcional offline,
não a concorrência, latência ou vazão do app com webcam. `--fps` substitui
o FPS declarado pelo arquivo quando necessário.

O EOF encerra a entrada sem repetir o último frame nem criar evidência.
A detecção é consumida com o atraso de um quadro do laço real, e folha
e tinta exigem persistência temporal. Vídeos muito curtos (inclusive um
único quadro) podem resultar em `no_canvas`; mantenha os frames de
aquecimento e assentamento da fonte para avaliação.

No caminho page-clean, os adaptadores reais de mão do app também são
reutilizados: `HandMasker` na worker (visibilidade) e `_make_hand_presence`
no laço (presença/movimento e cadência), com amostragem síncrona.
`--hand-model` ou `DRAW_HAND_MODEL` indicam o arquivo opcional `.task`.
Sem modelo ou MediaPipe, a mão continua sendo sinal indisponível,
como no app; o relatório declara a configuração de visibilidade e
quantas amostras de presença estavam disponíveis. A máscara sintética
de GT nunca substitui o detector de mão. No legado, uma máscara de mão
só é solicitada quando `--hand-model` é explícito.

O relatório JSON contém IoU, precisão, recall, TP/FP/FN, alinhamento,
dimensões, número de frames/detecções/passadas e o resultado por traço:

- `captured`: recall do traço ≥ `--stroke-recall` (padrão 0.5).
- `partial`: algum pixel recuperado, abaixo do limiar.
- `lost`: nenhum pixel recuperado.
- `unscorable`: nenhum pixel do traço sobrevive na grade avaliada.

As métricas são exatas por pixel, sem dilatação ou tolerância. O GT é
projetado para a grade do canvas pela **âncora realmente usada pela
worker**, sem alinhar pelo conteúdo. Para o legado usa-se o último quad
mediano de retificação; seu canvas pode integrar pequenas variações de
pose, como no app. Interseções contam na geometria de cada traço, portanto
o relatório por traço mede cobertura espacial, não prova em qual instante
o traço foi visto. Denominador vazio vale 1; IoU de duas máscaras vazias
vale 1. Sem canvas adquirido, o relatório declara `no_canvas` e toda tinta
do GT é perdida, sem inventar uma retificação.

## Avaliar um canvas já salvo

O runner salva máscara positiva em `canvas.npy` e âncora em
`canvas.corners.npy`. Para repetir exatamente a avaliação:

```sh
python tools/eval_capture.py --canvas captures/canvas.npy \
  --corners captures/canvas.corners.npy --gt captures/sim.gt.npz
```

PNG do app é interpretado como papel branco e tinta preta; cinza
provisório (176) não é tinta confirmada. NPY é máscara positiva (ou
probabilidade ≥0.5). `--canvas-kind mask|image` permite explicitar a
convenção. O canvas do app não guarda a âncora no PNG: sem `--corners`,
o bundle permite uma comparação **ideal_page**, que pressupõe uma
retificação perfeita e pode incluir erro de pose. Prefira a âncora do
runner para comparação reproduzível.

```sh
python tools/eval_capture.py --canvas captures/canvas_123.png \
  --gt captures/sim.gt.npz
python tools/eval_capture.py --video captures/sim.mp4 \
  --gt captures/sim_ink_full.npy --legacy-ink
```

O NPY isolado traz somente `ink_full` em espaço de câmera: permite
métricas globais, mas não relatório por traço. No modo `--video`, o runner
usa o quad detectado para projetá-lo ao canvas. Para canvas já retificado
é obrigatório informar `--corners`; para uma máscara que já está na
grade 512×512 da câmera, use `--canvas-space scene`.

## Fonte ao vivo em Python

```python
from tools.sim_source import SyntheticCapture
from tools.eval_capture import run_pipeline, evaluate_canvas

camera = SyntheticCapture("Olá!", seed=42, realtime=True)
camera.save_ground_truth("sim.gt.npz")
try:
    result = run_pipeline(camera, legacy_ink=True)
    report = evaluate_canvas(result["mask"], "sim.gt.npz", corners=result["corners"])
finally:
    camera.release()
```

`SyntheticCapture` implementa `isOpened/read/release/get`, gera cada frame
sob demanda e não armazena uma sequência de imagens em memória. `get`
expõe FPS, dimensões, contagem e posição em frames/ms; propriedades não
suportadas retornam 0. EOF retorna `(False, None)`. `realtime=True` limita
a entrega ao FPS no relógio de parede; o padrão offline não espera.
Também pode ser retornada por uma factory injetada nos utilitários de
captura existentes. O app permanece inalterado e recebe o MP4 pelo
`--source` que já existe.

O simulador reutiliza `_paper_base`, `_background`, `_sample_normal_quad`,
`_apply_stroke`, `_warp_paper`, composição da mão, `_ambiguity_ignore` e
`_sequence_photometric` de `tools/gen_synthetic.py`. A função
`compose_scene_sequence` materializa uma lista inteira e escolhe suas
próprias specs aleatórias; por isso a fonte lazy compõe com as mesmas
primitivas, em vez de chamar essa função ou mudar sua assinatura.

O planejador por texto fica em `tools/sim_text.py`. Ele complementa o
simulador procedural de corpus em `tools/sim_writer.py`, sem substituir
seus comandos `generate`/`audit` nem seus cenários de laboratório.
