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
