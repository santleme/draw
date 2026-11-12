# Taxonomia de falhas importantes do Draw

Documento de referência do projeto (frente B, 2026-11-05): o que conta
como **falha importante** no produto, para avaliar qualquer modelo ou
mudança pelo que importa na mesa — não por score agregado. Cada falha
tem: o sintoma como quem desenha vê, a causa provável na arquitetura
(com arquivo), a severidade, e **como se mede** — régua existente com
arquivo e limiar, ou `SEM RÉGUA AINDA` com o desenho da régua que falta.

Convenções:

- Severidade é do **produto** (camada 1 da `VISION.md`: captura fiel
  em tempo real), não do modelo isolado. **Bloqueante** = quebra a
  promessa central ou destrói conteúdo do usuário. **Maior** = degrada
  a captura de forma visível e recorrente. **Menor** = degrada em
  condição estreita ou com contorno conhecido.
- Limiares citados de `tools/judge.py` são os pré-registrados
  (2026-11-01/02): mudar um número depois de ver uma rodada é mover a
  trave — quem muda declara no pré-registro da rodada seguinte.
- A bancada (`~/workspace/draw-lab/judging-bench/`, scripts
  `judge_sheet.py`/`judge_ink.py`, protocolo em `JULGAMENTO.md`) só
  emite `REPROVADO_NA_BANCADA` ou `APTO_A_TESTE_AO_VIVO`; ela não
  promove ninguém. Promoção segue: bancada → holdout selado → teste
  ao vivo na mesa do Lucas → latência no Ryzen dele.

## Mapa rápido

### Insumo do holdout e tradução externa (2026-11-12)

@santleme relata revisão ruim das 25 páginas. A reconstrução somente leitura
das oito sessões originais no WSL reproduz cinco sessões/25 rascunhos com
fração 0,5; há exemplos visuais de teclado, mão e mesa. O delta de um teclado
marca 65,32% da imagem como tinta; o de uma cena com mão, 85,93%.
`final_minus_reference` mede escurecimento, não tinta semântica, e ignora
desenho já presente na primeira referência. A CLI registra decisões, não
oferece editor nem certifica distribuição. Não foram encontrados os ledgers
de revisão: não inventar contagens de aprovação ou perda do modelo.

`tools/selflearn/import_external.py` passa a traduzir o `frames.csv` do pool
catalogado e imagens/CSV de outros autores para a fila existente, em pool
externo próprio. Selo imutável por sessão/lote, metadados de treino anterior,
integridade de entrada/modelo e revisão humana obrigatória antes de exportar.
Pseudo-máscara do incumbente só no treino; holdout recebe suporte vazio para
rótulo humano independente e recusa aceite automático. O catálogo citado
descreve um pool privado de capturas, não vários datasets públicos; este
adaptador permite os demais sem alegar que já foram inventariados/aprovados.

Prova de integração no WSL, depois da criação da PR: oito fotos do
School Notebooks RU já disponível em arquivo local foram retificadas pelo
detector de produção e traduzidas pelo incumbente instalado (verificação
estrita pelo manifesto local). Resultado: oito propostas, um lote de
treino, zero rejeições `no_page`, zero revisões. Foi reservado só treino,
conservadoramente, pois a fonte já participa da linhagem de dados do projeto.
A fila privada ficou no laboratório local, fora do checkout; nenhum dado
foi enviado à VM nem publicado. Não é aprovação semântica das propostas.

**Validação de código (VM, 2026-11-12; não é teste ao vivo):**

| Execução completa | Passou | Falhou | Skip |
|---|---:|---:|---:|
| Base 3c00e29, worktree separado | 1.031 | 2 | 4 |
| Esta implementação, após revisão | 1.080 | 1 | 4 |

Foram adicionados 48 casos; o subconjunto de self-learning, registro,
arbitragem, remoção, replay e documentação passou 125/125. A falha comum
é `test_sim_writer.py::test_generate_scenario_corpus_contract`, que exige
fundos privados de laboratório ausentes na VM; ela também reprova o CI da
base. A outra falha da execução-base é a asserção de identidade do teste
threaded `test_new_sheet_does_not_inherit_the_previous_canvas`; passou na
execução desta branch, sem alteração desse teste. Não se alega tê-la corrigido.
Os mesmos 125 casos também passaram no WSL em worktree isolada; o pytest
foi instalado em diretório próprio de teste, sem alterar o venv do aplicativo.

`ruff check .`, compilação e formato dos fontes tocados passam. O formato
global já falha em 15 arquivos na base e ainda falha em 13 intocados aqui;
o CI `rapido` da base confirma isso. Não desabilitar os checks nem publicar
os fundos privados para deixá-los verdes. Essas pendências são anteriores
e separadas da régua de comportamento; não houve promoção de modelo.

Na revisão, preservou-se o `session_id` original (validado como nome simples)
para não ocultar holdout anterior; `split.json` prévio impede teacher no
holdout. IDs de lote permanecem hash, mesmo quando o selo contém caminhos:
uma chave do catálogo nunca vira caminho de saída. Na borracha E3, votos e
memórias dos pixels removidos são limpos sem reiniciar a fase global; as
regiões geométricas permanecem disponíveis para redesenho. Há regressões
para ruído rejeitado não confirmar após apagar e para redesenho em ambas
as trilhas laterais, com FAST ligado/desligado.

**Tentativa de teste nativo automatizado (não é prova ao vivo):** em
`01efd20`, modelos oficiais verificados e vídeo MJPEG sintético, com E3,
content guard e remoção ligados e arbitragem desligada. A folha foi
detectada, mas os 101 passes registrados produziram zero pixels de máscara
e de tinta confirmada; salvar o canvas gerou imagem branca. O pré-requisito
falhou: mover, apagar, redesenhar e casos negativos não foram executados.
Não atribuir isso à PR nem ao papel real sem evidência independente.
MediaPipe estava ausente, portanto não houve validação de oclusão por mão.
Sair com Q terminou em falha de segmentação (139), sem manifesto final.
A comparação nativa com a base `3c00e29`, mesma fonte/modelos/ambiente/flags,
reproduziu 139 e ausência de manifesto também na base. Ambos os traces
apontam o pump da captura durante o encerramento; a causa exata não foi
determinada. É uma falha herdada neste ambiente, não uma regressão provada
da PR. A comparação usou `01efd20`; não se repetiu a UI nas correções
posteriores de isolamento/E3. O subconjunto de código cobre essas correções,
mas não substitui uma fonte temporal reconhecida para validar mover/apagar.

| ID | Falha | Classe | Severidade | Régua |
|---|---|---|---|---|
| F-01 | Tinta repassada/reimpressa em camadas | acumulador/orquestração | bloqueante | parcial — ver F-01 |
| F-02 | Traço fino/fraco apagado (nunca chega ao canvas) | tinta | bloqueante | sim |
| F-03 | Mão confundida com papel | folha/cena | maior | sem régua dedicada |
| F-04 | Mão cobrindo o desenho: conteúdo some, atrasa ou duplica | cena/acumulador | maior | parcial — testes sintéticos, sem replay real |
| F-05 | Tinta fantasma em página vazia | tinta | maior | sim |
| F-06 | Impresso vazando como tinta | tinta | maior | sim |
| F-07 | Canvas muda sem tinta nova (invariância violada) | orquestração | bloqueante | parcial — ver F-07 |
| F-08 | Jitter/latência que impede desenhar | orquestração/folha | maior | parcial — ver F-08 |
| F-09 | Folha perdida com o caderno movido (e a tinta vai junto) | folha/orquestração | maior | sim, sem limiar |
| F-10 | Apagar no papel não apaga no digital | acumulador | maior | parcial — canal opt-in e testes sintéticos |
| F-11 | Página nova lida como a mesma (ou vice-versa) | orquestração | maior | parcial — ver F-11 |

## F-01 — Tinta repassada/reimpressa em camadas

**Sintoma (reportado pelo Lucas várias vezes):** o sistema detecta um
traço uma vez e depois fica "printando em cima" dele repetidamente —
o desenho assentado ganha camadas novas, engrossa ou pisca de novo,
como se fosse reimpresso a cada rodada.

**Causas prováveis na arquitetura (medidas, não hipótese solta):**

1. **Voto sobre warps que tremem.** Antes do descanso total, a sonda
   de 1 Hz alimentava o voto do acumulador sobre warps com jitter de
   geometria; cada alinhamento de fase confirmava uma camada nova
   sobre o desenho parado. Medido na régua de replay (modo hold):
   salto de **+9.999 px confirmados numa única passada** (a 89ª) sobre
   página parada. Registrado em `draw/ink_layers.py` (comentário de
   `_REST_MIN_GENERATION_PASSES`). Mitigações no código: porta de
   fronteira (`_INK_FRONTIER_RADIUS`, `draw/ink_layers.py` em
   `_LayeredInkAccumulator.add`) e descanso total
   (`_PageCleanWorker._at_rest`).
2. **Descanso total não fecha sem sinal de mão.** Com
   `hand_state is None` (clone sem MediaPipe), `_at_rest` nunca fecha
   e a sonda de 1 Hz segue alimentando o voto para sempre — a
   exposição 1 continua viva nessa configuração, só contida pela
   porta de fronteira.
3. **Reentrada de folha podia republicar a página inteira.** Antes da
   recuperação de identidade F-11, um `drop` seguido de reabertura
   chamava `new_generation()` e zerava o acumulador. Agora o app
   preserva uma referência e só reusa o canvas quando a primeira
   máscara registrada reencontra ≥65% da tinta confirmada, agora
   com dilatação e contenção ≥50%; conteúdo divergente começa
   limpo. Na baseline real histórica, anterior a essa correção,
   os 16/16 retornos decidiam `new_page` (F-09/F-11). A correção
   já passa na régua sintética; falta repetir a página real nessa
   revisão. O erro de comparação é separado do registro abaixo.
4. **Provisória → confirmada pinta o mesmo traço duas vezes.** A
   camada provisória aparece em cinza na primeira passada e a
   confirmada repinta em preto por cima quando o voto fecha
   (`_render_layered_canvas`, `draw/ink_layers.py`); com retratação
   (provisória que some e volta), o mesmo traço pisca várias vezes.
   É desenho intencional (feedback rápido × certeza), mas é parte do
   que o usuário lê como "reimprimindo".

**Severidade: bloqueante.** É a queixa recorrente número um do dono
do projeto e viola a regra dele de 2026-11-01/03: registrar uma vez e
adaptar por substituição, nunca acumular camada sobre desenho
assentado.

**Como se mede — régua parcial:**

- Testes do descanso e da fronteira no repo: `tests/test_ink_total_rest.py`,
  `tests/test_ink_rest_no_hand.py`, `tests/test_ink_rest_hand_present.py`,
  `tests/test_ink_frontier_gate.py`, `tests/test_provisional_ink.py`
  (comportamento da trabalhadora/acumulador em sequência sintética).
- Régua de replay do laboratório (modo hold, fora da bancada):
  página assentada sob jitter de warp, contar px confirmados novos
  por passada depois da quietude — o evento de +9.999 px nasceu dela.
- **Buraco:** a bancada não tem régua de app para F-01 (ela julga
  modelo por passada única, não o laço). A régua que falta está
  desenhada em F-07, que é a forma mensurável desta falha; F-01 é o
  sintoma, F-07 é o número.

## F-02 — Traço fino/fraco apagado

**Sintoma:** traço de lápis, traço fino de caneta ou hachura leve não
aparece no digital — ou aparece e some. O desenho perde justamente o
detalhe fino, que é onde a predição/correção mais erra (correção 2 do
Lucas, 2026-11-02).

**Causa provável:** a confirmação exige voto forte: janela de 3
passadas, `hits >= 3` com prob ≥ 0,5 e soma ≥ 1,4 (ou soma ≥ 1,9)
(`_INK_VOTE_*` em `draw/ink_layers.py`), sobre o limiar operacional
0,7 do acumulador. Traço fraco estável com prob entre 0,5 e 0,7
confirma na 3ª passada; abaixo de 0,5 **nunca confirma** — vive só
como provisória e é retratado (some). A porta de fronteira ainda
impede que evidência sub-limiar expanda a borda da tinta confirmada,
o que protege contra F-01 e cobra exatamente aqui. No modelo, é o
recall de passada única medido na régua: w8 marca 0,065 de recall de
núcleo real em passada única (relatório de desafiantes,
`~/workspace/draw-lab/challenger-ruler-2026-11-05/REPORT.md`) — o app
depende do acúmulo temporal para fechar o traço.

**Severidade: bloqueante.** Apagar traço do usuário é perda de
conteúdo; a própria régua de fino reprova candidato (TRI reprovado
com 0,249).

**Como se mede — régua existe:**

- `tools/judge.py`: `THIN_RECALL_MIN = 0.402` (recall de traço fino
  da w8, DIBCO pooled; diagnóstico IoU DIBCO nunca decide sozinho).
- Bancada `judge_ink.py`: veto "Recall de traço fino — DIBCO @0,7
  pooled ≥ 0,402" e recall de lápis no sintético (reportado na
  varredura de τ). Emenda de calibração da bancada: medido com
  precisão cheia, a w8 marca 0,40164 — veto que o incumbente também
  falha por poeira lê-se "empate", não reprovação.

## F-03 — Mão confundida com papel

**Sintoma:** com a mão sobre a mesa/folha, o quadrilátero da folha
cresce para cima da mão, a folha "anda" ou o sistema acha papel onde
é mão — o warp deforma e a tinta nasce deslocada.

**Causa provável:** no contrato instalado (FT-dyn, legado
`mask_logits`+`presence_logit`) **não existe classe mão**: a região
da folha é o que a máscara binária disser, e pele sobre a mesa pode
entrar nela. No contrato de 4 classes (T7), a região da folha é
`papel ∪ tinta` por construção (`draw/neural.py`,
`NeuralSheetDetector`: `page = (classes == _PAPER_CLASS) |
(classes == _INK_CLASS)`) — pixel de mão classificado como papel
entra na região do mesmo jeito e puxa o ajuste do quad. A memória de
arestas do detector (herança de reta quando a mão cobre uma aresta)
mitiga o caso "mão sobre a borda", não o caso "mão lida como papel".

**Severidade: maior.** Deforma a geometria de tudo que é desenhado
enquanto a mão está na cena — ou seja, o tempo todo.

**Como se mede — SEM RÉGUA DEDICADA AINDA.** As réguas de folha da
bancada (IoU/presença SmartDoc, presença Kinetics) medem o efeito
agregado, não a confusão mão×papel. Régua que falta: conjunto de
frames com mão sobre a folha e sobre a mesa ao lado da folha, com GT
de quad e de máscara de mão; métricas = fração de pixels de mão
dentro da região-folha do modelo e erro de canto contra os mesmos
frames sem mão. Frames reais de mesa já existem no laboratório
(30 frames da mesa no `judge_sheet.py` são relativos, sem GT);
frames SmartDoc com mão servem de partida.

## F-04 — Mão cobrindo o desenho: conteúdo some, atrasa ou duplica

**Sintoma:** a mão passa por cima do que já foi desenhado e, ao sair,
trechos somem, voltam atrasados ou aparecem duplicados/deslocados.

**Causas prováveis:**

- No caminho legado, o `InkAccumulator` (`draw/ink.py`) congela o
  pixel sob a mão (`visible_mask=False`: sem julgamento, sem carimbo)
  — por desenho, o conteúdo sob a mão é preservado. Mas se a máscara
  de mão erra (mão não detectada), o pixel é julgado como visível sem
  tinta e pode ser **apagado** pela histerese (fração cai a `exit`).
- No caminho page-clean, a máscara de visibilidade **já está integrada**
  (`draw/ink_layers.py`, `_PageCleanWorker`): com MediaPipe e o modelo
  de mão disponíveis, a trabalhadora obtém a máscara no warp e a
  registra junto com as probabilidades. Pixel coberto não vota e a
  provisória fica preservada sob a mão; a confirmada continua monotônica
  na geração. O modelo ainda recebe o warp cru. Sem máscara disponível,
  ou quando ela erra, pixels ocultos continuam entrando como visíveis,
  com risco de atraso, retração da provisória e falsa tinta.
- Se a mão cobre a borda da folha e o tracker perde a folha, vale
  F-09: a reabertura pede recuperação de identidade. O canvas é preservado
  se a tinta visível reencontra a referência; pouca visibilidade adia a
  decisão. Uma decisão errada de página nova ainda pode zerá-lo.

**Severidade: maior.** Desenhar é, por definição, mão sobre o desenho
metade do tempo.

**Como se mede — régua parcial.** `tests/test_ink_occlusion.py` injeta
visibilidade sintética e verifica abstenção de pixels ocultos, preservação
da provisória e recuperação de identidade só pela tinta visível (ou
adiamento quando quase toda a referência está coberta). Os quatro regimes
de mão da régua de re-registro continuam sendo sinais de presença/movimento,
não uma mão no warp. Régua que falta: replay de sequência real/sintética em que
uma mão (real, de vídeo; ou máscara recortada) cruza uma página já
desenhada; medir (a) px confirmados antes × depois da oclusão
(perda = qualquer queda), (b) atraso até tinta nova sob a mão
confirmar, (c) duplicação = crescimento de px confirmados sem tinta
nova no GT. A infraestrutura de replay do app existe
(`tools/replay_orchestration.py`); falta o corpus de oclusão e o
gabarito.

**Ataque de oclusão integrado; validação real pendente (frente separada):**
a evidência oculta já é tratada como desconhecida e a visibilidade
acompanha a geometria da probabilidade em `main`. Ainda falta cobrar perda
zero de confirmado, nenhuma cópia depois da desoclusão e atraso de
tinta nova sob a mão em passadas, incluindo erro/ausência do sinal
de mão. Isso não corrige o recall de F-11: o desencontro de níveis
de comparação acontece também sem oclusão. O experimento de tremor
abaixo não é validação de oclusão.

## F-05 — Tinta fantasma

**Sintoma:** página em branco (ou trecho limpo) ganha tinta que
ninguém desenhou — sujeira no canvas sem tinta nova. É o verso de
F-02 e o modo de falha que matou o desafiante pronto medido em
2026-11-05 (1,95% de FP em página vazia real, ~100× o da w8).

**Causa provável:** FP do modelo de tinta em textura de papel,
sombra e mobiliário, confirmado pelo voto quando a fase do jitter
alinha (mesma mecânica de F-01). A camada provisória existe para
medir esse risco: `retracted_px_total` conta quanta tinta especulativa
sumiu (`_LayeredInkAccumulator`).

**Severidade: maior.** Canvas que inventa tinta destrói a confiança
no "só muda com tinta real" (F-07).

**Como se mede — régua existe:**

- `tools/judge.py`: `EMPTY_FP_MAX = 0.00072` (0,072%, pior FP da w8
  nos warps vazios; mínimo de 3 warps).
- Bancada `judge_ink.py`: vetos "FP em vazios (sintético + f01–f03)
  ≤ 0,072%" e "FP médio nos warps vazios @0,7 ≤ 0,072%".

## F-06 — Impresso vazando como tinta

**Sintoma:** texto e linhas impressas da folha (pauta, formulário,
livro) são reproduzidos como se fossem desenho — o digital mistura o
que já estava impresso com o que a pessoa escreveu.

**Causa provável:** o modelo em produção (w8) é binário: separa por
aparência, não por identidade; impresso escuro é aparência de tinta.
Medido: vazamento real (DIBCO impresso) da w8 em 21,3%–39,5%
conforme o conjunto (bancada/relatório de desafiantes) contra a régua
de desenho ≤ 0,5%. A separação por identidade é a aposta da linhagem
de 3 classes (fundo/impresso/tinta, consome só P(tinta)) — ainda sem
candidato promovido.

**Severidade: maior.** Em folha impressa ou pautada — o papel real
das pessoas — o canvas nasce poluído antes do primeiro traço.

**Como se mede — régua existe:**

- `tools/judge.py`: `PRINT_LEAK_MAX = 0.005` (≤ 0,5%; mínimo de 5
  páginas) e invariância `INV_PRINT_CENTER_MIN = 0.995` (impresso no
  centro ≥ 99,5% recusado).
- Bancada `judge_ink.py`: veto "Print leak sintético (mask_print)
  ≤ 0,5%"; o leak **real** é reportado sem veto (baseline w8 21,3%,
  TRI 9,0%) — meta de desenho ≤ 9,0% com os vetos de recall passando.

## F-07 — Canvas muda sem tinta nova (invariância violada)

### Correção opt-in da duplicação dentro da geração (2026-11-12)

`DRAW_REG_CONTENT_GUARD=1` (padrão 0) compara suporte do canvas na
probabilidade original (colocação identidade) e na reprojetada. Com
deslocamento acima do gatilho existente de 8 px, massa mínima existente
e vantagem de identidade acima do delta existente, bloqueia o voto até
três poses assentadas; então troca só a âncora e usa a evidência original.
Não transporta nem soma o canvas. Geometria espúria com suporte superior
na reprojeção não troca âncora; identidade/virada em julgamento tem
precedência. O mecanismo está na trabalhadora real que o replay instancia;
o adaptador de contagem encaminha também o canal de remoção.

Régua parcial: `tests/test_ink_removal_registration.py` fixa página movida
sem duplicação, tinta nova após pouso, geometria espúria e paridade OFF.
Isto não reverte o veredito negativo ao vivo da arbitragem v3 nem prova
correção dos +1.911 a +9.383 px históricos no corpus privado. Não houve
retuning dos limiares existentes nem alteração de `DRAW_ECC_REFINE`.

**Sintoma:** com a página assentada e ninguém desenhando, o canvas
muda sozinho — tinta aparece, engrossa ou se move. É a forma
mensurável de F-01 e a violação direta da especificação do Lucas:
"mover o caderno = só re-registro, conteúdo idêntico; só alteração
real muda o canvas" (2026-11-01).

**Causa provável:** qualquer caminho que alimente o voto com a página
parada (sonda sem descanso, F-01), re-registro de geometria que
desloca o warp sob o canvas já publicado, ou republicação por reset
de geração. O portão correto é sobre a **saída confirmada**, nunca
sobre diff de quadro (o churn da câmera supera o sinal do lápis —
lição registrada em `draw/ink_layers.py`).

**Severidade: bloqueante.** É a invariante central do produto: se o
canvas não é estável parado, nenhuma camada acima (cursor, CAD)
pode confiar nele.

**Como se mede — régua parcial:**

- Testes de descanso no repo (mesma lista de F-01): zero passadas em
  repouso é comportamento testado da trabalhadora.
- Régua de replay do laboratório, modo hold (a que mediu +9.999 px):
  página assentada, N passadas, Δpx confirmados deve ser 0 após a
  quietude.
- **Régua de re-registro (2026-11-06), a segunda metade da de
  hold:** `tools/replay_reregister.py` (guarda em
  `tests/test_replay_reregister.py`). Replay em passadas, sem ONNX
  nem threads, que dirige o laço inteiro e não só o acumulador:
  câmera simulada (quad verdadeiro da página × quad do detector),
  o `_SheetTracker` real com relógio de cena, a trabalhadora de
  detecção espelhada (latência `--detect-ms`, padrão 36,7 ms do
  Ryzen; um quadro em voo, fila latest-wins de 1), a lógica de geração
  de `draw/app.py` espelhada (drop/reabertura → `new_generation()`,
  mediana de cantos, congelamento sem mão, virada atrás da flag) e
  a trabalhadora espelhada (cadência, descanso total, registro na
  âncora, `_LayeredInkAccumulator`) — o teste cobra que o espelho
  da trabalhadora confirma pixel a pixel o mesmo canvas que a
  `_PageCleanWorker` real. Quatro configurações de sinal de mão:
  sem_sinal (None), ausente (False), presente_parada e
  presente_mexendo (True). Controle parado (portão: nada muda) e
  três cenários — os de F-09 e F-11 abaixo. Página: a sessão real
  f01–f30 do cache quando o draw-lab existe; senão, página
  sintética determinística (`"source"` no JSON). Aproximação
  declarada: a passada re-warpa o mapa de probabilidade, o modelo
  não é re-rodado no quadro re-warpado.
- **Lacuna fechada — página real (2026-11-06):** `control_static`
  e `sway_negative` na configuração padrão foram aprovados com a
  página real do draw-lab. Fonte da rodada: ver baseline em F-09.
  A distinção entre aprovação com passadas e por vacuidade continua
  obrigatória. Ainda não há limiar pré-registrado em `tools/judge.py`;
  o modo hold com balanço realista segue fora da bancada e falta
  vídeo real de caderno movido, não a página real no replay.
- **Re-baseline v2 do ECC / #46 (2026-11-09):**
  `tools/replay_reregister_v2.py`, pré-inscrição congelada antes das páginas
  em `docs/F07_REPLAY_V2_A_R1_2026-11-09.json`, guarda em
  `tests/test_replay_reregister_v2.py`. Modo A: âncora fixa em `same_page`,
  helpers produtivos sem mudança; mapas já retificados expressos na base da
  passada antes do registro, sem injetar movimento físico duas vezes.
  Quads consecutivos são só diagnóstico. Uma página real f01–f30, mesmos
  inputs nos braços explícitos `DRAW_ECC_REFINE=0` e `1`, sem pooling.
  F-07 OFF: **12/15 decisivos**, 3 falhas, 5 vacuosos. ON: **20 adiados,
  denominador zero**, não «0/20 falhas»: todos os baselines tiveram rejeições
  ECC e o protocolo exclui episódios com baseline afetado de todos os votos.
  Rejeição nunca é voto de página nova. Números, hashes e limitações em
  `docs/F07_REPLAY_V2_A_R1_RESULTADOS_2026-11-09.md`. É régua de contrato sobre
  mapas, não replay físico nem recomendação sobre a flag; faltam as outras
  três páginas e frames brutos. A baseline v1 fica histórica, não comparável.

**Registro já ativo; o que falta é o portão.** Em
`draw/ink_layers.py`, `_PageCleanWorker._work` recalcula
`_registration_matrix(âncora, cantos)` em toda passada com ambos os
quads e aplica o warp sempre que não-identidade, tolerância `1e-4`.
A primeira passada fixa a âncora; sem cantos não há registro.
Não se trata de adicionar registro: a falha medida é aplicá-lo
também a movimento sustentado. O desenho do experimento está na
seção de pré-registro abaixo; produção permanece inalterada.

## F-08 — Jitter/latência que impede desenhar

**Sintoma:** o traço digital atrasa atrás da caneta, treme ou anda em
saltos; desenhar rápido vira adivinhação. (O Lucas já reportou
"travou menos" como melhora parcial em 2026-11-03 — velocidade
medida, reconhecimento ainda em aberto.)

**Causa provável:** tudo em série na CPU — folha (~42 ms/passada no
Ryzen medidos ao vivo; 512 ms/frame nesta VM, laboratório), passada
de tinta de ~168 ms no Ryzen com cadência de 250 ms
(`_PAGE_CLEAN_INTERVAL_SECONDS`, `draw/ink_layers.py`), presença de
mão assíncrona amostrada — e o jitter residual de canto do tracker
entra no warp e vira tremor da tinta (a dilatação de evidência e a
porta de fronteira existem para absorver parte dele).

**Severidade: maior.** Latência de captura é a diferença entre
"mesa digitalizadora" e "scanner lento" — a promessa da camada 1 é
tempo real.

**Como se mede — régua parcial:**

- Bancada `judge_sheet.py` reporta (sem veto) ms/frame, passo do
  centroide mediano (jitter) e tempo até a primeira confirmação;
  `judge_ink.py` reporta latência por warp. A regra do protocolo: o
  número que vale é o do **Ryzen do Lucas**, não o desta VM.
- `tools/step0_latency.py` e a telemetria por etapa da trabalhadora
  (`telemetry_snapshot` em `draw/ink_layers.py`: queue_wait/pass/
  submit_to_publish por passada).
- **Buraco:** não há limiar pré-registrado de latência ponta-a-ponta
  percebida (caneta → pixel na tela) nem de jitter de canvas; as
  réguas medem componentes. Régua que falta: orçamento nomeado por
  etapa + uma medida de ponta a ponta no replay (carimbo de captura
  × carimbo de publicação, que a telemetria já registra).

### M1 — jitter do quad de registro destrói o voto de tinta (auditoria de 2026-11-10)

**Mecanismo fechado** pelas provas de replay do ciclo de auditoria
(`~/workspace/draw-lab/live-audit-2026-11-10/7-prova-p2-registro.md`,
`~/workspace/draw-lab/live-audit-2026-11-10/9-p3a-variancia-quad.md`):
o voto de tinta (prob ≥ 0,5 no mesmo pixel em 3 passadas) é destruído
pelo jitter do quad que alimenta `_registration_matrix`
(`draw/ink_layers.py`) — fração de descarte 0,347/0,573/0,818 com
jitter de ±2/±4/±8 px por canto. A decomposição de variância (P3a)
mostrou que o jitter real é **evento discreto** (flips de modo do
detector, pousos de salto), não ruído: o rastreador chegou a
confirmar como âncora o quad espúrio do frame 0 e recusar todos os
seguintes.

Estado das saídas — **não repropor as refutadas sem evidência nova**:

| Saída | Prova | Veredito |
|---|---|---|
| ECC de registro (`DRAW_ECC_REFINE`) | P2 | refutada: fração de descarte no mesmo regime, IoU pior; segue OFF |
| Voto tolerante por dilatação da prob | P2b | refutada: infla o canvas 2–6× |
| Suavização EMA / One Euro do quad | P3b | reprovadas: erro simétrico (a tinta fica para trás quando a página anda de verdade; α = 0,15 não converge em 30 passadas) |
| Banda morta τ = 10 px no quad de registro (`DRAW_REG_DEADBAND`) | P3b (`~/workspace/draw-lab/live-audit-2026-11-10/10-p3b-estabilizadores.md`) | aprovada em replay; atrás da flag, padrão `0` |
| Âncora por mediana das 3 primeiras passadas (`DRAW_REG_ANCHOR_MEDIAN3`) | P3b | aprovada em replay; atrás da flag, padrão `0` |

As duas aprovadas são disciplina, não suavização. Banda morta: o
registro usa um quad congelado por geração enquanto o desvio máximo
de canto for < `_REG_DEADBAND_TAU_PX` (10); ≥ τ por 2 passadas
seguidas libera (contado em `reg_deadband_releases` da telemetria);
spike de 1 passada nunca libera (0 liberações espúrias no regime
evento-dirigido). τ = 3 foi medido e reprovado (libera no ruído; pior
inflação da classe): τ precisa exceder o diâmetro do jitter assentado.
Âncora-mediana: com passada 0 em modo espúrio (126–331 px/canto), a
âncora = 1ª amostra dá IoU 0,001 (falha silenciosa); a mediana-de-3
dá IoU 0,441–0,478 no mesmo caso, sem mexer em limiar. Ambas valem só
para o quad de registro: warp do app, rastreador e exibição
intocados. Régua no repo: `tests/test_reg_discipline.py` (contratos
das flags e paridade byte a byte com as duas desligadas). Ligar
qualquer uma por padrão exige teste ao vivo na mesa do Lucas.

**Interação das duas flags (prova composta, 2026-11-10).** Com as
duas ligadas no worker real, o braço composto pagava um pedágio de
abertura: descarte 0,285 em jitter de ±4 px (portão 0,25), com 100%
dele nascendo até a passada 5 e zero da 6 à 20. Mecanismo: a âncora
anda nas passadas 1–3 (mediana assentando) enquanto o congelado da
banda morta ficava preso na amostra 1 — as primeiras passadas
registravam contra um congelado que a própria âncora já abandonara.
Correção (sem constante nova): enquanto a janela de amostras da
mediana está aberta, o congelado é rebaseado para a âncora corrente
e o contador de liberação recomeca (`reg_deadband_rebased` na
telemetria); fechada a janela, vale a semântica da banda morta.
Depois da correção, o braço composto iguala o controle sem jitter
em ±2 e ±4 px (descarte 0,000, IoU 0,933, inflação 1,07×). Régua:
`tests/test_reg_rebase.py` + a bancada da prova composta
(`17-prova-composta-flags.md`, no laboratório).

### M2 — pouso sustentado: a segunda cópia é desenho, não bug (2026-11-10)

**Defeito.** Quando a folha anda de verdade e assenta em pose nova
no meio de uma geração (pouso sustentado, presença contínua), o
registro em `_registration_matrix` (`draw/ink_layers.py`) leva a
tinta nova para a âncora velha como se a folha tivesse ficado parada
e só o quad tivesse andado: a mesma tinta entra deslocada pelo
movimento e o voto confirma uma **segunda cópia**. É o comportamento
especificado do registro (feito para o tremor/flip do quad com a
folha parada), não um bug de implementação. Evidência:
`~/workspace/draw-lab/live-audit-2026-11-10/10-p3b-estabilizadores.md`
(no regime de eventos o seguidor duplica 1,55–1,72×; pouso não é
decidível por filtro de passada) e
`~/workspace/draw-lab/live-audit-2026-11-10/12-estudo-pouso.md`.

**Decisão do Lucas (2026-11-10): opção A** — re-ancorar e
re-registrar o canvas por substituição no pouso (registrar uma vez,
adaptar por substituição, nunca acumular). **Opção B descartada**
(fechar a geração e passar pela identidade) por defeito estrutural:
`new_generation()` não toca no acumulador, então no pouso B ou mantém
âncora velha e canvas intactos (a segunda cópia sobrevive) ou apaga o
canvas confirmado.

**Estado da implementação: atrás de `DRAW_ANCHOR_MIGRATION`, padrão
`0`, REPROVADA na prova de replay.** O mecanismo segue a
especificação: deslocamento médio da grade sob a matriz de registro
(`_matrix_grid_displacement`) > `_ANCHOR_MIGRATION_DISPLACEMENT_PX`
(8) por `_ANCHOR_MIGRATION_PASSES` (3) passadas com quad válido e sem
identidade pendente ⇒ `_LayeredInkAccumulator.migrate()` re-projeta
confirmada/provisória (vizinho mais próximo), janela do voto
(bilinear) e invalida os buffers derivados, com a matriz inversa do
registro âncora-nova→âncora-velha; a âncora nova é a mediana dos
quads do streak; a passada da migração força a publicação
(`poll()` devolve o canvas migrado na mesma passada); contador
`anchor_migrations` e eventos `anchor_migration_events` na
telemetria. Contratos em `tests/test_anchor_migration.py`; com a flag
desligada o canvas é byte-idêntico ao golden.

A prova (`tools/replay_anchor_migration.py`, fonte **sintética** — o
laboratório não estava nesta máquina; 5 séries × 3 braços, sem ECC,
contra o oráculo de substituição) reprovou com (8 px, K=3) e com o
único reajuste permitido (K=4): duplicação 0,69–1,14 do canvas
pré-pouso nos pousos de 40/120 px (a base mede 0,44–0,77), IoU
0,57–0,74, perda e retenção fora do portão nos 3 pousos, 2 migrações
no pouso de 120 px (o streak fecha no meio do movimento). Passaram:
0 migrações em flips e no controle quieto, isca de modo espúrio com
IoU 0,999, churn do quieto sem regressão no braço só-migração.

**Diagnóstico (medido, sem portão):** a inversa do registro assume
que a tinta ficou no mesmo ponto da câmera — certo para o modo
espúrio do detector (folha parada, quad errado), errado para o pouso
(a tinta anda com a folha e, na grade da página, não se mexe). Um
braço diagnóstico que só troca a âncora sem mover o estado inverte o
quadro: pousos com IoU 0,92 e retenção 1,0, mas a isca de modo cai
para IoU 0,74. As duas hipóteses são indistinguíveis pelo quad; o
próximo desenho precisa arbitrar pelo conteúdo (a evidência da
passada casa com o canvas na grade da página ou na da câmera?) —
decisão de desenho, não reajuste de limiar. Ligar a flag por padrão
está vetado até uma prova nova passar.

### M3 — arbitragem por conteúdo + abstenção: a v3 (2026-11-10)

**Teste ao vivo reprovado, relato de @santleme registrado em 2026-11-12:**
"ficou uma merda". Aprovação nos portões de replay não é aprovação na
mesa. Auditoria somente leitura dos manifestos do WSL:

| Sessão local | Execução da arbitragem | Cadência observada | Identidade |
|---|---|---|---|
| 1794341688-dbea1982 | 53 passadas, 6 abstenções; `spurious`, Sp=0,0201, Sc=0,0894, deslocamento 9,49 px | gap mediano 1,0001 s; inferência mediana 126,9 ms | nenhum evento |
| 1794443693-8096ad69 | 103 passadas, 5 abstenções; `landing_default`, canvas de só 23 px | gap mediano 1,0001 s; inferência mediana 120,0 ms, p95 136,8 ms | sete `new_page`, um `same_page` |

Os campos/eventos provam que a flag estava ligada nessas corridas de
produção. Não provam qual delas corresponde exatamente ao comentário do
usuário. A segunda registra 343 submits, 235 substituições latest-wins;
o relatório amostrado marca 93 amostras em idle e mão presente em 2%.
Isso favorece a hipótese de cadência ociosa sobre o orçamento nominal de
250 ms: três votos a 1 Hz levam segundos, mesmo com ONNX de ~120 ms.
Não há medição que atribua o atraso à soma serial folha+tinta; ambos têm
workers, competem por CPU e a régua deve medir os carimbos reais.

Há participação de F-11: na geração 3, recall 0,827 foi vetado pela
contenção 0,223 e decidiu `new_page`. Sem GT de página/tempo não se pode
chamar isso de falso split nem distinguir automaticamente F-07/F-09 de
tinta nova. O replay usa mapas prontos re-warpados, não reexecuta ONNX em
fotos com mão/luz; seus portões não cobrem essa mudança de distribuição.

**Observabilidade nova:** cada passada registra flag efetiva inclusive
OFF, sequência, gap real, queue/inferência/publicação, deslocamento e
identidade da matriz de registro, scores/streak/veredito de arbitragem,
decisão de identidade, visibilidade, pixels adicionados/removidos. O
gravador escreve `ink_passes.jsonl` continuamente, privado e limitado a
200.000 registros; deduplica sequências e conta lacunas/cortes no manifesto.
O início não desaparece mais no deque final de 256 passadas. Se houver
queda antes do fechamento, as linhas já gravadas continuam disponíveis.
Limiar de latência percebida e vídeo real anotado ainda faltam; não há
novo veredito ao vivo nesta mudança.

**Sucessora do M2.** A prova da arbitragem (frente 19, no
laboratório) separou as hipóteses com margem larga: no streak de
assentamento, a fração da massa da prob sobre o canvas confirmado
pré-evento na colocação **identidade** (Sp, hipótese página/pouso)
contra a colocação **registrada** (Sc, hipótese câmera/espúrio)
classificou 8/8 eventos com margens de 25–100× sobre δ = 0,05. Mas
a v2 (arbitrar sem mais nada) reprovou nos portões de duplicação
nas duas bancadas, por um mecanismo sondado passada a passada: **a
cópia deslocada confirma pelo voto enquanto o evento está aberto**,
antes de qualquer veredito — tinta confirmada não sai na geração,
então decidir a âncora depois é tarde.

**v3 = arbitragem + abstenção:** enquanto o evento está aberto, a
passada é registrada e pontuada, mas não vota (`visible = 0`). No
laboratório: aprovada em todos os portões na bancada sintética
(duplicação 0,004–0,030; IoU 0,991–0,998; retenção 1,0) e, na
bancada real, aprovada no razão do evento — duplicação **nascida
no evento** de 0,0005 (pouso de 40 px) e 0,039 (120 px), contra
0,211 da v2 e 0,915 da base; o resíduo total é um piso pré-evento
idêntico em todos os braços, nascido da âncora de amostra única
(defeito vizinho, correção no #57). POUSO troca a âncora **sem
mover o estado**; ESPÚRIO não faz nada; sem massa para julgar,
troca por default.

**Estado:** implementada atrás de `DRAW_ANCHOR_ARBITRATION`
(padrão `0`; flag nova — a da v1 segue vetada), contratos em
`tests/test_anchor_arbitration.py`. **Prova composta contra esta
implementação (frente 21, 2026-11-10): APROVADA em todos os
portões** no braço v3 + âncora-mediana, julgada na grade da âncora
final — duplicação nascida no evento **0,000 (0 px)** nos pousos
de 40 e 120 px, IoU 0,960/0,954, isca com IoU 0,987 e 0 ações,
ações exatas (1/1/3/0/0), retenção 1,000 e perda 0 por troca;
vereditos da telemetria 11/11 com margens de 15–200×. Dois limites
medidos e declarados: (1) pousos **repetidos** acumulam
re-expressão entre grades de eras (~1–3 px sobre o halo da prob
mole) — em 3 pousos o canvas termina ~2× o oráculo, com retenção
perfeita e duplicação canônica de 0,183 contra 3,164 da base;
(2) rotação ≥ ~45°: o mecanismo é cego à permutação posicional de
cantos (vereditos por default trocam para a grade permutada; o
canvas sobrevive, mas a frente de correspondência física de
cantos é pré-requisito para ligar por padrão). Ligar por padrão
segue decisão do Lucas, com teste ao vivo na mesa.

**Extensão cíclica (frentes 23/24, 2026-11-10).** A
correspondência física não vem de geometria (o quad posicional é
quase-invariante sob rotação: girar 180° custa 20–57 px e a
continuidade do tracker acerta 0/307 passos reais) nem de
classificador de orientação por warp (teto ~0,65 neste material:
spread, pauta ondulada, página em branco). Vem do mesmo sinal da
arbitragem, estendido às colocações cíclicas contra o snapshot.
A **primeira implementação foi reprovada pela prova da frente
24**: pontuar pelo registro com os cantos rolados marca 0,005
onde a permutação da imagem marca 0,23–0,27 na mesma passada
real (a construção do teste de contrato era auto-consistente e
fisicamente outra — lição registrada). A forma que a medição
manda, e a implementada: no streak, pontuar a **prob entregue
permutada por −k** (rot90 + resize ao canônico, a forma da
frente 23) contra o snapshot. Veredito `landing_rotated`: a
âncora nova é a mediana do streak **re-rotulada por k** (o
estado, como sempre na v3, não se move); a trabalhadora guarda
o roll corrente da âncora — o gatilho de deslocamento compara
em termos posicionais (uma âncora rolada não pode fazer toda
passada seguinte parecer evento) e uma **segunda rotação na
mesma geração abstém** (regime não medido). Os 4 escores vão
para a telemetria do evento. Contratos em
`tests/test_anchor_arbitration.py` (construção física: página
enchendo o quadro, prob permutada). **Prova (frente 26, física
do app, braço arbitragem+mediana): o pouso rodado ISOLADO está
aprovado** — rot_90/180/270 com IoU 0,967–0,976, `rotation_k`
correto em 100% dos eventos (margens 0,13–0,23 contra o teto de
ruído 0,012), âncora final a 1,2–1,5 px do quad físico, zero
eventos espúrios depois de cada pouso rodado, regressão reta
idêntica à régua da frente 25. Ressalva registrada: dup de
evento = 2 px em rot_90/rot_180 contra o portão absoluto de
zero — diagnóstico por fase mostra borda de ~1 px/fase do voto
através da âncora permutada (0 px na passada do veredito), não
segunda cópia. **Pouso depois de pouso rodado (pousos_3_rot): corrigido
(frente 27).** Correção à leitura da frente 26: na baseline a
era 3 **não planta cópia** — o canvas não muda um pixel na era
(os 1.635 px eram artefato do juiz na grade da âncora final com
a âncora atrasada; o fim da era 2 já lia 1.635 nessa grade). O
dano real era outro: âncora atrasada em 51 px e **611 px de
conteúdo legítimo que nunca confirmavam**. A decomposição,
medida sobre passadas capturadas: a evidência da era 3 é o
próprio S_roll (a prob entregue des-rotada pelo roll parado)
lido como **hipótese-página** — 0,193 contra 0,002 do Sc —
porque o warp entregue já está em coordenadas da página e só a
rotação parada está errada nele (permutação pura); parear
cantos no registro (a correção indicada pela F26) ataca o mapa
das passadas de voto e planta a cópia, medido na miniatura. A
correção implementada: com roll ≠ 0, `score_page` é o escore
compensado (com marca na telemetria do evento) e a ação do
pouso comum/default troca a âncora pela mediana do streak
**rolada pelo roll parado**, que se preserva. Suíte da F26
re-corrida: idêntica em tudo, exceto pousos_3_rot, que passa
(3/3 ações, 0 eventos após a última ação; os 611 px confirmam,
IoU canônico 0,832 → 0,856, ao custo de +15 px de dup na era);
restam só os FAILs pré-existentes de 2 px. Limites que ficam:
pouso reto com roll parado ímpar (1 ou 3) não medido; isca com
roll ≠ 0 sem série; segunda rotação na mesma geração segue
abstenção — agora com consequência benigna comprovada (sem
corrupção, só âncora latente). Rotação pura no lugar, sem
deslocamento que abra evento, segue invisível ao gatilho.

**Física da bancada (frente 25, 2026-11-10) — a disciplina
re-medida sob a física do app.** O app retifica cada frame pelo
quad da passada e submete (warp, quad) juntos; as bancadas
originais da disciplina usavam prob fixa com cantos variando.
Sob a física do app: a **banda morta PREJUDICA** (registrar com
quad congelado ≠ quad que retificou quebra o cancelamento
warp↔registro: descarte 0,66–0,93 no jitter, IoU 0,035 no
pouso_40 sozinha) — reprovada como componente, fica atrás da
flag desligada; a **mediana sozinha cobra pedágio de abertura**
(a grade anda nas passadas 1–3 sem re-expressar o estado) e só
sobrevive como parceira da arbitragem, onde limpa o piso do
pouso profundo (0,61 → 0,07); a **arbitragem SOBREVIVE** e a
composição medida é **arbitragem + mediana, sem banda morta**
(pousos dup-evento 0 px, IoU 0,960/0,954). A produção (OFF) é
quase perfeita onde a folha não anda — sua ferida é só o pouso.
Limites medidos da arbitragem: inanição por abstenção sob
jitter sustentado de ±8 px (19/20 abstenções; no ruído real
σ ≈ 1 px não morde) e o pedágio de descarte 0,16–0,37 da
mediana nas séries de evento.

## F-09 — Folha perdida com o caderno movido

**Sintoma:** a pessoa move o caderno (ou o detector perde a folha por
um instante) e a folha "some" — e com ela o canvas: ao voltar, a
página é tratada como nova e a tinta é redetectada do zero (ver
F-01, causa 3).

**Causa provável:** o tracker (`draw/sheet_track.py`,
`_SheetTracker.update`) descarta a folha quando os critérios de
perda passam (tempo e quadros sem confirmação); salto de posição
exige persistência antes de confirmar. Antes da mitigação F-11, a
reabertura sempre zerava o canvas. Agora o app preserva a referência
e pede a decisão por conteúdo na primeira máscara após o retorno;
na baseline real anterior à correção de identidade essa decisão
falhava em todos os retornos que chegavam a ela. A correção com
dilatação/contenção passa na régua sintética; faltam repetição real
nessa revisão e vídeo real. A troca de página
legítima e o movimento ainda compartilham o evento `drop`, dependendo
da comparação de conteúdo. (O gatilho fino de virada por cobertura existe mas está
**desligado por padrão** desde o hotfix de 2026-11-01: na mesa real
ele disparava a cada 2–5 s com o balanço de geometria do caderno
aberto — cantos oscilando até 314 px — zerando o canvas sem parar.)

**Severidade: maior.** Mover o caderno é gesto normal de quem
desenha; pela invariância dele (F-07), mover não pode custar o
conteúdo.

**Como se mede — régua parcial, sem limiar para quedas:**

- Bancada `judge_sheet.py`: conta **quedas e recuperações** por clip
  SmartDoc e na sequência da mesa, e presença na sequência — tudo
  reportado sem veto no `JULGAMENTO.md`.
- **Régua de re-registro (2026-11-06):** `tools/replay_reregister.py`
  (ver F-07). Cenário `move_same_canvas`: desenho assentado, o
  caderno anda (translação 30 px e 180 px, rotação 12°, escala
  0,85, combinado) e assenta. Portão: `canvas_identical_after_move`
  (canvas confirmado depois × antes, alinhados pela homografia
  estimada por ECC, tolerância de fronteira = `_INK_FRONTIER_RADIUS`),
  `generations_created = 0` e `new_confirmed_px_outside_tol = 0`.
  Cenário `drop_and_return`: a folha some por 6/30/150 quadros
  (detector falha com a folha à vista, ou a folha sai do quadro) e
  volta na mesma pose — agora com portão de sobrevivência, perda
  zero e crescimento residual ≤25 px. `--app-logic main`
  (padrão) espelha a recuperação de identidade de 87f52e4 (drop
  deixa a identidade pendente; a reabertura chama
  `new_generation(preserve_identity=True)` e a primeira passada
  mantém o canvas se o recall ≥ `_PAGE_IDENTITY_MIN_RECALL` e a
  contenção ≥ `_PAGE_IDENTITY_MIN_CONTAINMENT`, com dilatação
  pela tolerância de fronteira na correção já integrada);
  `--app-logic legacy` espelha o laço anterior (drop →
  `new_generation()`).
- Replay sintético de "moveu = mesmo canvas" também em
  `tests/test_ink_registration.py`; exige canvas confirmado idêntico
  pixel a pixel após re-registro.
- **Baseline (página sintética, 2026-11-06) — REPROVADO, como
  esperado; números para a correção posterior:**
  - moveu, com passadas (sem_sinal, presente_parada,
    presente_mexendo): 15/15 reprovados, igual nas duas lógicas (o
    movimento não passa por drop). Nenhum `new_generation()`
    e nenhum pixel perdido — mas **+3.460 a +8.127 px confirmados
    novos fora da tolerância** sobre um canvas de ~12.600–12.900 px
    (translação 30 px ≈ +8.000; rotação ≈ +7.600; escala e
    combinado ≈ +5.300; translação 180 px ≈ +3.500). A causa
    medida: o registro na âncora (`_registration_matrix`) lê a
    geometria nova como tremor e re-projeta a página que andou de
    volta para a posição antiga da câmera — o canvas ganha uma
    segunda cópia deslocada da tinta assentada (a homografia que
    melhor alinha é a identidade: o original fica e a cópia soma).
  - moveu, ausente: 5/5 "aprovados" **por vacuidade** — a
    trabalhadora está em descanso total e nada a acorda (zero
    passadas), e a mediana de cantos congela sem mão: a geometria
    da tinta fica 32–185 px longe da página (`ink_geometry_error_px`);
    o próximo traço cairia fora do lugar.
  - drop e volta, antes de 87f52e4 (`legacy`): 6 quadros sobrevivem
    (8/8). 30 e 150 quadros (≥ `_SHEET_LOST_*`): 16/16 com `drop` +
    reabertura = 2 `new_generation()`, canvas zerado e reconstruído
    com −919 a −2.527 px a menos que antes.
  - drop e volta, com a recuperação de identidade (`main`): antes da
    correção de identidade (2026-11-06), o mesmo 16/16 zerado, com 1
    `new_generation()` e decisão `new_page` em 16/16 — a mesma
    página, na mesma pose, dava recall 0,58–0,61, abaixo de 0,65.
    Depois da correção (comparação dilatada pela tolerância de
    fronteira + contenção dual, ver F-11): **24/24 sobrevivem** com
    folga — decisão `same_page` nos 16 drops (recall dilatado
    0,86–0,87), canvas sem perda, crescimento residual de 1–13 px
    fora da tolerância (teto do portão: 25 px). 6 quadros: 8/8
    sobrevivem (sem drop).
  - troca de conteúdo no retorno (`drop_and_turn`, página sintética de
    conteúdo distinto): 8/8 drops decidem `new_page` (recall
    dilatado ~0,025, contenção ~0,025) e o canvas não é herdado; os
    6 quadros curtos passam por vacuidade (sem drop).
  - sensibilidade histórica à latência da detecção, antes da correção
    de identidade: com 20 ms (um quadro), 36,7 ms e 100 ms os vereditos
    e as contagens da baseline original não mudam
    (15/15, 8/8 e 16/16); só os px mexem em até ~350.
- **Baseline histórica com página real (2026-11-06), lógica `main`
  em `d4690d5`, anterior à correção F-11 — MEDIDO
  EM LABORATÓRIO, não promoção:** rodada informada pelo Santleme,
  `source: "real"`, artefato local
  `~/workspace/draw-lab/replay-ruler-2026-11-01/baseline_real_main_2026-11-06.json`.
  O JSON permanece no laboratório, não é versionado. Estes números
  são o registro textual daquela rodada, não uma nova execução
  nesta máquina, seguindo a distinção de `tools/judge.py` entre
  medição de laboratório e fato validado na mesa.
  - `control_static`: **APROVADO** (4/4); `sway_negative` com a
    flag de virada desligada (padrão real): **APROVADO** (8/8).
  - `drop_and_return`: ausências média (30 quadros) e longa
    (150 quadros), nos dois modos de ausência e em todos os regimes
    de mão, dão **`new_page` em 16/16 retornos**, somados.
    Os eventos da rodada registram recall de decisão de
    0,093–0,099; a rodada levou 102 s.
    Ausência curta (6 quadros) não chega à decisão: o tracker
    segura. Confirma a falha F-09/F-11 da revisão medida: a
    recuperação por identidade nunca dispara nesses retornos.
    Não é o veredito da correção posterior já em `main`.
  - `move_same_canvas`: **REPROVADO**, zero gerações criadas, mas
    **+1.911 a +9.383 px confirmados novos fora da tolerância**
    com mão presente ou sem sinal; pior caso: mão parada no
    movimento combinado (**+9.383 px**). O registro lê a geometria
    nova como tremor e re-projeta a página que andou à posição
    antiga da câmera. A homografia que melhor alinha o confirmado
    continua sendo a identidade: o original fica e a cópia soma,
    **duplicação dentro da geração**, não reset.
  - `sway_negative` com evento de virada ligado **reprova**:
    **9 gerações no perfil biestável**. Não confundir esse teste
    de sensibilidade com a configuração real: a flag está
    desligada por padrão desde o hotfix de 2026-11-01; no padrão
    passa. A página real corrige, portanto, o negativo sintético
    que não expunha essa sensibilidade.

**Separação de causas — recall de identidade não é registro.** Na
baseline histórica, o recall **0,58–0,61** de `drop_and_return`,
abaixo do corte 0,65,
vem do nível de comparação: uma passada `>0,7` cobre só **57–62%**
do confirmado, que cresceu pelo halo da fronteira; `>0,3` cobre
**~95%**. Isso ocorre sem movimento, não demonstra desalinhamento
da homografia e não será curado por máscara de mão. Referência de
separação por conteúdo na página real:

| Comparação com o confirmado | Mesma página | Página diferente |
|---|---|---|
| Passada @0,5 | 0,86–0,88 | 0,10 |
| Núcleo erodido @0,7 | 0,84 | 0,08 |

Esses níveis são diagnósticos; a correção já integrada compara com
dilatação e contenção dual (F-11), sem baixar o recall de 0,65.
Não recalibrar F-11 junto com o portão de tremor.

- **Lacuna fechada:** falta da rodada com a página real do draw-lab.
  **Buracos restantes:** não há limiar de quedas aceitáveis; o
  movimento ainda é de quad sintético sobre mapa real, não vídeo
  real de caderno movido; a identidade corrigida tem prova sintética,
  ainda sem nova rodada com página real.

## F-10 — Apagar no papel não apaga no digital

### Canal de remoção opt-in (2026-11-12)

`DRAW_INK_REMOVAL=1`, padrão 0, acrescenta evidência negativa à confirmada.
Protocolo inicial fixado antes dos testes: probabilidade ≤0,1 em seis
passadas consecutivas **visíveis**, com pelo menos 65% do canvas antigo
ainda visto como tinta. A trabalhadora exige registro estável, sem
identidade/quarentena/abstenção em curso e sem ECC rejeitado. Oclusão,
geometria em movimento, ausência de visibilidade e perda global de tinta
não autorizam remoção. Uma trilha pendente impede descanso prematuro.

Ao remover, publica `confirmed_changed`, conta `removed_confirmed_px` e
limpa a memória de voto desses pixels (inclusive E3), para o traço não
ressuscitar da janela velha. Redesenhar continua possível. Reset e migração
transportam/zeram também a trilha negativa. OFF mantém a monotonicidade
anterior e o canvas byte a byte. Régua parcial em
`tests/test_ink_removal_registration.py`: borracha localizada, redesenho,
oclusão, flicker, virada, registro inseguro, E3/FAST e paridade OFF.

Limite explícito: apagar mais de 35% de uma vez ou toda a página fica
retido, pois a ausência total não diferencia borracha de página nova.
Também não há remoção sem máscara de visibilidade. Falta validar o canal
no papel real; o texto histórico de "sem canal" abaixo descreve o padrão
OFF, não a implementação opt-in.

**Sintoma:** a pessoa apaga um trecho com a borracha e o traço
continua no canvas para sempre (até trocar de folha/geração).

**Causa provável:** por construção. A camada confirmada do
acumulador é monotônica dentro da geração — "um pixel só entra
nela… e dela não sai dentro da geração" (`draw/ink_layers.py`,
`_LayeredInkAccumulator`). Não existe canal de remoção no caminho
page-clean: a borracha é invisível para o sistema, embora a
`VISION.md` a liste como entrada de primeira classe ("erases with
the eraser, corrects it on paper — and the correction is also
captured"). No caminho legado a histerese do `InkAccumulator` sabe
remover (fração cai a `exit`), mas esse caminho não é o da tinta em
produção.

**Severidade: maior.** Sem apagar, não há correção — e sem correção
não há o loop de feedback que é a visão do produto.

**Como se mede — SEM RÉGUA AINDA.** Régua que falta: replay com
apagamento real (trecho desenhado e apagado no papel, GT do trecho);
medir fração do trecho apagado que sai do canvas confirmado e tempo
até sair, com portão de não-regressão em F-02/F-05 (remover não pode
virar apagar traço fino vizinho nem ressuscitar fantasma).

## F-11 — Página nova lida como a mesma (ou vice-versa)

**Sintoma:** a pessoa troca a folha e o sistema continua somando
tinta no canvas da página anterior (desenhos de duas páginas
misturados) — ou, no verso, trata a mesma página como nova e perde
o acumulado (o disparo falso de 2026-11-01 era esta falha).

**Causa provável:** a identidade da página é inferida por eventos
frágeis: drop/reaparição da folha (F-09) ou o gatilho de cobertura
(`_INK_TURN_*` em `draw/ink_layers.py`, desligado por padrão). Não
há sinal positivo de "página trocou" provado contra balanço de
geometria realista.

**Severidade: maior.** Misturar duas páginas destrói as duas;
perder a atual destrói uma. É F-09/F-01 vistos pelo eixo da
identidade da página.

**Como se mede — régua parcial.** `tests/test_ink_registration.py`
reproduz os dois casos de identidade por conteúdo: página movida,
mesmo canvas confirmado pixel a pixel; conteúdo divergente, canvas
novo sem herdar os traços anteriores. `tests/test_app_session_state.py`
leva queda/retorno da mesma cena pelo laço do app e compara o canvas
entre gerações. A telemetria por sessão registra a decisão, o recall
e os pixels de referência da primeira máscara após a recuperação. O app
usa recall mínimo de 65% e contenção mínima de 50%, com comparação
dilatada pela tolerância de fronteira.

**Instrumentação Chamfer (opt-in, sem voto novo).**
`DRAW_IDENTITY_CHAMFER=1` acrescenta diagnóstico aos eventos de
identidade e aos records por passada de `draw/ink_layers.py`.
O parser, como no ECC, aceita `1/true/yes/on` (ignora caixa e espaços);
ausente ou qualquer outro valor deixa a flag OFF. Desligada, não calcula
distâncias nem acrescenta campos Chamfer/sombra. As decisões vivas continuam
usando recall 0,65 e contenção 0,5, sem limiares novos de distância.

- `chamfer_seen_to_reference_median_px` e
  `chamfer_reference_to_seen_median_px`: medianas das distâncias
  euclidianas, via `cv2.distanceTransform` (`DIST_L2`, `DIST_MASK_PRECISE`)
  sobre o complemento do alvo. Unidade: pixel da grade registrada na
  âncora, não pixel canônico reescalado. Com visibilidade, a referência
  é `prior_visible`; sem ela, `prior`. Os conjuntos não são dilatados.
- `chamfer_{seen_to_reference,reference_to_seen}_coverage_{2,4,8,16}px`:
  fração dos pontos da origem a distância **≤** ao raio, nos dois sentidos.
  Estes raios são apenas medidas de cobertura, nunca limiares de decisão.
  Origem vazia: mediana e coberturas `null`; alvo vazio com origem não
  vazia: mediana `null` (distância indefinida), coberturas zero. Referência
  indisponível: todas as métricas `null`, nunca sentinelas enormes do OpenCV.
- `ecc_refine_enabled`: estado explícito 0/1 de `DRAW_ECC_REFINE`.
  Emitido em **todos** os eventos de identidade e records de passada,
  mesmo com Chamfer desligado, assim como `ecc_rejected`. Não misturar
  distribuições de estados diferentes sem anotação.
- `shadow_decision`: `defer` se `ecc_rejected=True`; caso contrário,
  a decisão viva. Continua sendo diagnóstico opt-in: não vota por distância.
  Decisão/recall/contenção existentes continuam lado a lado; nos records
  por passada eles conservam os nomes `page_identity_*`.

**F-11 Block 2 — quarentena em produção.** Com `DRAW_ECC_REFINE=1`,
`ecc_rejected=True` adia a decisão viva para `deferred_ecc_rejected`, nunca
`new_page`. A referência e a âncora antigas ficam disponíveis para a
próxima passada, e a evidência rejeitada se abstém do voto e da virada:
o canvas não é contaminado nem zerado durante a quarentena. O adiamento
por oclusão continua como `deferred_occlusion`.

Eventos e records declaram `deferred`, `quarantine_reason` (`ecc_rejected`,
`occlusion` ou `null`) e `quarantine_frames`, número de passadas
adiadas nesta recuperação. O teto configurável no módulo
`_PAGE_IDENTITY_MAX_DEFERRALS = 5` é compartilhado pelos dois motivos;
a recuperação pendente não entra em descanso total. Uma decisão concluída
ou geração nova zera o contador. Se a próxima passada ainda adia após
atingir o teto, emite `quarantine_timeout` (não é voto de página nova),
descarta referência e canvas e perde a âncora, sem aceitar a evidência
daquela passada. A próxima passada começa com âncora e acumulador limpos.
O evento terminal preserva a contagem e o motivo; os estados internos
da quarentena são limpos. Este teto é um default operacional, não um
limiar de identidade calibrado nas quatro páginas reais.

**Cláusula C — consenso de troca.** Com referência disponível, `new_page`
só se consuma após **3 vereditos limpos consecutivos** de página diferente
(`_PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES = 3`). Nas duas primeiras passadas,
`deferred_switch_consensus` preserva referência, âncora e canvas, zera a
evidência da passada e mantém o descanso total bloqueado. Uma passada
adiada por ECC/oclusão quebra a consecutividade; `same_page` é imediato.
O consenso pendente não incrementa nem zera o contador do teto de **5**
adiamentos ECC+oclusão: é um adiamento separado, com `deferred=False`,
`quarantine_reason=null` e `quarantine_frames=0` no diagnóstico da passada.
O terceiro voto limpo consome a referência, troca a âncora e zera ambos os
contadores, publicando o canvas novo mesmo se estiver vazio; geração nova
também zera os contadores. `quarantine_timeout` nunca vota
`new_page`. Eventos e records de consenso mantêm `ecc_refine_enabled`
explícito como inteiro 0/1, com ou sem Chamfer. O flag ECC continua OFF
por padrão; recall ≥ 0,65 e contenção ≥ 0,5 não mudam.

O replay usa os mesmos nomes de diagnóstico nos eventos. Com a flag on,
`Episode.report` inclui `identity_events`, `identity_true_class` e
`identity_chamfer_by_ecc`: valores individuais, contagem, ausentes,
min/p25/mediana/p75/max para cada métrica, separados por estado ECC.
`identity_chamfer.by_true_class` agrega `move_same_canvas` e
`drop_and_return` como `same_page`, e `drop_and_turn` como `new_page`,
independentemente da decisão inferida. Episódios sem evento de identidade
continuam sem amostras (não são zeros). A divergência pré-existente do
replay permanece: não passa `visible`, usa `prior` inteiro e não adia por
oclusão. A rejeição ECC adia o replay vivo; a contagem de votos e a janela
de consenso de troca também espelham o worker.

Para medir sem calibrar corte, executar os dois estados separadamente:

```sh
DRAW_IDENTITY_CHAMFER=1 DRAW_ECC_REFINE=0 python tools/replay_reregister.py --synthetic --out chamfer-ecc0.json
DRAW_IDENTITY_CHAMFER=1 DRAW_ECC_REFINE=1 python tools/replay_reregister.py --synthetic --out chamfer-ecc1.json
```

Remover `--synthetic` só usa dados reais se o laboratório estiver presente;
conferir sempre `source`. Mesmo com `source=real`, `drop_and_turn` usa
espelho vertical como proxy, não uma segunda folha real. A amostra de
quatro páginas reais não permite promover qualquer limiar novo.

- `tools/replay_reregister.py`, cenário `drop_and_return` (ver F-09):
  antes da correção de identidade, `new_page` em 16/16 retornos da
  mesma página nas baselines sintética e real (recall 0,58–0,61 <
  0,65; níveis de comparação, não registro). Depois da correção,
  **resolvido na régua sintética** — a recuperação decide `same_page`
  nos 16 drops e o canvas sobrevive 24/24 (ver F-09 para os números).
  A correção: a comparação de identidade dilata os dois lados pela
  tolerância de fronteira (`_IDENTITY_DILATION_KERNEL = 2 ×
  _INK_FRONTIER_RADIUS + 1`) e exige contenção dual — o recall da
  tinta antiga reencontrada **e** a tinta da passada caindo sobre o
  canvas anterior, ambos ≥ `_PAGE_IDENTITY_MIN_RECALL`/`_PAGE_IDENTITY_MIN_CONTAINMENT`
  = 0,65 no recall e 0,5 na contenção. A contenção fica abaixo do
  recall porque tinta nova desenhada durante o drop é legítima:
  medição que calibrou (2026-11-06, 4 mãos × 3 estágios): mesma
  página recall/containment 0,876–0,925 / 1,0; página diferente
  0,218–0,240 / 0,231–0,251; mesma página com +75% de tinta nova
  cai para containment 0,65 — com 0,5 aceita ~1,8x de crescimento
  e mantém margem de 2x contra página diferente. Sem esse teto, a
  guarda rejeitaria a mesma página e perderia o canvas. Na página
  real (2026-11-06), uma passada @0,7 cobre só ~10% do canvas
  confirmado de 134.574 px, contra ~60% do sintético (~12.600 px)
  — a calibragem sintética do limiar não transfere sozinha.
- `tools/replay_reregister.py`, cenário `drop_and_turn` (guarda da
  correção): a página de conteúdo distinto que volta após o drop
  decide `new_page` em 8/8 drops efetivos na régua sintética
  (recall dilatado ~0,025), sem herdar o canvas — a dilatação não
  abriu falso positivo nesse teste. Na fonte real, a régua usa
  espelho vertical da página como proxy de conteúdo distinto, não
  vídeo de uma troca física de folha.
- `tools/replay_reregister.py`, cenário `sway_negative` (ver F-07):
  página parada, o ajuste do detector balança até 314 px de canto
  por 60 s de cena — perfil suave (senoides de 2–5 s por canto) e
  biestável (um lado do quad salta para a outra página do caderno
  aberto, permanência de 1–4 s). Portão: `false_page_turns = 0` na
  configuração padrão; com `DRAW_PAGE_TURN_EVENT=1` o resultado é
  reportado à parte, junto com `coverage_turn_signals`.
- Baseline sintética histórica (2026-11-06): 8/8 sem página nova nas duas
  configurações da flag, zero sinal de cobertura. O biestável leva a
  geometria da tinta a 314 px da página e o registro na âncora
  absorve; o suave nem chega à tinta (o candidato de salto do
  rastreador nunca persiste, excursão < 1 px). 4 dos 8 passam por
  vacuidade (ausente e presente_parada: zero passadas). Ressalva: o
  disparo falso de 2026-11-01 foi medido antes do registro e com o
  modelo rodando no quadro re-warpado; aqui o mapa é re-warpado.
- Baseline com página real (mesma data): balanço **aprovado no
  padrão** (8/8), mas evento de virada ligado cria **9 gerações
  no biestável** (2/8 reprovados). O resultado real substitui a leitura de que a flag
  ligada seria segura; ela continua desligada por padrão.
- **Buraco:** continua faltando o corpus de transições reais (troca
  real de folha × balanço/movimento sem troca, os dois em vídeo
  real), com portão duplo: toda troca detectada em ≤ N s **e** zero
  troca falsa por hora de balanço. O teste negativo tem que incluir
  balanço realista (314 px de canto medidos), não borracha em warp
  estático — a lição do hotfix está registrada no código.

## Pré-registro — portão de tremor sobre a homografia (F-07/F-09)

**Estado: experimento desenhado, ainda sem veredito.** Primeiro
implementar uma variante isolada na régua, depois executar no
laboratório; **não modificar `draw/ink_layers.py` neste passo**.
A hipótese é registrar apenas no regime de tremor; movimento
confirmado re-ancora a geometria **sem tocar o canvas confirmado**.
Não mudar identidade, descanso, fronteira, modelo ou flag de virada
no mesmo A/B: assim uma melhora não pode ser atribuída a outro fix.

### Sinal e limite de identificação

- Usar a trajetória dos quatro cantos que efetivamente chegam à
  passada, depois do tracker e da mediana do app. Projetar esses
  pontos pela homografia da âncora numa grade fixa; medir amplitudes
  e deslocamentos em pixels dessa grade, não pela norma dos
  coeficientes da homografia (dependente de escala).
- Numa janela de K passadas, por canto, medir caminho total L,
  deslocamento líquido D, razão D/L (zero quando L=0), excursão
  desde a âncora e reversões de direção. Tremor: pequena excursão,
  reversão/retorno e baixo D/L. Movimento sustentado: deslocamento
  líquido persistente, alto D/L e sem retorno; incluir deriva lenta
  acumulada, mesmo quando cada passo é subpixel. Usar o máximo
  entre cantos, não a média vetorial que anularia uma rotação ou
  esconderia a mão puxando um só canto.
- **Homografia sozinha não identifica a causa física.** Uma deriva
  monotônica pode ser o caderno andando ou viés do detector com a
  página parada. O caso de mão parada sobre um canto, documentado
  em `draw/ink_layers.py` (20.137 px congelados contra **71.548 px**
  sob viés sustentado, antes do registro), é um controle adversário
  obrigatório: re-ancorar todo viés monotônico pode recriar F-01.
  Os valores dessa régua anterior não são novos limites de aceite.
- Desambiguar na régua comparando **antes de votar** a evidência
  sem registro e a registrada com uma referência de tinta estável
  anterior à perturbação. Comparar núcleo e suporte com a mesma
  tolerância de fronteira, nos dois sentidos (recall e contenção),
  sem usar o confirmado já contaminado pela cópia. Se só o registro
  reencontra a referência, é evidência de erro geométrico; se o mapa
  sem registro reencontra e o registrado desloca, favorece movimento
  físico/re-ancoragem. Não reutilizar o recall F-11 @0,7 contra halo
  como se fosse teste de alinhamento. Sem tinta suficiente, ou se
  ambos os candidatos empatam/falham, marcar ambíguo; não inventar
  identidade ou declarar movimento resolvido só pela trajetória.

### Variante causal e parâmetros a congelar antes da rodada

1. **Tremor/estável:** manter a âncora e o registro atual quando a
   janela caracteriza oscilação reversível ou quando a comparação
   de conteúdo favorece apenas o registro (viés do detector).
2. **Suspeita:** ao sair do envelope, reter evidências em buffer
   limitado, sem alimentar o voto com mapas de referencial incerto.
   A classificação usa somente passado/presente; sem olhar quadros
   futuros e sem bloquear por tempo ilimitado para fingir invariância.
3. **Movimento confirmado:** persistência direcional mais conteúdo
   favorecendo o mapa sem registro autoriza atualizar a âncora aos
   cantos atuais. Não chamar `new_generation()`, limpar/re-warpar o
   confirmado ou mudar sua contagem/hash. Descartar apenas evidência
   transitória de referencial incompatível; as próximas passadas
   voltam a votar na mesma grade canônica. Se o conteúdo exigir uma
   transformação residual, esta tem de ser estimada e validada na
   régua, nunca presumida igual à identidade.
4. **Ambíguo:** conservar confirmado, registrar abstenção e aguardar
   no máximo a janela limite. Ao esgotá-la sem resolver, a célula é
   inconclusiva/reprovada por disponibilidade, não aprovada porque
   não votou. Queda/retorno continua pelo caminho `main` inalterado;
   descartar buffer/estado temporal na troca legítima de geração.

Grade inicial **proposta**, não calibrada: envelope de 1/2/4 px na
grade da página; K=3/6/12 passadas; D/L mínimo de 0,8/0,95 para
persistência, exigindo saída do envelope. Para comparar conteúdo,
margem mínima de 0,05 tanto em recall quanto em contenção entre
candidatos. Referência = núcleo erodido do confirmado antes da
perturbação, fixo até a decisão; suporte candidato = probabilidade
`>0,5`. Recall mede referência coberta pelo suporte dilatado;
contenção mede suporte coberto pela referência dilatada, ambos com
`_INK_FRONTIER_RADIUS`. Exigir ≥256 px em cada suporte, ou abster.
Se houver preferência pelo mapa sem registro mas ainda não houver
persistência, reter em suspeita, não votar com registro só por estar
dentro do envelope. Para voltar de suspeita a tremor, exigir duas
janelas consecutivas com D/L ≤0,35 e retorno ao envelope da âncora,
sem preferência de conteúdo pelo mapa sem registro. Após re-ancorar,
reiniciar a trajetória na nova âncora, não o canvas. Tempo máximo de
suspeita = 12 passadas. Registrar esses parâmetros no pré-registro
local **antes de executar**; mudança exige nova rodada declarada.
Nenhuma combinação dessa grade é vencedora por antecipação.
Separar sequências de calibração e confirmação antes
da varredura; congelar um candidato e executá-lo nas sequências
reservadas, sem retocar cortes após vê-las. Sem confirmação reservada,
reportar exploratório, não prova para produção.

### Protocolo no laboratório (medições ficam lá)

Usar `tools/replay_reregister.py` como infraestrutura e preservar o
espelho de `main` como braço A. A variante do portão é o braço B,
somente na régua. Entrada obrigatória: cache
`~/workspace/draw-lab/replay-ruler-2026-11-01/prob_cache.py` e warps
`~/workspace/draw-lab/real-negatives-2026-10-31/proc/warps` (f01–f30).
**Não aceitar fallback sintético como rodada real.** Guardar scripts,
hashes de código/modelo/entradas, parâmetros, mapas e JSON no lab;
somente protocolo e vereditos textuais viajam para o repo. Baselines
rodam ali sob demanda. A baseline real histórica de F-09 fixa os
números do problema de movimento. Como a identidade mudou em `main`,
repetir A com a revisão corrigida e entradas reais antes de comparar
B; manter a mesma identidade nos dois braços.

Executar A/B com as mesmas entradas, ruído, latência e quatro regimes
de mão, separados por célula, nunca só uma média:

- `control_static`, cinco movimentos de `move_same_canvas`,
  `drop_and_return` e `drop_and_turn` de 6/30/150 quadros nos dois modos e
  `sway_negative` suave/biestável até 314 px. Flag de virada
  desligada como configuração decisória; ligada apenas diagnóstico.
- Tremor reversível de ±1 px, deriva lenta de 3 px e viés sustentado
  de um canto com **página verdadeira parada**; repetir trajetórias
  com **página verdadeira movendo**. Incluir rampas lentas, movimento
  abrupto, rotação/escala, parada fora da âncora e ida-e-volta longa:
  não classificar o trecho inicial com conhecimento do retorno futuro.
  O balanço de 314 px é negativo obrigatório mesmo não cabendo no
  envelope de tremor pequeno.
- Após reassentar, inserir traço novo conhecido e verificar posição
  e confirmação. Repetir com passadas efetivas: `ausente` em descanso
  não prova re-ancoragem. Preservar a execução padrão para registrar
  vacuidade e usar um braço diagnóstico acordado para cobrar geometria.
- A régua re-warpa probabilidades, não reexecuta ONNX sob uma mão:
  não chamar esses regimes de corpus de oclusão nem de vídeo real.

Por passada registrar cantos, homografia proposta/aplicada, estado do
portão, D/L/reversões, decisão de conteúdo, re-ancoragens, buffer,
abstenções, passadas votadas/retidas, hash/contagem do confirmado,
px novos fora da tolerância, perdas, gerações e erro de geometria.
Contar px espúrios **durante todo o movimento**, não só no fim: uma
cópia transitória não pode desaparecer da avaliação por alinhamento
ECC posterior.

### Critérios de sucesso e saída

| Caso | Exigência para o portão |
|---|---|
| `move_same_canvas` | Eliminar os **+1.911 a +9.383 px**: `new_confirmed_px_outside_tol = 0`, nenhuma perda, canvas idêntico segundo a tolerância vigente e `generations_created = 0`; sem crescimento espúrio nas passadas intermediárias. |
| Re-ancoragem | Hash do confirmado inalterado no evento; próximo traço no lugar (`ink_geometry_error_px ≤ 5`, limite já usado na régua), confirmado em até 12 passadas após ficar visível. Não vale preservar canvas congelando a captura. |
| `drop_and_return` e `drop_and_turn` | Manter os portões e o comportamento de A na revisão corrigida: mesma página sobrevive sem perda e com crescimento residual ≤25 px; conteúdo distinto após drop decide `new_page`, sem herdar canvas. 6 quadros sem drop não provam identidade. Comparar eventos/recall/contenção com A; se A falhar na nova rodada real, registrar a falha separada e não atribuí-la ao portão de tremor. Os 16/16 `new_page` históricos não são alvo a preservar. |
| `sway_negative` e `control_static` | Continuar aprovados no padrão, sem geração espúria; manter explícitas as células sem passadas. A flag ligada continua diagnóstico separado. |
| Tremor, deriva e viés com página parada | Nenhuma cópia fora da tolerância nem perda contra a referência assentada; não trocar o ganho no movimento pela volta do mecanismo dos 71.548 px. |
| Disponibilidade | Abstenção limitada a 12 passadas; toda aprovação de movimento exige passadas após a perturbação e prova do traço novo. Empate/ausência de suporte não passa por vacuidade. |

Resultado: **REPROVADO** se violar um portão, **INCONCLUSIVO** se
faltarem dados/células/identificação, ou **APTO A PROPOR MUDANÇA EM
PRODUÇÃO** se passar na confirmação reservada. Nenhum desses estados
promove modelo ou altera o app. A mudança em produção só vem em passo
posterior, depois do veredito da régua e das provas restantes na mesa.

## O buraco de medição, resumido

Com régua e limiar hoje: **F-02, F-05, F-06** (tinta, via
`tools/judge.py` + `judge_ink.py`) e **F-09** sem limiar
(`judge_sheet.py`). Parciais: **F-01/F-07** (testes de unidade do
descanso + replay de laboratório fora da bancada + régua de
re-registro do laço, sem portão pré-registrado em `tools/judge.py`;
F-11 com portão na régua de re-registro, mas ainda sem vídeo real),
**F-08** (componentes medidos, sem orçamento de ponta a ponta),
**F-11** (replay de identidade e balanço de 314 px com página real,
sem vídeo real) e **F-09** (quedas contadas
sem limiar); **F-04** tem testes de visibilidade sintética, ainda sem
corpus de oclusão real. Sem régua dedicada: **F-03 e F-10** — falhas de
cena/acumulador, não de modelo isolado.

O "moveu = mesmo canvas" de F-07/F-09 deixou de ser buraco de
medição e virou falha medida: `tools/replay_reregister.py` reprova o
laço no cenário de movimento. A baseline **com página real**, lógica
`main` anterior à correção F-11, de
2026-11-06 confirma a sintética: movimento com mão presente ou sem
sinal soma **1,9–9,4 mil px** de cópia deslocada sem nenhum
`new_generation()`; após 30/150 quadros de ausência, a identidade
decidia `new_page` em **16/16**, por comparação de níveis, não por
registro. Com a correção de identidade já em `main`, a régua
**sintética** preserva a mesma página 24/24 e rejeita o conteúdo
distinto nos 8 drops efetivos. Ainda falta repetir a rodada de
página real nessa revisão corrigida. Controle e balanço passam no
padrão na baseline real histórica; o evento de virada
ligado cria 9 gerações no biestável. **A lacuna da página real está
fechada**. Falta vídeo real de caderno movido e provar o portão de
tremor pré-registrado acima; o ataque de oclusão é outra frente.
O padrão é o achado: a bancada julga bem o modelo de tinta; o que
não tem juiz é o comportamento do sistema com a mão na cena, com a
página trocando e com o tempo passando — que é onde moram as queixas
recorrentes do usuário.

**Buraco de observabilidade da aquisição (2026-11-10).** Na sessão ao
vivo de 2026-11-10 o app passou ~10 s lendo o mousepad do usuário como
folha e abriu 2 gerações sobre essa cena, e o sub-mecanismo ficou
indemonstrável: a sessão gravada só guardava warps por geração (a fase
sem folha e de posicionamento, onde o fantasma nasce, ficava cega) e a
telemetria não dizia, por aquisição, o `presence_logit`, a fonte do
desfecho combinado (neural × clássica × veto × presença-só) nem os
cantos. O replay posterior com o caminho combinado + rastreador sobre
todo o material salvo deu zero falsos positivos — o que só mostra que o
material não cobre essa fase. Desde então, toda sessão gravada
(`draw/selflearn/session.py`) guarda quadros crus esparsos em `raw/`
(1 a cada 30 passadas + toda transição `no_sheet`/`candidate`/`sheet`,
teto de 600 quadros ou 1 GiB, corte registrado no `manifest.json`) e um
registro por aquisição em `acquisitions.jsonl` e no evento `acquisition`
da telemetria (`draw/app.py`). É só observabilidade: nenhum limiar,
default ou veredito F-11 acima muda; defeito do caminho que esses
registros revelarem entra aqui como achado separado, sem conserto no
mesmo PR.

**Vigia de identidade v1: latência na virada e puxão indistinguível
(2026-11-10).** A prova da vigia v1 (`DRAW_IDENTITY_WATCH`) sobre as
sessões gravadas aprovou a regressão sintética e a telemetria, mas
reprovou dois portões reais com causa medida. (1) A virada da tarde
ficou a UMA passada do veredito: as duas primeiras passadas
pós-virada foram bloqueadas pelo portão de geometria quieta da
própria vigia (deslocamento de 9,53 px contra o teto de 8,0 px — a
perturbação da própria virada), e a gravação acabou com 2 de 3 votos.
(2) O puxão do caderno (mesma página, pose nova, spread enrolado) foi
partido como página nova: retificado na pose nova e registrado sobre
a âncora congelada, o conteúdo sobrepõe ~0,05 com o canvas anterior —
e a frente seguinte provou que NÃO existe separador de conteúdo
nesta régua: a cobertura em raio grande satura com a densidade da
tinta, não com a identidade (a página da substituição cobre a
referência melhor que a mesma página deformada em todo raio
8–48 px), e a condenação do puxão nasce antes dele, na era de
posicionamento (recall 0,19 medido sobre conteúdo ainda derivando
sobre a âncora). A v3 (mesma flag, default 0) responde sem inventar
evidência: a vigia conta só passada assentada (degrau entre quads
consecutivos ≤ 6 px) com deslocamento da âncora ≤ 16 px; o
armamento por ruptura semeia 1 voto no juiz (a evidência da vigia,
recall ≤ 0,30 ×2, é mais estrita que 1 voto, < 0,65); e o chamador
`pose_jump` só arma quando o episódio do salto foi DESCONTÍNUO
(candidato estabelecido substituído, veredito não-papel no meio do
episódio, ou candidato morto por perda de detecção — marcadores do
rastreador em `draw/sheet_track.py`), porque reposicionamento
contínuo é o gesto constante do usuário e partir um desenho em
andamento é o erro mais caro. Assimetria declarada: troca de folha
em episódio limpo direto pra pose nova está fora deste chamador; a
vigia de conteúdo e a porta da ausência seguem cobrindo o resto.

A prova da v3 nas sessões gravadas (frente 33) aprovou o puxão e a
manhã, mas reprovou a virada e a substituição da tarde por uma causa
arquitetural, não de limiar: sem o split falso da v1, nada movia a
âncora, e virada/substituição ficavam a 145–155 px da âncora
congelada, fora do teto da própria vigia; no braço com arbitragem, o
canvas congelou em 2.117 px por livelock (vereditos indeterminate em
ciclo, a divergência assentada mantendo o evento aberto). A frente 34
responde com a mobilidade da âncora por divergência assentada
(`DRAW_ANCHOR_FOLLOW`, default 0, em `draw/ink_layers.py`): pose
assentada (degrau entre quads consecutivos ≤ 6 px) e deslocada da
âncora (> 8 px de grade) por 3 passadas, sem identidade pendente,
re-ancora por SUBSTITUIÇÃO — âncora := mediana dos quads do streak e
estado intocado, porque a grade já é o referencial da folha e
deformar o estado pela homografia do raio da câmera é o carry
reprovado da migração v1; a arbitragem tem o primeiro veredito
(pouso decisivo pre-empta, spurious retém o episódio, indeterminate
deixa o FOLLOW disparar e fechar o evento). Na prova sobre as
mesmas sessões: puxão sem split com preservação 1,000 do canvas
pré-puxão, substituição dividindo na passada 28, virada dividindo na
passada 52 (a prevista pela simulação da frente 32), livelock
quebrado (página fechada de 27.162 px), manhã e regressão dígito a
dígito. Ressalva medida: na composição só com a vigia (sem a âncora
mediana), o mecanismo também segue micro-derivas assentadas da era
de posicionamento (re-ancoragens com Δ 11,7–16,7 px de câmera) e a
abstenção delas encolhe o canvas pré-puxão (2.032 contra 6.686 px
sem a flag); na composição com arbitragem+mediana o efeito não
aparece. Contratos em `tests/test_anchor_follow.py`.

**Réguas experimentais e o selado (2026-11-11).** `tools/seg_metrics.py`
ganhou clDice (`ink_cldice`, conectividade do traço — o F-02 partido
perde Tsens com IoU alta) e avaliação restrita a ROI
(`evaluate_in_roi`, `stroke_roi`, `interior_roi`), como diagnóstico ao
lado da IoU: nenhum portão as lê e nenhum limiar F-xx mudou. A
medição real segue presa ao holdout selado; a revisão dele agora tem
ferramenta (`tools/selflearn/review_holdout.py`), mas a revisão é humana.

Amarração com a bancada: o mapa ID → régua acima é a referência que o
`JULGAMENTO.md` pode citar; nenhuma régua da bancada foi alterada por
este documento. A guarda automática desta página é
`tests/test_falhas_doc.py`: ela falha se um caminho de repo citado
aqui deixar de existir ou se um ID do mapa sumir do texto.
