# Memória do projeto — decisões, situações e erros

_Este documento é a memória viva do Draw, mantida pela Julia (a IA que
trabalha no projeto desde o primeiro dia) a pedido do Santleme, em
2026-11-06. Ele registra o que foi decidido, o que aconteceu de
verdade e o que deu errado — principalmente as situações, porque é
nelas que as decisões fazem sentido. Números citados como "medidos"
foram medidos no ambiente indicado; número de laboratório não é
veredito — o veredito é o holdout selado e o teste ao vivo na mesa._

---

## 1. A ideia e a visão

- **2026-10-28 — a ideia (do Santleme).** Um sistema de visão que
  funciona como mesa digitalizadora sem mesa digitalizadora: a webcam
  sobre a mesa, a pessoa escreve/desenha no papel comum, e o sistema
  reproduz no digital o que a mão fez no papel. Repo `santleme/draw`
  criado neste dia, público.
- **Visão maior (2026-10-28).** Ponte em tempo real entre o desenho
  no papel e sistemas profissionais e simuladores (AutoCAD, NVIDIA
  Isaac Sim/Lab): desenhar e apagar no papel e ver o sistema
  interpretar os traços nesses ambientes, com feedback imediato de
  simulação (ex.: resistência de materiais) — sem desenhar com o
  mouse nesses softwares.
- **Primeira barreira (desde o dia 1).** A captura papel → digital:
  detectar a folha em qualquer papel, retificar, extrair a tinta.
  Tudo o resto vem depois dela.

## 2. Decisões de direção (do Santleme)

- **Sem folha-modelo (2026-10-28).** Nada de depender de folha
  impressa com marcadores: a detecção tem que funcionar em qualquer
  folha, caderno, bloco ou livro. Marcadores só como modo opcional.
- **A rota neural é a estrada principal (2026-10-28).** O detector
  clássico (contornos) é a base/fallback; a direção é modelo de visão
  neural, porque a meta é reconhecer qualquer tipo de papel. O medo
  declarado desde 2026-10-29: **generalização** — um modelo que só
  reconhece o papel e a mesa de uma pessoa não é sucesso.
- **YOLO fora (2026-11-02, por medida, não por princípio).** Não
  serve a nenhuma das três funções: layout por caixas não dá máscara
  de tinta e colapsa no frame real da mesa; caixa retangular não dá
  os 4 cantos precisos do warp; traço fino não é "objeto", tinta é
  segmentação pixel a pixel. Soma a licença AGPL.
- **Reinício "tudo de uma vez" (2026-10-31).** Reproduzir o desenho
  limpo ao lado, a partir da página inteira — não generativo:
  segmentação pixel-alinhada (página retificada → máscara de tinta).
  Um comando só (`python main.py`), detector neural como padrão.
- **Invariância de posição (2026-11-01, especificação dele).** Só
  alteração REAL muda o canvas. Mover o caderno = só re-registro,
  conteúdo idêntico. Registrar a tinta **uma vez** e adaptar por
  **substituição** (apagar, sobrepor, melhorar) — nunca acumular
  camada sobre camada. Página assentada com a mão parada = **zero**
  passadas de tinta (descanso total, estendido à mão presente parada
  em 2026-11-03 por decisão dele).
- **Sem aprendizado online dentro do app (2026-11-02).** O
  self-learning é um loop fora do app, sobre sessões gravadas.
- **Modelos SEPARADOS (2026-11-05, por interferência medida).** No
  modelo único da linhagem T7, tinta e folha brigavam (ver §5): a
  direção passou a ser um modelo por tarefa. **Em 2026-11-06 o
  Santleme propôs revisitar**: treinar um próximo modelo juntando
  folha + tinta + mão num só, com as regras dele — a avaliação está
  aberta e o modelo C (abaixo) é a primeira evidência nova.
- **Tinta × impresso = supervisão de 3 classes (2026-11-02).**
  Fundo (0) × impresso (1) × tinta (2); o descarte do impresso é por
  IDENTIDADE (o modelo de captura consome só P(tinta)), nunca por
  posição. Régua: vazamento de impresso ≤ 0,5%.

## 3. Decisões de processo (regras permanentes)

- **Commits diretos na main (2026-10-28).** Nada de PRs nem issues
  por padrão — cada um disparava um pedido de autorização que
  travava o avanço. Exceções só quando ele autoriza (aconteceu com
  as PRs #33–#35 e com a issue #36, o guarda-chuva do programa
  temporal, cuja implementação está **parada por ordem dele**).
- **Autonomia total (2026-10-30).** A Julia pode avançar o projeto
  como julgar melhor, inclusive treinar no Kaggle com GPU, sem pedir
  permissão a cada passo. Seguem sendo dele: troca de modelo em
  produção, tornar algo público, gastos.
- **Dataset mora no Kaggle, nunca no disco dele (2026-10-30, regra
  dura, nasceu de um erro — ver §6).** Fluxo: gerar em staging
  temporário → subir → confirmar → apagar local, no mesmo fluxo.
  Reforço de 2026-11-05: nada de baixar dataset no WSL dele; corpus
  de terceiros baixa de dentro de kernel do Kaggle.
- **Posicionamento por velocidade total (2026-10-29).** Cada
  trabalho roda onde o tempo total (setup + transferência +
  disponibilidade) é menor, e a Julia diz na mesma mensagem **onde**
  está rodando e por quê. Medido: geração de dados (CPU) é mais
  rápida no Ryzen dele (478 s × 1.587 s no kernel do Kaggle, no
  dataset da tinta); treino com GPU/TPU é Kaggle/Colab; a VM da
  Julia é para código, réguas e provas pequenas. O túnel VM↔WSL
  sustenta só ~0,5 MB/s: arquivo grande não atravessa por ela.
- **Comando no WSL dele quem roda é a Julia (2026-10-31).** Via
  `lssh`, em poucas conexões agrupadas, sem despejar blocos de
  comando pra ele colar. As mãos dele são só para o físico: mirar a
  câmera, desenhar, desplugar. Exceção: quando ele está dentro de
  uma sessão viva (Kaggle/Colab), ele recebe o comando exato — e a
  Julia confirma antes a plataforma e o acelerador reais da sessão
  (regra nascida dos erros de 2026-11-05, ver §6).
- **Teste ao vivo na mesa antes de refinamento de laboratório
  (2026-10-30).** Número de laboratório é número de laboratório: a
  Julia rotula como medido-em-laboratório, nunca como fato
  verificado por ele. Promoção de modelo exige **holdout selado**
  (≥ 5 sessões inteiras, ≥ 25 páginas, rótulo independente
  revisado à mão) **+ o teste ao vivo dele**.
- **Lançamento de treino = protocolo de portões (2026-11-02).**
  Auditoria de imports/device contra o runtime-alvo, probe do
  ambiente, **smoke end-to-end verde no mesmo runtime antes da
  corrida completa**, log visível + checkpoint + sync desde o
  minuto zero, uma alocação por vez, rota morta não se repete em
  loop. Utilização é portão (2026-11-03): smoke mede GPU util/VRAM,
  RAM e tempo/passo; loader estrangulando a GPU reprova o smoke.
  Treino conservador é defeito.
- **Agressividade é a regra padrão (2026-11-03).** Nas decisões
  discricionárias, sempre a versão rápida e ambiciosa primeiro.
  Cautela autoimposta é defeito. Não revoga os portões de medição
  que ele ordenou nem as decisões que são dele por autoridade.
- **Treino mora no repo (2026-11-05).** O corpo de treino vive em
  `training/`, versionado, melhorado no lugar — nada de scripts
  descartáveis por corrida. Kaggle/Colab são lançadores finos:
  clonar o repo e rodar (`training/run.py --backend auto` roteia
  CUDA/TPU). Lançar = conferir os pré-requisitos **antes** (cota
  real, runtime, acesso) — regra de 2026-11-05, nascida do erro da
  cota anunciada como saldo.
- **Informação atualizada é obrigatória (2026-11-05).** Nunca
  afirmar de memória de treino fato que muda com o tempo (hardware
  de plataforma, versões, técnicas, cotas, preços): pesquisar com
  fontes correntes e citar fonte e data. Quando ele pede pesquisa
  de 2026, as fontes são de **2026 mesmo** — fundação pré-2026 só
  isolada e nomeada como exceção.
- **Pagamento só via Pix (2026-10-29).** Sem cartão de crédito:
  qualquer rota de infraestrutura precisa funcionar com Pix, boleto
  ou sem método de pagamento. Modal/DigitalOcean descartados por
  exigirem cartão. Colab Pro dele (ativo desde 2026-11-05) foi via
  Google Play.
- **Texto público/externo:** ele é **@santleme / Santleme**, nunca
  o primeiro nome. Contribuição open source em inglês profissional,
  revisada por ele antes de qualquer envio.
- **Frames privados fora de repo e de dataset público, sempre.**
  Datasets públicos (decisão dele, 2026-11-05, virada irreversível
  no Kaggle): os sintéticos e os espelhos limpos de terceiros.
  Privados para sempre: as capturas reais da mesa dele e derivados.

## 4. Ambiente e máquinas (o palco das situações)

- **Máquina do Santleme:** Ryzen 7 7700X, 32 GB de RAM, **sem placa
  de vídeo**. Todo o ML roda no WSL dele (nada de ML no Windows).
  Câmera: webcam USB via usbipd-win ("1200W Camera"; MJPG 1280×720
  anda, YUYV 640×480 **trava** no usbipd — medido em 2026-10-31).
  Acesso da Julia: Tailscale + SSH direto no WSL (wrapper `lssh`).
  O WSL dele é systemd PID 1 e o sudo pede senha; cada conexão
  `lssh` religa a distro se ela estiver desligada — por isso as
  operações são agrupadas e nenhuma vigia faz polling nela.
- **VM da Julia:** 7,7 GB de RAM (avaliações ONNX de página cheia
  rodam **solo** — duas em concorrência derrubam o ritmo de ~13 s
  para ~52 s por par e a terceira morre por pressão de memória).
- **Kaggle:** contas guzzbr e pascalcoimbra (multi-conta é decisão
  dele, 2026-11-05). Cota de GPU: 30 h/semana por conta, reset às
  sextas 21:00 (horário de São Paulo). GPU atual: T4×2 (a P100 foi
  aposentada em 2026-09-15); no máximo 2 sessões de GPU em paralelo.
- **Colab Pro (dele):** varredura medida em 2026-11-05 com o passo
  real do T7: G4 é o acelerador mais rápido por passo (0,0445 s),
  depois A100 (0,0744 s), L4 (0,2315 s), T4 (0,3468 s, em fp16);
  H100 é recusado por entitlement mesmo no Pro. A sessão web dele
  **não** é endereçável pela CLI — mede-se por sessões gêmeas, sem
  tocar na viva.

## 5. Os modelos e seus vereditos (a bancada e a mesa mandam)

Produção hoje: **folha = FT dinâmico 384** (fine-tune de 2026-11-03,
instalado por decisão dele) e **tinta = w8**. Nenhum desafiante
passou nas duas instâncias (bancada + mesa) até agora.

- **Folha v3 + FT dinâmico.** O fine-tune dinâmico (forma leve) virou
  o padrão instalado; no teste ao vivo de 2026-11-03 ele relatou
  "travou menos" (detecção 42,2 ms/passada contra 81,5 ms da forma
  512; tinta ~105 ms). A w8 da tinta é **provisória** por decisão
  dele (2026-11-02): "tem coisa para melhorar real, a velocidade e
  tudo" — a ordem é treinar melhor, não cristalizar.
- **T7 v5e-1 (TPU Colab, 18 épocas).** Tinta mais que dobrou contra o
  v6e-1 (IoU de tinta 0,256/0,264), mas a folha piorou (bg05 0,156).
  **Reprovado na bancada.**
- **T7 v6e-1 (TPU Colab, 30 épocas).** IoU médio sintético
  0,7497/0,7423 — que escondia a tinta fina: IoU de tinta
  0,111/0,130 (recall exato igual; a 3 px de tolerância, 0,490/
  0,536: ele acha o traço, mas entrega fino e deslocado). Na
  bancada como folha: bg05 desabou (0,147, canto 710,8 px).
  **Reprovado.** A linhagem T7 com canal de tinta foi **arquivada**
  em 2026-11-05.
- **Noturnos A e B (2026-11-05/06, Kaggle, dieta corrigida).** A =
  com SmartDoc real; B = sem. Vereditos: **os dois reprovados** na
  bancada e no FP de ausência (A: FP em 182/187 cenas sem papel do
  Kinetics; B: 185/187; o incumbente: 11/187). Mas o A×B respondeu a
  pergunta: o SmartDoc **curou** o fundo hostil (bg05: IoU 0,917 e
  canto 21,8 px do A, melhor que o incumbente 0,882/31,5 px;
  B desabou para 0,199/592 px) — **dado real conserta geometria**;
  ausência é outra doença (negativos reais + mecanismo de
  presença). O A também trouxe instabilidade na mesa dele
  (centróide saltando 156 px × ~3,7 px do incumbente).
- **Modelo C (lançado e cancelado em 2026-11-06, antes do smoke; não
  está em treino).** Cabeça de
  **presença** (segunda saída ONNX) + ênfase em negativos duros
  (HNBS), desenhado de pesquisa só-2026 (OPCNet). É o ataque direto
  à doença da ausência. Morreu na geração de dados sintéticos, sem
  artefato e sem veredito; quem cancelou não foi identificado. Se for
  relançado e pousar um artefato, o julgamento é a bancada completa,
  com os portões de ausência e a estabilidade na mesa como
  perguntas decisivas.
- **Desafiantes prontos (régua de 2026-11-05, laboratório).** SAM 2.1
  tiny quebra no fundo hostil (IoU 0,801 × 0,919 do FT-dyn) custando
  ~40× a latência em CPU. SegFormer de OCR ganha recall de passada
  única mas vaza impresso 25,5% (a régua é ≤ 0,5%) e tinta fantasma
  1,95% (× 0,019% da w8). Nenhum substitui os de produção.

## 6. Situações e erros (a parte que importa)

- **O disco dele encheu 100% (2026-10-30).** Geração de dataset em
  staging na máquina dele lotou o C:. Nasceu aí a regra dura:
  dataset mora no Kaggle; gerar → subir → confirmar → **apagar
  local no mesmo fluxo**. (O espaço do C: só volta com
  `wsl --shutdown` + compactação do vhdx.)
- **Upload pro Kaggle quebrado por dois motivos (2026-11-03).**
  (1) O endpoint de blob espera `{"type": "DATASET", ...}`; com
  `fileName` o blob caía no lugar errado e o `create` devolvia
  "Path must be non-null". (2) Nesta VM o proxy come o
  Content-Length e o GCS responde 411: PUT vai por curl. Some-se:
  a CLI pula subpastas por padrão — o conserto foi um zip único
  por dataset, que o Kaggle extrai sozinho.
- **"Colab gratuito = TPU v2-8" — de cabeça, e errado (2026-11-05).**
  A Julia afirmou de memória de treino; a v2-8 tinha sido
  aposentada em set/2025 (v5e-1 no lugar). Nasceu a regra de
  informação atualizada obrigatória (§3).
- **Comandos trocados em sessão viva (2026-11-05).** A Julia passou
  comando de Kaggle GPU para uma sessão Colab TPU, e depois
  caminhos de Kaggle para o Colab — ele colou dois erros até vir o
  comando certo. Nasceu a regra: confirmar plataforma/acelerador
  reais pelos caminhos e saídas que ele cola **antes** de passar o
  comando.
- **A cota que era teto, não saldo (2026-11-05).** Push/lançamento
  anunciado sem conferir: as "30 h" eram o teto semanal já
  consumido. Nasceu a regra: lançar inclui verificar pré-requisitos
  antes, sempre.
- **Os kernels da noite morreram sem traceback (2026-11-05).**
  Causa raiz achada depois: `itertools.cycle` no loader de pares
  **cacheia todos os lotes que produz** — dezenas de GB de RAM da
  hospedeira, SIGKILL no meio da época 0, idêntico em lote 32 e 16.
  Matou os kernels A v3, B v1 e B v2. O conserto (reiterar em vez de
  ciclar) e o teste de regressão entraram na main em 2026-11-06.
  Lição irmã: a telemetria da API do Kaggle pode segurar logs por
  **horas** com a sessão rodando — log vazio ≠ kernel zumbi; e
  polling demais rende **429** (parar, não insistir em loop).
- **Recomendação de issue sem ler a thread (2026-10-31).** Uma
  auditoria OSS recomendou uma issue do Transformers que já tinha
  **dois PRs abertos** resolvendo — a recomendação morreu na
  execução. Regra desde então: recomendar issue exige ler a thread
  inteira e listar os PRs vinculados antes.
- **Faxina que chamou infraestrutura de morta (2026-11-04).** Uma
  auditoria marcou ~7.100 linhas como apagáveis medindo "o app não
  importa hoje" — incluía camadas decididas (selflearn, vectorize,
  fusion/bridge). Dormência ≠ morte: antes de classificar um
  módulo, cruzar com o histórico de decisões. Morto de verdade:
  ~200 linhas.
- **O nome dele num texto público (2026-10-31).** Um PR externo saiu
  com o primeiro nome dele no texto; ele mandou remover na hora.
  Desde então: @santleme em qualquer texto público, sem exceção.
- **O `pgrep` que se enxerga (2026-11-06).** Um minerador morreu
  numa reinicialização da VM e o `pgrep -f` casava com a própria
  linha de comando da sonda — a morte ficou invisível por ~45 min.
  Sonda de processo não pode conter o padrão que ela procura.

## 7. As falhas do produto (docs/FALHAS.md) e o diagnóstico do gargalo

A taxonomia F-01…F-11 nasceu em 2026-11-05. As que marcaram a
história:

- **F-01 — tinta repassada/reimpressa em camadas.** A tinta assentada
  era reimpressa sem fim: o acumulador votava com o quad **do
  momento**, e cada alinhamento da geometria confirmava a sua cópia
  do mesmo traço (na régua: 20.137 px com geometria congelada ×
  71.548 px com viés sustentado). Causa provada e consertada na
  PR #35: os votos passaram a ser ancorados na geometria da
  geração (−80%/−85% de crescimento espúrio, 100% da tinta
  verdadeira mantida). **Veredito ao vivo dele, 2026-11-06: "agora
  não tá passando mais" — resolvida na mesa real.**
- **F-09/F-11 — a folha que volta e a página lida como nova.** Ao
  ajeitar o papel, o app capturava a página no lugar novo **sem
  apagar a captura antiga**: o mesmo desenho duas vezes. É a
  violação direta da invariância de posição (§2). Em 2026-11-06 o
  próprio Santleme implementou a identidade positiva no clone dele:
  o drop não mata mais a geração; na volta, a primeira máscara
  registrada decide por **recall ≥ 65%** da tinta confirmada se é a
  mesma página (canvas reutilizado pixel a pixel) ou página nova
  (canvas limpo) — testes sintéticos verdes, conferidos pela Julia.
  A alternativa da Julia, com herança julgada pelo voto e migração
  de âncora, ficou arquivada na worktree da VM; a dele, de passada
  única, é a base escolhida. Correção da autoria invertida na revisão
  anterior. A baseline real histórica em `d4690d5` reprovou essa
  recuperação: `new_page` em 16/16 retornos médios/longos; níveis
  de comparação, não falta de registro. A correção posterior já em
  `main` usa dilatação/contenção dual e passa na régua sintética;
  falta repetir a página real nessa revisão (ver `docs/FALHAS.md`).
- **F-10 — apagar no papel não apaga no digital.** Por construção: a
  camada de tinta é monotônica dentro da geração. Segue aberta.
- **O gargalo de verdade não é o modelo (auditoria de 2026-11-06).**
  Pergunta dele: "eu não sei se o modelo é realmente o problema".
  Resposta medida, em ordem: (1) **medição/decisão** — holdout
  selado incompleto e decisões saindo de proxies; (2)
  **orquestração/estado** — F-01/F-07 eram código, F-10 é
  construção, F-11/F-04 sem régua; (3) **tradução de dados** —
  correção dele no mesmo dia: dado bruto **sobra** (datasets e
  vídeos no Kaggle); o que falta é **traduzir** esse material em
  exemplos rotulados treináveis e julgáveis, com o self-learning
  funcionando; (4) latência serial em CPU; (5) o modelo por último
  — IoU em distribuição saturou (0,974–0,977) e fine-tune curto com
  pouco dado real do alvo recupera a maior parte do gap. **Regra
  prática adotada:** treino longo do zero deixa de ser o padrão; o
  modelo se mexe com fine-tune curto a partir do incumbente quando
  há dado novo traduzido do alvo.

## 8. A esteira de tradução e o self-learning

### Revisão das 25 páginas — diagnóstico de 2026-11-12

@santleme revisou as 25 páginas com `review_holdout.py` e relatou resultado
ruim. Isso substitui o estado histórico de zero revisadas abaixo; não é um
veredito positivo do modelo. `reviews.json`, `reviews_log.jsonl` e o pool
processado não foram encontrados nem neste clone nem nas árvores Draw,
draw-lab e draw-data do WSL conectado. Portanto não há contagem verificável
de aceites/correções; não confundir inspeção visual relatada com um ledger
de aprovação persistido.

No WSL foi possível reconstruir os rascunhos a partir das oito sessões
originais, sem escrever/re-selar nada: fração 0,5 → cinco sessões holdout,
25 páginas, as mesmas quantidades do inventário. Inspeção de exemplos
confirmou teclado, mão/bandeja e mesa em vez de desenho. O rascunho de
1793654047/gen_0001 marca 65,32% dos pixels de um teclado; o de
1793666553/gen_0005 marca 85,93% de uma cena com mão. São diferenças de
cena/exposição, não rótulos de tinta. A ferramenta não está trocando o
arquivo: está mostrando uma subtração inadequada de entradas inadequadas.

A leitura do código confirma três limitações diferentes: o gravador usa o
primeiro warp da geração como referência (não necessariamente folha limpa);
`final_minus_reference` só rotula escurecimentos novos e perde desenho já
presente, além de confundir sombra/movimento com tinta; a CLI apenas registra
aceite ou máscara editada fora dela, sem editor visual ou julgamento de
representatividade. O inventário anterior já descrevia câmera/mesa, não
sessões representativas de desenho. Logo, completar 25 revisões não corrige
a distribuição e não prova erro do modelo; atribuir cada falha a uma dessas
causas ainda depende dos artefatos privados.

O catálogo `REVISAO_DADOS_KAGGLE_2026-11-06.md` descreve **um pool real de
capturas** (5.086 frames, 4.789 sem máscara), não um inventário de vários
datasets públicos de autores diferentes. Não confundir privacidade aprovada
com rótulo aprovado, nem as 2.518 imagens já usadas em treino com holdout novo.

### Tradução externa e régua ao vivo — 2026-11-12

`tools/selflearn/import_external.py` aceita o `frames.csv` desse pool e CSV
genérico/pasta de páginas de outros autores. Frames crus passam pelo detector
de folha e retificação; páginas já retificadas exigem declaração explícita.
Cada sessão/lote recebe selo imutável; o lote mora em pool próprio, não se
mistura com a saída de `process_sessions.py`. Entradas já usadas em treino
ficam só em treino. O incumbente gera propostas **só no treino**, com hash e
linhagem. A CLI existente revisa/exporta ambos os lados com `--split`.
Holdout sem referência exige máscara humana feita do zero; a máscara vazia
é apenas suporte para o editor e `--accept-draft` é recusado. Nenhum
pseudo-rótulo passa a ser independente por renomeação ou por aceite humano.

Integração real conferida em 2026-11-12: oito fotos de
`guzzbr/school-notebooks-ru` existentes no WSL passaram pelo detector/warp e
pelo incumbente, gerando oito propostas em um único lote de treino, nenhuma
aprovação/rejeição automática. O selo de treino foi conservador, não se
alega novidade dessa fonte frente aos treinos anteriores. A fila está fora
do checkout no laboratório local; a revisão humana continua pendente.

Exemplo para imagens já retificadas (para fotos cruas, retirar
`--images-are-pages` e indicar `--sheet-model` se necessário):

```bash
python tools/selflearn/import_external.py --root /dados/desenhos \
  --dataset autor/lote --manifest /dados/desenhos/frames.csv \
  --images-are-pages --out captures/external-desenhos
python tools/selflearn/review_holdout.py --pool captures/external-desenhos --split train status
python tools/selflearn/review_holdout.py --pool captures/external-desenhos --split holdout audit
```

CSV mínimo: `image_path,session_id` (ou `batch_id`); quando a fonte não tem
grupos identificáveis, usar um único `--batch` conservador para a pasta,
nunca inventar autor/sessão por imagem. No pool catalogado o CSV também
informa `session_used_for_prior_training`; para novos dados, é responsabilidade
do operador auditar treino anterior antes de reservá-los. O resumo informa
Original `session_id` nunca é renomeado: continua comparável ao holdout
dos checkpoints/rodadas anteriores. Para reutilizar sessões já divididas,
passar `--sealed-split /pool/anterior/split.json` (se estiver na raiz bruta,
é lido automaticamente); o selo anterior tem precedência e proíbe
pseudo-rotular holdout até com `--split train` explícito. O resumo informa
páginas em cada lado e rejeições `no_page`, sem certificar a qualidade da
retificação. `mark SESSAO GERACAO --label mascara.png` registra a revisão;
`export --out /dados/revisado` só leva sessões inteiramente revisadas do
`--split` escolhido. Nem a importação nem o aceite treinam/promovem modelo.

O relato do teste de arbitragem foi negativo: "ficou uma merda". Nos
manifestos locais há prova de execução ligada, não só env presumida:
1794341688 tem veredito `spurious`; 1794443693 tem `landing_default` e sete
`new_page`. A segunda sessão tem intervalo mediano de 1,0001 s entre
passadas e inferência mediana de 120 ms; a lentidão não se explica só pelos
168 ms históricos do modelo. Há indício de participação de F-11, mas sem
GT temporal não se pode afirmar que cada `new_page` é erro ou atribuir cada
pixel a F-07/F-09. A régua por passada agora é persistida durante a corrida
em `ink_passes.jsonl`, além do resumo final; ver diagnóstico em FALHAS.

As correções `DRAW_REG_CONTENT_GUARD` e `DRAW_INK_REMOVAL` são opt-in,
default 0; não promovem modelo nem substituem validação ao vivo. A primeira
evita votar a cópia reprojetada quando o conteúdo favorece identidade e
reancora só após assentamento. A segunda retira tinta confirmada com
ausência forte, visível e persistente e testemunho positivo da mesma página;
apagar a página inteira segue ambíguo com virada, portanto não é autorizado
automaticamente. O histórico anterior de F-10 aberta refere-se ao padrão
monotônico, que continua preservado com a flag desligada.

- **O gravador já estava lá (descoberto em 2026-11-06).** O app já
  gravava sessões por padrão (`draw/selflearn/session.py`): na
  máquina dele havia 9 sessões (60 MB, 04–06/11). Faltava a etapa de
  tradução — construída no mesmo dia: integridade por sha contra o
  manifesto, **selo treino × holdout persistido por sessão** (sessão
  selada nunca dessela; holdout = sessões inteiras, nunca frames),
  rascunho de rótulo independente (final − referência) e placar.
  Prova real: 8 sessões válidas → selo 5 holdout × 3 treino, 33
  páginas, 25 rascunhos de holdout, **0 revisados** — e o conteúdo
  era majoritariamente teste de câmera/mesa (teclado lido como
  página, bandeja, mãos, superexposição): a esteira está pronta, o
  insumo que falta é **sessão real de desenho**, que agora entra
  sozinha.
- **As 6 correções dele ao desenho do loop (2026-11-02).** Incerteza
  só no treino, nunca no holdout; correção da predição tem viés de
  ancoragem (rótulo do holdout é independente); pseudo-rótulo de
  professor da mesma família repete os erros dele (veredito é o
  holdout real); IoU de competição (DIBCO 0,13–0,22) é só
  diagnóstico — o portão é o replay real selado; coleta simples com
  referência automática; e a hipótese de latência dele (o consenso
  temporal do acumulador como primeiro suspeito dos 2,3–3,5 s).
- **Self-learning v2 (ordem de adoção dele, 2026-11-06).** (0)
  sessões reais para o selado → (1) **regra de parada no portão**
  (entregue no mesmo dia: 2 rodadas seguidas sem ganho, com IC de
  bootstrap por sessão excluindo o zero, param o loop; rodada abaixo
  do mínimo é inconclusiva e não conta) → (2) monitor de
  distribuição dos aceitos (PSI/KS) → (3) experimentos clDice/ROI →
  (4) conformal por último. Quem gera não julga: juízo triplo
  (conformal por classe, veto clássico, correção humana).
- **Itens 2–4 do v2 e a revisão do selado (2026-11-11, o "vai"
  dele).** Entraram na ordem dele, sem tocar na regra de parada e
  sem promover nada. (2) **Monitor dos aceitos**
  (`draw/selflearn/drift.py`): cada rodada de
  `tools/selflearn/train_round.py` grava `accepted_profile.json`
  (confiança do professor e fração aceita, por classe tinta/fundo) e
  `accepted_drift.json` contra a referência dada em
  `--accepted-reference` (a 1ª rodada ou um conjunto base). Limiares
  pré-registrados: PSI ≥ 0,10 atenção, ≥ 0,25 deriva; KS (p < 0,01)
  só sobe para atenção; amostra abaixo do mínimo é INSUFFICIENT,
  nunca OK. Achado de desenho: com 60 itens e 10 faixas o PSI de
  duas amostras iguais já dá ~0,3 — deriva fantasma —, então as
  faixas encolhem com o n (viés nulo ≤ 0,025). O status vai ao portão
  como sinal (`judge_candidate(accepted_drift=…)`): atenção/deriva
  pedem revisão humana e **não mudam o veredito**. Observar não
  consome RNG nem muda alvo/peso. (3) **clDice e ROI**
  (`tools/seg_metrics.py`): `cldice_binary`/`ink_cldice` (esqueleto
  Zhang-Suen em NumPy, idêntico ao OpenCV longe da borda; traço
  partido perde Tsens com IoU alta) e `evaluate_in_roi` com
  `stroke_roi`/`interior_roi` — régua experimental ao lado da IoU,
  nenhum portão lê. (4) **Conformal por classe**
  (`draw/selflearn/conformal.py`): quantil Mondrian por classe
  calibrado em rótulo independente de `--calibration-dir` (sessões
  que não treinam e não são do selado — a rodada recusa as duas), no
  professor inicial; o pixel só entra na consistência se o conjunto
  for singleton e concordar com o pseudo-rótulo — conformal e veto
  clássico só retiram, nunca acrescentam. Garantia é marginal sobre
  pixels permutáveis: meta, não certificado. **Selado:**
  `tools/selflearn/review_holdout.py` (`status` mostra
  revisadas/pendentes por sessão e o n do portão; `mark` exige prova
  explícita — `--accept-draft` ou `--label` corrigido — e grava
  `label_reviewed.png`, que o reprocessamento não sobrescreve, mais a
  trilha `reviews_log.jsonl`; `export` escreve só sessões inteiras
  revisadas no formato de `--holdout-dir`). **O que continua com
  ele:** olhar e corrigir as 25 páginas, desenhar sessões reais (as
  atuais são majoritariamente teste de câmera) e o teste ao vivo —
  código nenhum substitui isso, e promover segue exigindo holdout
  selado + teste ao vivo.
- **Negativos reais v1 (2026-11-06).** A régua de ausência já tinha
  consumido **todos** os 187 clips sem papel do corpus de escrita
  do Kinetics — reusar qualquer um contaminaria a régua. Mineradas
  6 classes novas (teclado, piano, montar computador, passar roupa,
  previsão do tempo, telejornal): 192 clips, verificação visual
  excluiu 25 (telejornal o mais sujo: 13/32 com papel na mão ou na
  bancada). Resultado: **167 cenas × 4 frames, disjunção provada
  por script** (interseção 0 com o corpus e 0 com a régua). Uso
  interno apenas — pixels de YouTube/Kinetics nunca sobem pro
  Kaggle nem se redistribuem. Detalhe de rota: o YouTube barra o
  cliente web no IP desta VM; o cliente `android` do yt-dlp passa.

## 9. Estado em 2026-11-06 (fim do dia)

- **Main em `fbd62126`** — depois de um dia de três pushes temáticos
  aprovados por ele via chat paralelo (fix do `itertools.cycle` +
  código do modelo C + pacote de eficiência; esteira do pool;
  regra de parada do portão), a Release de modelos foi refeita:
  **`models-2026-11-06`** traz a folha que ele realmente usa
  (FT-dyn, sha `e20a89f0…`) — a Release antiga tinha a v3 original e
  o MANIFEST da main também estava atrasado; os dois foram
  corrigidos e provados por download com sha idêntico ao do PC
  dele. A Release antiga fica como rollback.
- **Modelo C: lançado e cancelado em 2026-11-06** no Kaggle (conta
  pascalcoimbra, T4), antes do smoke — status `cancelAcknowledged`
  conferido na API. Não há Modelo C em treino e nenhum artefato foi
  produzido. Se for relançado e houver artefato, julgar
  ausência/estabilidade na bancada; o veredito final é a mesa dele.
- **F-01 mitigada ao vivo; movimento F-07 ainda reprovado na régua.**
  A base de identidade em passada única é do Santleme, não da Julia;
  a alternativa arquivada é dela. A página real confirmou duplicação
  sem reset e, antes da correção F-11, `new_page` nos 16 retornos
  médios/longos. A correção de identidade já em `main` preserva a
  mesma página 24/24 na régua sintética e rejeita conteúdo distinto
  em 8 drops efetivos; falta repetir a página real nessa revisão.
  O próximo experimento é o portão de tremor de `docs/FALHAS.md`,
  sem alterar produção neste passo.
- **Oclusão em frente separada:** patch relatado em outra worktree,
  ainda não integrado nesta `main`; presença de mão na régua não
  injeta oclusão nos pixels e não cura o recall F-11 sem mão.
- **Holdout selado:** 5 sessões × 25 páginas em rascunho, 0
  revisadas — a revisão e as próximas sessões reais de desenho são
  o caminho curto para destravar decisões de modelo. (2026-11-11: a
  ferramenta de revisão existe, `tools/selflearn/review_holdout.py`;
  a revisão em si segue com ele — ver §8.)
- **Frentes paradas por ordem dele:** implementação da issue #36
  (programa temporal) e os itens 2–4 do self-learning v2, até o
  "vai" dele. (2026-11-11: o "vai" veio e os itens 2–4 entraram —
  ver §8; a issue #36 segue parada.) A ideia do **modelo único** (folha + tinta + mão)
  entrou em avaliação no lugar da direção de modelos separados.

---

_Atualizações entram por commits diretos na main, como todo o resto
do projeto. Correções a este registro são bem-vindas do mesmo jeito:
a memória também erra — e quando erra, corrige por escrito, como
ficou registrado nas situações acima._
