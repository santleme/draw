# Plano de uso dos frames reais — sessão 20261029_130013

> Status: **plano, não execução.** Nenhuma máscara foi anotada aqui.
> Escopo: só decidir COMO esta sessão entra (ou não entra) no treino
> do modelo de segmentação 4 classes. Sem inventar rótulos.

## Decisão atualizada — 2026-10-30

Por solicitação do usuário, os **719 frames produtivos** selecionados das duas
sessões reais serão preparados como `real_train`, mantendo cada sessão inteira
no treino. A recomendação histórica abaixo de preservar holdout continua sendo
uma orientação metodológica, mas foi substituída para esta tranche pelo pedido
explícito de treinar com os dados reais. A validação/teste permanece sintética;
não há holdout real independente nestas duas sessões. Reserve uma nova sessão
para a próxima avaliação real.

As tags de imagem já revisadas não são máscaras pixel a pixel. Só inclua um
frame no pacote de treino depois de parear sua máscara aprovada, conferir
dimensões/checksum e validar valores `{0,1,2,3,255}`; `255` só pode ser usado
como ignorado porque o treinador agora o exclui da loss e das métricas.

## 1. O que a sessão realmente é (verificado, não presumido)

- **Origem:** `origin/main`, commit `38d6293` ("Update camera capture and
  dataset collection"); os frames entraram no commit `11d8127`
  ("Add camera training frame session").
- **Caminho:** `captures/training/session_20261029_130013_453419/`
- **Conteúdo:** **2.518 frames JPG + `frames.csv`** (2.519 linhas com
  cabeçalho), 66 MB no total.
- **Formato do CSV:** `frame_index,timestamp_ns,filename,width,height`.
  Todos os frames: **640×480**, sem exceção.
- **Duração:** timestamps de `1793289614045309270` a
  `1793289781830593109` ns → **167,8 s ≈ 2 min 48 s**, ou seja,
  **≈ 15 fps**. É um vídeo contínuo, não fotos avulsas.
- **Cena (amostras vistas, não só números):** mesma mesa, mesma câmera,
  teclado sempre no canto superior esquerdo. O conteúdo muda ao longo
  da tomada:
  - início: folha em branco com um quadradinho desenhado, mão borrada
    passando;
  - meio: folha impressa de cabeça para baixo ("CASA DOS ROLAMENTOS")
    com escrita à mão, mão + caneta sobre a folha;
  - fim: fone de ouvido cobrindo grande parte da folha.
  Brilho médio varia de ~72 a ~104 entre amostras; a diferença média
  entre frames consecutivos amostrados é baixa (mediana 2,3), com
  picos de até ~37 em mudanças de objeto/movimento. Ou seja: **uma
  tomada contínua, um único ambiente, mas com eventos reais**
  (troca de folha, mão, caneta, oclusão por objeto).

## 2. A regra que decide tudo: sessão única = fonte única

**Nunca dividir esta sessão por frame entre treino e teste.**

Frames vizinhos são quase idênticos: mesma mesa, mesma câmera, mesmo
ângulo, mesma iluminação, provavelmente a mesma mão. Se 1.500 frames
forem para treino e 1.018 para teste, o "teste" mede memorização
desta mesa, não generalização. A métrica ficaria bonita e falsa.

Por isso, a sessão inteira vai para **um único destino**:

| Rota | Destino da sessão inteira | Custo | O que prova |
|---|---|---|---|
| **A — Holdout real (recomendada)** | `real_holdout`, fora do treino e fora de qualquer augmentation | Zero para uso qualitativo; ~1–2 h para anotar um subconjunto de avaliação (ver §4) | O gap sintético × real, na mesa real, sem contaminar o treino |
| **B — Treino, só após anotação manual** | `real_train`, como fonte adicional ao sintético | Alto: 5–40 h de anotação conforme o subconjunto (§4), mais o risco de overfit a esta mesa | Que o modelo aprende com foto real — mas NÃO prova generalização para outra mesa/mão/luz |

Recomendação: **rota A agora; rota B só se vierem mais sessões**
(outras mesas, outras mãos, outras luzes). Com uma única sessão,
o valor dela como holdout é maior que como treino: ela é o único
termômetro real que o projeto tem, e treinar nela queima o termômetro.

Detalhe importante: mesmo na rota B, a validação real precisa vir de
**outra sessão** (outra mesa ou outra tomada). Treinar e validar na
mesma tomada, mesmo com frames diferentes, continua sendo vazamento.

## 3. O que dá para fazer SEM anotação manual completa

### 3.1 Pseudo-rótulo de papel pelo detector clássico — só como pré-anotação, nunca como verdade

Já existe `tools/label_frames.py`: roda `detect_sheet` em cada frame
e grava `labels.json` + overlays com o quadrilátero em verde para
revisão humana. É a ferramenta certa para gerar **candidatos**.

Mas, nesta sessão, o clássico é fraco — medido, não achado:

- Em 26 amostras (1 a cada 100 frames), o detector atual
  (`detect_sheet_with_confidence`, versão da branch) achou folha em
  **apenas 2, ambas com veredito `blank_maybe`**. Nos demais, `None`.
- Ou seja, um pseudo-rótulo clássico cobriria ~8% das amostras e só
  os casos fáceis: um dataset treinado nisso aprenderia "folha = só
  quando o clássico já acha", o viés exato que o modelo neural
  deveria superar.

E há um problema de formato: o clássico devolve um **quadrilátero**,
não uma máscara por pixel. Preencher o quadrilátero como `paper`
rotularia como papel tudo dentro dele — inclusive mão, caneta e
fone de ouvido. Para virar máscara, seria preciso subtrair as
regiões de mão/objeto à mão, o que já é anotação manual.

**Uso permitido:** gerar candidatos + overlays, revisar um por um,
corrigir os cantos, e só então usar como rótulo de papel. O overlay
aprovado por humano vira rótulo; o overlay não revisado não vira
nada.

### 3.2 Validação só de `paper_presence`, sem rótulo nenhum

Dá para rodar o modelo (ou o clássico) nos 2.518 frames e plotar a
presença de papel ao longo do tempo, sem nenhuma máscara:

- a presença deve ficar alta na folha em branco e na folha impressa;
- deve cair (ou ficar instável) quando o fone cobre a folha;
- quedas bruscas sem evento visível são suspeitas e valem inspeção.

Isso é um teste de sanidade útil e barato. **Não é IoU, não é
métrica por classe e não substitui o holdout anotado.** Serve para
achar falhas grosseiras, não para aprovar o modelo.

### 3.3 O que NÃO dá para fazer sem máscara

- **Treinar `hand`:** não há segmentação de mão confiável pronta
  aqui. Recortar mão por cor/pele ou pelo FreiHAND não se transfere
  para esta mesa sem revisão, e o resultado ensinaria o modelo a
  copiar os erros do pseudo-rotulador.
- **Treinar `ink`:** a extração clássica de tinta (threshold
  adaptativo) é justamente o que o modelo deve superar; usá-la como
  rótulo congela os erros do clássico (falha sob mão, sob sombra,
  em traço fino) dentro do modelo novo.
- **Treinar `paper` por quadrilátero preenchido:** ver §3.1 — mão e
  objetos dentro do quadrilátero virariam papel.
- **"Completar" a máscara por propagação temporal sem revisão:**
  propagar o quad por homografia/fluxo óptico entre keyframes
  anotados reduz custo, mas quebra exatamente nos eventos
  interessantes (troca de folha, fone entrando, mão cobrindo canto).
  Propagação só com revisão humana dos frames propagados.

## 4. Plano concreto de anotação (quando / se anotar)

### 4.1 Quantos frames valem a pena

**Não anotar os 2.518.** Frames vizinhos são duplicatas para fins
de treino/avaliação. Anotar tudo custaria semanas e não compraria
diversidade.

**Amostragem estratificada recomendada:**

1. **Base temporal:** 1 frame a cada 2 s ≈ **84 frames** (ou 1 por
   segundo ≈ 168, se houver orçamento). Use `timestamp_ns` do
   `frames.csv` para amostrar por tempo, não por índice cego.
2. **Eventos, sempre incluídos:** troca de folha, mão entrando,
   mão cobrindo canto, mão cobrindo centro, mão sobre traço, caneta
   visível, fone cobrindo a folha, folha parcialmente fora do
   quadro (se houver), início e fim da tomada.
3. **Mínimo viável para um holdout quantitativo: 60–100 frames**
   cobrindo todas as situações acima. Abaixo disso, a IoU real é
   ruído.
4. **Teto sensato para esta sessão: 150–200 frames.** Acima disso,
   o ganho é marginal e o dinheiro/hora é melhor gasto numa
   **segunda sessão** (outra mesa, outra mão, outra luz).

Estimativa de custo por frame (polígonos à mão, com revisão):
papel (4 pontos) ~30–60 s; mão (contorno) ~1–3 min; tinta (traçado
dos traços visíveis) ~2–5 min. Total **~3–8 min/frame** →
100 frames ≈ **5–13 h**; 200 frames ≈ **10–27 h**. É por isso que a
amostragem, e não a anotação total, é o plano.

### 4.2 Como anotar

- **Ferramenta:** LabelMe (polígonos, gratuito, exporta JSON por
  imagem) ou CVAT (melhor para vídeo, com interpolação entre
  keyframes — mas a interpolação precisa de revisão nos eventos).
  Não criar uma ferramenta nova só para isso; o gargalo é o olho
  humano, não o software.
- **Formato de saída:** máscara PNG de 1 canal, **na resolução
  original 640×480**, mesmo nome base do frame + `_mask.png`
  (ex.: `frame_00001251_mask.png`), valores **só em {0, 1, 2, 3}**:
  `0=background, 1=paper, 2=hand, 3=ink`. Na hora de treinar, a
  máscara segue o mesmo pré-processamento do modelo (esmagada para
  512×512, `INTER_NEAREST` — nunca bilinear, que criaria classes
  fracionárias).
- **Manifesto:** um JSON ao lado das máscaras com, por frame:
  `frame`, `annotator`, `date`, `source_session`
  (`session_20261029_130013_453419`), `split` (a sessão inteira num
  único split — ver §2) e `notes` (ex.: "fone cobre 60% da folha").
  Sem manifesto, as máscaras viram arquivos órfãos.
- **Controle de qualidade:** 10% das máscaras revisadas por uma
  segunda pessoa (ou pelo próprio anotador no dia seguinte), com
  foco em borda de mão e traço fino. Máscara com valor fora de
  {0,1,2,3} é erro e o lote é rejeitado — mesma regra do treino.

### 4.3 Regras de rótulo (as mesmas do projeto, mais dois casos desta sessão)

- **Regra visível, sem exceção:** o que a câmera vê. Mão sobre
  traço = `hand` (2), nunca `ink`. Papel sob a mão ou sob o fone =
  não é papel visível: é `hand` ou `background`, respectivamente.
- **Caneta:** `background`, não `hand` e nunca `ink`. A caneta é um
  objeto oclusor, não parte da mão e não traço. (Se o usuário
  decidir outra coisa, registrar a decisão no manifesto — mas não
  misturar as duas convenções no mesmo lote.)
- **Folha impressa:** a área da folha é `paper` (1), **mas o texto
  impresso NÃO é `ink`**. `ink` (3) é só o que o usuário acrescentou
  à mão (escrita e desenho). Se o anotador traçar o texto impresso
  como tinta, o modelo aprende "texto impresso = traço do usuário",
  o erro oposto ao objetivo do Draw. Este é o ponto mais fácil de
  errar neste lote e o mais importante de deixar explícito no
  briefing do anotador.
- **Fone de ouvido, teclado, mesa:** `background` (0), sempre.

## 5. Como isso se encaixa no pipeline existente

- O treino (`training/train.py`) já aceita um split com
  `images/` + `masks/`; um holdout real entra como um diretório à
  parte, avaliado por `tools/seg_metrics.py` (IoU/Dice/precision/
  recall por classe + tinta com tolerância 2–3 px), **com números
  reportados separados dos sintéticos**, nunca misturados.
- O `split_manifest.json` do treino deve registrar esta sessão como
  **uma fonte** (`source_id = session_20261029_130013_453419`,
  `split = real_holdout` ou `real_train`, conforme a rota), com os
  IDs dos frames anotados. Se um dia a sessão for dividida por
  engano, o manifesto é o alarme.
- Atenção de repositório: esta sessão já está commitada em `main`
  (66 MB). **Sessões futuras não devem ir para o git cru** — o repo
  incha a cada tomada. O caminho é dataset privado no Kaggle (como
  já se faz com o sintético) ou outro armazenamento de dataset, com
  o repo guardando só CSVs, manifesto e checksums. Siga o procedimento
  em [`REAL_CAPTURE_PUBLISHING.md`](REAL_CAPTURE_PUBLISHING.md) para
  dividir, publicar e verificar as partes. Este documento
  está na branch `feature/neural-seg-v1`; ao mesclar, reconciliar
  o caminho `captures/` com o que estiver em `main`.

## 6. Decisões que só o usuário pode tomar

1. **Rota A (holdout) ou rota B (treino)?** Recomendação: A.
2. **Quantos frames anotar?** Recomendação: 60–100 para holdout
   quantitativo, teto de 200, nunca os 2.518.
3. **Caneta = `background`?** Recomendação: sim (ver §4.3).
4. **Texto impresso ≠ `ink`?** Recomendação: sim, e deixar isso
   escrito no briefing do anotador.
5. **Vai haver mais sessões reais?** Se sim, de quais mesas/mãos/
   luzes — isso decide se a rota B faz sentido depois, e qual
   sessão vira a validação real.
6. **Quem anota e com qual ferramenta?** LabelMe ou CVAT, com a
   revisão de 10% do §4.2. Sem essa resposta, o plano não sai do
   papel — e, enquanto isso, a rota A qualitativa (§3.2) já pode
   rodar com custo zero.
