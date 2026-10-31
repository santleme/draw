# Fontes alternativas de mão — caso o FreiHAND não tenha máscara adequada

> Escopo deste documento: **somente** fontes de mão para recorte e
> inserção em cenas sintéticas do projeto neural do Draw
> (segmentação 0=background, 1=paper, 2=hand, 3=ink, regra visível:
> mão por cima de traço = `hand`).
>
> Este documento **não** altera o plano principal. Ele existe para o
> caso de a auditoria do FreiHAND (documento irmão
> `DATASET_FREIHAND.md`, outro agente) concluir que a cópia do Kaggle
> `danieldelro/freihand` não contém máscaras utilizáveis.
>
> Regra de honestidade aplicada aqui: licença só é afirmada quando
> encontrada em fonte oficial do dataset. Todo o resto está marcado
> como **NÃO CONFIRMADA** — não usar a fonte em produção antes de
> confirmar a licença na fonte oficial.
>
> Nenhum dataset inteiro foi baixado para esta pesquisa. Verificação
> feita por listagem de arquivos/metadata no Kaggle e por busca na web.

## 0. Ponto de partida: o que o FreiHAND oficial promete

Segundo a descrição oficial do FreiHAND (Kaggle, cópia
`danieldelro/freihand`, Freiburg University):

- 32.560 amostras únicas de treino, post-processadas em 4 versões
  (green screen / fundo / amostra / automático) = 130.240 imagens de
  treino, mais 3.960 de avaliação.
- Cada amostra de treino promete: imagem RGB **224×224**, **máscara de
  segmentação da mão 224×224**, matriz intrínseca K, escala da mão,
  21 keypoints 3D e forma 3D.
- Máscaras existem **só para o treino**; na avaliação, keypoints e
  forma são retidos.
- Termos declarados na página: **somente pesquisa, uso comercial
  proibido, sem garantia; é obrigatório citar o artigo**
  (Zimmermann et al., ICCV 2019).

Consequência: se a cópia do Kaggle contiver de fato a pasta
`training/mask/` (formato esperado: `training/mask/{id:08d}.jpg`,
máscara binária, sem versão — ver utilitário `fh_utils.read_msk`),
**não há necessidade de fonte alternativa para máscara**.
As alternativas abaixo só entram se essa verificação falhar, ou como
complemento de diversidade — e mesmo assim com as limitações listadas.

Limitações do FreiHAND que valem mesmo com máscara (motivo de existir
reserva):

- Mão centrada, 224×224, origem em green screen com fundos compostos:
  não é o ponto de vista do Draw (câmera de cima, mão entrando pela
  borda sobre uma folha A4).
- Máscara em JPG: borda com artefato de compressão; para recorte,
  limiarizar (ex.: > 127) e, se preciso, fechar morfologicamente.
- Resolução baixa para entrada 512×512 do modelo: upscale 224 → 512
  borra a borda da mão; testar se isso prejudica a classe `hand`.

## 1. Tabela comparativa

| Fonte | Máscara pronta? | Formato / tamanho | Ponto de vista | Licença / termos | Viabilidade de recorte para o Draw | Veredito |
|---|---|---|---|---|---|---|
| **FreiHAND (principal)** | Sim, segundo descrição oficial (treino) | RGB + máscara binária 224×224; Kaggle 3,9 GB; 130.240 treino / 3.960 avaliação | Mão centrada, green screen | Pesquisa apenas, comercial proibido, citar artigo | Alta, se a pasta `training/mask/` existir na cópia Kaggle; limiarizar JPG, upscale com cuidado | **1ª escolha — manter** |
| **Renderizar máscara a partir da mesh 3D do FreiHAND** | Não — é máscara derivada | Vértices 3D + K por amostra; renderizar projeção da mesh | Idêntico ao FreiHAND (mesmas imagens) | Mesma do FreiHAND (pesquisa apenas) | Média: exige projetar mesh com K, tratar oclusão dedo-dedo e borda; risco de divergir da máscara oficial; só vale se a máscara oficial faltar | **Fallback, não preferir** |
| **EgoHands (Indiana, Bambach et al., ICCV 2015)** | Sim, mas como polígonos, não PNG | 4.800 imagens, > 15.000 instâncias de mão; 48 locais, 100 imagens cada; anotação em `polygons.mat` por local; precisa rasterizar polígono → máscara binária | Egocêntrico, Google Glass, atividades: quebra-cabeça, xadrez, Jenga, cartas | **NÃO CONFIRMADA** na fonte original (a licença CC BY 4.0 vista é da cópia espelho no Roboflow Universe, não da página original — não usar como prova) | Alta para recorte, após rasterização; diversidade real de ambiente e iluminação é o melhor complemento ao FreiHAND; atenção: mãos de outras pessoas, interação social, não mão desenhando | **Reserva nº 1** |
| **RHD — Rendered Handpose Dataset (Zimmermann & Brox, 2017)** | Sim, por classe | 41.258 treino + 2.728 teste, 320×320; máscaras por: fundo, pessoa, 3 classes por dedo + palma; RGB + profundidade + 21 keypoints | Sintético, personagens Mixamo renderizados no Blender, fundo aleatório | Pesquisa apenas, comercial proibido, citar artigo (termos da página Kaggle do RHD) | Alta, após colapsar dedo+palma → `hand` binário; máscara limpa, sem artefato JPG; risco: domínio sintético (textura/pele) diferente de mão real | **Reserva nº 2 (aumento, não fonte única)** |
| **GTEA / EGTEA (Georgia Tech)** | Sim, pixel a pixel | GTEA: 663 imagens com anotação pixel (ou 625 com rótulo de duas mãos + 38 binárias, conforme a fonte); EGTEA: 12.799 binárias + 1.048 de duas mãos; 7 atividades, 4 sujeitos, cozinha | Egocêntrico, cabeça, cozinha, iluminação estática | **NÃO CONFIRMADA** | Média-baixa: pouca diversidade de ambiente/iluminação e de sujeitos; útil só como complemento pequeno | **Complemento pequeno, não reserva principal** |
| **EYTH — EgoYouTubeHands** | Sim, pixel a pixel | ~2.000 frames, ~2.600 instâncias (~1.800 primeira pessoa + ~800 terceira pessoa), de apenas 3 vídeos | YouTube, in-the-wild, diverso | **NÃO CONFIRMADA**; origem YouTube traz risco extra de redistribuição | Média: diversidade real, mas frames muito parecidos entre si (amostragem fixa) e poucos vídeos — alto risco de vazamento entre splits se não separar por vídeo | **Complemento, separar por vídeo** |
| **Ego2Hands (Lin et al., 2021)** | Sim (energia/segmentação, duas mãos) | Composição green-screen de mão direita + espelho como esquerda; avaliação 800×448, 8 sequências | Egocêntrico, duas mãos, movimento livre | **NÃO CONFIRMADA** | Média: única opção dedicada a **duas mãos**; composição green-screen pode deixar artefato de borda; usar só para as cenas de duas mãos do gerador | **Só para cenas de duas mãos** |
| **HandOverFace (HOF)** | Sim | Mãos sobre rosto, in-the-wild | Terceira pessoa, mão sobre rosto | **NÃO CONFIRMADA** | Baixa para o Draw: ponto de vista e contexto errados (rosto, não folha) | **Não usar** |
| **SynthHands (MPI, ICCV 2017)** | Máscara via renderização Unity (não PNG simples) | 63.530 frames de pose, 5 viewpoints egocêntricos, 48,5 GB em um único zip | Egocêntrico sintético, Unity, RealSense SR300 | Científico/não-comercial apenas, citar artigo (licença dentro do download) | Baixa: 48,5 GB, pipeline Unity, e o download inteiro é desproporcional ao ganho frente a RHD/FreiHAND | **Não usar** |
| **AnyHand (sintético, 2025)** | Sim (RGB, profundidade, máscara, bbox, pose) | 1,05M cenas single-hand / 2,1M cenas interact, 47.438 formas, 10.240 texturas | Sintético, múltiplas câmeras | **NÃO CONFIRMADA** (tamanho e licença não verificados em fonte oficial nesta pesquisa) | Não avaliável sem confirmar acesso/licença/tamanho; não basear o plano nele | **Não usar até confirmar** |
| **HaDR (sintético, domain randomization)** | Sim, RGB-D + máscara | 117 mil amostras (segundo comparativo da literatura) | Sintético, manipulação | **NÃO CONFIRMADA** | Não avaliável sem confirmar acesso/licença | **Não usar até confirmar** |
| **LSWH100 (Libras, Zenodo)** | Sim (máscara + profundidade, COCO) | 144.000 imagens (48.000 RGB × 3 conjuntos), 100 formas de mão, Blender, Unicamp | Mão centrada, sinalização, 4 perspectivas | **NÃO CONFIRMADA** (registro Zenodo 10.5281/zenodo.10635628 — verificar licença no registro antes de usar) | Baixa para o Draw: formas de Libras, mão centrada, não mão desenhando/escrevendo | **Não usar para o Draw** |
| **Mãos capturadas pelo próprio Draw (futuro)** | A anotar | A definir pelo Codex/usuário | **Exatamente o do Draw**: câmera de cima, mão sobre folha | Interna, consentimento do dono | Ideal a longo prazo, mas fora do escopo do Muse (não capturar câmera) | **Longo prazo, não agora** |

## 2. Recomendação

1. **Primeira escolha: FreiHAND direto.** Se `training/mask/` existir
   na cópia Kaggle, usar RGB + máscara binária, limiarizar, e registrar
   no manifesto do dataset: origem, versão, licença (pesquisa apenas)
   e a limitação de ponto de vista/resolução.
2. **Reserva nº 1: EgoHands.** Se FreiHAND falhar, ou para diversidade
   real além do FreiHAND: rasterizar `polygons.mat` → máscara PNG
   binária, separar splits **por local/vídeo** (nunca por frame), e
   **confirmar a licença na página original de Indiana antes de
   publicar qualquer derivado**. Acesso Kaggle verificado nesta
   pesquisa (sem baixar tudo): `himaniishah/egohands` (1,33 GB),
   `chaturvediabhay24/egohandssegmentation` (189 MB) — atenção:
   são re-uploads de terceiros, não a fonte oficial; preferir a
   fonte oficial quando possível.
3. **Reserva nº 2: RHD, só como aumento.** Máscara limpa e grande
   (41 mil), mas sintética: misturar com mão real, nunca treinar só
   nela, para não aprender textura sintética como `hand`.
4. **Mesh renderizada: só se a máscara oficial faltar.** Renderizar a
   mesh do FreiHAND com K é viável tecnicamente, mas é trabalho
   extra para reproduzir o que a máscara oficial já entrega — e pode
   divergir dela na borda. Se for esse o caminho, validar a máscara
   renderizada contra uma amostra visual antes de gerar o dataset.
5. **Duas mãos: Ego2Hands, só para esse caso.** Não usar como fonte
   principal de mão única.

## 3. Viabilidade de recorte (o que o gerador precisa, por fonte)

Para qualquer fonte, o gerador sintético precisa de, por amostra:

- RGB da mão (ou frame inteiro) + máscara binária **do mesmo tamanho**;
- conversão única: `hand_mask = (mask > limiar)`; para RHD,
  `hand_mask = (classe ∈ {dedos, palma})`; para EgoHands,
  `hand_mask = rasterizar(polígonos)`; para Ego2Hands, usar a
  segmentação fornecida, não a energia de detecção;
- corte com borda suave (feather de 1–2 px) para não deixar halo
  verde/branco do green screen;
- registro de proveniência por amostra (fonte, id original, split),
  para o split por fonte não vazar a mesma mão entre treino/val/teste.

## 4. O que falta confirmar (antes de usar qualquer reserva)

- [ ] FreiHAND na cópia Kaggle: existe `training/mask/`? Formato exato
      (JPG binário? valores 0/255?) — a cargo do agente de dataset.
- [ ] EgoHands original (Indiana): licença/termos de uso na página
      oficial; link de download oficial atual.
- [ ] GTEA/EGTEA, EYTH, Ego2Hands: licença oficial de cada um.
- [ ] RHD: confirmar se a licença "pesquisa apenas" permite gerar
      composições sintéticas derivadas para treino do Draw (não só
      reproduzir o dataset).
- [ ] Qualquer fonte com pessoas identificáveis: verificar se há
      restrição de redistribuição de imagens derivadas (recortes),
      não só do dataset original.
- [ ] Nenhuma fonte desta lista deve ser publicada/redistribuída
      junto com o modelo sem essa confirmação — o Draw é um projeto
      com potencial comercial futuro, e "pesquisa apenas" não cobre isso.

## 5. Fontes consultadas

- FreiHAND, Kaggle `danieldelro/freihand` (descrição oficial:
  RGB 224×224 + máscara 224×224 no treino, pesquisa apenas, citar
  Zimmermann et al. 2019) — verificado via listagem Kaggle
  (`datasets list --search freihand`, `datasets files` parcial).
- FreiHAND utils (`fh_utils.read_msk`: `training/mask/{id:08d}.jpg`,
  máscara binária, só treino).
- EgoHands: Bambach et al., ICCV 2015; 4.800 imagens, > 15.000
  instâncias, `polygons.mat`; Kaggle: `himaniishah/egohands`,
  `chaturvediabhay24/egohandssegmentation`.
- GTEA/EGTEA: 663 imagens pixel (GTEA) / 12.799 binárias + 1.048
  duas mãos (EGTEA); cozinha, 4 sujeitos.
- EYTH: ~2.000 frames / ~2.600 instâncias, 3 vídeos.
- RHD: Zimmermann & Brox 2017; 41.258 treino + 2.728 teste, 320×320,
  máscaras por dedo/palma; pesquisa apenas, citar artigo.
- SynthHands: MPI ICCV 2017; 48,5 GB, científico/não-comercial.
- Ego2Hands: Lin et al. 2021; composição green-screen, avaliação
  800×448.
- AnyHand / HaDR / LSWH100: tamanhos e licenças **não confirmados**
  em fonte oficial nesta pesquisa — listados apenas como candidatos
  descartados por enquanto.
