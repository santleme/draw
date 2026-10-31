# FreiHAND no Kaggle — verificação para o modelo de segmentação do Draw

Data da verificação: 2026-10-29. Verificado com a Kaggle CLI 2.2.5
(`.venv/bin/kaggle`), sem baixar o dataset inteiro: listagem por
paginação (`datasets files`, páginas de 200 arquivos, token de página)
+ download de amostra de 3 arquivos para `/tmp/freihand-sample`
(2 RGB + 1 máscara), + metadados oficiais (`datasets metadata`).
Token do Kaggle NÃO está neste documento e não foi exibido em nenhum
passo. Escopo deste agente: só este relatório e `tools/check_freihand.py`.
Nenhum commit, push, merge, webcam, app ou integração.

## Dataset verificado

- Kaggle: `danieldelro/freihand` — "FreiHAND - Hand Pose, Shape and
  Mask Dataset"
- Tamanho total informado: 3.885.249.922 bytes (~3,88 GB)
- Última atualização no Kaggle: 2022-03-03
- Downloads: 1.756 · votos: 27 · usabilidade: 0,9375 · visualizações: 9.211
- Origem: Universidade de Freiburg, grupo LMB (Zimmermann et al.,
  ICCV 2019, "FreiHAND: A Dataset for Markerless Capture of Hand Pose
  and Shape from Single RGB Images").

## Respostas diretas (fatos verificados, não suposições)

### Há imagens RGB utilizáveis? SIM
- `evaluation/rgb/00000000.jpg` e `training/rgb/00000000.jpg`
  baixados e lidos: **224 × 224 × 3, JPEG, uint8**, cor real
  (canais diferentes, médias por canal distintas).
- Contagem verificada por listagem: `evaluation/rgb` = **3.960 arquivos**,
  exatamente o número da descrição oficial para o split de avaliação.
- Treino, pela descrição oficial: 32.560 amostras únicas, cada uma
  processada de 4 formas para remover o fundo verde → 130.240 amostras
  de treino. A listagem completa do split de treino ainda estava em
  andamento ao fechar este relatório (12.200+ arquivos listados sem
  erro); os números de treino acima são os da descrição oficial, não
  uma contagem final própria — revalidar com
  `tools/check_freihand.py` sobre a cópia local completa antes de
  gerar o dataset sintético.

### Há máscaras reais de segmentação da mão? SIM — só no treino
- `training/mask/00000000.jpg` baixado e lido: **224 × 224, JPEG,
  cinza replicado nos 3 canais** (canais idênticos), com valores
  essencialmente binários.
- Amostra medida: 91,39% dos pixels = 0 (fundo), 7,60% dos pixels
  > 127 (mão). Valores intermediários existem só em pequena
  quantidade (1–7 e 248–254), típicos de artefato de compressão JPEG
  na borda da mão. **Regra de uso: binarizar com limiar 127**
  (`máscara = pixel > 127`); NÃO usar os valores JPEG crus como
  rótulo multi-classe.
- Contagem parcial verificada: **8.038 arquivos em `training/mask`**
  já listados nas primeiras 12.200 entradas, antes do fim da listagem.
- Avaliação: **não há máscara no split de avaliação nesta cópia**
  (só `evaluation/rgb` nas entradas listadas). Isso bate com a
  descrição original — as anotações de avaliação eram retidas para
  o servidor de avaliação — mesmo que a página do projeto mencione
  liberação posterior das anotações de avaliação no site original.
  Consequência prática: o split de avaliação do FreiHAND serve para
  inspeção visual, NÃO para métrica de mão com ground truth.

### Formato das máscaras
- Arquivo: JPEG grayscale (3 canais iguais), 224 × 224, 1 máscara por
  imagem RGB de treino com o **mesmo nome de arquivo**
  (`training/rgb/NNNNNNNN.jpg` ↔ `training/mask/NNNNNNNN.jpg`).
- NÃO é máscara gerada a partir de mesh nesta cópia: é a máscara de
  segmentação fornecida pelo dataset. Não há, nas entradas listadas
  até aqui, arquivos de keypoints 3D, matriz K, escala ou mesh
  (só `.jpg` e 2 arquivos `.json` na raiz, ainda não identificados —
  verificar antes de prometer keypoints/mesh a partir desta cópia).
  Keypoints/pose/mesh NÃO são necessários para o pipeline do Draw
  (só RGB + máscara), então essa ausência não bloqueia o trabalho.

### Resolução
- RGB e máscara: **224 × 224 pixels** (confirmado na amostra e na
  descrição oficial). Para o modelo alvo (entrada 512 × 512), as mãos
  serão reamostradas na composição sintética — registrar o
  reescalonamento no gerador, não assumir 224 como entrada final.

### Licença e termos — LEIA ANTES DE USAR EM PRODUTO
- Licença no Kaggle: **CC BY-NC 4.0** (Atribuição–NãoComercial 4.0
  Internacional).
- Termos do dataset original, na descrição oficial: **"for research
  purposes only", sem garantia, uso comercial proibido, e citação
  obrigatória** do artigo (Zimmermann et al., ICCV 2019 — referência
  completa na descrição do dataset).
- Implicação direta para o Draw: treinar com FreiHAND é adequado
  para esta etapa de pesquisa, com atribuição. **Se o Draw virar
  produto comercial, o uso de um modelo treinado com FreiHAND é um
  risco de licença a ser resolvido antes (trocar a fonte das mãos ou
  obter autorização)** — este relatório não resolve essa questão
  jurídica, só a registra. Não publicar as imagens FreiHAND
  redistribuídas sem checar os termos originais.
- Composições sintéticas que recortam mãos FreiHAND continuam sendo
  derivadas do dataset: mesma licença, mesma restrição, atribuição
  mantida no manifesto do dataset sintético.

## Decisão

**USAR O FreiHAND DIRETO, só para a máscara da mão no treino.**
Nenhuma invenção de rótulo é necessária: a máscara real existe,
é pareada por nome com o RGB e é binarizável com limiar 127.

- Usar: `training/rgb` + `training/mask` (treino) como matéria-prima
  das mãos nas cenas sintéticas.
- Não usar como métrica: `evaluation/rgb` (sem máscara nesta cópia).
- Não prometer a partir desta cópia: keypoints 3D, matriz K, escala
  e mesh — verificar os 2 `.json` da raiz antes de qualquer uso, e,
  se mesh for necessária no futuro, buscar a fonte original de
  Freiburg (mesma licença/termos), não esta cópia do Kaggle.
- Alternativa registrada (só se a máscara direta falhar em algum
  recorte): renderizar máscara a partir da mesh 3D do FreiHAND
  original, ou usar EgoHands (máscaras de mão pixel a pixel, uso
  para pesquisa — verificar licença antes de baixar). Nenhuma das
  duas é necessária agora.

## Regras obrigatórias para o gerador sintético (derivadas disto)

1. **Split por mão base, nunca por arquivo.** O treino original tem
   32.560 mãos únicas × 4 variantes de fundo. Dividir os arquivos
   aleatoriamente colocaria a mesma mão (com outro fundo) em treino
   e teste. Separar primeiro as fontes (mão base), depois gerar
   variações por split (80/10/10 sugerido no escopo).
2. **Índices recomeçam por split.** `00000000.jpg` existe em treino
   e em avaliação e NÃO é a mesma mão. Não usar nome de arquivo
   como chave de vazamento entre splits.
3. **Máscara sempre com limiar 127** após leitura do JPEG; guardar a
   máscara binarizada no dataset sintético, não o JPEG cru.
4. **Regra de oclusão do escopo:** na composição, onde a mão cobre
   papel ou traço, o rótulo final é `hand` (o que a câmera vê).
   Guardar `ink_full_mask` separada, fora do treino, só para avaliar
   memória futura.
5. **Manifesto:** registrar fonte (`danieldelro/freihand`, Kaggle),
   licença (CC BY-NC 4.0 + pesquisa somente), citação ICCV 2019,
   seeds e IDs das mãos base por split.

## Como reproduzir esta verificação

```bash
# Listagem (sem baixar tudo)
.venv/bin/kaggle datasets files danieldelro/freihand --page-size 1000 --format csv
# Amostra (3 arquivos, /tmp, nunca o dataset inteiro)
.venv/bin/kaggle datasets download danieldelro/freihand -f training/rgb/00000000.jpg --unzip -p /tmp/freihand-sample
.venv/bin/kaggle datasets download danieldelro/freihand -f training/mask/00000000.jpg --unzip -p /tmp/freihand-sample
.venv/bin/kaggle datasets download danieldelro/freihand -f evaluation/rgb/00000000.jpg --unzip -p /tmp/freihand-sample
# Checagem local (não baixa nada)
.venv/bin/python tools/check_freihand.py /tmp/freihand-sample
.venv/bin/python tools/check_freihand.py <raiz-completa-do-freihand>
```

Saída do checker na amostra (3 arquivos): estrutura OK, pares
treino 1/1, RGB 224×224×3, máscara 224×224 com 7,60% de cobertura
após limiar 127.
