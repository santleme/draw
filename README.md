<p align="center">
  <img src="docs/icon.svg" width="160" alt="Draw">
</p>

<h1 align="center">Draw</h1>

<p align="center">
  Uma mesa digitalizadora sem mesa digitalizadora.
</p>

Uma webcam aponta para a mesa. A pessoa escreve e desenha numa folha de papel comum, e o Draw reproduz o traço no digital, ao vivo. Sem caneta especial, sem superfície especial, sem folha-modelo impressa.

A visão completa do projeto fica em [VISION.md](VISION.md). Este documento cobre o estado atual, como rodar e como o sistema funciona por dentro.

## Status

O Draw roda em tempo real na mesa do autor. O detector do dia a dia é o modelo neural v3, com o detector clássico de reserva dentro do mesmo processo. O canvas digital já acumula tinta entre frames e trata a mão como região desconhecida.

O limite honesto do momento é a generalização. A v3 foi julgada nos frames da mesa do autor e em datasets públicos de documentos. Ela nunca passou por um conjunto de teste de outra mesa, outra câmera e outra luz. Esse conjunto ainda não existe, e nenhum número deste README mede outro setup.

## Início rápido

Pré-requisitos: Python (o autor roda 3.12), uma webcam e as dependências do projeto.

```bash
pip install -r requirements.txt
```

### Windows com WSL

O WSL2 não enxerga a webcam do Windows por padrão. Nenhum `/dev/video*` existe dentro dele até a câmera ser anexada.

O guia completo fica em [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md). O caminho curto roda de dentro do WSL.

```bash
bash scripts/setup_camera_wsl.sh
python main.py
```

O script prepara o usbipd, anexa a câmera ao WSL e confere o resultado. Ele também corrige o grupo `video` do usuário e religa os serviços de áudio que ele mesmo parou.

Se algo falhar, rode o diagnóstico antes de mexer em outra coisa.

```bash
python main.py --check
```

O `--check` valida o OpenCV, a existência de `/dev/video*`, a permissão do processo sobre o dispositivo e a leitura de um quadro real. Cada falha sai nomeada, com a correção na linha seguinte.

### Linux com câmera USB

```bash
python main.py
```

### Com o detector neural

O repositório não publica arquivo de modelo. Um clone novo roda só com o detector clássico até alguém colocar um ONNX na pasta `models/`. Na máquina do autor, o comando do dia a dia é este.

```bash
python main.py --neural models/draw_seg_v3_dyn.onnx
```

## Uso

| Flag | O que faz |
|---|---|
| `--source N`, URL ou arquivo | Webcam local por índice (padrão 0), stream MJPEG/RTSP, protocolo binário `tcp://host[:8765]` ou gravação `.drawcap` |
| `--neural [modelo.onnx]` | Detector neural como primário, com o clássico de reserva. Sem caminho, lê `DRAW_NEURAL_MODEL` |
| `--async-detect` | Roda a detecção da folha em thread dedicada de segundo plano, desacoplando o custo do ONNX da cadência de tinta e cursor |
| `--session-baseline` | Tinta da sessão: a página como está ao apertar `c` vira fundo (pauta, margem e desenho antigo somem); só entra o traço novo |
| `--hand-model [modelo.task]` | Máscara de mão e marcos 21-pontos do MediaPipe para o canvas acumulativo e fusão do cursor (`PenTipFusion`). Sem caminho, lê `DRAW_HAND_MODEL` ou `models/hand_landmarker.task` |
| `--ink-model [modelo.onnx]` | Gancho opcional para extrator aprendido de tinta; sem módulo instalado no pacote, avisa uma vez e segue com a heurística |
| `--markers` | Modo opcional com a folha-modelo impressa e marcadores ArUco |
| `--check` | Valida a fonte informada (`--source`) porta por porta e sai, sem abrir janela |
| `--save-dir pasta` | Pasta das capturas. O padrão é `captures` |

A variável `DRAW_ORT_THREADS` troca o número de threads da sessão ONNX sem tocar no código. O padrão é 4, medido na máquina do autor.

A máscara de mão é opcional e pede um pacote fora da instalação padrão.

```bash
pip install mediapipe
```

Sem o pacote ou sem o modelo, o app roda sem a máscara e avisa uma vez.

Teclas dentro do app:

- `s` salva o canvas digital como PNG em `captures/` (e `canvas_<ts>_stamps.npz` com os carimbos temporais do acumulador quando há tinta consolidada)
- `c` congela ou reinicia a página base de 12 quadros quando `--session-baseline` está ativo
- `f` salva o frame cru atual em `captures/frames/`
- `g` liga ou desliga a captura automática, um frame cru a cada 10 quadros
- `q` ou `Esc` sai

O modo padrão é o sem marcadores, em qualquer folha ou página de caderno. Ele é o padrão desde a v0.2. O modo com marcadores pede a folha-modelo gerada por `python template/generate_template.py`.

## Como funciona

O pipeline tem quatro etapas por frame. A câmera entrega o frame. Um detector encontra a folha e devolve os quatro cantos. A homografia retifica a folha para a vista de cima, em 850 por 1200 px. A etapa de tinta separa o traço do papel nessa vista e atualiza o canvas digital.

### Detecção neural

O detector primário é um SegFormer-B0 exportado para ONNX. Ele devolve duas saídas. Um logit de presença diz se há papel na cena. Uma máscara diz onde o papel está. O quadrilátero da folha sai do maior contorno da máscara, com escada de epsilon, reajuste de cada aresta como reta e refino subpixel nos cantos. Uma memória de arestas guarda a última reta confiável de cada lado. Quando a mão cobre uma aresta, a reta herdada segura o canto no lugar em vez de projetar a borda do braço.

Quando o modelo não vê folha no frame, o detector clássico assume aquele frame. Quando o modelo vê folha e o ajuste do quadrilátero falha, o retângulo de área mínima da máscara entra como reserva.

O ONNX dinâmico aceita qualquer resolução de entrada. O app roda esse arquivo a 384 por 384. O ONNX original, de formato fixo, impõe os 512 por 512 dele. A seção de desempenho traz os números dessa troca.

### Detecção clássica

O detector clássico é visão computacional em OpenCV, em duas etapas.

A primeira etapa propõe candidatos por geometria. O frame passa por Canny e pela árvore completa de contornos. Um contorno vira candidato quando fecha um quadrilátero convexo, com área de pelo menos 12% do frame e razão de aspecto entre 0,4 e 2,5. A aproximação do polígono usa uma escada de epsilons de 0,01 a 0,05, porque cantos arredondados e a lombada de um caderno não fecham num epsilon só. Os cantos aceitos ganham refino subpixel. No loop do app, a detecção roda de forma espaçada e os cantos passam por estabilização temporal.

A segunda etapa é um portão de textura. Ele mede o interior encolhido de cada candidato e devolve um de três vereditos.

- `paper` quando o interior parece papel
- `blank_maybe` quando o candidato é branco o bastante, mas liso demais para afirmar papel
- rejeitado quando o candidato nem branco o bastante é

O portão mede a fração de branco, a densidade de bordas internas e a variância do Laplaciano. Um candidato vira `paper` quando a fração de branco passa de 0,5, a variância do Laplaciano passa de 30 e uma de duas condições de textura vale. A densidade de bordas precisa de pelo menos 0,03, ou a variância do Laplaciano precisa de pelo menos 200. Um candidato reprovado não encerra a busca. O contorno seguinte ainda pode ser a folha.

Um candidato `blank_maybe` recebe tratamento mais duro no app. Ele precisa do dobro de detecções consistentes para travar, 6 em vez de 3. Enquanto a trava vem desse veredito, a tela mostra o aviso de baixa confiança.

O portão existe por um histórico medido. A versão só geométrica aceitava qualquer quadrilátero convexo grande. Numa auditoria com 10 cenas sem papel nenhum, ela disparou em 6. Tapete de mesa, caixa de papelão, monitor e bandeja branca viraram folha. O portão de textura cortou esses casos nas cenas medidas, sem modelo nenhum. Os limiares atuais são calibração de laboratório e precisam ser refeitos com capturas reais.

### Tinta e canvas acumulativo

Um limiar adaptativo separa a tinta do papel na folha retificada. Um veto separável roda junto. Um pixel só vira tinta quando está pelo menos 6 níveis abaixo das medianas da linha e da coluna inteiras dele. O veto mata a pauta do caderno e a textura do papel, que o limiar sozinho marcava como tinta.

O canvas não é uma foto do frame atual. Ele acumula evidência com histerese sobre uma janela de 15 frames. Um pixel entra na tinta com fração de amostras visíveis de 0,45 e sai com 0,20, com um mínimo de 4 amostras visíveis. O traço sobrevive à mão passando por cima. A borracha também funciona por esse caminho. A tinta apagada some do canvas depois de amostras limpas visíveis o bastante, não no frame seguinte.

Com a máscara de mão ligada, a região sob a mão vira desconhecida (não ganha nem perde tinta enquanto está coberta) e o marco 8 (ponta do indicador) projetado no espaço do papel alimenta `draw/fusion.py` (`PenTipFusion` com One Euro Filter + centroide do delta fresco por quadro) para estimar posição e toque da caneta sem rodar o MediaPipe duas vezes.

Com `--session-baseline`, `SessionBaselineInk` captura a mediana visível de 12 quadros ao apertar `c`, realinha cada quadro à referência em escala nativa por correlação de fase (`512x864`, largura total da folha) e normaliza o ganho global apenas sobre os pixels visíveis fora da mão, preservando traços finos de lápis sem transformar pauta ou margem impressa em falsa tinta.

### Vetorização offline e primitivas geométricas (Camada 2)

`tools/vectorize.py` converte o PNG do canvas salvo (e opcionalmente o `_stamps.npz` correspondente) em polilinhas simplificadas, primitivas geométricas em milímetros do A4 (`line`, `circle`, `rectangle`, `polygon`, `polyline`), SVG semântico e JSON estruturado:

```bash
python tools/vectorize.py captures/canvas_1730000000.png --clean --stamps captures/canvas_1730000000_stamps.npz --primitives --json saida.json
```

## Desempenho medido

Números medidos em 2026-10-29 no Ryzen 7 7700X do autor, com o pipeline completo do app sobre frames reais. Medianas por frame.

| Configuração | Tempo por frame | Taxa |
|---|---|---|
| Sessão padrão do ONNX Runtime, modelo fixo a 512 | 212,8 ms | 4,7 fps |
| 4 threads intra-op, modelo fixo a 512 | 183,5 ms | 5,4 fps |
| 4 threads intra-op, modelo dinâmico a 384 | 127,2 ms | 7,9 fps |

A sessão sozinha explica quase tudo. No padrão do ONNX Runtime, `session.run` marcava 248 ms no mesmo tensor. Com 4 threads, marca 122 ms a 512 e 58,3 ms a 384. Com 16 threads, piora para 210 ms. INT8 dinâmico foi medido e rejeitado. A qualidade dele se manteve, mas o tempo subiu para 366 ms, três vezes o FP32 dessa máquina.

A troca de resolução custou pouco em qualidade. Nos 49 frames do harness, a máscara a 384 tem IoU médio de 0,984 contra a máscara a 512. Nenhum veredito de presença trocou nos positivos. As três frames de uma caixa branca sobre a mesa, que a 512 eram falso positivo fraco, a 384 viram negativo.

## Treino do detector

Todo o treino roda no Kaggle, em GPU. Os frames reais do autor ficam em datasets privados e nunca entram neste repositório.

A mistura da v3 combina quatro fontes. MIDV-2020 photos traz 1.000 fotos reais de documentos com cantos anotados. SmartDoc 2015 traz 24.889 frames com gabarito. O dataset school-notebooks-RU traz páginas de caderno. Os 49 frames da mesa do autor entram com rótulo de presença para todos e polígonos desenhados à mão para 11 deles.

O detector clássico chegou a ser o professor planejado para rotular os frames. Nos 49 frames reais ele rotulou zero, porque o caderno sai cortado pela borda superior do quadro e os quatro cantos nunca aparecem juntos. O gabarito humano substituiu o professor.

O harness `tools/eval_real_frames.py` é o portão de qualquer modelo novo. Nenhum modelo entra no app sem passar por ele. A v3 marca presença em 43 de 43 positivos e 0 falsos positivos nos 6 negativos. Nas 11 anotações, o IoU de máscara é 0,962 de média e 0,841 de mínimo a 512. Na configuração de produção, a 384, presença é 49 de 49 e o IoU é 0,959 de média e 0,851 de mínimo.

O kernel da v3 levou 44,7 minutos numa T4. O kernel da v2 levou 3 horas e 51 minutos. A diferença não veio de dados menores. Veio de pipeline assíncrono com workers, precisão mista, batch 16 e uma agenda de 20 épocas com parada por régua, no lugar de 40 épocas fixas.

## Limites conhecidos

- Uma folha totalmente em branco e lisa mede igual a qualquer retângulo branco e liso, como uma bandeja branca. Nenhuma estatística de interior separa os dois casos para o detector clássico. Quem separa é identidade de objeto aprendida, e é por isso que o neural é o primário.
- A v3 nunca foi julgada na mesa de outra pessoa. O conjunto de teste externo é a régua que falta para chamar o detector de geral.
- A ponte de captura do Windows em `bridge/` nunca foi testada numa máquina real.
- Traço fino de 1 a 3 px sofre com o jitter da retificação. A média medida é de 5,4 px por amostra no replay do autor. O canvas acumulativo reduz o dano, mas não elimina a causa.

## Arquitetura

- `draw/` é o pacote principal:
  - `app.py` tem o loop da janela OpenCV, atalhos de teclado e diagnóstico `--check`.
  - `pipeline.py` concentra o `DrawPipeline` sem GUI (estabilização `_SheetTracker`, cadência uniforme 1/1 de tinta/mão/cursor, trabalhador opcional `--async-detect` e persistência de PNG + `_stamps.npz`).
  - `paper.py` tem a detecção clássica, o portão de textura em ROI com mapas preguiçosos e a homografia (`warp_sheet`, `perspective_matrix`, `warp_points`).
  - `neural.py` tem o detector neural por ONNX, com presença, máscara, memória de arestas e guardas de retidão/corpo rígido contra oclusão de braço.
  - `ink.py` tem a heurística de tinta (`extract_ink`), a tinta diferencial da sessão (`SessionBaselineInk` + `align_to_reference`) e o canvas acumulativo em inteiros/LUT (`InkAccumulator`).
  - `hand.py` tem a máscara de oclusão e extração dos 21 marcos da mão via MediaPipe (`HandMasker.detect`).
  - `fusion.py` tem a fusão temporal marco 8 + delta fresco de tinta (`PenTipFusion`, `OneEuroFilter2D`).
  - `protocol.py` implementa o protocolo binário `bridge/PROTOCOL.md` e o adaptador `HelperCapture` para arquivos `.drawcap` e sockets `tcp://host[:8765]`.
  - `config.py` guarda o tamanho da folha retificada (`850x1200`) e os dados dos marcadores.
- `template/` tem o gerador da folha-modelo imprimível, usado só pelo modo `--markers`. O PNG gerado não entra no versionamento.
- `bridge/` tem a especificação do protocolo binário de captura (`bridge/PROTOCOL.md`) e instruções da ponte de captura no Windows ([bridge/README.md](bridge/README.md)).
- `scripts/` guarda `setup_camera_wsl.sh`, o instalador da câmera do WSL.
- `tests/` guarda toda a bateria sem webcam (`run_all.py`, `selftest.py`, testes diferenciais do acumulador e carimbos, `SessionBaselineInk`, guardas neurais, `DrawPipeline`/fontes, fusão e vetorização v2).
- `tools/` reúne as ferramentas de apoio e bancada: `vectorize.py` (vetorização + primitivas da Camada 2), `helper_receiver.py` (gravação/inspeção `.drawcap` e TCP), `step0_latency.py`, `run_log.py`, `label_frames.py` e `eval_real_frames.py`.
- `verification/` guarda a skill e o verificador geométrico da homografia (`run_verification.py`, cobrindo o caminho com marcadores, o caminho sem marcadores em 10 poses e a contraprova `--sabotage`).
- `docs/` tem o guia do Windows com WSL e os registros técnicos de bancada.
- `main.py` é o ponto de entrada.
- `VISION.md` é o documento de visão do projeto.

## Desenvolvimento

Sem webcam por perto, a bateria inteira (selftest, 198 sondas de homografia + contraprova `--sabotage`, testes unitários/diferenciais e selfchecks de todas as ferramentas em `tools/`) roda num comando só:

```bash
python tests/run_all.py
pytest tests/ -q
```

Para rotular frames capturados:

```bash
python tools/label_frames.py captures/frames dataset/rotulos
```

A saída é um `labels.json` com os frames rotulados e os rejeitados. Cada frame rotulado ganha uma imagem de revisão na subpasta `overlays`. Os frames de origem nunca são modificados.

## Roadmap

Feito:

- Captura em tempo real, com folha retificada, tinta extraída e canvas digital ao vivo
- Fonte de vídeo flexível, por índice local ou por URL de stream
- Câmera do Windows anexada ao WSL pelo script de instalação
- Diagnóstico de câmera porta por porta com `--check`
- Detecção sem marcadores como modo padrão
- Portão de textura com veredito de baixa confiança para folha lisa
- Segmentação de papel treinada no Kaggle, com sinal de presença, nas versões v1, v2 e v3
- Datasets reais de documentos e frames da mesa do autor no treino
- Canvas acumulativo com histerese em LUT uint8, carimbos temporais e máscara de mão opcional
- Tinta diferencial da sessão (`--session-baseline`) com realinhamento subpixel por correlação de fase e ganho mascarado por visibilidade
- Pipeline desacoplado (`DrawPipeline`) com cadência uniforme 1/1, detecção assíncrona (`--async-detect`), fusão de ponta de caneta (`PenTipFusion`) e suporte direto a `.drawcap` / `tcp://`
- Vetorização offline v2 (`tools/vectorize.py`) com indexação O(H*W), poda 8-mínima de esporões, costura em junções, ordenação temporal e classificação de primitivas geométricas (Camada 2: reta, círculo, retângulo)
- Pacote de desempenho com threads fixadas e modelo dinâmico a 384

Em andamento:

- Conjunto de teste externo, com frames de outros setups, para julgar a generalização da v3
- Fase do traço, com a fidelidade ao vivo já medida e a ponte de cursor para o Windows em construção
- Modelo aprendido de tinta, com a formulação fechada e o dataset sintético em montagem

Planejado:

- Traço vetorial, com ordem, caminho e suavização dos traços
- Interpretação de formas, letras e diagramas
- Integração com um primeiro sistema alvo, CAD ou simulador
- Feedback de simulação em tempo real
