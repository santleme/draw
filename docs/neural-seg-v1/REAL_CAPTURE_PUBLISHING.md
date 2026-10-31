# Publicar lotes de captura real

## Padrão

- Os frames e arquivos de arquivo (`.tar`) ficam no Kaggle em um dataset
  **privado**. O Git guarda somente `frames.csv`, `takes.csv`, o manifesto do
  lote e os checksums dos arquivos publicados.
- Não use `git add -f captures/` para contornar o `.gitignore`. O fato de
  `captures/` estar ignorado é intencional.
- Agrupe os JPEGs em arquivos `.tar` de aproximadamente 200 MB, sem
  recomprimir as imagens. Prefira fechar cada parte no limite entre takes;
  registre no manifesto quais takes e frames estão em cada parte. Se um take
  sozinho exceder o tamanho-alvo, divida-o e registre os intervalos de frames.
- Mantenha partes numeradas e reproduzíveis (`part-000.tar`, `part-001.tar`,
  …). Não inclua credenciais Kaggle, arquivos de configuração com tokens ou
  imagens soltas no commit.

## Preparar e conferir um lote

1. Preserve a sessão capturada e confirme que `frames.csv` e `takes.csv`
   cobrem todos os JPEGs. Não renomeie os frames depois de gerar os CSVs.
2. Monte as partes por take, sem compressão. Antes de publicar, confira que
   cada arquivo pode ser listado/extraído e gere SHA-256 para cada parte.
3. Crie ou atualize o dataset Kaggle como **privado**. Use a divisão em partes
   de ~200 MB que já funcionou para os lotes sintéticos. Guarde o slug e a
   versão do dataset no manifesto do lote.
4. Depois do upload, confira no Kaggle o estado `ready`, a lista e o tamanho
   das partes. Baixe ao menos uma parte e compare seu SHA-256 com o local.
5. Faça commit dos CSVs, do manifesto e dos checksums. Antes do commit,
   confirme com `git status --short` que nenhum JPEG ou `.tar` será incluído.

O manifesto deve registrar pelo menos: `batch_id`, caminho local da sessão,
data/hora, câmera, resolução/FPS solicitados e observados quando disponíveis,
contagem de frames, takes completos e parciais, intervalos de frames por parte,
bytes e SHA-256 de cada parte, slug/versão/visibilidade do Kaggle e eventuais
lacunas ou falhas da captura. Use caminhos relativos e nunca registre tokens.

## Fallback: Git

Use este caminho somente quando o armazenamento privado não estiver disponível
ou quando o usuário pedir explicitamente Git. Agrupe JPEGs em `.tar` sem
compressão, mantendo cada arquivo abaixo de 90 MiB para ficar com margem sob
o limite por arquivo do GitHub. Commite e faça push a cada 10 takes,
registrando o intervalo de takes em cada commit. Confira o push antes de
iniciar o próximo grupo. O Git continua público se o repositório for público; não envie dados
pessoais sem autorização explícita.

SSH pode ser escolhido para novos pushes depois de confirmar que a chave está
configurada e que `git ls-remote git@github.com:GUZZBR1/draw.git` funciona. Não
troque o protocolo nem reinicie um push grande que já está avançando: uma
tentativa nova precisa transferir os objetos novamente.

## Estado do primeiro lote

O lote `batch_01` foi commitado como frames JPEG individuais antes deste
procedimento. Não reescreva seu histórico nem replique esse método nos próximos
lotes. A transferência desse lote deve ser verificada pelo SHA remoto antes de
ser considerada concluída.
