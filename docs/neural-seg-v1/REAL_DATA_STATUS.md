# Status dos dados reais para treinamento

## Estado desta proposta

Este PR adiciona ferramentas e instruções para preparar dados reais para o
treinamento do segmentador. Ele **não entrega um dataset real validado nem um
modelo treinado** e não afirma melhora de qualidade ou acurácia.

| Item | Estado |
|---|---|
| Frames selecionados | 719 frames produtivos de duas sessões; 41 duplicatas claras foram excluídas. |
| Revisão visual | Revisão em miniaturas concluída; 21 erros claros em regiões grandes foram corrigidos. |
| Máscaras pixel a pixel | Revisão incompleta. Bordas finas de tinta e casos ambíguos ainda precisam de revisão em resolução completa. |
| Rótulos | As máscaras candidatas são pseudo-rótulos, não ground truth. Pixels `255` representam regiões desconhecidas e são ignorados pelo treinador. |
| Dados publicados neste PR | Nenhuma captura original nem máscara gerada está incluída. |
| Treino e resultado do modelo | Nenhum treino foi executado; nenhum checkpoint ou modelo treinado é entregue. |
| Testes automatizados | Após a reconciliação: 29 passaram e 2 testes dependentes de PyTorch completo foram ignorados, pois o ambiente local não tem PyTorch utilizável. O smoke CPU e as verificações de sintaxe/configuração passaram; treino e exportação ONNX completos continuam pendentes. |

## Uso permitido neste estágio

O pacote de máscaras pode ser usado para experimentos exploratórios. Resultados
de treino devem ser identificados como exploratórios e separados das métricas
de validação sintética. As duas sessões usadas para treino não constituem um
holdout real independente.

## Antes de alegar validação dos dados ou do modelo

1. Revisar cada par imagem/máscara em resolução completa, incluindo tinta,
   texto impresso, mão sobre tinta e limites ambíguos.
2. Confirmar integridade dos 719 pares: IDs, hashes, dimensões, canal único e
   valores permitidos `{0, 1, 2, 3, 255}`; rejeitar máscaras sem pixels
   supervisionados.
3. Executar os testes automatizados e o preflight do dataset.
4. Treinar e publicar métricas com rastreabilidade de configuração e dados.
5. Reservar uma nova sessão real para avaliação independente antes de fazer
   afirmações de desempenho em dados reais.
