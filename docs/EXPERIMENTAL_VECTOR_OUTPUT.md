# Exportação vetorial e formas experimentais

Esta integração adiciona uma API de exportação em escala física e uma
classificação geométrica experimental. Ela não altera os atalhos nem o
comportamento do app.

O app ainda não chama essas APIs. Para uma integração opcional posterior,
consulte [INTEGRATION_WITH_APP.md](INTEGRATION_WITH_APP.md), que explica
como selecionar a máscara correta por modo e preservar a semântica do canvas
acumulado.

## Exportar traços para SVG e DXF

`draw.export` recebe a lista produzida por `draw.strokes.extract_strokes`.
O SVG e o DXF usam escala uniforme em milímetros; o DXF inverte o eixo Y
para o sistema de coordenadas CAD. A escala padrão presume papel A4 e canvas
850 × 1200 px. Se o papel ou o tamanho do canvas forem diferentes, informe
esses valores explicitamente.

```python
from draw import export, strokes

found = strokes.extract_strokes(ink_mask)
svg_path, json_path = export.export_strokes_scaled(
    found, "saida/desenho", paper="A4"
)
dxf_path = export.export_strokes_dxf(found, "saida/desenho", paper="A4")
```

O exportador preserva a ordem recebida. O snapshot atual ordena traços por
posição; isso não representa a ordem temporal do desenho.

## Classificação de formas

`draw.shapes.classify_strokes` agrupa traços próximos e retorna linha,
círculo, retângulo ou `unknown`, com confiança e referência aos traços de
origem. Os limiares foram calibrados em geometrias sintéticas. Os testes
confirmam esses casos controlados, não a precisão em desenhos reais. Trate
os rótulos como experimentais e preserve os traços originais; não use os
rótulos como geometria CAD aprovada sem uma avaliação com exemplos reais
rotulados.

```python
from draw import shapes

predicted = shapes.classify_strokes(found)
shapes.export_shapes(predicted, "saida/formas.json")
```
