# Guia opcional: ligar a exportação ao app

Este PR entrega APIs para extrair traços, exportar SVG/DXF e classificar
formas. **Não liga essas APIs ao loop da câmera, não adiciona atalhos e não
altera a interface.** Essa separação é intencional: o app tem mais de uma
fonte possível para a tinta, e a integração precisa respeitar o modo ativo e
o estado acumulado da sessão.

Este guia é uma opção para quem quiser fazer essa integração depois. O
mantenedor pode pedir ao Codex que siga estes passos; aceitar o PR não implica
aceitar também o wiring.

## Passo a passo

1. **Leia o `draw/app.py` da branch que será alterada.** Não copie o `app.py`
   de outra branch. Confirme como essa versão seleciona a folha, produz a
   máscara, acumula a sessão e trata as teclas. O código pode ter mudado desde
   a criação deste guia.

2. **Escolha explicitamente a imagem que será exportada.**
   - No modo com marcadores, a fonte é a máscara limpa produzida para a folha
     retificada (`extract_ink(..., clear_markers=True)`).
   - No modo sem marcadores, se o usuário está vendo um canvas acumulado,
     exporte esse estado acumulado para corresponder ao que aparece na tela.
     O canvas renderizado tem fundo branco e tinta preta; converta-o em uma
     máscara binária com tinta `255` antes de chamar a extração:

     ```python
     gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
     ink_mask = cv2.threshold(
         gray, 127, 255, cv2.THRESH_BINARY_INV
     )[1]
     ```

   - Se a versão atual não estiver mostrando um canvas acumulado, use a máscara
     de tinta que já alimenta a renderização daquele modo. Não passe a imagem
     RGB/BGR da câmera, nem o warp da folha, diretamente para
     `extract_strokes`.

   O caminho sem marcadores pode atualizar o acumulador somente quando a
   detecção roda. Ao integrar, decida se o requisito é exportar o último
   canvas confirmado (mais fiel ao preview) ou reprocessar o frame atual; não
   misture silenciosamente essas duas semânticas.

3. **Execute a vetorização sob demanda.** Faça a extração quando o usuário
   pedir a exportação, não em todos os frames do loop. Isso evita acrescentar
   afinamento e percurso do esqueleto à latência normal da câmera.

   ```python
   from draw import export, strokes

   found = strokes.extract_strokes(ink_mask)
   svg_path, json_path = export.export_strokes_scaled(
       found, base_path, paper="A4"
   )
   dxf_path = export.export_strokes_dxf(
       found, base_path, paper="A4"
   )
   ```

4. **Defina escala e destino.** A escala padrão assume folha A4 e canvas
   `850 × 1200 px`. Para outras dimensões, crie uma escala com
   `export.sheet_scale(canvas_width=..., canvas_height=..., paper=...)` e
   passe-a às duas funções. Crie a pasta de saída antes de escrever e use um
   nome único por sessão para não sobrescrever exportações anteriores. O DXF
   está em milímetros e inverte o eixo Y para coordenadas CAD.

5. **Preserve os atalhos existentes.** Adicione uma tecla diferente de `s`
   (que já salva PNG), `f`, `g`, `c` e `q`/Esc. Mostre no terminal ou na
   interface os caminhos efetivamente salvos e trate canvas ausente/vazio sem
   encerrar o loop da câmera.

6. **Mantenha shapes opt-in e experimentais.** Se a integração também chamar
   `shapes.classify_strokes`, salve a classificação em saída separada, preserve
   os traços sem classificação e aceite `unknown`. Os limiares foram
   calibrados com geometrias sintéticas; não mostre os rótulos como
   reconhecimento confiável ou geometria CAD validada sem avaliação em dados
   reais rotulados. A sequência extraída é espacial/determinística, não a
   ordem temporal em que a pessoa desenhou.

7. **Valide a integração na branch de destino.** Acrescente testes sem webcam
   que cubram: seleção da máscara em cada modo; conversão do canvas acumulado;
   acionamento do atalho e nomes/unidades dos arquivos; canvas vazio; e
   preservação dos atalhos atuais. Use captura e escrita simuladas nos testes.
   Rode também os testes existentes de traços, exportação e formas. Faça uma
   sessão manual com desenho aberto, forma fechada, traço apagado e perda ou
   reaquisição da folha antes de tornar a ação padrão.
