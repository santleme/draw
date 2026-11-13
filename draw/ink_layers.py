"""Tinta em camadas: acumulador com voto e a trabalhadora page-clean.

O acumulador guarda a tinta de uma geracao em duas camadas — a
provisoria, que a primeira passada viu, e a confirmada, que o voto
entre passadas fechou (com a porta de fronteira contra o ombro do
tremor). A trabalhadora roda as passadas em thread propria com a
cadencia de _PAGE_CLEAN_INTERVAL_SECONDS, a porta de ociosidade
sem mao, o descanso total de pagina assentada e o evento de
virada atras da flag DRAW_PAGE_TURN_EVENT.

Extraido de draw/app.py na modularizacao de 2026-11-05; o app
re-exporta todos os nomes deste modulo. As constantes de cadencia
e descanso sao lidas deste modulo em tempo de execucao: quem
precisa muda-las (os testes) patcheia aqui, onde a leitura mora.
"""

import os
import threading
import time
from collections import deque

import cv2
import numpy as np

from . import ink as ink_module


# Porta de ociosidade sem mao: no maximo uma sonda a cada
# _IDLE_PROBE_SECONDS. A porta e a presenca da mao, nao um
# detector de mudanca por diff de warp: medido nesta webcam, o
# churn entre quadros parados (1,5-2,0) supera o sinal do lapis,
# entao diff cego dispara a toa ou engole o traco fraco. Escrever
# exige mao no papel; sem mao, a pagina esta assentada e so a
# sonda periodica confere.
_IDLE_PROBE_SECONDS = 1.0


# Sem sinal de mao (clone sem MediaPipe), a porta de ociosidade
# nunca fecha e a trabalhadora rodaria a 250 ms para sempre —
# CPU queimada a toa num app que o dono acha "lento". O descanso,
# nesse caso, vem da tinta parada: depois de
# _NO_HAND_REST_AFTER_STILL_PASSES passadas seguidas sem nenhuma
# mudanca no acumulador (nada confirmou, nada provisoriamente
# mudou), a cadencia cai para a da sonda ate algo mudar. O sinal
# e o do proprio acumulador, nao diff de quadro — o diff cego
# nesta webcam ja foi medido e descartado (churn de exposicao
# empata com o traco). Com sinal de mao presente, nada muda.
_NO_HAND_REST_AFTER_STILL_PASSES = 4


# Descanso total (especificacao do Lucas, 2026-11-01 e de novo
# 2026-11-03): folha assentada e saida confirmada quieta = ZERO
# passadas, nao sonda de 1 Hz para sempre. A sonda de 1 Hz
# seguia alimentando o voto sobre warps que tremem, e cada
# alinhamento de fase confirmava uma camada nova sobre o
# desenho parado (regua de replay, modo hold: salto de +9.999
# px na passada 89). O descanso entra quando a geracao ja rodou
# uma janela de voto inteira (_REST_MIN_GENERATION_PASSES) e as
# ultimas _REST_QUIET_PASSES passadas nao confirmaram nada novo
# — o portao e sobre a SAIDA CONFIRMADA do acumulador, nunca
# sobre diff de quadro (licao de 2026-11-01: o churn da camera
# supera o sinal do lapis). Acorda com evento que anuncia tinta
# possivel: a mao volta ou se mexe (set_hand_presence), a folha
# troca (new_generation) ou a geracao vira. Sem sinal de mao
# (None) o descanso total NAO fecha: nao existe evento que
# anuncie tinta nova, e a sonda de 1 Hz segue sendo o mecanismo
# de acordar desse clone — com a porta de fronteira, ela nao
# acumula mais.
_REST_MIN_GENERATION_PASSES = 3


_REST_QUIET_PASSES = 2


# Descanso total tambem COM a mao presente (decisao do Lucas,
# 2026-11-03): ele segurando o caderno parado, a cadencia de
# 250 ms seguia eterna. A extensao usa a mesma maquinaria de
# quietude da saida confirmada (_REST_*), mais a mao parada por
# _HAND_REST_STILL_SECONDS. Tinta nova com a mao ja no quadro
# nao se anuncia pela tinta — quem acorda e a mao se mexendo,
# medida na assinatura da mascara que o adaptador ja computa
# (area e centroide), nunca por diff do quadro da pagina (o
# churn da camera supera o sinal do lapis, licao de 2026-11-01).
# Nao ha sonda periodica de tinta neste caminho: sonda e
# passada, e o ponto do descanso e zero passadas.
_HAND_REST_STILL_SECONDS = 1.0


# Evento geometrico que acorda o descanso (frente do cenario E3,
# atras de DRAW_INK_FOLLOW_MOVE, default desligado): a premissa do
# descanso sem mao e "a pagina nao muda" — mas a folha que desliza
# sem mao na cena mostra exatamente o contrario (medido em
# 2026-11-12: o warp congelado seguiu vendo mesa e a trabalhadora
# dormiu no meio do movimento, porque submit nao acorda). Um quad
# submetido a mais de _GEO_WAKE_MOVE_PX da ancora e evidencia de
# cena nova — enquanto o deslocamento persistir, a quietude nao
# acumula e a sonda segue; quando a geometria retorna a moldura,
# _REST_QUIET_PASSES recoloca a trabalhadora para dormir.
# 25 px casa com o destrave do historico do laco: abaixo disso o
# rastreador ainda trata como jitter de deteccao.
_GEO_WAKE_MOVE_PX = 25.0


# Cadencia maxima da tinta: uma passada completa a cada ~250 ms.
# A passada custa 168 ms no Ryzen com 8 threads (perfil em
# draw-lab/page-clean-v0/PERF_BASELINE_RYZEN.md), entao quem manda
# no ritmo do canvas e a cadencia, nao o modelo — e o laco
# principal nao paga nenhum dos dois.
_PAGE_CLEAN_INTERVAL_SECONDS = 0.25


# Teto da espera pela trabalhadora de tinta na saida do app.
_PAGE_CLEAN_STOP_TIMEOUT_SECONDS = 5.0


# Virada (U3): canvas com pelo menos _INK_TURN_MIN_CANVAS_PX e a
# mascara da passada cobrindo <= _INK_TURN_MAX_OVERLAP dele por
# _INK_TURN_PASSES passadas seguidas = pagina trocada (conteudo
# descorrelacionado). Apagar um trecho a mao nao derruba a
# cobertura da pagina inteira a <=30%: o apagado continua no canvas
# e o resto do desenho segue coberto pela mascara.
_INK_TURN_MIN_CANVAS_PX = 2000


_INK_TURN_MAX_OVERLAP = 0.30


_INK_TURN_PASSES = 2


# HOTFIX 2026-11-01 (draw-lab/page-turn-falsefire-2026-11-01): o
# gatilho de cobertura acima foi REPROVADO no teste ao vivo. Na
# mesa real, o ajuste da folha do caderno aberto oscila (o proprio
# codigo registra oscilacao de ate 314 px nos cantos): o warp
# balanca, a saida do modelo desencontra do canvas acumulado e a
# cobertura desaba SEM pagina nenhuma virar — o evento disparava
# a cada 2-5 s, salvando snapshot e zerando o canvas sem parar.
# A prova offline usou warps estaticos com tremor de +-1-3 px,
# mansos demais pra expor isso; o teste negativo era borracha,
# nao balanco de geometria. Decisao: o disparo automatico por
# cobertura fica DESLIGADO por padrao, atras da flag
# DRAW_PAGE_TURN_EVENT=1, ate existir um gatilho provado contra
# serie de balanco de geometria realista. O codigo do evento e o
# snapshot ficam intactos (versao estrita futura); com a flag
# desligada, nenhum colapso de cobertura zera o canvas — a troca
# de pagina segue pelo caminho pre-existente de drop/reaparicao
# da folha, que abre geracao nova do mesmo jeito.
_ENV_TURN_EVENT_VAR = "DRAW_PAGE_TURN_EVENT"


def _page_turn_event_enabled():
    """True so com DRAW_PAGE_TURN_EVENT=1 (ver o hotfix acima)."""
    return os.environ.get(_ENV_TURN_EVENT_VAR, "").strip() == "1"


# Confirmacao entre passadas (regras medidas no diagnostico de
# 2026-11-01, draw-lab/stall-diagnosis-2026-11-01): sobre a soma
# das probabilidades das ultimas _INK_VOTE_WINDOW passadas, um
# pixel confirma quando (a) a soma chega a _INK_VOTE_SCORE_STRONG
# (tinta forte confirma na 2a passada) ou (b) ele aparece com
# prob >= _INK_VOTE_HIT nas 3 passadas da janela e a soma chega a
# _INK_VOTE_SCORE (o lapis fraco estavel confirma na 3a). Copia
# deslocada de tremor e FP de mobiliario aparecem de forma
# intermitente por posicao e nao fecham nem (a) nem (b).
_INK_VOTE_WINDOW = 3


_INK_VOTE_SCORE = 1.4


_INK_VOTE_SCORE_STRONG = 1.9


_INK_VOTE_HIT = 0.5


_INK_VOTE_MIN_HITS = 3


# Porta de fronteira da confirmacao (medida na regua de replay,
# draw-lab/replay-ruler-2026-11-01, modo hold sobre o codigo sem
# ela): com a pagina desenhada parada, o canvas confirmado
# crescia em degraus a cada alinhamento de fase entre o ciclo do
# warp e a janela do voto — um salto de +9.999 px na passada 89
# e 6.144 px confirmados que passada nenhuma viu acima do
# limiar. O mecanismo: o warp treme, copias deslocadas do mesmo
# traco somam probabilidade SUB-limiar (0,5-0,7) no ombro do
# traco e o voto fecha sobre a soma, camada sobre camada, sem
# nenhuma passada ter chamado aquilo de tinta.
# A porta: pixel dentro da faixa do ombro — dilatacao de
# _INK_FRONTIER_RADIUS px sobre (tinta confirmada ∪ tinta que
# a janela viu acima do limiar) — so confirma se alguma
# passada da janela o viu acima do limiar: evidencia nova
# SUBSTITUI a posicao da antiga, nao soma um ombro novo ao
# redor dela. O raio e o ombro medido nos mapas reais do
# cache da regua: a banda 0,5-0,7 mora a p50 = 1 px e
# p90 ≈ 3 px do nucleo > 0,7. Longe do ombro o voto segue
# intacto: e ele que descobre o lapis fraco consistente
# (0,55 x 3) em regiao nova. Varredura offline sobre os warps
# f28-f30 em ciclo, 300 passadas: fora-da-referencia cai de
# 8.567 px (sem porta) para 419 px (raio 3 ancorado), e o
# crescimento total satura no teto da uniao das passadas.
_INK_FRONTIER_RADIUS = 3


_INK_FRONTIER_KERNEL = np.ones(
    (2 * _INK_FRONTIER_RADIUS + 1, 2 * _INK_FRONTIER_RADIUS + 1),
    dtype=np.uint8,
)


# Regra de voto E3 (frente 89, desenho + prototipos offline em
# draw-lab/live-audit-2026-11-10/89-regra-do-voto.md; implementacao
# da frente 90): a candidata vencedora da F89, porto fiel da bancada
# dela (frente89/f89_bench4.py, classe CommitE3). A regra atual
# confirma sobre janelas deslizantes sobrepostas e o canvas cresce
# 2,90x na regua de 17 passadas (relatorios 86/89): a franja que
# pisca 2x na mesma posicao dentro de alguma janela confirma. A E3
# separa a geracao em duas fases:
# - Fase 1 (as _INK_VOTE_E3_PHASE1_PASSES primeiras passadas): o
#   canvas nasce UMA vez, do nucleo de consenso espacial das 3
#   passadas — tinta a <= _INK_FRONTIER_RADIUS px do pixel nas 3
#   passadas, vista acima do limiar ao menos 1x no proprio pixel e
#   soma das 3 probabilidades >= _INK_VOTE_E3_CORE_MIN_SUM (6.539 px
#   e contencao 0,904 na regua). A uniao de tudo que foi visto acima
#   do limiar nessas passadas vira "conteudo velho" da geracao.
# - Fase 2: so entra tinta de componente conexo NOVO — componente
#   (acima do limiar, area >= _INK_VOTE_E3_MIN_COMPONENT_PX) com
#   menos de metade da area sobre o conteudo velho dilatado e fora
#   das trilhas ja abertas. Componente sobre o velho e absorvido
#   pelo velho (copia deslocada e franja nunca abrem trilha). A
#   regiao do componente novo (dilatada) abre uma trilha lateral
#   onde so a evidencia da regiao vota, com persistencia
#   consecutiva estrita dividida por tamanho: componente grande
#   (>= _INK_VOTE_E3_BIG_COMPONENT_PX) confirma com
#   _INK_VOTE_E3_BIG_WINDOW passadas consecutivas acima do limiar e
#   soma >= _INK_VOTE_E3_BIG_SCORE; componente pequeno confirma com
#   _INK_VOTE_E3_SMALL_WINDOW e soma >= _INK_VOTE_E3_SMALL_SCORE.
# Custos medidos da F89, assumidos: o canvas final cobre menos da
# tinta de cada passada que o da regra atual (completude 0,308 x
# 0,634 no prototipo offline — a regua formal e da frente 90);
# marca pequena confirma devagar (trilha lenta); conteudo das 3
# primeiras passadas fora do nucleo e "velho" pra sempre na
# geracao. Atras de DRAW_INK_VOTE_E3 (padrao 0): com a flag
# desligada o acumulador roda a regra atual sem tocar em nenhum
# estado da E3. A flag e LIDA UMA VEZ por geracao (na primeira
# passada apos o reset): trocar a variavel no meio da geracao nao
# mistura as duas regras no mesmo canvas.
_INK_VOTE_E3_ENV = "DRAW_INK_VOTE_E3"
_INK_VOTE_E3_PHASE1_PASSES = 3
_INK_VOTE_E3_CORE_MIN_SUM = 1.0
_INK_VOTE_E3_MIN_COMPONENT_PX = 20
_INK_VOTE_E3_BIG_COMPONENT_PX = 100
_INK_VOTE_E3_BIG_WINDOW = 3
_INK_VOTE_E3_BIG_SCORE = 2.1
_INK_VOTE_E3_SMALL_WINDOW = 5
_INK_VOTE_E3_SMALL_SCORE = 3.5
_INK_VOTE_E3_DILATE_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (2 * _INK_FRONTIER_RADIUS + 1, 2 * _INK_FRONTIER_RADIUS + 1),
)
_INK_VOTE_E3_REGION_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))


def _ink_vote_e3_enabled():
    value = os.environ.get(_INK_VOTE_E3_ENV, "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


# Regra de voto E5-S (frentes 106/107, implementacao da frente 108;
# draw-lab/live-audit-2026-11-10/106-voto-e4-sucessor-e3.md e
# 107-voto-e5-subdivisao.md): sucessora da E3, atras de
# DRAW_INK_VOTE_E4 (o nome da flag e o da familia E4 desenhada na
# F106; o mecanismo implementado e a E5-S = E4j + subdivisao por
# nascimento, a configuracao que a F107 validou em prototipo).
# A E3 confunde "visto" com "velho" (a uniao de tudo que foi visto
# na fase 1 vira barreira e conteudo nascido encostado nela nunca
# abre trilha: hachura progressiva 0,072 no simulador). A E5-S
# separa as duas coisas e subdivide o resto:
# - "Velho" = so o instantaneo de ABERTURA da geracao: a mascara
#   da 1a passada congelada, que nunca expande por absorcao, mais
#   o MAPA de probabilidade da abertura — componente com >=50%
#   da area sobre o velho dilatado (r3) OU com >=50% dos pixels
#   com evidencia de abertura >= _INK_VOTE_E4_OPENING_EVIDENCE
#   e pre-existente naquela passada, sem virar estado.
# - Fase 1 identica a da E3 (nucleo de consenso espacial das 3
#   primeiras passadas, _e3_phase1_core), mas o residuo dela —
#   visto, nao confirmado — NAO vira barreira: na fase 2 ele e
#   reclassificado como qualquer componente novo.
# - Novidade e cobertura medidas contra o CONFIRMADO dilatado e
#   as regioes de trilhas ja abertas, nunca contra a uniao do
#   visto. Componente com <50% sobre a cobertura abre trilha pela
#   propria forma/distancia: longe do confirmado (>=50% da area
#   a mais de 8 px dele) abre trilha por tamanho — grande
#   (>=100 px, 3 consecutivas, soma >= 2,1, regiao r6), pequena
#   (20-99 px, 5 consecutivas, soma >= 3,5, regiao r6) ou FINA
#   (forma de linha: lado maior >= 50 e area <= 3,5x o lado;
#   voto com tolerancia espacial de 1 px, 3 consecutivas,
#   soma >= 2,1, regiao r3 — a linha de 1 px tremula +-1 px e o
#   hit estrito no mesmo pixel nao fecha); na banda 3-8 px do
#   confirmado abre a trilha LENTA (5 consecutivas, soma >= 3,5,
#   regiao r3) — a persistencia estrita separa a copia deslocada
#   do traco novo encostado.
# - SUBDIVISAO POR NASCIMENTO (a peca da F107): componente com
#   >=50% sobre a cobertura nao morre inteiro — o RESIDUO fora
#   da cobertura e re-componentizado e cada sub-componente
#   >= 20 px abre a SUA trilha, classificado pela sua propria
#   forma/distancia, com regiao r3. E o que fecha a classe da
#   absorcao da hachura (bloco de um componente so: as linhas
#   novas, minoria dele, nunca abriam trilha): prototipo
#   hachura 0,749 -> 0,996, cursiva 0,886 -> 0,997, CAD
#   0,943 -> 1,000. A parte coberta segue ignorada.
# Portoes e ressalvas assumidas (F107, re-registro do Lucas de
# 2026-11-12): o crescimento no simulador e julgado como
# "<= default em recall equalizado, por cenario" (nao na classe
# da E3); 1 px segue teto de modelo (forte 0,641 no prototipo;
# medio com teto 0,35); eventos crescem um pouco sobre a E3
# (3,73 x 3,34 no prototipo — copia deslocada e indistinguivel
# de conteudo novo para a regra local; e problema de registro).
# Coexistencia com a E3: as duas flags podem estar ligadas; a
# PRECEDENCIA DECLARADA e E4 > E3 — com as duas ligadas roda so
# a E5-S (a telemetria e a da E4). Como a E3, a flag e LIDA UMA
# VEZ por geracao; com ela desligada o acumulador roda a regra
# de sempre sem tocar em nenhum estado da E4.
_INK_VOTE_E4_ENV = "DRAW_INK_VOTE_E4"
_INK_VOTE_E4_OPENING_EVIDENCE = 0.35
_INK_VOTE_E4_MIN_COMPONENT_PX = 20
_INK_VOTE_E4_BIG_COMPONENT_PX = 100
_INK_VOTE_E4_BIG_WINDOW = 3
_INK_VOTE_E4_BIG_SCORE = 2.1
_INK_VOTE_E4_SMALL_WINDOW = 5
_INK_VOTE_E4_SMALL_SCORE = 3.5
_INK_VOTE_E4_SLOW_WINDOW = 5
_INK_VOTE_E4_SLOW_SCORE = 3.5
_INK_VOTE_E4_THIN_WINDOW = 3
_INK_VOTE_E4_THIN_SCORE = 2.1
_INK_VOTE_E4_THIN_MIN_LENGTH = 50
_INK_VOTE_E4_THIN_MAX_AREA_RATIO = 3.5
_INK_VOTE_E4_FAR_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
_INK_VOTE_E4_TOLERANCE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _ink_vote_e4_enabled():
    value = os.environ.get(_INK_VOTE_E4_ENV, "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


# Cadencia ativa por atividade (frente 92): com a flag ligada, a
# sonda ociosa de _IDLE_PROBE_SECONDS nao governa enquanto a
# saida do acumulador esta mudando — ver _PageCleanWorker.
_INK_ACTIVE_CADENCE_ENV = "DRAW_INK_ACTIVE_CADENCE"


def _ink_active_cadence_enabled():
    value = os.environ.get(_INK_ACTIVE_CADENCE_ENV, "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


# Corte de latencia voto->canvas (frente 94, 2026-11-12;
# draw-lab/live-audit-2026-11-10/94-latencia-voto-canvas.md).
# A medicao da F94 separou o que as F89/F90 mediam fundido: a
# tinta FORTE ja fica VISIVEL na 1a passada nos dois bracos (a
# provisoria pinta o que a janela viu acima do limiar), mas so
# CONFIRMA na 2a (regra atual) ou na 3a/5a (E3: nucleo/trilha
# grande/trilha pequena); a tinta FRACA canonica (0,55, o lapis
# do comentario do voto) NAO fica visivel antes de confirmar —
# a provisoria so mostra o que passou do limiar de tinta (0,7),
# e a evidencia de hit (>= _INK_VOTE_HIT) nao pintava nada: na
# regra atual ela surge de supetao na 3a passada, ja confirmada;
# na E3 nunca surge (componente <= limiar nao abre trilha nem
# entra no nucleo).
# O corte que a F94 ENTREGA, atras de DRAW_INK_VOTE_FAST
# (padrao 0, lida uma vez por geracao junto com a da E3), e so
# na camada provisoria (exibicao): pixel com
# >= _INK_VOTE_FAST_PROV_MIN_HITS hits na janela e soma
# >= _INK_VOTE_FAST_PROV_MIN_SCORE entra na provisoria antes de
# confirmar — a tinta fraca fica visivel na 2a passada nos dois
# bracos, em vez de surgir confirmada na 3a (regra atual) ou
# nunca (E3). A confirmada NAO muda: quem consome canvas
# (poll) nao ve pixel nenhum a mais, e as reguas de crescimento/
# contencao/completude ficam byte-identicas nos dois bracos; a
# provisoria que nao confirma segue retratando ao sair da
# janela, e a contagem de retratados mede o custo.
# O corte de CONFIRMACAO (regra forte de 2 passadas — soma
# >= 1,9, a paridade com _INK_VOTE_SCORE_STRONG — nas trilhas
# e no nucleo da fase 1 da E3) foi implementado e MEDIDO nesta
# frente e REPROVADO na regua: crescimento 1,216-1,554x contra
# o portao de 1,15x em todas as variantes (so trilhas, so
# nucleo, so trilha grande, soma 1,96, com consenso espacial —
# a condicao de vizinhanca e vazia na franja, que sempre tem
# tinta a <= 3 px). O tremor entre passadas vizinhas e
# correlacionado: 2 passadas fortes consecutivas no mesmo
# pixel acontecem na franja do flicker, nao so no traco novo.
# A latencia de confirmacao da E3 esta no piso da evidencia
# que os portoes permitem; nao ha corte de confirmacao nesta
# camada sem devolver o crescimento que a E3 eliminou.
_INK_VOTE_FAST_ENV = "DRAW_INK_VOTE_FAST"
_INK_VOTE_FAST_PROV_MIN_HITS = 2
_INK_VOTE_FAST_PROV_MIN_SCORE = 1.0


def _ink_vote_fast_enabled():
    value = os.environ.get(_INK_VOTE_FAST_ENV, "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _e3_phase1_core(maps, threshold):
    """Nucleo de consenso espacial das primeiras passadas (F89).

    Confirma o pixel que tem tinta (prob > limiar) a
    <= _INK_FRONTIER_RADIUS px em TODAS as passadas da fase 1, foi
    visto acima do limiar ao menos uma vez no proprio pixel e cuja
    soma de probabilidades na fase chega a _INK_VOTE_E3_CORE_MIN_SUM.
    """
    total = maps[0].copy()
    for item in maps[1:]:
        np.add(total, item, out=total)
    hits = np.zeros(maps[0].shape, dtype=np.int32)
    neighbours = np.zeros(maps[0].shape, dtype=np.int32)
    for item in maps:
        ink = item > threshold
        np.add(hits, ink, out=hits)
        np.add(
            neighbours,
            cv2.dilate(ink.astype(np.uint8), _INK_VOTE_E3_DILATE_KERNEL) > 0,
            out=neighbours,
        )
    return (
        (neighbours >= len(maps)) & (hits >= 1) & (total >= _INK_VOTE_E3_CORE_MIN_SUM)
    )


class _E3SideVote:
    """Voto estrito de uma trilha lateral da E3 (bancada da F89).

    Janela deslizante das ultimas `window` passadas ALIMENTADAS (a
    trilha so e alimentada nas passadas em que a regiao dela
    existe); o pixel confirma quando foi visto acima do limiar em
    TODAS as passadas da janela cheia e a soma chega a min_score —
    persistencia consecutiva, sem a janela sobreposta da regra
    principal. A porta de fronteira e a mesma da regra principal
    (kernel quadrado de _INK_FRONTIER_RADIUS sobre visto u
    confirmado da trilha): dentro do ombro, so confirma quem a
    janela viu acima do limiar.
    """

    def __init__(self, window, min_score, threshold):
        self._window_len = int(window)
        self._min_score = float(min_score)
        self._threshold = float(threshold)
        self.reset()

    def reset(self):
        self._maps = deque(maxlen=self._window_len)
        self._confirmed = None

    def migrate(self, matrix, size):
        if self._confirmed is None:
            return
        self._confirmed = (
            cv2.warpPerspective(
                self._confirmed.astype(np.uint8),
                matrix,
                size,
                flags=cv2.INTER_NEAREST,
                borderValue=0,
            )
            > 0
        )
        self._maps = deque(
            (
                cv2.warpPerspective(
                    item, matrix, size, flags=cv2.INTER_LINEAR, borderValue=0.0
                )
                for item in self._maps
            ),
            maxlen=self._window_len,
        )

    def add(self, prob):
        if self._confirmed is None or self._confirmed.shape != prob.shape:
            self._confirmed = np.zeros(prob.shape, dtype=bool)
            self._maps.clear()
        self._maps.append(prob.copy())
        maps = list(self._maps)
        score = maps[0].copy()
        for item in maps[1:]:
            np.add(score, item, out=score)
        hits = (maps[0] > self._threshold).astype(np.int32)
        for item in maps[1:]:
            np.add(hits, item > self._threshold, out=hits)
        seen = hits > 0
        rule = (hits >= self._window_len) & (score >= self._min_score)
        if np.any(rule):
            anchor = seen | self._confirmed
            near_anchor = cv2.dilate(anchor.astype(np.uint8), _INK_FRONTIER_KERNEL) > 0
            rule &= seen | ~near_anchor
        self._confirmed |= rule
        return self._confirmed


class _E4ThinSideVote:
    """Voto da trilha fina da E5-S: estrito com tolerancia de 1 px.

    Porto do SideTol dos prototipos das F106/F107: a linha de
    1 px tremula +-1 px entre passadas e o hit no mesmo pixel em
    toda a janela nao fecha. Aqui o hit conta na vizinhanca r1
    (vizinhanca >= janela), o pixel precisa de >= 1 hit proprio
    e a soma da janela sobre os mapas dilatados r1 fecha em
    min_score. A porta de fronteira e a mesma das outras
    trilhas. Interface identica a do _E3SideVote (reset/migrate/
    add; _confirmed/_maps com os mesmos nomes) para o caminho de
    remocao (DRAW_INK_REMOVAL) limpar as duas familias igual.
    """

    def __init__(self, window, min_score, threshold):
        self._window_len = int(window)
        self._min_score = float(min_score)
        self._threshold = float(threshold)
        self.reset()

    def reset(self):
        self._maps = deque(maxlen=self._window_len)
        self._confirmed = None

    def migrate(self, matrix, size):
        if self._confirmed is None:
            return
        self._confirmed = (
            cv2.warpPerspective(
                self._confirmed.astype(np.uint8),
                matrix,
                size,
                flags=cv2.INTER_NEAREST,
                borderValue=0,
            )
            > 0
        )
        self._maps = deque(
            (
                cv2.warpPerspective(
                    item, matrix, size, flags=cv2.INTER_LINEAR, borderValue=0.0
                )
                for item in self._maps
            ),
            maxlen=self._window_len,
        )

    def add(self, prob):
        if self._confirmed is None or self._confirmed.shape != prob.shape:
            self._confirmed = np.zeros(prob.shape, dtype=bool)
            self._maps.clear()
        self._maps.append(prob.copy())
        maps = list(self._maps)
        if len(maps) < self._window_len:
            return self._confirmed
        score = None
        hits = None
        neighbours = None
        seen = None
        for item in maps:
            dilated = cv2.dilate(item, _INK_VOTE_E4_TOLERANCE_KERNEL)
            score = dilated.copy() if score is None else score + dilated
            hit = (item > self._threshold).astype(np.int32)
            hits = hit if hits is None else hits + hit
            near_hit = (
                cv2.dilate(
                    (item > self._threshold).astype(np.uint8),
                    _INK_VOTE_E4_TOLERANCE_KERNEL,
                )
                > 0
            ).astype(np.int32)
            neighbours = near_hit if neighbours is None else neighbours + near_hit
            seen = (hit > 0) if seen is None else (seen | (hit > 0))
        rule = (
            (neighbours >= self._window_len) & (hits >= 1) & (score >= self._min_score)
        )
        if np.any(rule):
            anchor = seen | self._confirmed
            near_anchor = cv2.dilate(anchor.astype(np.uint8), _INK_FRONTIER_KERNEL) > 0
            rule &= seen | ~near_anchor
        self._confirmed |= rule
        return self._confirmed


# Registro da evidencia na geometria ancora da geracao (medido
# na regua desta frente, 2026-11-05, draw-lab/parallel-2026-11-05/
# repassada): o acumulador vive numa grade fixa da pagina, mas a
# geometria que retifica cada passada anda — a cadeia rastreador
# + mediana do app sustenta deslocamentos de varios px por
# dezenas de passadas quando um vies persistente (a mao sobre um
# canto) puxa o quad, e ate o tremor de +-1 px entre passadas
# basta. Sem registro, a evidencia entra na grade na posicao em
# que AQUELE quad a projetou: a janela do voto e de 0,75 s e nao
# distingue "tinta nova" de "a mesma tinta re-projetada 2 px ao
# lado" — cada alinhamento confirma a sua copia e a confirmada
# degenera para a uniao de todas as alinhagens, a tinta assentada
# reimpressa camada sobre camada. Na regua (pagina real assentada
# do cache de 2026-11-01, 300 passadas, nenhum conteudo novo):
# geometria congelada confirma 20.137 px e para; tremor de +-1 px
# confirma 45.528 px; deriva lenta de 3 px, 42.930 px; vies de
# canto sustentado, 71.548 px — contra unioes de 47.472/42.711/
# 76.302 px das mascaras das alinhagens: o voto, sob geometria
# movel, e a uniao antiga com outro nome.
# O registro: a primeira passada da geracao fixa a geometria
# ancora; o mapa de cada passada seguinte e re-projetado para a
# grade da ancora (M = A @ C^-1, homografias camera->pagina da
# ancora e da passada) antes do voto. Mover o caderno vira so
# re-registro: o conteudo confirmado nao anda nem duplica
# (registrar uma vez, adaptar por substituicao). O custo e um
# warpPerspective por passada sobre o mapa float, ~ms contra os
# ~168 ms da passada do modelo; geometria igual a da ancora (o
# caso assentado, com o quad congelado) pula o warp.
_REGISTRATION_IDENTITY_TOL = 1e-4


# Ao retomar apos um drop do tracker, a primeira mascara e registrada
# na grade antiga e precisa reencontrar a maior parte da tinta existente.
# Evidencia nova pode acrescentar tracos, mas uma pagina diferente nao
# deve herdar o canvas anterior.
_PAGE_IDENTITY_MIN_RECALL = 0.65
# Guarda dual da identidade: alem do recall da tinta antiga, a
# tinta da passada tem de cair sobre o canvas anterior (mesma
# tolerancia dilatada). O limiar e menor que o do recall porque a
# tinta nova desenhada durante o drop e legitima: medido na regua
# (2026-11-06), mesma pagina com +75% de tinta nova da
# containment 0,65 e perderia o canvas; com 0,5 aceita ~1,8x de
# crescimento. Pagina diferente mede 0,23-0,25 (sintetica), margem
# de 2x contra falso positivo.
_PAGE_IDENTITY_MIN_CONTAINMENT = 0.5
# Maximo de passadas adiadas por recuperacao, somando ECC e oclusao.
_PAGE_IDENTITY_MAX_DEFERRALS = 5
# Troca de pagina exige esta janela de vereditos limpos consecutivos.
_PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES = 3
# O canvas confirmado inclui o halo da fronteira, que uma passada
# sozinha nunca cobre no limiar do voto: a comparacao dilata os
# dois lados pela mesma tolerancia do comparador da regua.
_IDENTITY_DILATION_KERNEL = 2 * _INK_FRONTIER_RADIUS + 1

_ECC_REFINE_ENV = "DRAW_ECC_REFINE"


def _ecc_refinement_enabled():
    value = os.environ.get(_ECC_REFINE_ENV, "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


_IDENTITY_CHAMFER_ENV = "DRAW_IDENTITY_CHAMFER"
_IDENTITY_CHAMFER_RADII = (2, 4, 8, 16)


def _identity_chamfer_enabled():
    value = os.environ.get(_IDENTITY_CHAMFER_ENV, "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _identity_chamfer_fields(seen, reference, decision, ecc_info):
    """Diagnostico apenas: distancias euclidianas na grade da ancora."""
    fields = {
        "ecc_refine_enabled": int(_ecc_refinement_enabled()),
        "shadow_decision": "defer" if ecc_info.get("ecc_rejected") else decision,
    }
    for direction, source, target in (
        ("seen_to_reference", seen, reference),
        ("reference_to_seen", reference, seen),
    ):
        distances = None
        if (
            source is not None
            and target is not None
            and np.any(source)
            and np.any(target)
        ):
            distance_map = cv2.distanceTransform(
                (~target).astype(np.uint8),
                cv2.DIST_L2,
                cv2.DIST_MASK_PRECISE,
            )
            distances = distance_map[source]
        prefix = f"chamfer_{direction}"
        fields[f"{prefix}_median_px"] = (
            None if distances is None else float(np.median(distances))
        )
        for radius in _IDENTITY_CHAMFER_RADII:
            fields[f"{prefix}_coverage_{radius}px"] = (
                float(np.mean(distances <= radius))
                if distances is not None
                else 0.0
                if (source is not None and np.any(source) and target is not None)
                else None
            )
    return fields


# Oclusao na decisao de identidade: o recall so julga a tinta
# anterior que a passada consegue VER (a mao sobre a pagina nao e
# evidencia de pagina nova). Abaixo deste minimo de tinta anterior
# visivel, a decisao e ADIADA para a proxima passada em vez de
# sair no chute com a pagina coberta.
_IDENTITY_MIN_VISIBLE_FRACTION = 0.25
_IDENTITY_MIN_VISIBLE_PX = 32

# Julgamento de virada sob oclusao: a cobertura e medida sobre a
# tinta confirmada VISIVEL; com menos que isto de tinta visivel, a
# passada se abstem — nem avanca nem zera a sequencia de virada.
_OCCLUSION_TURN_MIN_VISIBLE_PX = 200

# Vigia de identidade no meio da geracao (frente 28, 2026-11-10):
# a fronteira de geracao existente so julga a identidade na VOLTA
# da folha (drop -> new_generation(preserve_identity=True)). Com a
# folha sempre presente, virar a pagina somava as duas no mesmo
# canvas — medido ao vivo: decisao de identidade None em 53/53
# passadas, 87% do canvas confirmado depois da troca. A vigia NAO
# decide nada: ela ARMA o juiz de identidade existente no meio da
# geracao, com a referencia = o canvas confirmado ate aqui. Quem
# decide e o juiz (recall/containment dilatados, consenso de 3,
# adiamento por oclusao/ECC, quarentena) — o gatilho cru de
# cobertura do U3 foi reprovado ao vivo em 2026-11-01 justamente
# por decidir sozinho sob balanco de geometria; aqui o balanco
# nem arma (a vigia exige geometria quieta) e quem arma ainda
# passa pelo juiz. Dois chamadores: (a) salto grande de pose
# confirmado pelo tracker, sinalizado pelo app via
# arm_identity_check("pose_jump"); (b) ruptura de conteudo: o
# recall dilatado do canvas na passada registrada <=
# _IDENTITY_WATCH_MAX_RECALL por _IDENTITY_WATCH_PASSES passadas
# seguidas com massa de tinta nova >= _IDENTITY_WATCH_MIN_SEEN_PX
# e deslocamento da ancora <= _IDENTITY_WATCH_MAX_DISPLACEMENT_PX.
# Virada para pagina em branco nao arma (sem massa nova nao ha o
# que somar; quando a escrita comeca na pagina nova, a vigia
# dispara). Atras de DRAW_IDENTITY_WATCH=1, desligado por padrao.
_ENV_IDENTITY_WATCH_VAR = "DRAW_IDENTITY_WATCH"
_IDENTITY_WATCH_MIN_CANVAS_PX = 2000
_IDENTITY_WATCH_MIN_SEEN_PX = 200
_IDENTITY_WATCH_MAX_RECALL = 0.30
_IDENTITY_WATCH_PASSES = 2
# Portoes de geometria da vigia — v3 (frente 32, 2026-11-10): a v1
# exigia deslocamento da ancora <= 8 px e isso (i) bloqueou as
# passadas da virada real (9,53 px, a perturbacao da propria
# virada) e (ii) cegava a vigia sob deriva. A v3 conta a passada
# quando a geometria esta ASSENTADA (degrau entre quads
# consecutivos <= _IDENTITY_WATCH_MAX_STEP_PX, a definicao da
# arbitragem) E o deslocamento acumulado da ancora segue <=
# _IDENTITY_WATCH_MAX_DISPLACEMENT_PX (trava de deriva lenta: a
# era de posicionamento que se condenou sozinha estava a >= 25 px;
# a virada precisa de >= 9,53 — margens de ~1,7x dos dois lados,
# amarradas ao material gravado). E o armamento por ruptura
# SEMEIA 1 voto de troca no juiz: as 2 passadas da vigia medem
# recall <= 0,30 — evidencia mais estrita em recall que 1 voto
# do juiz (< 0,65); o juiz segue decidindo com o consenso dele.
_IDENTITY_WATCH_MAX_DISPLACEMENT_PX = 16.0
_IDENTITY_WATCH_MAX_STEP_PX = 6.0
_IDENTITY_WATCH_SEED_VOTES = 1


def _identity_watch_enabled():
    """True so com DRAW_IDENTITY_WATCH=1 (ver o desenho acima)."""
    return os.environ.get(_ENV_IDENTITY_WATCH_VAR, "").strip() == "1"


def _corners_page_matrix(corners, width, height):
    """Homografia camera->pagina de um quad, na grade do mapa.

    Mesma convencao de draw.paper.perspective_matrix (os cantos
    na ordem do rastreador, casando posicionalmente com o destino
    (0,0), (W,0), (W,H), (0,H)), com o destino no tamanho do
    mapa da passada: e nessa grade que ancora e passada
    precisam conversar.
    """
    destination = np.array(
        [(0, 0), (width, 0), (width, height), (0, height)],
        dtype=np.float32,
    )
    source = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return cv2.getPerspectiveTransform(source, destination)


def _permute_prob(prob, k):
    """A prob como o warp posicional a entrega sob rotacao de k x 90
    graus: rot90 + resize de volta ao canonico nos k impares (a
    forma medida na frente 23 — NAO e o registro com cantos
    rolados, que a frente 24 provou nao reconstruir a evidencia
    entregue: escores 0,005 contra 0,23-0,27 da permutacao)."""
    k %= 4
    if k == 0:
        return prob
    if k == 2:
        return np.rot90(prob, 2).copy()
    rotated = np.rot90(prob, 1 if k == 1 else -1)
    return cv2.resize(
        rotated,
        (prob.shape[1], prob.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )


def _registration_matrix(anchor_corners, corners, shape):
    """M que leva o mapa desta passada para a grade da ancora.

    Um ponto fisico da pagina tem UMA coordenada na ancora:
    p_ancora = A @ C^-1 @ p_passada, com A e C as homografias
    camera->pagina da ancora e desta passada.
    """
    height, width = shape
    anchor = _corners_page_matrix(anchor_corners, width, height)
    current = _corners_page_matrix(corners, width, height)
    return anchor @ np.linalg.inv(current)


def _registration_is_identity(matrix):
    """True se a matriz nao move pixel nenhum (tol sub-pixel)."""
    return bool(
        np.max(np.abs(matrix - np.eye(3, dtype=np.float64)))
        < _REGISTRATION_IDENTITY_TOL
    )


def _experimental_enabled(name):
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


# Protocolo inicial F-10: ausência forte, visível e persistente, com
# testemunho positivo da mesma página. Não confundir apagar tudo com virada.
_REMOVAL_LOW_PROB = 0.1
_REMOVAL_PASSES = 6
_REMOVAL_RETAINED_FRACTION = 0.65
# Piso absoluto de testemunho: a fração de 65% veda a remoção quando
# mais de ~35% do canvas confirmado está velho (cópia defasada após
# migração de âncora, contorno de mão confirmado) — os próprios pixels
# errados impedem a testemunha de existir, trava de catch-22 medida no
# sim-hand (retained 0,33%, canvas congelado com folha limpa à vista).
# Um modelo que re-testemunha >= piso px do canvas está comprovadamente
# vendo tinta: a evidência negativa dele é real. O caso que a fração
# protege (saída do modelo escurecendo por inteiro) continua vedado:
# modelo escuro não retém piso nenhum. Os pixels fora de vista seguem
# imunes pela máscara `visible` dentro de `missing`.
_REMOVAL_WITNESS_FLOOR_PX = 4000


# Disciplina de registro (M1, auditoria de 2026-11-10, draw-lab/
# live-audit-2026-11-10/, provas P2/P3a/P3b): o voto de tinta e
# destruido pelo jitter do quad de registro, e o jitter real e
# evento discreto (flip de modo do detector, pouso de salto), nao
# ruido. Duas correcoes aprovadas em replay, ambas atras de flag e
# so sobre o quad que alimenta _registration_matrix (warp do app,
# rastreador e exibicao intocados):
# - DRAW_REG_ANCHOR_MEDIAN3: a ancora da geracao e a mediana por
#   componente dos cantos das _REG_ANCHOR_MEDIAN_SAMPLES primeiras
#   passadas com quad valido (mediana das que houver antes disso),
#   nao a 1a amostra — um quad espurio na passada 0 nao projeta a
#   pagina inteira no quadro errado.
# - DRAW_REG_DEADBAND: o registro usa um quad congelado por geracao
#   enquanto o deslocamento maximo de canto contra ele for
#   < _REG_DEADBAND_TAU_PX; >= tau por _REG_DEADBAND_RELEASE_PASSES
#   passadas seguidas libera (o congelado salta para o quad da
#   passada). Spike de 1 passada nunca libera. tau precisa exceder o
#   diametro do jitter assentado: tau=3 foi medido e reprovado
#   (libera no ruido). EMA/One Euro foram reprovados na P3b (a tinta
#   fica para tras quando a pagina anda de verdade).
_REG_ANCHOR_MEDIAN_ENV = "DRAW_REG_ANCHOR_MEDIAN3"
_REG_ANCHOR_MEDIAN_SAMPLES = 3
_REG_DEADBAND_ENV = "DRAW_REG_DEADBAND"
_REG_DEADBAND_TAU_PX = 10
_REG_DEADBAND_RELEASE_PASSES = 2


def _reg_anchor_median_enabled():
    value = os.environ.get(_REG_ANCHOR_MEDIAN_ENV, "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _reg_deadband_enabled():
    value = os.environ.get(_REG_DEADBAND_ENV, "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


# Migracao de ancora intra-geracao (decisao A do Lucas, 2026-11-10;
# draw-lab/live-audit-2026-11-10/12-estudo-pouso.md): quando a folha
# se move de verdade e assenta em pose nova no meio da geracao (pouso
# sustentado), o registro na ancora velha projeta a tinta nova como
# segunda copia. Atras de DRAW_ANCHOR_MIGRATION (padrao 0): passadas
# seguidas com deslocamento medio da grade sob a matriz de registro >
# _ANCHOR_MIGRATION_DISPLACEMENT_PX contam; na
# _ANCHOR_MIGRATION_PASSES-esima, todo o estado do acumulador e
# re-projetado para a grade nova (substituicao, nunca acumulo) e a
# ancora passa a ser a mediana dos quads do streak. O tremor do
# regime de registro (<= ~3 px) fica abaixo do limiar e spike de 1
# passada nao sustenta.
_ANCHOR_MIGRATION_ENV = "DRAW_ANCHOR_MIGRATION"
_ANCHOR_MIGRATION_DISPLACEMENT_PX = 8.0
_ANCHOR_MIGRATION_PASSES = 3
_ANCHOR_MIGRATION_GRID = 16


def _anchor_migration_enabled():
    value = os.environ.get(_ANCHOR_MIGRATION_ENV, "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


# Arbitragem de ancora por conteudo (v3; frente 19, 2026-11-10;
# draw-lab/live-audit-2026-11-10/19-migracao-v2-arbitragem.md): a
# migracao da v1 foi reprovada na propria prova (FALHAS M2) porque a
# inversa do registro assume a tinta parada na camera — certo pro
# modo espurio do detector, errado pro pouso, em que a tinta anda
# com a folha. As duas hipoteses sao indistinguiveis pelo quad; a
# v3 arbitra pelo CONTEUDO: num streak de assentamento (passadas
# deslocadas > _ANCHOR_MIGRATION_DISPLACEMENT_PX, mutuamente
# consistentes a _ARB_SETTLE_PX, K = _ANCHOR_MIGRATION_PASSES),
# compara a fracao da massa da prob que cai sobre o canvas
# confirmado pre-evento na colocacao identidade (Sp, hipotese
# pagina/pouso) e na registrada (Sc, hipotese camera/espurio).
# POUSO troca a ancora SEM mover o estado; ESPURIO nao faz nada.
# Enquanto o evento esta aberto, a passada NAO vota (visible=0): a
# copia deslocada confirmava pelo voto antes de qualquer veredito
# (mecanismo que reprovou a v2 sem abstencao). Atras de
# DRAW_ANCHOR_ARBITRATION (padrao 0); a flag da v1 segue vetada.
_ANCHOR_ARBITRATION_ENV = "DRAW_ANCHOR_ARBITRATION"
_ARB_SETTLE_PX = 6.0
_ARB_DELTA = 0.05
_ARB_MIN_CANVAS_PX = 200
_ARB_MIN_PASS_PX = 200


def _anchor_arbitration_enabled():
    value = os.environ.get(_ANCHOR_ARBITRATION_ENV, "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


# Mobilidade da ancora por divergencia assentada (frente 34,
# 2026-11-10; draw-lab/live-audit-2026-11-10/34-desenho-ancora-
# movel.md): a vigia v3 fechou o split errado do puxao e, ao
# fechar, expos a pergunta que nenhum mecanismo respondia — quem
# move a ancora quando ninguem divide a pagina? Sem resposta, a
# ancora congela na pose de abertura, a vigia de identidade fica
# cega a ~150 px pelo resto da geracao (F33, G1/G3) e a
# arbitragem cicla sem veredito com o canvas congelado (o
# livelock da F33). O FOLLOW e a regra do Lucas (2026-11-01) como
# mecanismo: mover o caderno = so re-registro por substituicao.
# Divergencia assentada sustentada (deslocamento da grade >
# _FOLLOW_DISPLACEMENT_PX, degrau entre quads consecutivos <=
# _FOLLOW_SETTLE_PX por _FOLLOW_PASSES passadas) sem decisao de
# identidade pendente RE-ANCORA: ancora := mediana dos quads do
# streak, estado do acumulador INTOCADO — a grade ja e a da
# folha, e o carry que deforma o estado pela homografia do raio
# da camera e o da migracao v1, reprovado no FALHAS M2. Enquanto
# o evento esta aberto a passada nao vota (a copia deslocada e o
# defeito que a abstencao da arbitragem ja fechou). Precedencia:
# a arbitragem tem o primeiro veredito de cada passada — veredito
# decisivo com acao pre-empta; spurious retem o FOLLOW pelo
# episodio inteiro; indeterminate ou ausencia de veredito
# (creep: a consistencia par-a-par nunca fecha) deixam a
# geometria decidir. Atras de DRAW_ANCHOR_FOLLOW (padrao 0);
# com ele ligado, o passo da migracao v1 nao roda.
_ANCHOR_FOLLOW_ENV = "DRAW_ANCHOR_FOLLOW"
_FOLLOW_DISPLACEMENT_PX = _ANCHOR_MIGRATION_DISPLACEMENT_PX
_FOLLOW_PASSES = _ANCHOR_MIGRATION_PASSES
_FOLLOW_SETTLE_PX = 6.0


def _anchor_follow_enabled():
    value = os.environ.get(_ANCHOR_FOLLOW_ENV, "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _matrix_grid_displacement(matrix, shape):
    """Deslocamento medio (px) de uma grade densa da pagina sob matrix."""
    height, width = shape
    xs = np.linspace(0.0, float(width), _ANCHOR_MIGRATION_GRID)
    ys = np.linspace(0.0, float(height), _ANCHOR_MIGRATION_GRID)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(grid, np.asarray(matrix, dtype=np.float64))
    return float(np.mean(np.linalg.norm(moved - grid, axis=-1)))


def _median_anchor(samples):
    """Mediana por componente de uma lista de quads (4, 2)."""
    return np.median(np.stack(samples), axis=0).astype(np.float32)


def _deadband_step(frozen, corners, over):
    """Um passo da banda morta: (quad de registro, congelado, over, liberou, desvio).

    over conta as passadas seguidas com desvio >= tau contra o
    congelado; a liberacao so ocorre na _REG_DEADBAND_RELEASE_PASSES-esima.
    """
    shift = float(np.max(np.linalg.norm(corners - frozen, axis=1)))
    if shift < _REG_DEADBAND_TAU_PX:
        return frozen, frozen, 0, False, shift
    over += 1
    if over >= _REG_DEADBAND_RELEASE_PASSES:
        frozen = corners.copy()
        return frozen, frozen, 0, True, shift
    return frozen, frozen, over, False, shift


# Fonte fixada pelos hashes da pre-inscricao do replay v2.
# fmt: off
def _refine_ink_ecc(reference, observed, hand_mask=None):
    """Refina somente o residuo na grade da ancora; devolve mapa, H e custo."""
    started = time.perf_counter()
    identity = np.eye(3, dtype=np.float64)
    info = {
        "ecc_rejected": False,
        "ecc_converged": False,
        "ecc_mode": "bypass",
    }

    def finish(aligned, matrix):
        info["ecc_ms"] = (time.perf_counter() - started) * 1000.0
        return aligned, matrix, info

    if reference is None or reference.shape != observed.shape:
        return finish(observed, identity)
    ref = np.asarray(reference, dtype=np.float32)
    obs = np.asarray(observed, dtype=np.float32)
    if hand_mask is not None:
        ref = ref.copy()
        obs = obs.copy()
        ref[hand_mask > 0] = 0.0
        obs[hand_mask > 0] = 0.0
    if np.array_equal(ref, obs):
        return finish(obs, identity)
    if (not np.isfinite(ref).all() or not np.isfinite(obs).all()
            or np.count_nonzero(ref > 0.7) < 500
            or np.count_nonzero(obs > 0.7) < 500):
        return finish(obs, identity)
    height, width = obs.shape
    half_size = (width // 2, height // 2)
    if min(half_size) < 2:
        return finish(obs, identity)
    ref_half = cv2.resize(ref, half_size, interpolation=cv2.INTER_AREA)
    obs_half = cv2.resize(obs, half_size, interpolation=cv2.INTER_AREA)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    info["ecc_mode"] = "homography"
    try:
        _, estimated = cv2.findTransformECC(
            ref_half, obs_half, np.eye(3, dtype=np.float32),
            cv2.MOTION_HOMOGRAPHY, criteria, None, 5,
        )
        info["ecc_converged"] = True
    except cv2.error:
        info["ecc_mode"] = "affine"
        try:
            _, affine = cv2.findTransformECC(
                ref_half, obs_half, np.eye(2, 3, dtype=np.float32),
                cv2.MOTION_AFFINE, criteria, None, 5,
            )
            info["ecc_converged"] = True
            estimated = np.vstack((affine, [0.0, 0.0, 1.0]))
        except cv2.error:
            info["ecc_rejected"] = True
            return finish(obs, identity)
    scale = np.diag([half_size[0] / width, half_size[1] / height, 1.0])
    try:
        full = np.linalg.inv(scale) @ estimated @ scale
        align = np.linalg.inv(full)
        if not np.isfinite(align).all() or abs(align[2, 2]) < 1e-12:
            raise ValueError("homografia nao normalizavel")
        align /= align[2, 2]
        corners = np.array(
            [[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]],
            dtype=np.float64,
        )
        projected = corners @ align.T
        if (not np.isfinite(align).all() or np.linalg.det(align) <= 1e-6
                or np.any(projected[:, 2] <= 1e-6)):
            raise ValueError("homografia degenerada")
        projected = projected[:, :2] / projected[:, 2:]
        # A margem e medida na grade titular 850x1200, inclusive
        # quando ferramentas/testes usam uma pagina reduzida.
        displacement = np.linalg.norm(
            (projected - corners[:, :2]) * [850.0 / width, 1200.0 / height],
            axis=1,
        )
        if not np.isfinite(displacement).all() or displacement.max() > 55.0:
            raise ValueError("residuo fora da margem")
    except (np.linalg.LinAlgError, ValueError):
        info["ecc_rejected"] = True
        return finish(obs, identity)
    aligned = cv2.warpPerspective(
        obs, align, (width, height), flags=cv2.INTER_LINEAR,
        borderValue=0.0,
    )
    return finish(aligned, align)


def _register_ink_probability(prob, anchor, corners, reference, visible=None):
    """Registra na ancora; o refinamento ECC fica opt-in via ambiente."""
    info = {
        "ecc_ms": 0.0,
        "ecc_rejected": False,
        "ecc_converged": False,
        "ecc_mode": "bypass",
    }
    if visible is not None:
        prob = prob.copy()
        prob[~visible] = 0.0
    if anchor is None or corners is None:
        return prob, visible, info
    matrix = _registration_matrix(anchor, corners, prob.shape)
    if not _registration_is_identity(matrix):
        size = (prob.shape[1], prob.shape[0])
        prob = cv2.warpPerspective(
            prob, matrix, size, flags=cv2.INTER_LINEAR, borderValue=0.0,
        )
        if visible is not None:
            visible = cv2.warpPerspective(
                visible.astype(np.uint8), matrix, size,
                flags=cv2.INTER_NEAREST, borderValue=0,
            ) > 0
    hand_mask = None if visible is None else ~visible
    if not _ecc_refinement_enabled():
        info["ecc_mode"] = "disabled"
        return prob, visible, info
    prob, align, info = _refine_ink_ecc(reference, prob, hand_mask)
    if visible is not None and not _registration_is_identity(align):
        visible = cv2.warpPerspective(
            visible.astype(np.uint8), align, (prob.shape[1], prob.shape[0]),
            flags=cv2.INTER_NEAREST, borderValue=0,
        ) > 0
    return prob, visible, info

# fmt: on


class _LayeredInkAccumulator:
    """Tinta de uma geracao em duas camadas: provisoria e confirmada.

    Cada passada entrega o mapa de probabilidade do modelo. A
    camada provisoria e o que as passadas da janela viram acima do
    limiar e ainda nao confirmou: ela existe para a tela mostrar a
    tinta nova na primeira passada, e so para a tela. A camada
    confirmada e o canvas: um pixel so entra nela pelas regras de
    voto acima com a porta de fronteira (evidencia sub-limiar nao
    expande a borda da tinta ja confirmada), e dela nao sai
    dentro da geracao.

    Pixel provisorio que sai da janela sem confirmar e retratado:
    some da camada provisoria e entra na contagem de retratados.
    E essa contagem que mede o risco declarado da publicacao
    especulativa, o page-clean pegar mobiliario: a regua de replay
    le quanta tinta provisoria some depois.

    So a trabalhadora de tinta toca nesta classe.
    """

    def __init__(self, threshold=0.7):
        self.threshold = float(threshold)
        self.reset()

    def reset(self):
        self._window = deque(maxlen=_INK_VOTE_WINDOW)
        self._removal_streak = None
        self._confirmed = None
        self._provisional = None
        self._provisional_ever = None
        self._retracted_ever = None
        self._low_overlap_streak = 0
        self._turn_fired = False
        self._score_buf = None
        self._hits_buf = None
        self._seen_buf = None
        # Regra E3 (DRAW_INK_VOTE_E3): o modo da geracao e decidido
        # na primeira passada apos o reset (None = nao decidido) e
        # todo o estado da regra zera junto com o da geracao — uma
        # geracao nova nao herda nucleo, uniao velha nem trilhas.
        self._e3_active = None
        # Regra E5-S (DRAW_INK_VOTE_E4): idem — decidida na
        # primeira passada da geracao, com precedencia sobre a
        # E3 quando as duas flags estao ligadas (ver a constante
        # _INK_VOTE_E4_ENV).
        self._e4_active = None
        # Corte de latencia (DRAW_INK_VOTE_FAST): decidido junto
        # com o modo E3, na mesma primeira passada da geracao.
        self._fast_active = None
        self._reset_e3_state()
        self._reset_e4_state()

    def _reset_e3_state(self):
        self._e3_passes = 0
        self._e3_phase1_maps = []
        self._e3_old_union = None
        self._e3_region_big = None
        self._e3_region_small = None
        self._e3_side_big = _E3SideVote(
            _INK_VOTE_E3_BIG_WINDOW, _INK_VOTE_E3_BIG_SCORE, self.threshold
        )
        self._e3_side_small = _E3SideVote(
            _INK_VOTE_E3_SMALL_WINDOW, _INK_VOTE_E3_SMALL_SCORE, self.threshold
        )
        self._e3_tracks_big = 0
        self._e3_tracks_small = 0

    def _e3_step(self, evidence):
        """Uma passada da regra E3; devolve a mascara confirmada.

        Porto fiel da bancada vencedora da F89 (CommitE3): fase 1
        confirma o nucleo de consenso espacial ao fechar a 3a
        passada; fase 2 classifica cada componente conexo novo
        contra a uniao velha dilatada e as regioes ja abertas
        (fracoes medidas contra os instantaneos do inicio da
        passada, como na bancada) e alimenta as trilhas por
        tamanho com a evidencia mascarada pela regiao.
        """
        shape = evidence.shape
        if self._e3_old_union is None or self._e3_old_union.shape != shape:
            self._e3_old_union = np.zeros(shape, dtype=bool)
            self._e3_region_big = np.zeros(shape, dtype=bool)
            self._e3_region_small = np.zeros(shape, dtype=bool)
            self._e3_phase1_maps = []
            self._e3_passes = 0
            self._e3_side_big.reset()
            self._e3_side_small.reset()
            self._e3_tracks_big = 0
            self._e3_tracks_small = 0
        ink = evidence > self.threshold
        confirmed = self._confirmed > 0
        if self._e3_passes < _INK_VOTE_E3_PHASE1_PASSES:
            self._e3_phase1_maps.append(evidence.copy())
            self._e3_old_union |= ink
            if self._e3_passes == _INK_VOTE_E3_PHASE1_PASSES - 1:
                confirmed = confirmed | _e3_phase1_core(
                    self._e3_phase1_maps, self.threshold
                )
            self._e3_passes += 1
            return confirmed
        old_dilated = (
            cv2.dilate(self._e3_old_union.astype(np.uint8), _INK_VOTE_E3_DILATE_KERNEL)
            > 0
        )
        regions = self._e3_region_big | self._e3_region_small
        regions_dilated = (
            cv2.dilate(regions.astype(np.uint8), _INK_VOTE_E3_DILATE_KERNEL) > 0
        )
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            ink.astype(np.uint8), 8
        )
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < _INK_VOTE_E3_MIN_COMPONENT_PX:
                continue
            component = labels == index
            frac_old = float(np.count_nonzero(component & old_dilated) / area)
            frac_region = float(np.count_nonzero(component & regions_dilated) / area)
            if frac_old >= 0.5:
                self._e3_old_union |= component
            elif frac_region < 0.5:
                dilated = (
                    cv2.dilate(component.astype(np.uint8), _INK_VOTE_E3_REGION_KERNEL)
                    > 0
                )
                if area >= _INK_VOTE_E3_BIG_COMPONENT_PX:
                    self._e3_region_big |= dilated
                    self._e3_tracks_big += 1
                else:
                    self._e3_region_small |= dilated
                    self._e3_tracks_small += 1
        if np.any(self._e3_region_big):
            masked = np.where(self._e3_region_big, evidence, 0.0).astype(np.float32)
            confirmed = confirmed | self._e3_side_big.add(masked)
        if np.any(self._e3_region_small):
            masked = np.where(self._e3_region_small, evidence, 0.0).astype(np.float32)
            confirmed = confirmed | self._e3_side_small.add(masked)
        self._e3_passes += 1
        return confirmed

    def _reset_e4_state(self):
        self._e4_passes = 0
        self._e4_phase1_maps = []
        self._e4_opening = None
        self._e4_opening_prob = None
        self._e4_region_big = None
        self._e4_region_small = None
        self._e4_region_slow = None
        self._e4_region_thin = None
        self._e4_side_big = _E3SideVote(
            _INK_VOTE_E4_BIG_WINDOW, _INK_VOTE_E4_BIG_SCORE, self.threshold
        )
        self._e4_side_small = _E3SideVote(
            _INK_VOTE_E4_SMALL_WINDOW, _INK_VOTE_E4_SMALL_SCORE, self.threshold
        )
        self._e4_side_slow = _E3SideVote(
            _INK_VOTE_E4_SLOW_WINDOW, _INK_VOTE_E4_SLOW_SCORE, self.threshold
        )
        self._e4_side_thin = _E4ThinSideVote(
            _INK_VOTE_E4_THIN_WINDOW, _INK_VOTE_E4_THIN_SCORE, self.threshold
        )
        self._e4_tracks_big = 0
        self._e4_tracks_small = 0
        self._e4_tracks_slow = 0
        self._e4_tracks_thin = 0
        self._e4_tracks_sub = 0

    def _e4_open_track(self, component, area, width, height, far_mask, sub=False):
        """Abre a trilha de um componente (ou sub-componente) novo.

        Classificacao do prototipo da F107 (E5._open): a banda de
        distancia ao confirmado decide lenta x longe; longe, a
        forma de linha decide fina x por-tamanho. Sub-componente
        de nascimento abre com regiao r3 (a regiao r6 colaria a
        sub-trilha no conteudo coberto que a gerou).
        """
        frac_far = float(np.count_nonzero(component & far_mask) / area)
        dil3 = cv2.dilate(component.astype(np.uint8), _INK_VOTE_E3_DILATE_KERNEL) > 0
        if frac_far < 0.5:
            self._e4_region_slow |= dil3
            self._e4_tracks_slow += 1
        elif (
            max(width, height) >= _INK_VOTE_E4_THIN_MIN_LENGTH
            and area <= _INK_VOTE_E4_THIN_MAX_AREA_RATIO * max(width, height)
        ):
            self._e4_region_thin |= dil3
            self._e4_tracks_thin += 1
        elif area >= _INK_VOTE_E4_BIG_COMPONENT_PX:
            self._e4_region_big |= (
                dil3
                if sub
                else cv2.dilate(component.astype(np.uint8), _INK_VOTE_E3_REGION_KERNEL)
                > 0
            )
            self._e4_tracks_big += 1
        else:
            self._e4_region_small |= (
                dil3
                if sub
                else cv2.dilate(component.astype(np.uint8), _INK_VOTE_E3_REGION_KERNEL)
                > 0
            )
            self._e4_tracks_small += 1
        if sub:
            self._e4_tracks_sub += 1

    def _e4_step(self, evidence):
        """Uma passada da regra E5-S; devolve a mascara confirmada.

        Porto fiel do prototipo validado da F107 (frente107/
        f107_proto.py, classe E5 com use_subdiv=True,
        use_weak=False, use_thinweak=False — so a subdivisao por
        nascimento sobre a E4j): fase 1 = nucleo da E3 com a
        abertura (mascara + mapa de prob da 1a passada) congelada
        como unico "velho"; fase 2 classifica cada componente
        contra abertura, confirmado e regioes (instantaneos do
        inicio da passada, como no prototipo) e alimenta as
        quatro trilhas com a evidencia mascarada pela regiao.
        """
        shape = evidence.shape
        if self._e4_opening is None or self._e4_opening.shape != shape:
            self._e4_opening = np.zeros(shape, dtype=bool)
            self._e4_opening_prob = np.zeros(shape, dtype=np.float32)
            self._e4_region_big = np.zeros(shape, dtype=bool)
            self._e4_region_small = np.zeros(shape, dtype=bool)
            self._e4_region_slow = np.zeros(shape, dtype=bool)
            self._e4_region_thin = np.zeros(shape, dtype=bool)
            self._e4_phase1_maps = []
            self._e4_passes = 0
            self._e4_side_big.reset()
            self._e4_side_small.reset()
            self._e4_side_slow.reset()
            self._e4_side_thin.reset()
            self._e4_tracks_big = 0
            self._e4_tracks_small = 0
            self._e4_tracks_slow = 0
            self._e4_tracks_thin = 0
            self._e4_tracks_sub = 0
        ink = evidence > self.threshold
        confirmed = self._confirmed > 0
        if self._e4_passes == 0:
            self._e4_opening = ink.copy()
            self._e4_opening_prob = evidence.copy()
        if self._e4_passes < _INK_VOTE_E3_PHASE1_PASSES:
            self._e4_phase1_maps.append(evidence.copy())
            if self._e4_passes == _INK_VOTE_E3_PHASE1_PASSES - 1:
                confirmed = confirmed | _e3_phase1_core(
                    self._e4_phase1_maps, self.threshold
                )
            self._e4_passes += 1
            return confirmed
        confirmed_d3 = (
            cv2.dilate(confirmed.astype(np.uint8), _INK_VOTE_E3_DILATE_KERNEL) > 0
        )
        confirmed_d8 = (
            cv2.dilate(confirmed.astype(np.uint8), _INK_VOTE_E4_FAR_KERNEL) > 0
        )
        far_mask = ~confirmed_d8
        opening_d3 = (
            cv2.dilate(self._e4_opening.astype(np.uint8), _INK_VOTE_E3_DILATE_KERNEL)
            > 0
        )
        opening_evidence = self._e4_opening_prob >= _INK_VOTE_E4_OPENING_EVIDENCE
        regions = (
            self._e4_region_big
            | self._e4_region_small
            | self._e4_region_slow
            | self._e4_region_thin
        )
        regions_dilated = (
            cv2.dilate(regions.astype(np.uint8), _INK_VOTE_E3_DILATE_KERNEL) > 0
        )
        covered = confirmed_d3 | regions_dilated
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            ink.astype(np.uint8), 8
        )
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < _INK_VOTE_E4_MIN_COMPONENT_PX:
                continue
            component = labels == index
            if (
                float(np.count_nonzero(component & opening_d3) / area) >= 0.5
                or float(np.count_nonzero(component & opening_evidence) / area) >= 0.5
            ):
                # Pre-existente na abertura: ignorado NESTA
                # passada, sem virar estado — nada absorve pra
                # sempre (a diferenca central contra a E3).
                continue
            frac_covered = float(np.count_nonzero(component & covered) / area)
            if frac_covered < 0.5:
                self._e4_open_track(
                    component,
                    area,
                    int(stats[index, cv2.CC_STAT_WIDTH]),
                    int(stats[index, cv2.CC_STAT_HEIGHT]),
                    far_mask,
                )
                continue
            # Subdivisao por nascimento: o residuo fora da
            # cobertura e re-componentizado e cada pedaco abre a
            # sua trilha; a parte coberta segue ignorada.
            residual = component & ~covered
            if np.count_nonzero(residual) < _INK_VOTE_E4_MIN_COMPONENT_PX:
                continue
            sub_count, sub_labels, sub_stats, _sub = cv2.connectedComponentsWithStats(
                residual.astype(np.uint8), 8
            )
            for sub_index in range(1, sub_count):
                sub_area = int(sub_stats[sub_index, cv2.CC_STAT_AREA])
                if sub_area < _INK_VOTE_E4_MIN_COMPONENT_PX:
                    continue
                self._e4_open_track(
                    sub_labels == sub_index,
                    sub_area,
                    int(sub_stats[sub_index, cv2.CC_STAT_WIDTH]),
                    int(sub_stats[sub_index, cv2.CC_STAT_HEIGHT]),
                    far_mask,
                    sub=True,
                )
        for region, side in (
            (self._e4_region_big, self._e4_side_big),
            (self._e4_region_small, self._e4_side_small),
            (self._e4_region_slow, self._e4_side_slow),
            (self._e4_region_thin, self._e4_side_thin),
        ):
            if np.any(region):
                masked = np.where(region, evidence, 0.0).astype(np.float32)
                confirmed = confirmed | side.add(masked)
        self._e4_passes += 1
        return confirmed

    def migrate(self, matrix):
        """Re-projeta todo o estado da geracao para outra grade.

        matrix leva a grade atual para a nova (convencao do
        cv2.warpPerspective). Confirmada e provisoria sao mascaras:
        vizinho mais proximo, sem ombro sintetico na borda da tinta.
        A janela do voto guarda probabilidades: bilinear, borda zero
        (o mesmo tratamento da evidencia ao vivo). Os buffers de
        estatistica derivados sao invalidados e a sequencia de baixa
        cobertura zera: o conteudo deles era da grade velha.
        """
        if self._confirmed is None:
            return
        matrix = np.asarray(matrix, dtype=np.float64)
        height, width = self._confirmed.shape
        size = (width, height)

        def nearest(mask):
            return cv2.warpPerspective(
                mask, matrix, size, flags=cv2.INTER_NEAREST, borderValue=0
            )

        self._confirmed = nearest(self._confirmed)
        if self._removal_streak is not None:
            self._removal_streak = nearest(self._removal_streak)
        self._provisional = nearest(self._provisional)
        self._provisional_ever = nearest(self._provisional_ever.astype(np.uint8)) > 0
        self._retracted_ever = nearest(self._retracted_ever.astype(np.uint8)) > 0
        self._window = deque(
            (
                cv2.warpPerspective(
                    item, matrix, size, flags=cv2.INTER_LINEAR, borderValue=0.0
                )
                for item in self._window
            ),
            maxlen=self._window.maxlen,
        )
        self._score_buf = None
        self._hits_buf = None
        self._seen_buf = None
        self._low_overlap_streak = 0
        if self._e3_active:
            # Estado da E3 acompanha a grade: mascaras por vizinho
            # mais proximo, mapas de evidencia/trilha bilineares —
            # o mesmo tratamento do resto do estado da geracao.
            if self._e3_old_union is not None:
                self._e3_old_union = nearest(self._e3_old_union.astype(np.uint8)) > 0
                self._e3_region_big = nearest(self._e3_region_big.astype(np.uint8)) > 0
                self._e3_region_small = (
                    nearest(self._e3_region_small.astype(np.uint8)) > 0
                )
                self._e3_phase1_maps = [
                    cv2.warpPerspective(
                        item,
                        matrix,
                        size,
                        flags=cv2.INTER_LINEAR,
                        borderValue=0.0,
                    )
                    for item in self._e3_phase1_maps
                ]
                self._e3_side_big.migrate(matrix, size)
                self._e3_side_small.migrate(matrix, size)
        if self._e4_active:
            # Estado da E5-S acompanha a grade, como o da E3:
            # abertura e regioes por vizinho mais proximo, mapa
            # de probabilidade da abertura e mapas de evidencia/
            # trilha bilineares.
            if self._e4_opening is not None:
                self._e4_opening = nearest(self._e4_opening.astype(np.uint8)) > 0
                self._e4_region_big = nearest(self._e4_region_big.astype(np.uint8)) > 0
                self._e4_region_small = (
                    nearest(self._e4_region_small.astype(np.uint8)) > 0
                )
                self._e4_region_slow = (
                    nearest(self._e4_region_slow.astype(np.uint8)) > 0
                )
                self._e4_region_thin = (
                    nearest(self._e4_region_thin.astype(np.uint8)) > 0
                )
                self._e4_opening_prob = cv2.warpPerspective(
                    self._e4_opening_prob,
                    matrix,
                    size,
                    flags=cv2.INTER_LINEAR,
                    borderValue=0.0,
                )
                self._e4_phase1_maps = [
                    cv2.warpPerspective(
                        item,
                        matrix,
                        size,
                        flags=cv2.INTER_LINEAR,
                        borderValue=0.0,
                    )
                    for item in self._e4_phase1_maps
                ]
                self._e4_side_big.migrate(matrix, size)
                self._e4_side_small.migrate(matrix, size)
                self._e4_side_slow.migrate(matrix, size)
                self._e4_side_thin.migrate(matrix, size)

    def _window_stats(self):
        """(soma, hits, visto-como-tinta) da janela, sem empilhar.

        O caminho antigo fazia np.stack da janela (3 mapas float32
        de pagina inteira, ~12 MB) e somava/contava/maximizava
        sobre a pilha a cada passada — o maior transitorio do
        acumulador (pesquisa de performance de 2026-11-03, T10).
        Aqui as tres reducoes saem por acumulacao em buffers
        reusados: a soma adiciona os mesmos mapas na mesma ordem
        (resultado float identico), hits e visto sao contagens
        inteiras/booleanas exatas. Os buffers sao devolvidos para
        leitura imediata; a proxima chamada os sobrescreve.
        """
        window = list(self._window)
        shape = window[0].shape
        if self._score_buf is None or self._score_buf.shape != shape:
            self._score_buf = np.zeros(shape, dtype=np.float32)
            self._hits_buf = np.zeros(shape, dtype=np.int32)
            self._seen_buf = np.zeros(shape, dtype=bool)
        score = self._score_buf
        np.copyto(score, window[0])
        for item in window[1:]:
            np.add(score, item, out=score)
        hits = self._hits_buf
        hits[...] = window[0] >= _INK_VOTE_HIT
        for item in window[1:]:
            np.add(hits, item >= _INK_VOTE_HIT, out=hits)
        seen = self._seen_buf
        seen[...] = window[0] > self.threshold
        for item in window[1:]:
            np.logical_or(seen, item > self.threshold, out=seen)
        return score, hits, seen

    def add(self, prob, visible=None, *, removal_allowed=True):
        prob = np.asarray(prob, dtype=np.float32)
        if visible is not None:
            visible = np.asarray(visible, dtype=bool)
            if visible.shape != prob.shape:
                visible = None
        if self._confirmed is None or self._confirmed.shape != prob.shape:
            self._removal_streak = None
            self._confirmed = np.zeros(prob.shape, dtype=np.uint8)
            self._provisional = np.zeros(prob.shape, dtype=np.uint8)
            self._provisional_ever = np.zeros(prob.shape, dtype=bool)
            self._retracted_ever = np.zeros(prob.shape, dtype=bool)
            self._window.clear()
            self._low_overlap_streak = 0
            self._turn_fired = False
            self._e3_active = None
            self._e4_active = None
            self._fast_active = None
            self._reset_e3_state()
            self._reset_e4_state()
        if self._e3_active is None:
            # A regra da geracao decide na primeira passada e nao
            # muda no meio dela (ver _INK_VOTE_E3_ENV). O corte de
            # latencia da F94 decide junto, na mesma passada.
            # Precedencia declarada: DRAW_INK_VOTE_E4 ligada vence
            # a E3 — com as duas ligadas roda so a E5-S.
            self._e4_active = _ink_vote_e4_enabled()
            self._e3_active = _ink_vote_e3_enabled() and not self._e4_active
            self._fast_active = _ink_vote_fast_enabled()
        confirmed_before = self._confirmed > 0
        canvas_before = int(np.count_nonzero(self._confirmed))
        # Oclusao: pixel que a passada nao pode ver (a mao sobre a
        # pagina) nao entrega evidencia, nem a favor nem contra. A
        # probabilidade dele entra zerada na janela — soma zero no
        # escore, nao conta hit nem "visto como tinta" —, o
        # julgamento de virada mede a cobertura so sobre a tinta
        # confirmada visivel (ou se abstem quando quase tudo esta
        # coberto: mao segurando a pagina nao e pagina trocada) e a
        # camada provisoria congela onde nao ha visao (ver abaixo).
        evidence = (
            prob if visible is None else np.where(visible, prob, 0.0).astype(np.float32)
        )
        mask_now = evidence > self.threshold
        turn = False
        if canvas_before >= _INK_TURN_MIN_CANVAS_PX and not self._turn_fired:
            if visible is None:
                judge_turn = True
                turn_low = (
                    int(np.count_nonzero(mask_now & confirmed_before))
                    <= _INK_TURN_MAX_OVERLAP * canvas_before
                )
            else:
                visible_confirmed = confirmed_before & visible
                visible_px = int(np.count_nonzero(visible_confirmed))
                judge_turn = (
                    visible_px >= _OCCLUSION_TURN_MIN_VISIBLE_PX
                    and visible_px >= 0.5 * canvas_before
                )
                turn_low = judge_turn and (
                    int(np.count_nonzero(mask_now & visible_confirmed))
                    <= _INK_TURN_MAX_OVERLAP * visible_px
                )
            if judge_turn:
                if turn_low:
                    self._low_overlap_streak += 1
                    if self._low_overlap_streak >= _INK_TURN_PASSES:
                        turn = True
                        self._turn_fired = True
                else:
                    self._low_overlap_streak = 0
        self._window.append(evidence.copy())
        if self._e4_active:
            # Regra E5-S (DRAW_INK_VOTE_E4): a confirmacao sai de
            # _e4_step; a provisoria segue a janela corrente, como
            # nas outras regras — quem muda e quem confirma.
            score, hits, seen_as_ink = self._window_stats()
            new_confirmed = self._e4_step(evidence) & ~confirmed_before
        elif self._e3_active:
            # Regra E3 (DRAW_INK_VOTE_E3): a confirmacao sai de
            # _e3_step; o "visto" da camada provisoria segue sendo a
            # janela corrente, como na regra principal — a E3 muda
            # quem confirma, nao o que a tela mostra como tinta nova.
            score, hits, seen_as_ink = self._window_stats()
            new_confirmed = self._e3_step(evidence) & ~confirmed_before
        else:
            score, hits, seen_as_ink = self._window_stats()
            confirmed_rule = (score >= _INK_VOTE_SCORE_STRONG) | (
                (hits >= _INK_VOTE_MIN_HITS) & (score >= _INK_VOTE_SCORE)
            )
            if np.any(confirmed_rule):
                # Porta de fronteira (ver _INK_FRONTIER_RADIUS):
                # dentro da faixa do ombro da tinta (confirmada ou
                # vista acima do limiar nesta janela), so confirma
                # quem alguma passada viu como tinta; a soma
                # sub-limiar das copias deslocadas nao engrossa o
                # traco nem poe camada nova sobre ele.
                anchor = seen_as_ink | confirmed_before
                near_anchor = (
                    cv2.dilate(
                        anchor.astype(np.uint8),
                        _INK_FRONTIER_KERNEL,
                    )
                    > 0
                )
                confirmed_rule &= seen_as_ink | ~near_anchor
            new_confirmed = confirmed_rule & ~confirmed_before
        new_confirmed_px = int(np.count_nonzero(new_confirmed))
        if new_confirmed_px:
            self._confirmed[new_confirmed] = 255
        removed_px = 0
        if _experimental_enabled("DRAW_INK_REMOVAL"):
            if self._removal_streak is None:
                self._removal_streak = np.zeros(prob.shape, dtype=np.uint8)
            # Sem máscara de visibilidade não há evidência negativa confiável.
            witnessed = False
            if removal_allowed and visible is not None and canvas_before:
                retained = np.count_nonzero(confirmed_before & visible & mask_now)
                witnessed = (
                    retained >= _REMOVAL_RETAINED_FRACTION * canvas_before
                    or retained >= _REMOVAL_WITNESS_FLOOR_PX
                )
            missing = (
                confirmed_before
                & visible
                & np.isfinite(prob)
                & (prob <= _REMOVAL_LOW_PROB)
                if witnessed
                else np.zeros(prob.shape, dtype=bool)
            )
            self._removal_streak[~missing] = 0
            self._removal_streak[missing] = np.minimum(
                self._removal_streak[missing].astype(np.uint16) + 1, _REMOVAL_PASSES
            ).astype(np.uint8)
            removed = self._removal_streak >= _REMOVAL_PASSES
            removed_px = int(np.count_nonzero(removed))
            if removed_px:
                self._confirmed[removed] = 0
                self._removal_streak[removed] = 0
                # Nenhuma memória de voto pode ressuscitar o traço apagado.
                for item in self._window:
                    item[removed] = 0
                self._score_buf = self._hits_buf = self._seen_buf = None
                if self._e3_active:
                    for item in self._e3_phase1_maps:
                        item[removed] = 0
                    if self._e3_old_union is not None:
                        self._e3_old_union[removed] = False
                    # Regiões são suporte geométrico para redesenhar, não votos.
                    for side in (self._e3_side_big, self._e3_side_small):
                        if side._confirmed is not None:
                            side._confirmed[removed] = False
                        for item in side._maps:
                            item[removed] = 0
                if self._e4_active:
                    # Acomodacao com o DRAW_INK_REMOVAL do PR #67:
                    # a memoria de voto da E5-S tambem nao pode
                    # ressuscitar o traco apagado.
                    for item in self._e4_phase1_maps:
                        item[removed] = 0
                    if self._e4_opening is not None:
                        self._e4_opening[removed] = False
                        self._e4_opening_prob[removed] = 0
                    for mask in (
                        self._e4_region_big,
                        self._e4_region_small,
                        self._e4_region_slow,
                        self._e4_region_thin,
                    ):
                        if mask is not None:
                            mask[removed] = False
                    for side in (
                        self._e4_side_big,
                        self._e4_side_small,
                        self._e4_side_slow,
                        self._e4_side_thin,
                    ):
                        if side._confirmed is not None:
                            side._confirmed[removed] = False
                        for item in side._maps:
                            item[removed] = 0
                seen_as_ink[removed] = False
                score[removed] = 0
                hits[removed] = 0
        confirmed_now = self._confirmed > 0
        provisional_now = seen_as_ink & ~confirmed_now
        if self._fast_active:
            # Provisoria rapida (F94, SO exibicao): a evidencia de
            # hit acumulada (>= 2 hits na janela, soma >= 1,0) pinta
            # a tinta fraca antes da confirmacao. A confirmada nao
            # muda — poll() nao ve este pixel; se o voto nao
            # fechar, a provisoria retrata ao sair da janela, como
            # qualquer provisoria.
            provisional_now |= (
                (hits >= _INK_VOTE_FAST_PROV_MIN_HITS)
                & (score >= _INK_VOTE_FAST_PROV_MIN_SCORE)
            ) & ~confirmed_now
        previous_provisional = self._provisional > 0
        if visible is not None:
            # Sem visao nao ha retracao nem estreia: o provisorio
            # coberto fica como estava. Retrair e afirmar que a
            # tinta sumiu — e quem esta sobre ela e a mao, nao o
            # apagador; sem este congelamento, toda oclusao virava
            # retracao na regua e a tinta piscava ao reaparecer.
            provisional_now = provisional_now | (previous_provisional & ~visible)
        retracted = previous_provisional & ~provisional_now & ~confirmed_now
        if np.any(retracted):
            self._retracted_ever |= retracted
        self._provisional_ever |= provisional_now
        provisional_changed = bool(np.any(provisional_now != previous_provisional))
        self._provisional = np.where(provisional_now, 255, 0).astype(np.uint8)
        result = {
            "confirmed": self._confirmed.copy(),
            "provisional": self._provisional.copy(),
            "confirmed_changed": new_confirmed_px > 0 or removed_px > 0,
            "provisional_changed": provisional_changed,
            "new_confirmed_px": new_confirmed_px,
            "confirmed_px": int(np.count_nonzero(self._confirmed)),
            "provisional_px": int(np.count_nonzero(self._provisional)),
            "provisional_ever_px": int(np.count_nonzero(self._provisional_ever)),
            "retracted_px_total": int(np.count_nonzero(self._retracted_ever)),
            "turn": turn,
        }
        if _experimental_enabled("DRAW_INK_REMOVAL"):
            result["removed_confirmed_px"] = removed_px
        if self._e3_active:
            # Telemetria da E3: so existe com a flag ligada, como os
            # demais mecanismos atras de flag deste arquivo.
            result["ink_vote_e3"] = True
            result["ink_vote_e3_passes"] = self._e3_passes
            result["ink_vote_e3_tracks_big"] = self._e3_tracks_big
            result["ink_vote_e3_tracks_small"] = self._e3_tracks_small
        if self._e4_active:
            # Telemetria da E5-S: idem, so com DRAW_INK_VOTE_E4
            # ligada (com E3+E4 ligadas, a precedencia e da E4 e
            # a telemetria e esta).
            result["ink_vote_e4"] = True
            result["ink_vote_e4_passes"] = self._e4_passes
            result["ink_vote_e4_tracks_big"] = self._e4_tracks_big
            result["ink_vote_e4_tracks_small"] = self._e4_tracks_small
            result["ink_vote_e4_tracks_slow"] = self._e4_tracks_slow
            result["ink_vote_e4_tracks_thin"] = self._e4_tracks_thin
            result["ink_vote_e4_tracks_sub"] = self._e4_tracks_sub
        if self._fast_active:
            # Telemetria do corte de latencia (F94): idem, so com
            # DRAW_INK_VOTE_FAST ligada.
            result["ink_vote_fast"] = True
        return result


# Tamanho dos historicos de telemetria da trabalhadora: registros
# por passada e eventos de identidade. Replays que medem episodios
# longos passam (None, None) para guardar tudo.
_WORKER_HISTORY = (256, 128)


# Folga do teste de cadencia no pump(): relogio de cena em float
# (indice * 1/30 s) nao pode perder uma passada por um ulp.
_SYNC_CADENCE_EPS_SECONDS = 1e-9


class _PageCleanWorker:
    """Tinta pagina->limpa em thread propria, latest-wins.

    O laco principal entrega o warp mais recente com submit() e
    segue: a fila tem tamanho 1 e um warp nao processado e
    substituido pelo proximo, nunca acumulado. A trabalhadora roda
    passadas completas do modelo com pelo menos
    _PAGE_CLEAN_INTERVAL_SECONDS entre inicios — e nenhuma
    passada em descanso total (folha assentada, saida confirmada
    quieta e mao fora ou parada; _at_rest) — e alimenta o
    _LayeredInkAccumulator da geracao: a tinta nova sai na hora
    como camada provisoria para a exibicao (poll_display) e so
    vira canvas pelos votos de confirmacao (poll). Publicar cedo
    sem voto era a uniao antiga, que integrava o tremor e o
    mobiliario para sempre; confirmar antes de mostrar era o
    atraso de segundos da cadeia assentada. As duas camadas
    separam as duas perguntas.

    Geracao e o contrato de sessao da folha: folha nova ou folha
    que sai chama new_generation(), que descarta pendente e
    publicado da geracao anterior; a uniao da trabalhadora zera na
    primeira passada da geracao nova. Resultado de geracao velha
    nunca entra no canvas novo, mesmo com uma passada em voo na
    troca. Dentro da geracao, a evidencia de cada passada e
    registrada na geometria da primeira (a ancora) antes do voto:
    o quad que retifica anda entre passadas, e sem o registro cada
    alinhamento confirmava a sua copia da mesma tinta — a
    reimpressao medida na regua de 2026-11-05 (ver
    _REGISTRATION_IDENTITY_TOL). Blocos sujos por diff de pixels NAO entram aqui: medido
    nesta webcam, o churn entre quadros consecutivos parados
    (media de |diff| por bloco ~1,5-2,0 niveis; maximo pos-blur
    4-6) empata ou supera o sinal de um traco novo no bloco
    (lapis: maximo pos-blur 6-7), entao pular bloco por diff
    arriscaria engolir justamente o lapis. A economia vem da
    assincronia e da cadencia, nao de pular bloco.
    """

    def __init__(
        self,
        page_clean,
        hand_masker=None,
        clock=time.monotonic,
        history=_WORKER_HISTORY,
    ):
        self._page_clean = page_clean
        # Relogio monotonico injetavel: o padrao e o do processo;
        # replays com pump() passam o relogio da cena.
        self._clock = clock
        self._hand_masker = hand_masker
        self._condition = threading.Condition()
        self._thread = threading.Thread(
            target=self._work, name="draw-page-clean", daemon=True
        )
        self._pending = None
        self._generation = 0
        self._published = None
        self._version = 0
        self._stopped = False
        self._error = None
        # Estado privado da trabalhadora (so ela toca):
        self._worker_generation = -1
        # Geometria ancora da geracao: os cantos da primeira
        # passada com cantos. Toda evidencia posterior e
        # registrada nela antes do voto (ver _registration_matrix).
        # None enquanto a geracao nao teve passada com cantos —
        # chamadores que nao informam cantos (ferramentas, testes
        # antigos) seguem com a evidencia na propria geometria.
        self._anchor_corners = None
        self._content_guard_quads = []
        # Disciplina de registro (flags DRAW_REG_*, desligadas por
        # padrao): amostras de cantos da ancora-mediana e o quad
        # congelado da banda morta, ambos por geracao.
        self._anchor_samples = []
        self._reg_frozen_corners = None
        self._reg_deadband_over = 0
        self._reg_deadband_releases = 0
        # Migracao de ancora (flag DRAW_ANCHOR_MIGRATION): streak de
        # passadas deslocadas e os quads delas, por geracao.
        self._migration_streak = 0
        self._migration_quads = []
        self._anchor_migrations = 0
        self._migration_events = deque(maxlen=128)
        # Arbitragem de ancora (flag DRAW_ANCHOR_ARBITRATION): o
        # evento aberto, o streak de assentamento com seus escores
        # por passada e o snapshot do canvas pre-evento.
        self._arb_event_open = False
        self._arb_streak_quads = []
        self._arb_scores = []
        self._arb_snapshot = None
        self._arb_actions = 0
        self._arb_events = deque(maxlen=128)
        # Mobilidade da ancora (flag DRAW_ANCHOR_FOLLOW): o evento
        # de divergencia assentada, o streak de quads assentados e
        # a retencao por veredito spurious da arbitragem.
        self._follow_event_open = False
        self._follow_streak = 0
        self._follow_streak_quads = []
        self._follow_hold = False
        self._anchor_reanchors = 0
        self._follow_events = deque(maxlen=128)
        # Rotulacao ciclica que a ancora corrente carrega em relacao
        # aos quads posicionais (0 = posicional; k = cantos rolados
        # por k apos um pouso rodado julgado pela arbitragem).
        self._anchor_roll = 0
        self._identity_reference = None
        self._identity_reference_corners = None
        self._identity_recovery_requested = False
        self._identity_defer_count = 0
        self._identity_switch_votes = 0
        # Vigia de identidade (frente 28): armamento no meio da
        # geracao + evento de divisao para o app colher.
        self._identity_armed_mid_generation = False
        self._identity_arm_reason = None
        self._identity_split_event = None
        self._identity_watch_streak = 0
        self._identity_watch_prev_corners = None
        self._accumulator = _LayeredInkAccumulator(
            threshold=getattr(page_clean, "threshold", 0.7)
        )
        self._last_pass_start = None
        self._passes_total = 0
        self._last_pass_ms = None
        self._provisional_px = 0
        self._confirmed_px = 0
        self._retracted_px_total = 0
        self._published_display = None
        self._display_version = 0
        self._hand_state = None
        self._confirm_once = False
        self._still_passes = 0
        # Descanso total (so a trabalhadora e o new_generation
        # tocam): passadas desta geracao e passadas seguidas sem
        # confirmar nada novo.
        self._generation_passes = 0
        self._quiet_passes = 0
        # Passadas seguidas sem a saida (confirmada ou provisoria)
        # mudar, contadas SO pelas passadas — nunca zeradas por
        # set_hand_presence, que o laco chama a cada amostra de
        # presenca e zeraria o sinal entre passadas. E o sinal que
        # a cadencia por atividade (frente 92) le; _still_passes
        # nao serve porque a presenca o zera.
        self._activity_quiet_passes = 0
        # Passadas retiradas da fila e descartadas ANTES do modelo
        # por a geracao ter trocado (ou a trabalhadora ter parado)
        # entre a retirada e a execucao: custo de modelo poupado,
        # contado para a telemetria nao confundir com passada.
        self._telemetry_skipped_stale = 0
        # Relogio de quietude da mao (carimbo monotonico do
        # ultimo movimento/aparicao reportado): o descanso com a
        # mao presente exige a mao parada alem de
        # _HAND_REST_STILL_SECONDS. None = nenhum movimento
        # reportado ainda — nao conta como parada.
        self._hand_motion_at = None
        # Estado do evento geometrico (frente do cenario E3, atras
        # de DRAW_INK_FOLLOW_MOVE): True enquanto o quad submetido
        # segue fora da moldura da ancora — o periodo em que o
        # descanso nao pode fechar.
        self._geo_away = False
        # Telemetria por etapa (U0 do loop v2): contadores e os
        # carimbos das ultimas passadas, para medir onde o tempo
        # submit->publicacao mora sem mudar comportamento nenhum.
        # Quem le usa telemetry_snapshot(); quem escreve e o submit
        # (laco) e a trabalhadora, sempre sob a Condition.
        self._telemetry_submits = 0
        self._telemetry_replaced = 0
        self._telemetry_passes = 0
        self._telemetry_publishes = 0
        self._previous_pass_started_at = None
        self._telemetry_records = deque(maxlen=history[0])
        self._identity_events = deque(maxlen=history[1])
        # Evento de virada (desligado por padrao): pendente
        # (geracao, canvas confirmado anterior).
        self._turn_event = None

    def start(self):
        self._thread.start()
        return self

    def new_generation(self, preserve_identity=False):
        """Abre geracao; opcionalmente identifica a pagina anterior.

        preserve_identity guarda uma copia do canvas confirmado para
        que a primeira passada possa decidir, por conteudo registrado,
        se uma folha que voltou e a mesma pagina em outra pose.
        """
        with self._condition:
            self._identity_recovery_requested = bool(preserve_identity)
            self._identity_defer_count = 0
            self._identity_switch_votes = 0
            self._identity_armed_mid_generation = False
            self._identity_arm_reason = None
            self._identity_split_event = None
            self._identity_watch_streak = 0
            self._identity_watch_prev_corners = None
            if preserve_identity:
                reference = self._accumulator._confirmed
                self._identity_reference = (
                    None if reference is None else reference.copy()
                )
                self._identity_reference_corners = (
                    None
                    if self._anchor_corners is None
                    else self._anchor_corners.copy()
                )
            else:
                self._identity_reference = None
                self._identity_reference_corners = None
            self._generation += 1
            self._anchor_samples = []
            self._content_guard_quads = []
            self._reg_frozen_corners = None
            self._reg_deadband_over = 0
            self._migration_streak = 0
            self._migration_quads = []
            self._arb_event_open = False
            self._arb_streak_quads = []
            self._arb_scores = []
            self._arb_snapshot = None
            self._follow_event_open = False
            self._follow_streak = 0
            self._follow_streak_quads = []
            self._follow_hold = False
            self._anchor_roll = 0
            self._still_passes = 0
            self._generation_passes = 0
            self._quiet_passes = 0
            self._activity_quiet_passes = 0
            self._pending = None
            self._published = None
            self._published_display = None
            self._turn_event = None
            self._condition.notify_all()
            return self._generation

    def submit(self, warped, generation, corners=None, visible=None, coverage=None):
        """Entrega o warp mais recente (latest-wins).

        corners e o quad que retificou ESTE warp (os mesmos
        cantos que o chamador passou ao warp_sheet): a
        trabalhadora registra a evidencia da passada na
        geometria ancora da geracao antes do voto. Cantos fora
        do formato (4, 2) sao tratados como ausentes — um quad
        malformado nao pode derrubar a thread da tinta; sem
        cantos, a passada entra na propria geometria.

        visible e a mascara de visibilidade DESTE warp (True onde
        a camera ve a pagina; False sob a mao), na mesma grade do
        warp. Onde ela e False, a passada se abstem: a evidencia
        nao vota, a virada nao julga e a decisao de identidade so
        olha a tinta anterior visivel. Formato divergente do warp
        e tratado como ausente, pela mesma razao dos cantos.

        coverage e a cobertura de folha DESTE warp (True onde a
        retificacao cobre papel de verdade; False onde o quad
        admitiu o que esta fora da folha — mesa que entra por
        quad largo ou atrasado depois de movimento). Onde ela e
        False, a passada so vota sobre a tinta que a ancora ja
        conhece: re-confirmar e retrair o que existe, sim; tinta
        nova nascendo fora da folha, nunca. Assim o defeito
        semantico da mascara (traco denso virando "fundo" numa
        pose nova) nao congela a tinta real — ela ja e tinta —
        e a banda de mesa, que nunca foi tinta, segue vedada.
        """
        if corners is not None:
            try:
                corners = np.asarray(corners, dtype=np.float32)
            except (TypeError, ValueError):
                corners = None
            if corners is not None and corners.shape != (4, 2):
                corners = None
        if visible is not None:
            try:
                visible = np.asarray(visible, dtype=bool)
            except (TypeError, ValueError):
                visible = None
            if visible is not None and visible.shape != warped.shape[:2]:
                visible = None
        if coverage is not None:
            try:
                coverage = np.asarray(coverage, dtype=bool)
            except (TypeError, ValueError):
                coverage = None
            if coverage is not None and coverage.shape != warped.shape[:2]:
                coverage = None
        with self._condition:
            if self._stopped or generation != self._generation:
                return
            if (
                _experimental_enabled("DRAW_INK_FOLLOW_MOVE")
                and corners is not None
                and self._anchor_corners is not None
            ):
                # Evento geometrico: o quad deste warp saiu da
                # moldura da ancora (folha que se moveu de
                # verdade, nao jitter — quem envia aqui ja e a
                # geometria estabilizada do laco). Enquanto o
                # deslocamento persistir, a premissa do descanso
                # ("pagina parada, nada para ver") esta refutada:
                # a vista pode ter evidencia que o canvas nunca
                # contou. A quietude nao acumula — a sonda fica no
                # regime de 1 Hz do modo sem sinal de mao — e o
                # descanso so volta quando a geometria retorna a
                # moldura ou a saida prova que nao ha nada novo.
                away = (
                    float(
                        np.max(np.linalg.norm(corners - self._anchor_corners, axis=1))
                    )
                    > _GEO_WAKE_MOVE_PX
                )
                if away:
                    self._quiet_passes = 0
                    self._activity_quiet_passes = 0
                    self._still_passes = 0
                self._geo_away = away
            if self._pending is not None:
                # Latest-wins: um warp pendente substituido nunca
                # vira passada; a telemetria conta a substituicao
                # porque ela e atraso real de atualizacao.
                self._telemetry_replaced += 1
            self._telemetry_submits += 1
            self._pending = (
                warped,
                generation,
                self._clock(),
                corners,
                visible,
                coverage,
            )
            self._condition.notify_all()

    def set_hand_presence(self, present, moved=False):
        """Alimenta a porta de ociosidade: True, False ou None.

        A transicao para False arma exatamente uma passada de
        confirmacao: a mao saiu, a pagina assentou, e essa passada
        fecha o voto do que foi escrito. Depois dela, sem mao, a
        trabalhadora sonda a no maximo 1 Hz enquanto a saida
        confirmada ainda muda — e entra em descanso total (zero
        passadas) quando ela para de mudar; ver _at_rest.

        moved=True e o sinal de atividade da mascara (ver
        _MediaPipeHandPresence.sample): reseta o relogio de
        quietude da mao e acorda o descanso com a mao presente.
        Transicao de presenca tambem conta como movimento: a mao
        chegando ou saindo nunca e mao parada.
        """
        with self._condition:
            if present is False and self._hand_state is not False:
                self._confirm_once = True
            if present is not None:
                self._still_passes = 0
            if present is not None and (moved or present != self._hand_state):
                self._hand_motion_at = self._clock()
            self._hand_state = present
            self._condition.notify_all()

    def _hand_still(self):
        """Mao presente e sem movimento alem da janela de quietude."""
        return (
            self._hand_motion_at is not None
            and self._clock() - self._hand_motion_at >= _HAND_REST_STILL_SECONDS
        )

    def _at_rest(self):
        """Descanso total: nenhuma passada ate um evento acordar.

        Sem mao: sinal presente e False, passada de confirmacao
        da saida da mao ja gasta, janela de voto da geracao
        completa e as ultimas passadas sem confirmar nada novo.
        Com mao (decisao do Lucas, 2026-11-03): a mesma quietude
        da saida confirmada, mais a mao parada alem de
        _HAND_REST_STILL_SECONDS — acorda com o movimento da
        propria mao, reportado pela assinatura da mascara.
        Ver _REST_MIN_GENERATION_PASSES/_REST_QUIET_PASSES.
        """
        if (
            _experimental_enabled("DRAW_INK_REMOVAL")
            and self._accumulator._removal_streak is not None
            and np.any(self._accumulator._removal_streak)
        ):
            return False
        if (
            _experimental_enabled("DRAW_REG_CONTENT_GUARD")
            and self._content_guard_quads
        ):
            return False
        if (
            _ink_active_cadence_enabled()
            and self._activity_quiet_passes < _NO_HAND_REST_AFTER_STILL_PASSES
        ):
            # Com a cadencia por atividade (frente 92), a
            # quietude so da confirmada nao basta para descansar:
            # o criterio conta PASSADAS, nao tempo — a 4
            # passadas/s, 2 passadas quietas sao 0,5 s de cena e
            # a trabalhadora descansava no meio do desenho lento,
            # sem evento que a acordasse (submit nao acorda o
            # descanso). Enquanto a camada provisoria — a tinta
            # que o usuario ve chegando — mudou dentro da janela
            # de atividade, a pagina nao assentou.
            return False
        if self._identity_recovery_requested and self._identity_reference is not None:
            return False
        if (
            self._generation_passes < _REST_MIN_GENERATION_PASSES
            or self._quiet_passes < _REST_QUIET_PASSES
        ):
            return False
        if self._hand_state is False:
            return not self._confirm_once
        if self._hand_state is True:
            return self._hand_still()
        return False

    def _pass_interval(self):
        if self._at_rest():
            return None
        if (
            _ink_active_cadence_enabled()
            and self._hand_state is False
            and not self._confirm_once
            and self._activity_quiet_passes < _NO_HAND_REST_AFTER_STILL_PASSES
        ):
            # Cadencia por atividade (frente 92, atras de
            # DRAW_INK_ACTIVE_CADENCE, default OFF): a sonda
            # ociosa pressupoe "sem mao, sem tinta nova" — mas
            # quem desmente o pressuposto e a propria saida: na
            # sessao viva 1794443693 a confirmada cresceu em
            # quase toda pasada por geracoes inteiras com o
            # intervalo travado em 1,0 s (93 de 102 intervalos),
            # porque o sinal de presenca dizia "sem mao" enquanto
            # o conteudo mudava. Enquanto a saida mudou nas
            # ultimas _NO_HAND_REST_AFTER_STILL_PASSES passadas,
            # a cadencia e a ativa; assentou, a sonda volta. O
            # sinal e o do acumulador (a mesma familia que o
            # descanso total usa), nunca diff de quadro. O regime
            # sem sinal (None) ja tem a regra gemea via
            # _still_passes, mais abaixo, e nao muda.
            return _PAGE_CLEAN_INTERVAL_SECONDS
        if self._hand_state is False and not self._confirm_once:
            return _IDLE_PROBE_SECONDS
        if (
            self._hand_state is None
            and self._still_passes >= _NO_HAND_REST_AFTER_STILL_PASSES
        ):
            return _IDLE_PROBE_SECONDS
        return _PAGE_CLEAN_INTERVAL_SECONDS

    def _anchor_migration_step(
        self,
        generation,
        corners,
        shape,
        identity_owned,
        reg_anchor_median,
        reg_deadband,
    ):
        """Avanca o streak do pouso; migra na passada que o completa.

        Chamado sob a Condition, com a flag ligada. Presenca perdida
        (sem quad valido) e julgamento de identidade em curso nao sao
        pouso: zeram o streak — a fronteira de geracao/identidade e a
        dona desses eventos. Devolve o evento da migracao ou None.
        """
        old_anchor = self._anchor_corners
        if identity_owned or corners is None or old_anchor is None:
            self._migration_streak = 0
            self._migration_quads = []
            return None
        matrix = _registration_matrix(old_anchor, corners, shape)
        displacement = _matrix_grid_displacement(matrix, shape)
        if displacement <= _ANCHOR_MIGRATION_DISPLACEMENT_PX:
            self._migration_streak = 0
            self._migration_quads = []
            return None
        self._migration_streak += 1
        self._migration_quads.append(corners.copy())
        if self._migration_streak < _ANCHOR_MIGRATION_PASSES:
            return None
        new_anchor = _median_anchor(self._migration_quads)
        # Inversa do registro ancora-nova -> ancora-velha: leva o
        # estado da grade velha para a nova.
        migration = np.linalg.inv(_registration_matrix(old_anchor, new_anchor, shape))
        self._accumulator.migrate(migration)
        self._anchor_corners = new_anchor
        if reg_anchor_median:
            self._anchor_samples = [quad.copy() for quad in self._migration_quads]
        if reg_deadband:
            self._reg_frozen_corners = new_anchor.copy()
            self._reg_deadband_over = 0
        self._anchor_migrations += 1
        event = {
            "generation": generation,
            "pass_index": self._generation_passes,
            "displacement_px": displacement,
            "streak": self._migration_streak,
            "anchor_delta_px": float(
                np.max(np.linalg.norm(new_anchor - old_anchor, axis=1))
            ),
            "anchor_delta_mean_px": _matrix_grid_displacement(migration, shape),
        }
        self._migration_events.append(event)
        self._migration_streak = 0
        self._migration_quads = []
        return event

    def _arb_close_event(self):
        self._arb_event_open = False
        self._arb_streak_quads = []
        self._arb_scores = []
        self._arb_snapshot = None

    def _content_registration_guard(
        self, raw, registered, visible, registered_visible, corners, identity_owned
    ):
        """Identidade de conteúdo vence reprojeção só com suporte positivo.

        Enquanto a pose não assenta, não há voto. No pouso, só a âncora
        muda: confirmado e janela continuam nas coordenadas da página.
        """
        info = {"reg_content_guard": True, "reg_content_blocked": False}
        confirmed = self._accumulator._confirmed
        if (
            identity_owned
            or corners is None
            or self._anchor_corners is None
            or self._anchor_roll
            or confirmed is None
        ):
            self._content_guard_quads = []
            return registered, registered_visible, info
        displacement = _matrix_grid_displacement(
            _registration_matrix(self._anchor_corners, corners, raw.shape), raw.shape
        )
        if displacement <= _ANCHOR_MIGRATION_DISPLACEMENT_PX:
            self._content_guard_quads = []
            return registered, registered_visible, info
        snapshot = confirmed > 0
        raw_evidence = raw if visible is None else np.where(visible, raw, 0.0)
        reg_evidence = (
            registered
            if registered_visible is None
            else np.where(registered_visible, registered, 0.0)
        )
        raw_mass, reg_mass = float(np.sum(raw_evidence)), float(np.sum(reg_evidence))
        sp = float(np.sum(raw_evidence * snapshot)) / max(raw_mass, 1e-9)
        sc = float(np.sum(reg_evidence * snapshot)) / max(reg_mass, 1e-9)
        info.update(
            reg_content_score_identity=sp,
            reg_content_score_projected=sc,
            reg_content_displacement_px=displacement,
        )
        enough = (
            np.count_nonzero(snapshot) >= _ARB_MIN_CANVAS_PX
            and np.count_nonzero(raw_evidence > self._accumulator.threshold)
            >= _ARB_MIN_PASS_PX
        )
        if not enough or sp <= sc + _ARB_DELTA:
            self._content_guard_quads = []
            return registered, registered_visible, info
        if any(
            np.max(np.linalg.norm(corners - q, axis=1)) > _ARB_SETTLE_PX
            for q in self._content_guard_quads
        ):
            self._content_guard_quads = []
        self._content_guard_quads.append(corners.copy())
        info["reg_content_streak"] = len(self._content_guard_quads)
        if len(self._content_guard_quads) < _ANCHOR_MIGRATION_PASSES:
            info["reg_content_blocked"] = True
            return registered, np.zeros(raw.shape, dtype=bool), info
        self._anchor_corners = _median_anchor(self._content_guard_quads)
        self._anchor_samples = [q.copy() for q in self._content_guard_quads]
        self._reg_frozen_corners = self._anchor_corners.copy()
        self._reg_deadband_over = 0
        self._content_guard_quads = []
        info["reg_content_identity_won"] = True
        return raw, visible, info

    def _follow_close_event(self):
        self._follow_event_open = False
        self._follow_streak = 0
        self._follow_streak_quads = []
        self._follow_hold = False

    def _anchor_arbitration_step(
        self,
        generation,
        corners,
        shape,
        identity_owned,
        reg_anchor_median,
        reg_deadband,
        raw_prob,
        registered_prob,
        visible=None,
    ):
        """Avanca o evento de arbitragem; arbitra no fecho do streak.

        Chamado sob a Condition, com DRAW_ANCHOR_ARBITRATION ligada.
        Devolve (status, evento): "closed" (sem evento; a passada
        vota), "open" (evento aberto sem veredito; a passada NAO
        vota — a copia deslocada nao pode confirmar pelo voto
        enquanto a hipotese nao foi julgada) ou "verdict" (o evento
        fechou nesta passada; ela vota, re-registrada na ancora nova
        se o veredito foi pouso). Presenca perdida e identidade em
        curso fecham o evento sem veredito: a fronteira de
        geracao/identidade e a dona desses casos.
        """
        old_anchor = self._anchor_corners
        if identity_owned or corners is None or old_anchor is None:
            self._arb_close_event()
            return "closed", None
        # O gatilho compara em termos POSICIONAIS: uma ancora que ja
        # carrega rotulos rolados (pouso rodado anterior) nao pode
        # fazer toda passada seguinte parecer deslocada pela
        # permutacao — o deslocamento que abre evento e o da pose,
        # nao o da rotulacao.
        trigger_anchor = (
            old_anchor
            if self._anchor_roll == 0
            else np.roll(old_anchor, -self._anchor_roll, axis=0)
        )
        matrix = _registration_matrix(trigger_anchor, corners, shape)
        displacement = _matrix_grid_displacement(matrix, shape)
        if displacement <= _ANCHOR_MIGRATION_DISPLACEMENT_PX:
            self._arb_close_event()
            return "closed", None
        if not self._arb_event_open:
            self._arb_event_open = True
            confirmed = self._accumulator._confirmed
            self._arb_snapshot = None if confirmed is None else confirmed > 0
            self._arb_streak_quads = []
            self._arb_scores = []
        consistent = all(
            float(np.max(np.linalg.norm(corners - quad, axis=1))) <= _ARB_SETTLE_PX
            for quad in self._arb_streak_quads
        )
        if not consistent:
            # O movimento continua: a pose candidata e a mais
            # recente e o snapshot renova (o canvas nao mudou — as
            # passadas do evento nao votam).
            confirmed = self._accumulator._confirmed
            self._arb_snapshot = None if confirmed is None else confirmed > 0
            self._arb_streak_quads = []
            self._arb_scores = []
        self._arb_streak_quads.append(corners.copy())
        snapshot = self._arb_snapshot
        raw_total = float(np.sum(raw_prob))
        registered_total = float(np.sum(registered_prob))
        if snapshot is None or raw_total <= 0.0 or registered_total <= 0.0:
            score_page = 0.0
            score_camera = 0.0
            cyclic = (0.0, 0.0, 0.0)
        else:
            # Com roll parado (frente 27): a evidencia da pagina e o
            # S_roll da familia ciclica, reclassificado — o warp
            # entregue ja esta em coordenadas da pagina e a unica
            # coisa errada nele e a rotacao parada (permutacao
            # pura), entao a hipotese-pagina le a prob des-rotada
            # pelo roll corrente. Medido na era 3 capturada: 0,193
            # contra 0,002 do Sc. Os ciclicos de rotacao NOVA
            # seguem zerados (segunda rotacao = abstencao).
            page_prob = (
                raw_prob
                if self._anchor_roll == 0
                else _permute_prob(raw_prob, -self._anchor_roll)
            )
            score_page = float(np.sum(page_prob * snapshot) / raw_total)
            score_camera = float(np.sum(registered_prob * snapshot) / registered_total)
            # Colocacoes ciclicas (frentes 23/24): a hipotese "a
            # folha assentou rodada de k x 90 graus" pontuada como a
            # medicao manda — a prob ENTREGUE permutada por -k
            # (rot90 + resize) contra o snapshot. A forma anterior
            # (registrar com os cantos rolados) marcava 0,005 onde
            # esta marca 0,23-0,27 na mesma passada real: reprovada
            # e substituida. So dentro do evento, e so a partir de
            # uma ancora posicional (roll == 0): uma segunda rotacao
            # na mesma geracao e regime nao medido — abstem.
            if self._anchor_roll != 0:
                cyclic = (0.0, 0.0, 0.0)
            else:
                cyclic = tuple(
                    float(np.sum(_permute_prob(raw_prob, -k) * snapshot) / raw_total)
                    for k in (1, 2, 3)
                )
        pass_mass = int(np.count_nonzero(raw_prob > 0.5))
        self._arb_scores.append(
            (score_page, score_camera, cyclic[0], cyclic[1], cyclic[2], pass_mass)
        )
        if len(self._arb_streak_quads) < _ANCHOR_MIGRATION_PASSES:
            return "open", None
        mean_page = float(np.mean([s[0] for s in self._arb_scores]))
        mean_camera = float(np.mean([s[1] for s in self._arb_scores]))
        mean_cyclic = [
            float(np.mean([s[2 + k] for s in self._arb_scores])) for k in range(3)
        ]
        mean_mass = float(np.mean([s[5] for s in self._arb_scores]))
        canvas_px = (
            0
            if self._arb_snapshot is None
            else int(np.count_nonzero(self._arb_snapshot))
        )
        rotation_k = 0
        if canvas_px < _ARB_MIN_CANVAS_PX or mean_mass < _ARB_MIN_PASS_PX:
            # Sem massa para julgar nao ha copia a fabricar nem a
            # proteger: trocar a ancora e custo zero nas duas
            # hipoteses.
            verdict = "landing_default"
        else:
            family = [
                ("landing", mean_page),
                ("spurious", mean_camera),
                ("landing_rotated", mean_cyclic[0]),
                ("landing_rotated", mean_cyclic[1]),
                ("landing_rotated", mean_cyclic[2]),
            ]
            order = sorted(range(len(family)), key=lambda i: family[i][1], reverse=True)
            best, second = order[0], order[1]
            if family[best][1] - family[second][1] <= _ARB_DELTA:
                verdict = "indeterminate"
            else:
                verdict = family[best][0]
                if verdict == "landing_rotated":
                    rotation_k = best - 1
        event = {
            "generation": generation,
            "pass_index": self._generation_passes,
            "verdict": verdict,
            "action": verdict in ("landing", "landing_default", "landing_rotated"),
            "score_page": mean_page,
            "score_camera": mean_camera,
            "scores_cyclic": [mean_camera, *mean_cyclic],
            "rotation_k": rotation_k,
            "score_page_roll_compensated": self._anchor_roll != 0,
            "canvas_px": canvas_px,
            "pass_mass_px": mean_mass,
            "displacement_px": displacement,
            "streak": len(self._arb_streak_quads),
        }
        self._arb_events.append(event)
        if event["action"]:
            new_anchor = _median_anchor(self._arb_streak_quads)
            if rotation_k:
                # Pouso com rotacao: a ancora nova carrega os rotulos
                # fisicos — os cantos da mediana re-rotulados por k,
                # a mesma permutacao que venceu a pontuacao. O estado
                # nao se move (a v3 inteira e troca de ancora).
                new_anchor = np.roll(new_anchor, rotation_k, axis=0)
            elif self._anchor_roll != 0:
                # Pouso comum (ou default) sobre a grade rolada
                # (frente 27, item 3): a mediana posicional do
                # streak rolada pelo roll parado — sem isso o par
                # ancora/roll fica incoerente e o pareamento do
                # voto e do gatilho quebra.
                new_anchor = np.roll(new_anchor, self._anchor_roll, axis=0)
            self._anchor_corners = new_anchor
            # So o pouso rodado muda a rotulacao; nos demais
            # vereditos o roll parado se preserva.
            if verdict == "landing_rotated":
                self._anchor_roll = rotation_k
            if reg_anchor_median:
                self._anchor_samples = [quad.copy() for quad in self._arb_streak_quads]
            if reg_deadband:
                self._reg_frozen_corners = new_anchor.copy()
                self._reg_deadband_over = 0
            self._arb_actions += 1
        self._arb_close_event()
        return "verdict", event

    def _anchor_follow_step(
        self,
        generation,
        corners,
        shape,
        identity_owned,
        reg_anchor_median,
        reg_deadband,
        step_px,
        arb_event,
    ):
        """Avanca o evento de divergencia assentada; re-ancora no
        fecho do streak.

        Chamado sob a Condition, com DRAW_ANCHOR_FOLLOW ligada, DEPOIS
        do passo da arbitragem (arb_event e o veredito dela nesta
        passada, ou None). Devolve (status, evento): "closed" (sem
        evento; a passada vota), "open" (divergencia assentada em
        curso sem re-ancoragem; a passada NAO vota) ou "reanchored"
        (a ancora trocou nesta passada; ela vota, re-registrada na
        ancora nova pelo chamador). O desenho e a precedencia estao
        no comentario das constantes _FOLLOW_* e em
        34-desenho-ancora-movel.md.
        """
        old_anchor = self._anchor_corners
        if identity_owned or corners is None or old_anchor is None:
            self._follow_close_event()
            return "closed", None
        # Precedencia da arbitragem (veredito DESTA passada): quem
        # decidiu por conteudo nao e atravessado pela geometria.
        if arb_event is not None and arb_event["verdict"] != "indeterminate":
            if arb_event["action"]:
                # A ancora ja se moveu pelo conteudo (pouso, pouso
                # rodado ou default): a divergencia acabou.
                self._follow_close_event()
                return "closed", None
            # spurious: o conteudo decidiu que a folha nao se
            # moveu — o episodio e do juizo de conteudo e o FOLLOW
            # fica retido (sem fogo, sem abstencao propria) ate ele
            # fechar.
            self._follow_event_open = False
            self._follow_streak = 0
            self._follow_streak_quads = []
            self._follow_hold = True
            return "closed", None
        # O gatilho compara em termos POSICIONAIS, como o da
        # arbitragem: o roll parado nao e deslocamento de pose.
        trigger_anchor = (
            old_anchor
            if self._anchor_roll == 0
            else np.roll(old_anchor, -self._anchor_roll, axis=0)
        )
        matrix = _registration_matrix(trigger_anchor, corners, shape)
        displacement = _matrix_grid_displacement(matrix, shape)
        if displacement <= _FOLLOW_DISPLACEMENT_PX:
            self._follow_close_event()
            return "closed", None
        if self._follow_hold:
            # Episodio retido pelo veredito spurious: a geometria
            # nao decide nem abstee enquanto o episodio durar.
            return "closed", None
        self._follow_event_open = True
        if step_px is not None and step_px <= _FOLLOW_SETTLE_PX:
            self._follow_streak += 1
            self._follow_streak_quads.append(corners.copy())
        else:
            # A folha ainda esta andando: o evento segue aberto
            # (a passada abstee), o streak recomeca.
            self._follow_streak = 0
            self._follow_streak_quads = []
        if self._follow_streak < _FOLLOW_PASSES:
            return "open", None
        new_anchor = _median_anchor(self._follow_streak_quads)
        if self._anchor_roll != 0:
            # Pouso comum sobre a grade rolada: a mediana
            # posicional rolada pelo roll parado (a coerencia
            # ancora/roll da arbitragem); o FOLLOW nunca cria roll.
            new_anchor = np.roll(new_anchor, self._anchor_roll, axis=0)
        closed_arb_event = self._arb_event_open
        if closed_arb_event:
            # A divergencia que a arbitragem julgava foi resolvida
            # pela geometria; ela recomeca contra a ancora nova.
            self._arb_close_event()
        self._anchor_corners = new_anchor
        if reg_anchor_median:
            self._anchor_samples = [quad.copy() for quad in self._follow_streak_quads]
        if reg_deadband:
            self._reg_frozen_corners = new_anchor.copy()
            self._reg_deadband_over = 0
        # A geometria sob a vigia mudou: a sequencia dela recomeca
        # (o recall por passada segue valido; a ancora nova e o
        # referencial corrente que o portao dela pressupoe).
        self._identity_watch_streak = 0
        self._anchor_reanchors += 1
        event = {
            "generation": generation,
            "pass_index": self._generation_passes,
            "displacement_px": displacement,
            "streak": self._follow_streak,
            "anchor_delta_px": float(
                np.max(np.linalg.norm(new_anchor - old_anchor, axis=1))
            ),
            "closed_arb_event": closed_arb_event,
        }
        self._follow_events.append(event)
        self._follow_close_event()
        return "reanchored", event

    def telemetry_snapshot(self):
        """Copia da telemetria por etapa (somente leitura).

        Contadores desde a criacao da trabalhadora e, por passada
        recente, os carimbos monotonicos e as duracoes derivadas:
        queue_wait_s (submit->inicio da passada; inclui a espera
        da cadencia de _PAGE_CLEAN_INTERVAL_SECONDS), pass_s
        (inicio->fim do modelo) e submit_to_publish_s (submit->
        fim da passada que decide a publicacao). Cada registro
        traz tambem interval_s (o intervalo que governou a espera
        daquela passada; None na primeira ou na acordada do
        descanso), hand_state (o sinal de presenca na retirada) e
        activity_quiet_passes (quietude da saida ao fim da
        passada) — sem esses tres, o orcamento de cadencia nao
        distingue sonda ociosa de cadencia ativa (frente 92). Na reconciliacao
        com a orquestracao em camadas, passes conta cada passada
        do modelo e publishes conta so as passadas que publicam
        canvas confirmado novo no poll(). Eventos de identidade de
        pagina ficam numa lista separada para sobreviverem ao buffer
        curto de passadas. Nenhum campo muda comportamento.
        """
        with self._condition:
            snapshot = {
                "anchor_arbitration_enabled": _anchor_arbitration_enabled(),
                "submits": self._telemetry_submits,
                "submits_replaced": self._telemetry_replaced,
                "passes": self._telemetry_passes,
                "publishes": self._telemetry_publishes,
                "passes_skipped_stale": self._telemetry_skipped_stale,
                "records": [dict(r) for r in self._telemetry_records],
                "page_identity_events": [
                    dict(event) for event in self._identity_events
                ],
            }
            if _reg_deadband_enabled() or self._reg_deadband_releases:
                snapshot["reg_deadband_releases"] = self._reg_deadband_releases
            if _anchor_migration_enabled() or self._anchor_migrations:
                snapshot["anchor_migrations"] = self._anchor_migrations
                snapshot["anchor_migration_events"] = [
                    dict(event) for event in self._migration_events
                ]
            if _anchor_arbitration_enabled() or self._arb_actions:
                snapshot["anchor_arbitration_actions"] = self._arb_actions
                snapshot["anchor_arbitration_events"] = [
                    dict(event) for event in self._arb_events
                ]
            if _anchor_follow_enabled() or self._anchor_reanchors:
                snapshot["anchor_reanchors"] = self._anchor_reanchors
                snapshot["anchor_reanchor_events"] = [
                    dict(event) for event in self._follow_events
                ]
            return snapshot

    def poll(self, generation, after_version):
        """(versao, copia do canvas confirmado) se ha publicado
        novo da geracao depois de after_version; senao None.

        Este e o canal dos consumidores de canvas (snapshot,
        cursor, Paint, ponte): so tinta confirmada passa por aqui.
        A camada provisoria nunca sai por este poll.
        """
        with self._condition:
            published = self._published
            if (
                published is None
                or published[0] != generation
                or published[1] <= after_version
            ):
                return None
            return published[1], published[2].copy()

    def poll_display(self, generation, after_version):
        """(versao, confirmado, provisorio) para a tela, ou None.

        As camadas vao separadas para a exibicao pintar a
        provisoria em estilo distinto. Este canal e so de
        exibicao: quem consome canvas usa poll().
        """
        with self._condition:
            published = self._published_display
            if (
                published is None
                or published[0] != generation
                or published[1] <= after_version
            ):
                return None
            return (
                published[1],
                published[2].copy(),
                published[3].copy(),
            )

    def take_turn_event(self, generation):
        """Canvas anterior se a virada (U3) disparou nesta geracao.

        Devolve a copia do canvas da pagina que acabou e limpa o
        evento; None se nao ha evento pendente pra esta geracao.
        Quem chama (o laco) salva o snapshot e abre geracao nova.
        So dispara com DRAW_PAGE_TURN_EVENT=1 (ver o portao acima).
        """
        with self._condition:
            if self._turn_event is None or self._turn_event[0] != generation:
                return None
            _gen, canvas = self._turn_event
            self._turn_event = None
            return canvas

    def arm_identity_check(self, reason):
        """Arma o juiz de identidade no meio da geracao corrente.

        A referencia do julgamento e o canvas confirmado ate aqui:
        se as proximas passadas casarem com ele (recall/containment
        do juiz existente), era a mesma pagina; se o consenso disser
        pagina nova, o canvas atual fecha como pagina terminada
        (ver take_identity_split) e a geracao recomeca do zero sem
        trocar de geracao no app ate ele colher o evento. Devolve
        True se armou. Sem efeito com a vigia desligada, com um
        julgamento ja em curso, ou com canvas confirmado abaixo do
        piso (pagina quase vazia nao tem identidade a defender —
        o caso degenerado das decisoes contra reference_px = 0).
        """
        with self._condition:
            return self._arm_identity_check_locked(reason)

    def _arm_identity_check_locked(self, reason, seed_votes=0):
        if not _identity_watch_enabled():
            return False
        if self._identity_reference is not None:
            return False
        confirmed = self._accumulator._confirmed
        if confirmed is None:
            return False
        reference_px = int(np.count_nonzero(confirmed > 0))
        if reference_px < _IDENTITY_WATCH_MIN_CANVAS_PX:
            return False
        self._identity_reference = confirmed.copy()
        self._identity_reference_corners = (
            None if self._anchor_corners is None else self._anchor_corners.copy()
        )
        self._identity_recovery_requested = True
        self._identity_defer_count = 0
        self._identity_switch_votes = int(seed_votes)
        self._identity_armed_mid_generation = True
        self._identity_arm_reason = reason
        self._identity_events.append(
            {
                "generation": self._generation,
                "decision": "armed",
                "reason": reason,
                "recall": None,
                "containment": None,
                "reference_px": reference_px,
                "visible_reference_px": None,
                "geometry_registered": False,
                "threshold": _PAGE_IDENTITY_MIN_RECALL,
                "containment_threshold": _PAGE_IDENTITY_MIN_CONTAINMENT,
                "ecc_refine_enabled": _ecc_refinement_enabled(),
                "quarantine_reason": None,
                "quarantine_frames": 0,
                "deferred": False,
            }
        )
        return True

    def take_identity_split(self, generation):
        """(canvas, motivo) se o juiz dividiu a geracao no meio.

        Veredito new_page de um julgamento ARMADO no meio da
        geracao: o canvas devolvido e a pagina que acabou (copia,
        feita antes do reset do acumulador); quem chama salva o
        snapshot e abre geracao nova, como no evento de virada.
        None se nao ha divisao pendente pra esta geracao. O timeout
        de quarentena NAO divide: segue a semantica existente
        (perda de lock, sem veredito) e nada e guardado aqui.
        """
        with self._condition:
            if (
                self._identity_split_event is None
                or self._identity_split_event[0] != generation
            ):
                return None
            _gen, canvas, reason = self._identity_split_event
            self._identity_split_event = None
            return canvas, reason

    def _identity_watch_step(
        self, registered_prob, registered_visible, corners, step_px
    ):
        """Um passo da vigia de conteudo (lock ja segurado).

        Mede o recall dilatado do canvas confirmado na passada
        registrada — a mesma metrica do juiz, contra o canvas VIVO.
        Devolve (recall, suspeita): recall None quando a passada
        nao conta (oclusao da tinta anterior: nao conta e nao zera;
        os demais casos zeram a sequencia); suspeita True quando a
        passada mediu ruptura ou o juiz acabou de ser armado — a
        passada suspeita NAO vota, para a tinta da pagina nova nao
        confirmar no canvas da antiga antes do veredito (a medicao
        acontece antes do voto, na mesma passada). Ruptura
        sustentada arma o juiz via _arm_identity_check_locked; a
        vigia nunca decide.
        """
        confirmed = self._accumulator._confirmed
        if confirmed is None or corners is None or self._anchor_corners is None:
            self._identity_watch_streak = 0
            return None, False
        prior = confirmed > 0
        prior_px = int(np.count_nonzero(prior))
        if prior_px < _IDENTITY_WATCH_MIN_CANVAS_PX:
            self._identity_watch_streak = 0
            return None, False
        displacement = float(
            np.max(np.linalg.norm(corners - self._anchor_corners, axis=1))
        )
        if displacement > _IDENTITY_WATCH_MAX_DISPLACEMENT_PX:
            self._identity_watch_streak = 0
            return None, False
        if step_px is None or step_px > _IDENTITY_WATCH_MAX_STEP_PX:
            # Geometria nao assentada (a folha ainda esta mexendo
            # entre passadas): conteudo nao e decidivel neste
            # material — nem conta, nem arma.
            self._identity_watch_streak = 0
            return None, False
        seen = registered_prob > self._accumulator.threshold
        if int(np.count_nonzero(seen)) < _IDENTITY_WATCH_MIN_SEEN_PX:
            self._identity_watch_streak = 0
            return None, False
        if registered_visible is not None:
            prior_visible = prior & registered_visible
            prior_visible_px = int(np.count_nonzero(prior_visible))
            if prior_visible_px < max(
                _IDENTITY_MIN_VISIBLE_PX,
                int(_IDENTITY_MIN_VISIBLE_FRACTION * prior_px),
            ):
                return None, False
            prior = prior_visible
            prior_px = prior_visible_px
        kernel = np.ones((_IDENTITY_DILATION_KERNEL,) * 2, np.uint8)
        seen_d = cv2.dilate(seen.astype(np.uint8), kernel) > 0
        recall = float(np.count_nonzero(prior & seen_d) / prior_px)
        if recall <= _IDENTITY_WATCH_MAX_RECALL:
            self._identity_watch_streak += 1
        else:
            self._identity_watch_streak = 0
        if self._identity_watch_streak >= _IDENTITY_WATCH_PASSES:
            self._identity_watch_streak = 0
            return recall, self._arm_identity_check_locked(
                "content_rupture", seed_votes=_IDENTITY_WATCH_SEED_VOTES
            )
        return recall, self._identity_watch_streak >= 1

    def _work(self):
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._pending is not None or self._stopped
                )
                if self._stopped:
                    return
                # Cadencia: o intervalo corre com o pendente na
                # mao; um submit durante a espera o substitui
                # (latest-wins) e e ele que roda ao vencer. Se a
                # geracao trocou, new_generation limpou o pendente
                # e esta volta nao roda nada.
                while self._last_pass_start is not None:
                    interval = self._pass_interval()
                    if interval is None:
                        # Descanso total: espera sem prazo por um
                        # evento que acorde (a mao voltar, a mao
                        # presente se mexer, geracao nova, stop).
                        # Submit NAO acorda: ele so troca o
                        # pendente, que e reavaliado quando um
                        # evento de verdade chegar. Sem isto a
                        # sonda de 1 Hz rodava para sempre sobre a
                        # pagina parada e o voto acumulava camadas.
                        self._condition.wait()
                        if self._stopped:
                            return
                        if self._pending is None:
                            break
                        continue
                    remaining = interval - (self._clock() - self._last_pass_start)
                    if remaining <= 0 or self._pending is None:
                        break
                    self._condition.wait(timeout=remaining)
                    if self._stopped:
                        return
                if self._pending is None:
                    continue
                job = self._take_pending_locked()
            if not self._run_pass(job):
                return

    def _take_pending_locked(self):
        """Retira o pendente e abre a passada (com a Condition na mao)."""
        (
            warped,
            generation,
            submitted_at,
            corners,
            visible,
            coverage,
        ) = self._pending
        self._pending = None
        # Intervalo que governou a espera desta passada (o regime
        # de cadencia em vigor na retirada, antes de _last_pass_start
        # andar): vai ao registro de telemetria para o orcamento de
        # latencia separar espera de cadencia de custo de passada.
        # None na primeira passada da trabalhadora.
        scheduled_interval = (
            self._pass_interval() if self._last_pass_start is not None else None
        )
        scheduled_hand_state = self._hand_state
        self._last_pass_start = self._clock()
        pass_started_at = self._last_pass_start
        if self._confirm_once:
            self._confirm_once = False
        return (
            warped,
            generation,
            submitted_at,
            corners,
            visible,
            coverage,
            pass_started_at,
            scheduled_interval,
            scheduled_hand_state,
        )

    def pump(self):
        """Execucao sincrona (sem thread): roda a passada se ja venceu.

        Para quem injeta um relogio controlado (replays, testes) e
        nao chama start(): a mesma decisao de cadencia e descanso
        do _work, avaliada no instante do relogio, seguida da mesma
        passada. True se uma passada rodou.
        """
        with self._condition:
            if self._stopped or self._pending is None:
                return False
            if self._last_pass_start is not None:
                interval = self._pass_interval()
                if interval is None:
                    return False
                elapsed = self._clock() - self._last_pass_start
                if elapsed < interval - _SYNC_CADENCE_EPS_SECONDS:
                    return False
            job = self._take_pending_locked()
        self._run_pass(job)
        return True

    def _run_pass(self, job):
        """Uma passada do modelo + voto. False quando a trabalhadora para."""
        (
            warped,
            generation,
            submitted_at,
            corners,
            visible,
            coverage,
            pass_started_at,
            scheduled_interval,
            scheduled_hand_state,
        ) = job
        # Geracao trocou (ou a trabalhadora parou) entre a retirada
        # e a execucao: a passada esta morta antes de nascer — o
        # pos-modelo a descartaria do mesmo jeito, mas pagando o
        # custo inteiro do modelo por um resultado que nunca entra
        # no canvas. Descarta aqui e conta a economia; o desfecho
        # (True segue, False para) e o mesmo do descarte tardio.
        with self._condition:
            if self._stopped:
                return False
            if generation != self._generation:
                self._telemetry_skipped_stale += 1
                return True
        # Uma excecao do modelo (shape errado, sessao morta, ONNX
        # sem memoria) matava esta thread em silencio: o canvas
        # congelava no ultimo quadro e nada era impresso. O erro
        # fica guardado, e o laco principal o reporta.
        pass_started = self._clock()
        identity_decision = None
        identity_recall = None
        identity_containment = None
        identity_reference_px = None
        identity_visible_reference_px = None
        identity_deferred = False
        identity_timed_out = False
        identity_switch_pending = False
        quarantine_reason = None
        quarantine_frames = 0
        ecc_refine_enabled = int(_ecc_refinement_enabled())
        identity_geometry_registered = False
        identity_chamfer = {}
        reg_anchor_median = _reg_anchor_median_enabled()
        reg_deadband = _reg_deadband_enabled()
        anchor_migration = _anchor_migration_enabled()
        anchor_arbitration = _anchor_arbitration_enabled()
        anchor_follow = _anchor_follow_enabled()
        force_publish = False
        reg_info = {}
        ecc_info = {
            "ecc_ms": 0.0,
            "ecc_rejected": False,
            "ecc_converged": False,
            "ecc_mode": "bypass",
        }
        try:
            if visible is None and self._hand_masker is not None:
                hand_mask = self._hand_masker.mask(warped)
                if hand_mask is not None:
                    visible = hand_mask == 0
            ink_prob = getattr(self._page_clean, "ink_prob", None)
            if ink_prob is not None:
                prob = ink_prob(warped)
            else:
                prob = (self._page_clean.ink_mask(warped) > 0).astype(np.float32)
            # Cobertura de folha (DRAW_SHEET_COVERAGE) viaja em
            # canal proprio ate depois do registro: o veto nao e
            # cego nem se confunde com oclusao de mao — ver o
            # gate de re-confirmacao abaixo, no espaco da ancora.
            pass_ended_at = self._clock()
            with self._condition:
                if self._stopped:
                    return False
                if generation != self._generation:
                    return True
                first_pass = generation != self._worker_generation
                recovering = self._identity_reference is not None and (
                    first_pass or self._identity_recovery_requested
                )
                anchor = (
                    self._identity_reference_corners
                    if recovering
                    else corners
                    if first_pass
                    else self._anchor_corners
                )
                registration_corners = corners
                if not recovering and corners is not None:
                    anchor_rebased = False
                    if (
                        reg_anchor_median
                        and not first_pass
                        and 0 < len(self._anchor_samples) < _REG_ANCHOR_MEDIAN_SAMPLES
                    ):
                        self._anchor_samples.append(corners.copy())
                        self._anchor_corners = _median_anchor(self._anchor_samples)
                        anchor = self._anchor_corners
                        anchor_rebased = True
                    if reg_anchor_median:
                        reg_info["reg_anchor_samples"] = max(
                            1 if first_pass else 0, len(self._anchor_samples)
                        )
                    if reg_deadband:
                        if self._reg_frozen_corners is None:
                            self._reg_frozen_corners = corners.copy()
                            self._reg_deadband_over = 0
                        if anchor_rebased:
                            # Rebase (interacao das duas flags,
                            # prova composta de 2026-11-10): a
                            # ancora ainda esta assentando (a
                            # janela da mediana acabou de andar);
                            # o congelado acompanha a ancora em
                            # vez de ficar preso na amostra 1, e o
                            # contador de liberacao recomeca.
                            # Fechada a janela, vale a semantica
                            # da banda morta do #57.
                            self._reg_frozen_corners = self._anchor_corners.copy()
                            self._reg_deadband_over = 0
                            registration_corners = self._reg_frozen_corners
                            shift = float(
                                np.max(
                                    np.linalg.norm(
                                        corners - self._reg_frozen_corners,
                                        axis=1,
                                    )
                                )
                            )
                            reg_info.update(
                                {
                                    "reg_deadband_shift_px": shift,
                                    "reg_deadband_held": True,
                                    "reg_deadband_pending": 0,
                                    "reg_deadband_released": False,
                                    "reg_deadband_rebased": True,
                                }
                            )
                        else:
                            (
                                registration_corners,
                                self._reg_frozen_corners,
                                self._reg_deadband_over,
                                released,
                                shift,
                            ) = _deadband_step(
                                self._reg_frozen_corners,
                                corners,
                                self._reg_deadband_over,
                            )
                            if released:
                                self._reg_deadband_releases += 1
                            reg_info.update(
                                {
                                    "reg_deadband_shift_px": shift,
                                    "reg_deadband_held": not released,
                                    "reg_deadband_pending": self._reg_deadband_over,
                                    "reg_deadband_released": released,
                                }
                            )
                confirmed = self._accumulator._confirmed
                anchor_ink = (
                    confirmed > 0
                    if confirmed is not None and confirmed.shape == prob.shape
                    else None
                )
                ecc_reference = (
                    confirmed.astype(np.float32) / 255.0
                    if (
                        confirmed is not None
                        and ink_prob is not None
                        and (not first_pass or recovering)
                    )
                    else None
                )
            # ECC nao segura o lock que submit/poll/new_generation usam.
            registered_prob, registered_visible, ecc_info = _register_ink_probability(
                prob, anchor, registration_corners, ecc_reference, visible
            )
            if coverage is not None and coverage.shape == prob.shape:
                # Gate de re-confirmacao (DRAW_SHEET_COVERAGE): onde
                # o warp nao e folha, so a tinta que a ancora JA
                # conhece vota — re-confirmar e retrair o que
                # existe, sim; tinta nova nascendo na mesa, nunca.
                # e o que separa o buraco semantico (a faixa de
                # texto que o segmentador marca como fundo na pose
                # nova, E3 2026-11-12) da banda de mesa: os dois
                # sao "fora da mascara perto dela", mas so um ja
                # era tinta. Sem tinta de ancora o gate e o veto
                # seco de sempre.
                if anchor is not None and registration_corners is not None:
                    reg_matrix = _registration_matrix(
                        anchor, registration_corners, prob.shape
                    )
                    if not _registration_is_identity(reg_matrix):
                        reg_coverage = (
                            cv2.warpPerspective(
                                coverage.astype(np.uint8),
                                reg_matrix,
                                (prob.shape[1], prob.shape[0]),
                                flags=cv2.INTER_NEAREST,
                                borderValue=0,
                            )
                            > 0
                        )
                    else:
                        reg_coverage = coverage
                else:
                    reg_coverage = coverage
                if anchor_ink is None:
                    registered_prob = registered_prob * reg_coverage
                else:
                    registered_prob = np.where(
                        reg_coverage | anchor_ink,
                        registered_prob,
                        0.0,
                    ).astype(np.float32)
            if anchor is not None and registration_corners is not None:
                matrix = _registration_matrix(anchor, registration_corners, prob.shape)
                reg_info["registration_displacement_px"] = _matrix_grid_displacement(
                    matrix, prob.shape
                )
                reg_info["registration_is_identity"] = _registration_is_identity(matrix)
        except Exception as error:  # noqa: BLE001 - vai para o laco
            with self._condition:
                self._error = error
                self._stopped = True
                self._condition.notify_all()
            return False
        with self._condition:
            if self._stopped:
                return False
            if generation != self._generation:
                return True
            first_pass = generation != self._worker_generation
            if first_pass:
                self._worker_generation = generation
                self._generation_passes = 0
                self._quiet_passes = 0
            # Degrau geometrico entre passadas consecutivas
            # (a vigia v3 so conta passada assentada).
            watch_step_px = None
            if corners is not None:
                if self._identity_watch_prev_corners is not None:
                    watch_step_px = float(
                        np.max(
                            np.linalg.norm(
                                corners - self._identity_watch_prev_corners,
                                axis=1,
                            )
                        )
                    )
                self._identity_watch_prev_corners = corners.copy()
            # A decisao de identidade roda na primeira passada e,
            # enquanto estiver em quarentena, nas seguintes
            # — a referencia so e consumida quando a decisao sai.
            identity_pending = (
                self._identity_recovery_requested
                and self._identity_reference is not None
            )
            if first_pass or identity_pending:
                identity_reference = self._identity_reference
                identity_corners = self._identity_reference_corners
                identity_recovery_requested = self._identity_recovery_requested
                keep_page = False
                if identity_reference is not None:
                    candidate = registered_prob
                    identity_visible = registered_visible
                    if identity_corners is not None and corners is not None:
                        identity_geometry_registered = True
                    prior = identity_reference > 0
                    prior_px = int(np.count_nonzero(prior))
                    identity_reference_px = prior_px
                    if prior_px == 0:
                        seen_any = candidate > self._accumulator.threshold
                        if identity_visible is not None:
                            seen_any = seen_any & identity_visible
                        keep_page = not np.any(seen_any)
                        identity_recall = 1.0 if keep_page else 0.0
                    else:
                        seen = candidate > self._accumulator.threshold
                        kernel = np.ones((_IDENTITY_DILATION_KERNEL,) * 2, np.uint8)
                        seen_d = cv2.dilate(seen.astype(np.uint8), kernel) > 0
                        if identity_visible is not None:
                            prior_visible = prior & identity_visible
                            prior_visible_px = int(np.count_nonzero(prior_visible))
                            identity_visible_reference_px = prior_visible_px
                            if prior_visible_px < max(
                                _IDENTITY_MIN_VISIBLE_PX,
                                int(_IDENTITY_MIN_VISIBLE_FRACTION * prior_px),
                            ):
                                # Tinta anterior quase toda
                                # coberta: decidir agora seria
                                # condenar a pagina pela mao.
                                # Adia; a referencia fica para a
                                # proxima passada.
                                identity_deferred = True
                                quarantine_reason = "occlusion"
                            else:
                                prior_d = (
                                    cv2.dilate(
                                        prior_visible.astype(np.uint8),
                                        kernel,
                                    )
                                    > 0
                                )
                                identity_recall = (
                                    np.count_nonzero(prior_visible & seen_d)
                                    / prior_visible_px
                                )
                                seen_px = int(np.count_nonzero(seen))
                                identity_containment = (
                                    0.0
                                    if seen_px == 0
                                    else (np.count_nonzero(seen & prior_d) / seen_px)
                                )
                                keep_page = (
                                    identity_recall >= _PAGE_IDENTITY_MIN_RECALL
                                    and identity_containment
                                    >= _PAGE_IDENTITY_MIN_CONTAINMENT
                                )
                        else:
                            prior_d = cv2.dilate(prior.astype(np.uint8), kernel) > 0
                            identity_recall = (
                                np.count_nonzero(prior & seen_d) / prior_px
                            )
                            seen_px = int(np.count_nonzero(seen))
                            identity_containment = (
                                0.0
                                if seen_px == 0
                                else (np.count_nonzero(seen & prior_d) / seen_px)
                            )
                            keep_page = (
                                identity_recall >= _PAGE_IDENTITY_MIN_RECALL
                                and identity_containment
                                >= _PAGE_IDENTITY_MIN_CONTAINMENT
                            )
                if (
                    identity_reference is not None
                    and ecc_refine_enabled
                    and ecc_info.get("ecc_rejected")
                ):
                    identity_deferred = True
                    quarantine_reason = "ecc_rejected"
                if identity_deferred:
                    self._identity_defer_count += 1
                    quarantine_frames = self._identity_defer_count
                    identity_timed_out = (
                        quarantine_frames > _PAGE_IDENTITY_MAX_DEFERRALS
                    )
                if identity_deferred:
                    self._identity_switch_votes = 0
                elif identity_reference is not None and not keep_page:
                    self._identity_switch_votes += 1
                    identity_switch_pending = (
                        self._identity_switch_votes
                        < _PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES
                    )
                else:
                    self._identity_switch_votes = 0
                if identity_recovery_requested:
                    if identity_timed_out:
                        identity_decision = "quarantine_timeout"
                    elif quarantine_reason == "ecc_rejected":
                        identity_decision = "deferred_ecc_rejected"
                    elif identity_deferred:
                        identity_decision = "deferred_occlusion"
                    elif identity_switch_pending:
                        identity_decision = "deferred_switch_consensus"
                    else:
                        identity_decision = "same_page" if keep_page else "new_page"
                    if _identity_chamfer_enabled():
                        metric_seen = metric_reference = None
                        if identity_reference is not None:
                            metric_seen = candidate > self._accumulator.threshold
                            metric_reference = (
                                prior
                                if identity_visible is None
                                else prior & identity_visible
                            )
                        identity_chamfer = _identity_chamfer_fields(
                            metric_seen,
                            metric_reference,
                            identity_decision,
                            ecc_info,
                        )
                    self._identity_events.append(
                        {
                            "generation": generation,
                            "decision": identity_decision,
                            "recall": identity_recall,
                            "containment": identity_containment,
                            "reference_px": identity_reference_px,
                            "visible_reference_px": (identity_visible_reference_px),
                            "geometry_registered": (identity_geometry_registered),
                            "threshold": _PAGE_IDENTITY_MIN_RECALL,
                            "containment_threshold": (_PAGE_IDENTITY_MIN_CONTAINMENT),
                            **ecc_info,
                            **identity_chamfer,
                            "ecc_refine_enabled": ecc_refine_enabled,
                            "quarantine_reason": quarantine_reason,
                            "quarantine_frames": quarantine_frames,
                            "deferred": identity_deferred,
                        }
                    )
                if (
                    identity_deferred and not identity_timed_out
                ) or identity_switch_pending:
                    # A referencia NAO e consumida: a proxima
                    # passada decide com registro e visao
                    # confiaveis e consenso. ECC rejeitado nao vota.
                    pass
                else:
                    if identity_timed_out:
                        # Perda de lock, nao voto de pagina nova.
                        # A proxima passada estabelece outra ancora.
                        self._accumulator.reset()
                        self._anchor_corners = None
                        self._anchor_samples = []
                        self._reg_frozen_corners = None
                        self._reg_deadband_over = 0
                        self._migration_streak = 0
                        self._migration_quads = []
                        self._arb_event_open = False
                        self._arb_streak_quads = []
                        self._arb_scores = []
                        self._arb_snapshot = None
                        self._follow_event_open = False
                        self._follow_streak = 0
                        self._follow_streak_quads = []
                        self._follow_hold = False
                        self._worker_generation = -1
                        self._generation_passes = 0
                        self._quiet_passes = 0
                        self._still_passes = 0
                    elif not keep_page:
                        if (
                            self._identity_armed_mid_generation
                            and self._accumulator._confirmed is not None
                        ):
                            # Divisao no meio da geracao (vigia,
                            # frente 28): a pagina que acabou sai
                            # como evento pro app salvar ANTES do
                            # reset — sem isto o canvas terminado
                            # se perdia com o reset do veredito.
                            self._identity_split_event = (
                                generation,
                                self._accumulator._confirmed.copy(),
                                self._identity_arm_reason,
                            )
                        self._accumulator.reset()
                        self._anchor_corners = corners
                        self._anchor_samples = (
                            [corners.copy()]
                            if reg_anchor_median and corners is not None
                            else []
                        )
                    # Keep the old anchor when identity matched: all future
                    # observations remain registered into the same canvas.
                    self._identity_reference = None
                    self._identity_reference_corners = None
                    self._identity_recovery_requested = False
                    self._identity_defer_count = 0
                    self._identity_switch_votes = 0
                    self._identity_armed_mid_generation = False
                    self._identity_arm_reason = None
            identity_owned = (
                first_pass
                or identity_pending
                or identity_deferred
                or identity_switch_pending
            )
            arbitration_abstain = False
            arb_event = None
            if anchor_arbitration:
                arb_status, arb_event = self._anchor_arbitration_step(
                    generation,
                    corners,
                    prob.shape,
                    identity_owned,
                    reg_anchor_median,
                    reg_deadband,
                    prob,
                    registered_prob,
                    visible,
                )
                arbitration_abstain = arb_status == "open"
                reg_info["arb_event_open"] = self._arb_event_open
                reg_info["arb_abstained"] = arbitration_abstain
                reg_info["arb_streak"] = len(self._arb_streak_quads)
                if self._arb_scores:
                    scores = self._arb_scores[-1]
                    reg_info["arb_pass_score_page"] = scores[0]
                    reg_info["arb_pass_score_camera"] = scores[1]
                    reg_info["arb_pass_scores_cyclic"] = list(scores[2:5])
                if arb_event is not None:
                    reg_info["arbitration_event"] = dict(arb_event)
                    reg_info["arb_verdict"] = arb_event["verdict"]
                    reg_info["arb_score_page"] = arb_event["score_page"]
                    reg_info["arb_score_camera"] = arb_event["score_camera"]
                    reg_info["arb_rotation_k"] = arb_event["rotation_k"]
                if (
                    arb_status == "verdict"
                    and arb_event is not None
                    and arb_event["action"]
                ):
                    # Pouso: a passada corrente entra ja
                    # registrada na ancora nova, por cima do
                    # estado INTOCADO (a v3 nao move o estado;
                    # com ECC ligado, a referencia e o canvas
                    # como esta).
                    arb_reference = (
                        None
                        if ecc_reference is None
                        else self._accumulator._confirmed.astype(np.float32) / 255.0
                    )
                    registered_prob, registered_visible, ecc_info = (
                        _register_ink_probability(
                            prob,
                            self._anchor_corners,
                            (self._reg_frozen_corners if reg_deadband else corners),
                            arb_reference,
                            visible,
                        )
                    )
                    force_publish = True
            elif anchor_migration and not anchor_follow:
                migration = self._anchor_migration_step(
                    generation,
                    corners,
                    prob.shape,
                    first_pass or identity_pending,
                    reg_anchor_median,
                    reg_deadband,
                )
                if migration is not None:
                    # A passada corrente entra por cima do estado
                    # migrado, ja registrada na ancora nova (com ECC
                    # contra o canvas migrado, se ligado). Raro: uma
                    # vez por pouso, entao pode segurar o lock.
                    migrated_reference = (
                        None
                        if ecc_reference is None
                        else self._accumulator._confirmed.astype(np.float32) / 255.0
                    )
                    registered_prob, registered_visible, ecc_info = (
                        _register_ink_probability(
                            prob,
                            self._anchor_corners,
                            (self._reg_frozen_corners if reg_deadband else corners),
                            migrated_reference,
                            visible,
                        )
                    )
                    force_publish = True
                reg_info["anchor_migration_streak"] = self._migration_streak
                reg_info["anchor_migrated"] = migration is not None
            follow_abstain = False
            if anchor_follow:
                # Mobilidade da ancora (frente 34): roda DEPOIS
                # da arbitragem — o veredito dela nesta passada
                # (arb_event) e a precedencia que o passo le.
                follow_status, follow_event = self._anchor_follow_step(
                    generation,
                    corners,
                    prob.shape,
                    identity_owned,
                    reg_anchor_median,
                    reg_deadband,
                    watch_step_px,
                    arb_event,
                )
                follow_abstain = follow_status == "open"
                reg_info["follow_event_open"] = self._follow_event_open
                reg_info["follow_abstained"] = follow_abstain
                reg_info["follow_streak"] = self._follow_streak
                reg_info["follow_hold"] = self._follow_hold
                if follow_event is not None:
                    reg_info["follow_reanchored"] = True
                    reg_info["follow_displacement_px"] = follow_event["displacement_px"]
                    reg_info["follow_closed_arb_event"] = follow_event[
                        "closed_arb_event"
                    ]
                    if follow_event["closed_arb_event"]:
                        reg_info["arb_event_open"] = False
                    # A passada corrente entra ja registrada na
                    # ancora nova, por cima do estado INTOCADO
                    # (a troca e so de ancora; com ECC ligado, a
                    # referencia e o canvas como esta) — e o
                    # evento de arbitragem que a abstinha foi
                    # fechado pela re-ancoragem.
                    follow_reference = (
                        None
                        if ecc_reference is None
                        else self._accumulator._confirmed.astype(np.float32) / 255.0
                    )
                    registered_prob, registered_visible, ecc_info = (
                        _register_ink_probability(
                            prob,
                            self._anchor_corners,
                            (self._reg_frozen_corners if reg_deadband else corners),
                            follow_reference,
                            visible,
                        )
                    )
                    arbitration_abstain = False
                    force_publish = True
            identity_watch_suspect = False
            if (
                _identity_watch_enabled()
                and not identity_pending
                and not first_pass
                and not self._arb_event_open
            ):
                # Vigia de conteudo (frente 28): sem julgamento
                # em curso, ruptura sustentada com geometria
                # quieta arma o juiz de identidade. A passada
                # corrente ainda nao votou neste ponto.
                (
                    reg_info["identity_watch_recall"],
                    identity_watch_suspect,
                ) = self._identity_watch_step(
                    registered_prob, registered_visible, corners, watch_step_px
                )
                reg_info["identity_watch_streak"] = self._identity_watch_streak
            content_blocked = False
            if _experimental_enabled("DRAW_REG_CONTENT_GUARD"):
                registered_prob, registered_visible, content_info = (
                    self._content_registration_guard(
                        prob,
                        registered_prob,
                        visible,
                        registered_visible,
                        corners,
                        identity_owned
                        or arbitration_abstain
                        or follow_abstain
                        or identity_watch_suspect,
                    )
                )
                content_blocked = content_info["reg_content_blocked"]
                force_publish = force_publish or content_info.get(
                    "reg_content_identity_won", False
                )
                reg_info.update(content_info)
            if (
                (first_pass or identity_pending)
                and not identity_deferred
                and not identity_switch_pending
                and not keep_page
            ):
                if visible is not None:
                    prob = np.where(visible, prob, 0.0).astype(np.float32)
            else:
                prob = registered_prob
                visible = registered_visible
            if identity_switch_pending or (
                identity_deferred
                and (quarantine_reason == "ecc_rejected" or identity_timed_out)
            ):
                visible = np.zeros(prob.shape, dtype=bool)
            if arbitration_abstain:
                # Evento de arbitragem aberto: a passada e
                # registrada e pontuada, mas nao vota.
                visible = np.zeros(prob.shape, dtype=bool)
            if follow_abstain:
                # Evento de divergencia assentada aberto: a
                # passada foi medida, mas nao vota — registrada
                # na ancora velha, ela depositaria a copia
                # deslocada que a re-ancoragem existe para
                # nao fabricar.
                visible = np.zeros(prob.shape, dtype=bool)
            if identity_watch_suspect:
                # Suspeita de ruptura (vigia): a passada mediu o
                # colapso do recall ou armou o juiz; ela nao
                # vota, para a tinta da pagina nova nao
                # confirmar no canvas da antiga antes do
                # veredito de identidade.
                visible = np.zeros(prob.shape, dtype=bool)
            result = self._accumulator.add(
                prob,
                visible,
                removal_allowed=(
                    not identity_owned
                    and not identity_watch_suspect
                    and not arbitration_abstain
                    and not follow_abstain
                    and not content_blocked
                    and not ecc_info.get("ecc_rejected")
                    and corners is not None
                    and watch_step_px is not None
                    and watch_step_px <= _ARB_SETTLE_PX
                    and reg_info.get("registration_displacement_px", float("inf"))
                    <= _ANCHOR_MIGRATION_DISPLACEMENT_PX
                ),
            )
            if result.get("ink_vote_e3"):
                # Telemetria da regra E3 (so com DRAW_INK_VOTE_E3
                # ligada): fase da geracao e trilhas abertas.
                reg_info["ink_vote_e3_passes"] = result["ink_vote_e3_passes"]
                reg_info["ink_vote_e3_tracks_big"] = result["ink_vote_e3_tracks_big"]
                reg_info["ink_vote_e3_tracks_small"] = result[
                    "ink_vote_e3_tracks_small"
                ]
            if result.get("ink_vote_e4"):
                # Telemetria da regra E5-S (so com
                # DRAW_INK_VOTE_E4 ligada): fase e trilhas.
                reg_info["ink_vote_e4_passes"] = result["ink_vote_e4_passes"]
                reg_info["ink_vote_e4_tracks_big"] = result["ink_vote_e4_tracks_big"]
                reg_info["ink_vote_e4_tracks_small"] = result[
                    "ink_vote_e4_tracks_small"
                ]
                reg_info["ink_vote_e4_tracks_slow"] = result["ink_vote_e4_tracks_slow"]
                reg_info["ink_vote_e4_tracks_thin"] = result["ink_vote_e4_tracks_thin"]
                reg_info["ink_vote_e4_tracks_sub"] = result["ink_vote_e4_tracks_sub"]
            if result.get("ink_vote_fast"):
                # Telemetria do corte de latencia (so com
                # DRAW_INK_VOTE_FAST ligada).
                reg_info["ink_vote_fast"] = True
            if (
                result["confirmed_changed"]
                or result["provisional_changed"]
                or result["turn"]
            ):
                self._still_passes = 0
                self._activity_quiet_passes = 0
            else:
                self._still_passes += 1
                self._activity_quiet_passes += 1
            # Contadores do descanso total: so a saida
            # CONFIRMADA interessa (provisoria tremula com o
            # warp e nao pode segurar a trabalhadora acordada).
            self._generation_passes += 1
            if result["confirmed_changed"] or result["turn"]:
                self._quiet_passes = 0
            else:
                self._quiet_passes += 1
            self._passes_total += 1
            self._last_pass_ms = (self._clock() - pass_started) * 1000.0
            self._provisional_px = result["provisional_px"]
            self._confirmed_px = result["confirmed_px"]
            self._retracted_px_total = result["retracted_px_total"]
            # Virada: o acumulador detecta a cobertura sobre o
            # canvas confirmado; o evento so e guardado atras
            # de _page_turn_event_enabled(), desligado por
            # padrao (hotfix 2026-11-01).
            if (
                result["turn"]
                and _page_turn_event_enabled()
                and self._turn_event is None
            ):
                self._turn_event = (
                    generation,
                    result["confirmed"],
                )
            publish_confirmed = (
                first_pass
                or identity_timed_out
                or identity_decision == "new_page"
                or result["confirmed_changed"]
                # Migracao: o canvas migrado nao "muda" para o
                # voto, mas a tela mostraria a grade velha.
                or force_publish
            )
            if publish_confirmed:
                self._version += 1
                self._published = (
                    generation,
                    self._version,
                    result["confirmed"],
                )
            if publish_confirmed or result["provisional_changed"]:
                self._display_version += 1
                self._published_display = (
                    generation,
                    self._display_version,
                    result["confirmed"],
                    result["provisional"],
                )
            # Telemetria por etapa (loop v2) sobre o caminho em
            # camadas: um registro por passada; publishes conta
            # so as passadas que publicam canvas confirmado
            # novo, e o campo de mascara da passada vira
            # confirmed_px (a uniao nao existe mais).
            published_at = self._clock()
            self._telemetry_passes += 1
            if publish_confirmed:
                self._telemetry_publishes += 1
            self._telemetry_records.append(
                {
                    **ecc_info,
                    **identity_chamfer,
                    **reg_info,
                    "ecc_refine_enabled": ecc_refine_enabled,
                    "anchor_arbitration_enabled": anchor_arbitration,
                    "sequence": self._telemetry_passes,
                    "pass_gap_s": None
                    if self._previous_pass_started_at is None
                    else pass_started_at - self._previous_pass_started_at,
                    "corner_step_px": watch_step_px,
                    "quarantine_reason": quarantine_reason,
                    "quarantine_frames": quarantine_frames,
                    "deferred": identity_deferred,
                    "generation": generation,
                    "version": self._version,
                    "page_identity_decision": identity_decision,
                    "page_identity_recall": identity_recall,
                    "page_identity_containment": identity_containment,
                    "page_identity_reference_px": identity_reference_px,
                    "page_identity_visible_reference_px": (
                        identity_visible_reference_px
                    ),
                    "page_identity_geometry_registered": (identity_geometry_registered),
                    "submitted_at": submitted_at,
                    "pass_started_at": pass_started_at,
                    "pass_ended_at": pass_ended_at,
                    "published_at": published_at,
                    "queue_wait_s": pass_started_at - submitted_at,
                    "pass_s": pass_ended_at - pass_started_at,
                    "submit_to_publish_s": published_at - submitted_at,
                    "interval_s": scheduled_interval,
                    "hand_state": scheduled_hand_state,
                    "activity_quiet_passes": self._activity_quiet_passes,
                    "mask_px": int(
                        np.count_nonzero(prob > self._accumulator.threshold)
                    ),
                    "confirmed_px": result["confirmed_px"],
                    "new_confirmed_px": result["new_confirmed_px"],
                    "removed_confirmed_px": result.get("removed_confirmed_px", 0),
                    "visible_fraction": 1.0
                    if visible is None
                    else float(np.mean(visible)),
                }
            )
            self._previous_pass_started_at = pass_started_at
            self._condition.notify_all()
        return True

    def stats(self):
        """Contadores para a telemetria: passadas e custo da ultima."""
        with self._condition:
            return {
                "passes_total": self._passes_total,
                "pass_ms": (
                    None if self._last_pass_ms is None else round(self._last_pass_ms, 1)
                ),
                "canvas_px": self._confirmed_px,
                "provisional_px": self._provisional_px,
                "retracted_px_total": self._retracted_px_total,
                "worker_mode": (
                    "rest"
                    if self._at_rest()
                    else "active"
                    if self._hand_state
                    else "idle"
                    if self._hand_state is False
                    else "unknown"
                ),
                "resting": self._at_rest(),
                "hand_present": self._hand_state,
            }

    @property
    def error(self):
        """A excecao que matou a trabalhadora, ou None."""
        with self._condition:
            return self._error

    def stop(self):
        # O teto cobre uma passada em voo: a trabalhadora so ve o
        # stop entre passadas, e sair com ela dentro do ONNX
        # derruba o processo na finalizacao (visto na bancada
        # desta VM, passada de segundos). No Ryzen a passada e
        # 168 ms e o join volta muito antes do teto.
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join(timeout=_PAGE_CLEAN_STOP_TIMEOUT_SECONDS)


def _render_layered_canvas(confirmed, provisional, out=None):
    """Canvas de exibicao: confirmada em preto, provisoria em cinza.

    A provisoria e tinta que a primeira passada viu e o voto
    ainda nao confirmou. Ela aparece mais clara para quem desenha
    saber que pode sumir, e nunca entra no canvas que poll()
    entrega aos consumidores.
    """
    canvas = ink_module.render_canvas(confirmed, out=out)
    if provisional is not None:
        light = (provisional > 0) & (confirmed == 0)
        if np.any(light):
            canvas[light] = (176, 176, 176)
    return canvas
