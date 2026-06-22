"""
Bot de Poker — player_bdHANDY (v2, Fase 2)
==========================================

Um jogador profissional heads-up: Tight-Aggressive (TAG) com small-ball
disciplinado e adaptacao por oponente entre partidas.

Fase 2: nucleo escolhido por bake-off empirico contra o campo real (venceu a
versao anterior por +9pp de WR medio). Sobre o nucleo TAG ha UMA melhoria de
exploit, gatilhada por COMPORTAMENTO observado (nao por nome do oponente):
o ramo "calling station" — quando o oponente paga muito nossos raises
(fold_eq baixo) mas raramente toma a iniciativa, zeramos o blefe, afinamos e
engrossamos o valor (ele paga com pior) e mantemos disciplina de fold contra a
agressao polarizada dele (margem maior). E upside-only: so dispara com leitura
de fold_eq baixa; contra qualquer outro perfil, joga o baseline TAG.

Como ESTE motor realmente funciona (verificado em src/game/game.py):
  - O flop (3 cartas) ja esta visivel na PRIMEIRA rodada de apostas. Os estados
    de board sao: 3 cartas (duas rodadas), 4 cartas (turn), 5 cartas (river).
    Logo, NAO existe "pre-flop as cegas": decidimos sempre com uma mao de 5
    cartas formada. A selecao de maos e baseada em equity desde a 1a decisao.
  - `current_bet` PERSISTE entre as ruas dentro da mesma mao (so zera a cada
    nova mao). Quem so paga, paga `current_bet` (>= big blind) TODA rua — nao
    existe check gratis pos-flop. Cada decisao e: pagar a aposta de pe, aumentar,
    ou foldar. Por isso: foldar barato quando atras e pressionar com a aposta que
    "carrega" para a proxima rua sao alavancas centrais.
  - O dealer e o BIG BLIND e age por ULTIMO em todas as ruas (esta em posicao).
    A posicao alterna a cada mao; nosso assento e fixo dentro de uma partida.
  - Uma instancia nova e criada por partida, MAS o objeto de classe persiste por
    todas as 2000 partidas de um confronto -> memoria de classe permite aprender
    o oponente entre partidas (a alavanca contra os "nits").

Restricoes: apenas stdlib permitida + modulos da engine. Trabalho por decisao e
limitado (<= 21 avaliacoes de 5 cartas) para nunca estourar o timeout de 50 ms.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import random
from collections import Counter
from itertools import combinations

from players.player import Player
from game.game_view import GameView
from cards.cards import Hand
from cards.sequences import (
    score_cinco_cartas,
    avaliar_cinco_cartas,
    VALORES,
    RANK_CARTA_ALTA,
    RANK_UM_PAR,
    RANK_DOIS_PARES,
    RANK_TRINCA,
    RANK_STRAIGHT,
    RANK_FLUSH,
    RANK_FULL_HOUSE,
    RANK_QUADRA,
)


# ─────────────────────────────────────────────────────────────────────────────
# Avaliador inteiro rapido (Tier 4: Monte Carlo de equity dentro do budget)
#   carta -> int = rank*4 + naipe ; _eval5/_eval7 dao score comparavel.
# ─────────────────────────────────────────────────────────────────────────────
_R2I = {"2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, "9": 7,
        "10": 8, "J": 9, "Q": 10, "K": 11, "A": 12}
_S2I = {"s": 0, "h": 1, "d": 2, "c": 3}
_ALL52 = tuple(range(52))
_B5 = 15 ** 5


def _ci(card) -> int:
    return _R2I[card.value] * 4 + _S2I[card.suit]


def _eval5(cs) -> int:
    r = [c >> 2 for c in cs]
    r.sort(reverse=True)
    counts = [0] * 13
    for x in r:
        counts[x] += 1
    maxc = max(counts)
    flush = (cs[0] & 3) == (cs[1] & 3) == (cs[2] & 3) == (cs[3] & 3) == (cs[4] & 3)
    straight = False
    high = r[0]
    if maxc == 1:
        if r[0] - r[4] == 4:
            straight = True
        elif r[0] == 12 and r[1] == 3 and r[2] == 2 and r[3] == 1 and r[4] == 0:
            straight = True
            high = 3
    if flush and straight:
        return 8 * _B5 + high
    if maxc == 4:
        quad = counts.index(4)
        kick = next(x for x in r if x != quad)
        return 7 * _B5 + quad * 15 + kick
    trip = pa = pb = -1
    for x in range(12, -1, -1):
        c = counts[x]
        if c == 3 and trip == -1:
            trip = x
        elif c == 2:
            if pa == -1:
                pa = x
            elif pb == -1:
                pb = x
    if trip >= 0 and pa >= 0:
        return 6 * _B5 + trip * 15 + pa
    if flush:
        s = 0
        for x in r:
            s = s * 15 + x
        return 5 * _B5 + s
    if straight:
        return 4 * _B5 + high
    if trip >= 0:
        ks = [x for x in r if x != trip][:2]
        return 3 * _B5 + trip * 225 + ks[0] * 15 + ks[1]
    if pa >= 0 and pb >= 0:
        k = next(x for x in r if x != pa and x != pb)
        return 2 * _B5 + pa * 225 + pb * 15 + k
    if pa >= 0:
        ks = [x for x in r if x != pa][:3]
        return 1 * _B5 + pa * 3375 + ks[0] * 225 + ks[1] * 15 + ks[2]
    s = 0
    for x in r:
        s = s * 15 + x
    return s


def _eval7(seven) -> int:
    best = 0
    for combo in combinations(seven, 5):
        s = _eval5(combo)
        if s > best:
            best = s
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Modelo de oponente — acumulado entre partidas (memoria de classe)
# ─────────────────────────────────────────────────────────────────────────────
class _Opp:
    """Estatisticas robustas e diretamente observaveis de um oponente.

    Nada de metricas "censuradas" (ex: 'meu pote grande venceu?'): so contamos
    coisas que enxergamos com certeza dentro de decision()."""

    def __init__(self) -> None:
        self.hands = 0            # maos observadas
        self.my_raises = 0        # vezes que NOS demos um raise que exigiu resposta
        self.fold_to_raise = 0    # dessas, quantas o oponente foldou
        self.opp_actions = 0      # decisoes do oponente observadas
        self.opp_aggr = 0         # dessas, quantas foram raise/aposta do oponente
        self.deep_hands = 0       # maos que chegaram a 4/5 cartas (turn/river)
        self.ag_fast = 0.5        # T3: EMA rapida de agressao (reage rapido)
        self.ag_slow = 0.5        # T3: EMA lenta (leitura estavel)

    def shift(self) -> float:
        """T3 (anti-poison): divergencia fast-slow = oponente MUDANDO de estilo
        (ex.: camaleao que finge passividade e depois blefa). Usado p/ amortecer
        nossos desvios e nao sermos envenenados."""
        return abs(self.ag_fast - self.ag_slow)

    # frequencia com que o oponente fica passivo/foldando vs nossa pressao
    def fold_eq(self) -> float:
        if self.my_raises < 1:
            return 0.5
        return self.fold_to_raise / self.my_raises

    def aggression(self) -> float:
        if self.opp_actions < 1:
            return 0.5
        return self.opp_aggr / self.opp_actions

    def deep_rate(self) -> float:
        if self.hands < 1:
            return 0.5
        return self.deep_hands / self.hands

    def archetype(self) -> str:
        """Classifica so quando ha amostra suficiente; senao 'unknown'."""
        if self.hands < 12 or self.my_raises < 6:
            return "unknown"
        fe = self.fold_eq()
        ag = self.aggression()
        if ag > 0.52:
            return "maniac"
        if fe > 0.58 and ag < 0.42:
            return "nit"
        if fe < 0.26:
            return "station"
        return "reg"


class BdHandy(Player):
    """TAG profissional com small-ball anti-nit e adaptacao por oponente."""

    # memoria compartilhada entre todas as partidas do confronto
    _MEM: dict[str, _Opp] = {}

    # ── parametros base (ajustaveis) ────────────────────────────────────────
    _PUSHFOLD_BB = 12       # abaixo disso entra em push/fold
    _BASE_BLUFF = 0.08      # frequencia base de blefe quando temos fold equity
    _BASE_VALUE_THR = 0.58  # equity minima para tratar como value
    _ODDS_MARGIN = 0.05     # folga exigida sobre pot odds para pagar

    # ── TOGGLES DE HARDENING ANTI-EXPLOIT (Fase 2) ───────────────────────────
    # A/B-testaveis via test_tiers.py. Default: so o T1a (jitter) validado ON.
    USE_SIZE_JITTER   = True    # T1a: descorrelaciona tamanho<->forca (VALIDADO)
    USE_BOUNDARY_MIX  = False   # T1b: borra limiares call/fold/raise (anti-leitura)
    USE_BLUFF_BALANCE = False   # T2 : range de blefe balanceado + bluff-catch
    USE_ANTIPOISON    = False # T3 : amortece desvios qd opp muda de estilo (camaleao)
    USE_RANGE_MC      = False  # T4 : equity por Monte Carlo (mistura no _equity)

    def __init__(self, name, hand, chips) -> None:
        super().__init__(name, hand, chips)
        self._my_seat: int | None = None     # assento absoluto (0/1), fixo na partida
        self._opp: _Opp | None = None        # perfil do oponente (memoria de classe)
        self._opp_name: str | None = None

        # estado da mao corrente
        self._prev_board = -1
        self._prev_pot = -1     # o pote so cresce dentro de uma mao; cai -> nova mao
        self._hand_open = False
        self._raised_this_hand = False        # demos algum raise nesta mao?
        self._i_folded = False                # nos desistimos nesta mao?
        self._saw_deep = False                # vimos board 4/5 nesta mao?
        self._raise_board = -1                # nivel do board no nosso ultimo raise
        self._last_seen_board = -1            # ultimo nivel de board em que agimos
        self._start_chips = chips             # fichas no inicio da mao

    # ── utilidades de cartas ────────────────────────────────────────────────
    @staticmethod
    def _v(card) -> int:
        return VALORES[card.value]

    def _best_score(self, cards) -> int:
        """Melhor score de 5 cartas dentre as `cards` (5, 6 ou 7 cartas)."""
        if len(cards) < 5:
            return 0
        if len(cards) == 5:
            return score_cinco_cartas(list(cards))
        best = 0
        for cinco in combinations(cards, 5):
            s = score_cinco_cartas(list(cinco))
            if s > best:
                best = s
        return best

    def _best_rank(self, cards) -> int:
        """Rank (categoria) da melhor mao de 5 dentre `cards`."""
        if len(cards) < 5:
            return RANK_CARTA_ALTA
        best_rank = -1
        for cinco in combinations(cards, 5):
            r, _ = avaliar_cinco_cartas(list(cinco))
            if r > best_rank:
                best_rank = r
        return best_rank

    def _count_outs(self, hole, board) -> int:
        """Outs aproximados para flush/straight draw + overcards. Limitado."""
        cards = list(hole) + list(board)
        outs = 0
        # flush draw: 4 do mesmo naipe
        suits = Counter(c.suit for c in cards)
        for s, q in suits.items():
            if q == 4:
                outs = max(outs, 9)
        # straight draw
        vals = sorted({self._v(c) for c in cards})
        # considera A como 1 tambem (wheel)
        if 14 in vals:
            vals = sorted(set(vals) | {1})
        run_outs = 0
        for low in range(1, 11):  # janelas de 5 valores consecutivos
            window = set(range(low, low + 5))
            have = len(window & set(vals))
            if have == 4:
                run_outs = max(run_outs, 8)  # open-ended-ish
            elif have == 3 and run_outs < 4:
                pass
        outs = max(outs, run_outs)
        # overcards (so quando nao temos par feito) — valor pequeno
        return min(outs, 15)

    # ── motor de equity (deterministico, barato) ────────────────────────────
    def _equity(self, gv: GameView):
        """Retorna (equity, rank, draw_equity, plays_board)."""
        hole = list(gv.my_hand)
        board = list(gv.board)
        cards = hole + board
        bl = len(board)

        rank = self._best_rank(cards)

        # "jogar o board": minha melhor mao usa 0 cartas da mao (so empata)
        plays_board = False
        if bl >= 5:
            board_score = self._best_score(board)
            full_score = self._best_score(cards)
            if full_score <= board_score:
                plays_board = True

        made = self._made_equity(hole, board, rank)
        if plays_board:
            made = min(made, 0.16)

        # equity de draw (cartas por vir): board 3 -> 2 cartas, board 4 -> 1
        draw_eq = 0.0
        if bl < 5 and rank <= RANK_TRINCA:
            outs = self._count_outs(hole, board)
            mult = 4 if bl == 3 else 2
            draw_eq = min(0.85, outs * mult / 100.0)

        equity = max(made, made + 0.5 * max(0.0, draw_eq - 0.12))
        # T4: mistura equity por Monte Carlo (vs range uniforme) nos spots com
        # board >= 3. Mais preciso em spots fechados; gated por flag.
        if self.USE_RANGE_MC and bl >= 3:
            mc = self._mc_equity(hole, board, {3: 120, 4: 160, 5: 200}.get(bl, 120))
            if mc is not None:
                equity = 0.5 * equity + 0.5 * mc
        equity = max(0.02, min(0.99, equity))
        return equity, rank, draw_eq, plays_board

    def _mc_equity(self, hole, board, trials):
        """T4: equity REAL por Monte Carlo — amostra a mao do opp (range uniforme)
        + cartas que faltam, mede a fracao de vitorias (empate=0.5). Avaliador
        inteiro rapido (~50us/amostra) p/ caber no budget de 50 ms."""
        try:
            my2 = [_ci(c) for c in hole]
            b = [_ci(c) for c in board]
            known = set(my2) | set(b)
            rem = [c for c in _ALL52 if c not in known]
            need = 5 - len(b)
            win = 0.0
            sample = random.sample
            for _ in range(trials):
                s = sample(rem, 2 + need)
                full = b + s[2:]
                ms = _eval7(my2 + full)
                os_ = _eval7(s[:2] + full)
                win += 1.0 if ms > os_ else (0.5 if ms == os_ else 0.0)
            return win / trials
        except Exception:
            return None

    def _made_equity(self, hole, board, rank) -> float:
        """Equity heuristica da mao FEITA contra 1 oponente que continua."""
        if rank >= RANK_QUADRA:
            return 0.98
        if rank == RANK_FULL_HOUSE:
            return 0.93
        if rank == RANK_FLUSH:
            return 0.88
        if rank == RANK_STRAIGHT:
            return 0.82
        if rank == RANK_TRINCA:
            return 0.77
        if rank == RANK_DOIS_PARES:
            return 0.66
        if rank == RANK_UM_PAR:
            return self._pair_equity(hole, board)
        # carta alta
        hi = max(self._v(c) for c in hole)
        return 0.24 + (hi - 2) * 0.010   # ~0.24..0.36

    def _pair_equity(self, hole, board) -> float:
        """Distingue overpair / top / middle / weak / par de bolso baixo."""
        board_vals = sorted((self._v(c) for c in board), reverse=True)
        h0, h1 = self._v(hole[0]), self._v(hole[1])
        pocket = (h0 == h1)
        top_board = board_vals[0] if board_vals else 0

        if pocket:
            if h0 > top_board:
                return 0.70          # overpair
            return 0.46              # par de bolso abaixo do board
        # par com o board: qual valor pareou?
        paired_val = 0
        for hv in (h0, h1):
            if hv in board_vals:
                paired_val = max(paired_val, hv)
        kicker = max(h0, h1) if max(h0, h1) != paired_val else min(h0, h1)
        if paired_val == 0:
            return 0.30              # sem par real (fallback)
        if paired_val >= top_board:
            return 0.60 + min(0.06, (kicker - 9) * 0.01 if kicker > 9 else 0.0)
        if paired_val >= (board_vals[1] if len(board_vals) > 1 else 0):
            return 0.48              # middle pair
        return 0.40                  # par fraco / baixo

    # ── posicao ─────────────────────────────────────────────────────────────
    def _in_position(self, gv: GameView) -> bool:
        if self._my_seat is None:
            return gv.opponents[0].current_bet_in_round > 0
        return gv.dealer_position == self._my_seat

    def _lock_seat(self, gv: GameView) -> None:
        if self._my_seat is not None:
            return
        ip = gv.opponents[0].current_bet_in_round > 0
        self._my_seat = gv.dealer_position if ip else (1 - gv.dealer_position)

    # ── rastreio de mao / oponente ──────────────────────────────────────────
    def _bind_opp(self, gv: GameView) -> None:
        if self._opp is not None:
            return
        name = gv.opponents[0].name
        self._opp_name = name
        self._opp = BdHandy._MEM.setdefault(name, _Opp())

    def _on_new_hand(self, gv: GameView) -> None:
        """Fecha a mao anterior (atualiza fold equity) e abre uma nova."""
        opp = self._opp
        if opp is not None and self._hand_open:
            opp.hands += 1
            if self._saw_deep:
                opp.deep_hands += 1
            # so contamos fold equity em maos onde NOS demos um raise
            if self._raised_this_hand:
                opp.my_raises += 1
                # se a mao terminou na MESMA rua do nosso ultimo raise (nao
                # agimos numa rua mais avancada) e nao fomos nos que desistimos,
                # entao o oponente foldou aquele raise. Captura folds em qualquer
                # rua (flop/turn/river), nao so no board de 3 cartas.
                if not self._i_folded and self._last_seen_board <= self._raise_board:
                    opp.fold_to_raise += 1
        # abre nova mao
        self._hand_open = True
        self._raised_this_hand = False
        self._i_folded = False
        self._saw_deep = False
        self._raise_board = -1
        self._last_seen_board = -1
        self._start_chips = gv.my_chips

    def _observe_opp(self, gv: GameView) -> None:
        """Registra agressao do oponente nesta decisao."""
        opp = self._opp
        if opp is None:
            return
        opp.opp_actions += 1
        # oponente foi agressivo se a aposta de pe passou de 1 big blind
        aggr_now = (gv.current_bet > gv.big_blind
                    and gv.opponents[0].current_bet_in_round >= gv.current_bet)
        if aggr_now:
            opp.opp_aggr += 1
        a = 1.0 if aggr_now else 0.0          # T3: alimenta EMAs fast/slow
        opp.ag_fast += 0.25 * (a - opp.ag_fast)
        opp.ag_slow += 0.05 * (a - opp.ag_slow)

    # ── helpers de aposta ───────────────────────────────────────────────────
    def _raise_to(self, gv: GameView, pot_frac: float, min_extra_bb: int = 1) -> int:
        """Calcula um total-da-rua para aumentar ~pot_frac do pote. Faz all-in se
        o alvo cobre o stack. Garante exceder a aposta atual."""
        bb = gv.big_blind
        extra = max(min_extra_bb * bb, int(round(pot_frac * gv.pot)))
        # ANTI-LEITURA (Fase 2): jitter no tamanho para DESCORRELACIONAR
        # tamanho<->forca. Sem isto, nosso codigo publico revela "aposta grande =
        # valor, aposta pequena = blefe" e um oponente adaptativo folda/flutua de
        # graca. O jitter borra o limiar (valor e blefe passam a se sobrepor em
        # tamanho) com custo de EV minimo (media ~1.0). Centro levemente > 1 para
        # nao encolher o sizing medio de valor.
        if self.USE_SIZE_JITTER:
            extra = int(round(extra * random.uniform(0.82, 1.26)))
            extra = max(min_extra_bb * bb, extra)
        target = gv.current_bet + extra
        # total que eu ainda preciso por nesta rua (ja investi current_bet-to_call)
        my_invested = gv.current_bet - gv.to_call
        needed = target - my_invested
        if needed >= gv.my_chips:
            return gv.my_chips + my_invested  # all-in (total da rua = tudo)
        return target

    def _call_or_fold(self, gv: GameView, equity: float, margin: float) -> int:
        to_call = gv.to_call
        if to_call <= 0:
            return 0
        pot_odds = to_call / (gv.pot + to_call)
        if equity >= pot_odds + margin:
            return 0
        return -1

    # ── decisao principal ───────────────────────────────────────────────────
    def decision(self, gv: GameView) -> int:
        try:
            action = self._decide(gv)
            if action == -1:
                self._i_folded = True
            return action
        except Exception:
            return 0  # fallback seguro: call/check

    def _decide(self, gv: GameView) -> int:
        self._lock_seat(gv)
        self._bind_opp(gv)

        bl = len(gv.board)
        # Deteccao de nova mao: dentro de uma mao o pote so cresce e o board so
        # aumenta; uma queda no pote (reset para os blinds) ou no board marca uma
        # mao nova. Isso captura tambem maos consecutivas que ficam no board de 3
        # cartas (oponente foldou cedo) — caso que a deteccao por board perdia.
        if self._prev_board < 0 or bl < self._prev_board or gv.pot < self._prev_pot:
            self._on_new_hand(gv)
        if bl >= 4:
            self._saw_deep = True
        self._prev_board = bl
        self._prev_pot = gv.pot
        self._last_seen_board = bl   # ultima rua em que efetivamente agimos

        self._observe_opp(gv)

        equity, rank, draw_eq, plays_board = self._equity(gv)
        in_pos = self._in_position(gv)
        to_call = gv.to_call
        bb = gv.big_blind
        eff_bb = gv.my_chips / max(1, bb)

        # Realidade desta engine + campo: nao ha check gratis e a aposta "carrega"
        # entre ruas, e os oponentes valorizam muito e blefam pouco (agressao
        # observada 0.03-0.15). Logo o correto e: tight, value pesado, blefe baixo
        # e DISCIPLINA DE FOLD (margem alta sobre as pot odds — quem aposta forte
        # aqui quase sempre TEM a mao). So afrouxamos a margem contra maniacos
        # (que blefam) e aumentamos o roubo contra quem realmente folda demais.
        ag = self._opp.aggression() if self._opp else 0.5
        fe = self._opp.fold_eq() if self._opp else 0.5
        known = self._opp.hands >= 12 if self._opp else False

        bluff = self._BASE_BLUFF
        value_thr = self._BASE_VALUE_THR
        margin = 0.08              # base disciplinada: foldar para agressao
        steal_frac = 0.50
        value_frac = 0.72
        anti_nit = False

        if known and ag > 0.45:
            # maniaco: blefa muito -> arma trap, paga mais leve, valoriza
            arche = "maniac"
            bluff = 0.03
            margin = 0.0
            value_thr = 0.56
        elif known and ag < 0.12:
            # value-bot passivo (papa/tubarao): aposta = forca. Foldar muito,
            # quase nunca blefar; roubar so quando ele demonstra fraqueza.
            arche = "nit"
            anti_nit = True
            bluff = 0.05
            value_thr = 0.64
            margin = 0.12          # paga so com folga grande -> nao paga value
            if fe > 0.13:          # ele larga raises: rouba um pouco mais
                bluff = 0.10
        elif known and fe < 0.24 and ag < 0.45:
            # CALLING STATION (ex.: Pinguim_Rei): paga nossos raises (fold_eq
            # baixo) mas raramente toma a iniciativa. Logo NAO blefar (ele nao
            # larga), valor FINO e mais grosso (ele paga com pior), e margem
            # leve porque a agressao real dele e rara mas honesta. Gated so por
            # fold_eq observado -> upside-only e robusto a outros oponentes.
            arche = "station"
            bluff = 0.0
            value_thr = 0.50
            value_frac = 0.80
            margin = 0.10
        else:
            arche = "reg"
            bluff = 0.08
            value_thr = 0.58
            margin = 0.07

        # ── HARDENING ANTI-EXPLOIT (Fase 2, gated) ───────────────────────────
        if self.USE_BLUFF_BALANCE:
            # T2: nunca um range puro de valor (senao auto-foldam contra nos) +
            # bluff-catch contra agressao (paga mais leve quem blefa muito).
            if arche != "station":
                bluff = max(bluff, 0.14)
            if known and ag > 0.30:
                margin = min(margin, 0.03)
        if self.USE_ANTIPOISON and known:
            # T3: encolhe desvios do baseline quando o opp esta mudando de estilo
            # (fast vs slow divergem) -> camaleao nao consegue nos envenenar.
            damp = max(0.4, 1.0 - 2.0 * self._opp.shift())
            bluff = 0.08 + (bluff - 0.08) * damp
            value_thr = 0.58 + (value_thr - 0.58) * damp
            margin = 0.07 + (margin - 0.07) * damp
        if self.USE_BOUNDARY_MIX:
            # T1b: borra os limiares p/ nao serem lidos deterministicamente.
            margin += random.uniform(-0.02, 0.02)
            value_thr += random.uniform(-0.02, 0.02)

        facing_bet = to_call > 0
        real_bet = to_call > bb    # aposta acima do blind = agressao real
        pot_odds = to_call / (gv.pot + to_call) if facing_bet else 0.0

        # equity ajustada para PAGAR: o range que aposta forte e mais forte que
        # o nosso quando so temos um par. Desconto por categoria + por tamanho.
        e_call = equity
        if real_bet:
            if rank == RANK_UM_PAR:
                e_call *= 0.72
            elif rank == RANK_DOIS_PARES:
                e_call *= 0.88
            if to_call > gv.pot * 0.6:
                e_call *= 0.88

        # ── PUSH / FOLD (stack curto) ────────────────────────────────────────
        if eff_bb < self._PUSHFOLD_BB:
            shove_thr = 0.68 if anti_nit else (0.54 if in_pos else 0.58)
            if equity >= shove_thr or rank >= RANK_TRINCA:
                return self._mark_raise(gv.my_chips + (gv.current_bet - to_call))
            if not facing_bet:
                return 0
            return 0 if e_call >= pot_odds else -1

        # ── VALUE forte: trinca+ ou equity altissima -> construir pote ───────
        if rank >= RANK_TRINCA or equity >= 0.85:
            # slowplay ocasional em posicao com casa+/quadra para induzir
            if in_pos and facing_bet and rank >= RANK_FULL_HOUSE and random.random() < 0.30:
                return 0
            # contra nit com so trinca media, aposta controlada (nao "cego")
            frac = value_frac if not anti_nit else 0.60
            return self._mark_raise(self._raise_to(gv, frac))

        # ── VALUE medio: dois pares ou par forte ─────────────────────────────
        if rank == RANK_DOIS_PARES or equity >= value_thr:
            if not facing_bet:
                frac = value_frac if not anti_nit else 0.55
                return self._mark_raise(self._raise_to(gv, frac))
            # enfrentando aposta: protege com raise so se barato, forte e nao-nit
            if (not anti_nit and rank >= RANK_DOIS_PARES
                    and to_call <= gv.pot * 0.6 and e_call >= pot_odds + 0.12
                    and random.random() < 0.45):
                return self._mark_raise(self._raise_to(gv, value_frac))
            return 0 if e_call >= pot_odds + margin else -1

        # ── DRAW forte: semi-blefe seletivo / pagar por odds ─────────────────
        if draw_eq >= 0.32 and not anti_nit:
            if not facing_bet:
                if in_pos and random.random() < 0.40:
                    return self._mark_raise(self._raise_to(gv, steal_frac))
                return 0
            if draw_eq >= pot_odds or to_call <= gv.pot * 0.25:
                return 0
            return -1
        if draw_eq >= 0.32 and anti_nit:
            # contra nit, paga draw so se MUITO barato (implied odds baixas)
            if facing_bet and to_call <= gv.pot * 0.20:
                return 0
            return -1 if facing_bet else 0

        # ── MARGINAL / AR ────────────────────────────────────────────────────
        if not facing_bet:
            # iniciativa (raro nesta engine): rouba so com fold equity real
            steal_chance = bluff + (0.06 if in_pos else 0.0)
            if random.random() < steal_chance:
                return self._mark_raise(self._raise_to(gv, steal_frac))
            return 0 if to_call <= 0 else -1

        # enfrentando aposta com mao fraca: bluff-catch barato so se as contas
        # fecham e o oponente nao e nit; senao fold disciplinado.
        if not real_bet and e_call >= pot_odds + margin and not anti_nit:
            return 0
        if (arche == "reg" and in_pos and to_call <= gv.pot * 0.4
                and random.random() < bluff * 0.5):
            return self._mark_raise(self._raise_to(gv, steal_frac))
        return -1

    def _mark_raise(self, total: int) -> int:
        """Registra que demos um raise nesta mao (para medir fold equity).
        Se o alvo nao supera a aposta atual, o motor trata como call -> nao conta."""
        self._raised_this_hand = True
        self._raise_board = self._prev_board   # nivel do board deste raise
        return total


def create_player() -> Player:
    return BdHandy("player_bdHANDY", Hand(), 0)
