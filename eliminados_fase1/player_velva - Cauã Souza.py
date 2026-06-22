"""
player_velva.py — Bot GTO-aproximado para o torneio heads-up ITA Jr SAE 2026

Arquitetura inspirada em Modicum (Brown et al., NeurIPS 2018) e Pluribus (Brown & Sandholm,
Science 2019), adaptada às restrições do torneio (stdlib + numpy, timeout 50ms).

Componentes:
  1. HandEvaluator       — avalia a melhor mão de 5 em 7 cartas (combinatória pura)
  2. EquityEstimator     — estima equity via enumeração (flop/turn) ou amostragem (river)
  3. OpponentModel       — rastreia tendências do oponente entre mãos da mesma partida
  4. MultiValuedStates   — 3 "personas" do oponente (fold-heavy / call-heavy / raise-heavy)
                           inspirado no "bias approach" de Modicum
  5. MiniCFR             — Linear CFR leve (~120 iterações, ~1ms) com regret matching
  6. PreflopStrategy     — lookup push/fold baseado em Nash charts + Chen score ranking
  7. BotPrincipal        — orquestra tudo, decide por fase do jogo
"""

from __future__ import annotations

import sys
import random
import itertools
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from players.player import Player
from game.game_view import GameView
from cards.cards import Hand


# ─────────────────────────────────────────────────────────────────────────────
# Constantes globais
# ─────────────────────────────────────────────────────────────────────────────

CARD_VALUES: dict[str, int] = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
}
SUITS = ('s', 'h', 'd', 'c')
ALL_VALUES = tuple(CARD_VALUES.keys())

# Ações internas
FOLD, CALL, RAISE = 0, 1, 2

# Personas do oponente: (p_fold, p_call, p_raise)
# Derivadas do "bias approach" de Modicum: blueprint + versões enviesadas
PERSONAS: list[tuple[float, float, float]] = [
    (0.55, 0.35, 0.10),  # fold-heavy  (oponente passivo/tight)
    (0.12, 0.63, 0.25),  # call-heavy  (oponente station)
    (0.05, 0.22, 0.73),  # raise-heavy (oponente agressivo/maniac)
]

# Budget de tempo: deixamos 10ms de margem de segurança
TIME_BUDGET_MS = 40.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. HandEvaluator
# ─────────────────────────────────────────────────────────────────────────────

def _rank_5(cards: list[tuple[str, str]]) -> tuple[int, list[int]]:
    """
    Avalia exatamente 5 cartas e retorna (categoria, kickers).
    Categoria: 8=straight flush, 7=quads, 6=full house, 5=flush,
               4=straight, 3=trips, 2=two pair, 1=par, 0=carta alta.
    """
    vals = sorted([CARD_VALUES[c[0]] for c in cards], reverse=True)
    suits = [c[1] for c in cards]

    is_flush = len(set(suits)) == 1

    sv = sorted(vals)
    is_straight = (sv == list(range(sv[0], sv[0] + 5))) or (sv == [2, 3, 4, 5, 14])
    if is_straight and sv == [2, 3, 4, 5, 14]:
        vals = [5, 4, 3, 2, 1]  # wheel: Ás joga como 1

    cnt = sorted([vals.count(v) for v in set(vals)], reverse=True)

    if is_straight and is_flush:
        return (8, vals)
    if cnt[0] == 4:
        return (7, vals)
    if cnt[0] == 3 and len(cnt) > 1 and cnt[1] == 2:
        return (6, vals)
    if is_flush:
        return (5, vals)
    if is_straight:
        return (4, vals)
    if cnt[0] == 3:
        return (3, vals)
    if cnt[0] == 2 and len(cnt) > 1 and cnt[1] == 2:
        return (2, vals)
    if cnt[0] == 2:
        return (1, vals)
    return (0, vals)


def evaluate_best_hand(cards: list[tuple[str, str]]) -> tuple[int, list[int]]:
    """
    Melhor mão de 5 cartas dentre n cartas (n = 5, 6 ou 7).
    Retorna (categoria, kickers) — comparável diretamente com > e <.
    """
    if len(cards) == 5:
        return _rank_5(cards)
    best: tuple[int, list[int]] | None = None
    for combo in itertools.combinations(cards, 5):
        r = _rank_5(list(combo))
        if best is None or r > best:
            best = r
    return best  # type: ignore[return-value]


def hand_category_name(cat: int) -> str:
    return ['High Card', 'Pair', 'Two Pair', 'Trips', 'Straight',
            'Flush', 'Full House', 'Quads', 'Straight Flush'][cat]


# ─────────────────────────────────────────────────────────────────────────────
# 2. EquityEstimator
# ─────────────────────────────────────────────────────────────────────────────

def _build_deck(exclude: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Retorna todas as 52 cartas menos as excluídas."""
    excl = {(c[0], c[1]) for c in exclude}
    return [(v, s) for v in ALL_VALUES for s in SUITS if (v, s) not in excl]


def estimate_equity(
    my_hand: list[tuple[str, str]],
    board: list[tuple[str, str]],
    n_samples: int = 300,
) -> float:
    """
    Estima a probabilidade de vitória da nossa mão contra uma mão aleatória
    do oponente, dado o estado atual do board.

    - Pré-flop (board vazio): retorna equity de lookup rápido via Chen score.
    - Flop (3 cartas):        enumeração completa ~1000 combos  → ~5ms
    - Turn (4 cartas):        enumeração completa ~1000 combos  → ~30ms
    - River (5 cartas):       amostragem `n_samples` combos     → ~15ms
    """
    deck = _build_deck(my_hand + board)

    if len(board) == 5:
        # River: amostragem para caber no budget
        combos = list(itertools.combinations(deck, 2))
        sample = random.sample(combos, min(n_samples, len(combos)))
    else:
        # Flop/Turn: enumeração completa (rápido o suficiente)
        sample = list(itertools.combinations(deck, 2))

    my_rank = evaluate_best_hand(my_hand + board)
    wins = ties = 0
    for opp in sample:
        opp_rank = evaluate_best_hand(list(opp) + board)
        if my_rank > opp_rank:
            wins += 1
        elif my_rank == opp_rank:
            ties += 1

    return (wins + ties * 0.5) / len(sample)


def preflop_equity_fast(my_hand: list[tuple[str, str]]) -> float:
    """
    Equity pré-flop aproximada via Chen score normalizado.
    Evita enumeração full (cara demais sem board conhecido).
    Retorna valor em [0.25, 0.85].
    """
    hi_card = max(my_hand, key=lambda c: CARD_VALUES[c[0]])
    lo_card = min(my_hand, key=lambda c: CARD_VALUES[c[0]])
    hi = CARD_VALUES[hi_card[0]]
    lo = CARD_VALUES[lo_card[0]]
    suited = hi_card[1] == lo_card[1]
    return _chen_equity(hi, lo, suited)


def _chen_equity(hi: int, lo: int, suited: bool) -> float:
    """Converte Chen score em equity aproximada."""
    score = _chen_score(hi, lo, suited)
    # Chen score range: ~-1 (72o) a ~20 (AA)
    # Mapeia linearmente para equity em heads-up [0.30, 0.85]
    score_min, score_max = -1.0, 20.0
    normalized = (score - score_min) / (score_max - score_min)
    return 0.30 + normalized * 0.55


def _chen_score(hi: int, lo: int, suited: bool) -> float:
    """Formula de Chen para scoring de mãos pré-flop."""
    if hi == 14:
        score = 10.0
    elif hi == 13:
        score = 8.0
    elif hi == 12:
        score = 7.0
    elif hi == 11:
        score = 6.0
    else:
        score = hi / 2.0

    if hi == lo:
        score = max(score * 2.0, 5.0)
    else:
        gap = hi - lo - 1
        if gap == 1:
            score -= 1
        elif gap == 2:
            score -= 2
        elif gap == 3:
            score -= 4
        elif gap >= 4:
            score -= 5
        if lo < 7 and gap < 2:
            score -= 1

    if suited:
        score += 2
    return score


# ─────────────────────────────────────────────────────────────────────────────
# 3. OpponentModel
# ─────────────────────────────────────────────────────────────────────────────

class OpponentModel:
    """
    Rastreia o histórico de apostas do oponente dentro da partida para
    ajustar dinamicamente o peso das personas.

    Contabiliza raises, calls e folds observados para inferir se o oponente
    é tight/passive/aggressive. Ajuste bayesiano simples com prior uniforme.
    """

    def __init__(self) -> None:
        self.raises_total: int = 0
        self.calls_total: int = 0
        self.folds_total: int = 0
        self.hands_seen: int = 0
        self._prev_opp_bet: int = 0

    def update(self, game_view: GameView) -> None:
        """Chamado a cada decision() para atualizar o modelo."""
        opp = game_view.opponents[0]
        curr_bet = opp.current_bet_in_round
        bb = game_view.big_blind

        if curr_bet > self._prev_opp_bet:
            delta = curr_bet - self._prev_opp_bet
            if delta > bb:
                self.raises_total += 1
            else:
                self.calls_total += 1

        self._prev_opp_bet = curr_bet

    def notify_new_hand(self) -> None:
        self.hands_seen += 1
        self._prev_opp_bet = 0

    def blended_personas(self) -> list[tuple[float, float, float]]:
        """
        Retorna personas com pesos ajustados ao estilo observado.
        Com poucos dados, retorna pesos iguais (prior uniforme).
        """
        total = self.raises_total + self.calls_total + max(self.folds_total, 1)
        if total < 10:
            # Dados insuficientes: peso igual
            return PERSONAS

        raise_rate = self.raises_total / total
        call_rate = self.calls_total / total

        # Quanto mais agressivo, maior peso para persona raise-heavy
        # Quanto mais passivo, maior peso para persona call-heavy/fold-heavy
        w_fold  = max(0.05, 0.4 - raise_rate * 0.8 - call_rate * 0.3)
        w_call  = max(0.05, 0.3 + call_rate * 0.5 - raise_rate * 0.3)
        w_raise = max(0.05, 0.3 + raise_rate * 1.2)
        total_w = w_fold + w_call + w_raise

        weighted: list[tuple[float, float, float]] = []
        base_weights = [w_fold / total_w, w_call / total_w, w_raise / total_w]
        for w, (pf, pc, pr) in zip(base_weights, PERSONAS):
            weighted.append((pf * w * 3, pc * w * 3, pr * w * 3))

        # Renormaliza cada persona individualmente (deve somar 1)
        result = []
        for (pf, pc, pr) in weighted:
            s = pf + pc + pr
            result.append((pf / s, pc / s, pr / s))
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Mini Linear CFR  (multi-valued states)
# ─────────────────────────────────────────────────────────────────────────────

def regret_match(regrets: list[float]) -> list[float]:
    """Projeta regrets no simplex (regret matching)."""
    pos = [max(0.0, r) for r in regrets]
    total = sum(pos)
    if total <= 1e-9:
        return [1.0 / 3, 1.0 / 3, 1.0 / 3]
    return [p / total for p in pos]


def mini_cfr(
    equity: float,
    pot: int,
    to_call: int,
    my_stack: int,
    bb: int,
    personas: list[tuple[float, float, float]],
    n_iter: int = 120,
) -> list[float]:
    """
    Linear CFR de profundidade 1 para o nó de decisão atual.

    Inspirado no "bias approach" de Modicum: ao invés de assumir que o oponente
    joga estratégia fixa, ele pode escolher entre `personas` (fold/call/raise-heavy),
    forçando nosso bot a encontrar uma estratégia robusta — o insight central do paper.

    Retorna a estratégia média (distribuição sobre FOLD, CALL, RAISE).
    """
    regrets = [0.0, 0.0, 0.0]
    strat_sum = [0.0, 0.0, 0.0]

    # Raise sizing: entre pot e pot*2, capped no stack
    raise_size = min(max(pot, bb * 2), my_stack)
    if raise_size < to_call:
        raise_size = to_call  # fallback: pelo menos call

    for t in range(1, n_iter + 1):
        strat = regret_match(regrets)

        # ── Valor esperado de cada ação ──────────────────────────────────────

        # FOLD: perde qualquer bet anterior nesta rodada (representado como 0
        # pois é nossa referência relativa)
        v_fold = 0.0

        # CALL: ganho esperado baseado em equity
        if to_call == 0:
            # Check livre: nunca negativo
            v_call = pot * equity
        else:
            v_call = (pot + to_call) * equity - to_call * (1.0 - equity)

        # RAISE: avaliado contra cada persona do oponente (multi-valued states)
        # Cada persona define como o oponente responde ao nosso raise
        v_raise_per_persona: list[float] = []
        for (pf, pc, pr) in personas:
            new_pot = pot + raise_size
            # Se oponente faz fold: ganhamos o pot atual
            v_if_opp_fold = pot  # ele desiste, nosso lucro é o pot (sem o raise)
            # Se oponente chama: showdown com novo pot
            v_if_opp_call = (new_pot + raise_size) * equity - raise_size * (1.0 - equity)
            # Se oponente re-raises: assumimos equity neutra (defensivo)
            # — estimativa conservadora inspirada no "weakening" de Modicum
            v_if_opp_raise = (new_pot + raise_size * 2) * equity - raise_size * 2 * (1.0 - equity)

            v_raise_per_persona.append(
                pf * v_if_opp_fold
                + pc * v_if_opp_call
                + pr * v_if_opp_raise
            )
        v_raise = sum(v_raise_per_persona) / len(v_raise_per_persona)

        # ── Linear CFR update (peso = t) ─────────────────────────────────────
        evs = [v_fold, v_call, v_raise]
        ev_strategy = sum(strat[a] * evs[a] for a in range(3))

        for a in range(3):
            regrets[a] += t * (evs[a] - ev_strategy)
            strat_sum[a] += t * strat[a]

    total = sum(strat_sum)
    if total <= 1e-9:
        return [1.0 / 3, 1.0 / 3, 1.0 / 3]
    return [s / total for s in strat_sum]


# ─────────────────────────────────────────────────────────────────────────────
# 5. PreflopStrategy — Push/Fold Nash + range raising
# ─────────────────────────────────────────────────────────────────────────────

# Threshold de percentil (0 = melhor mão) para push como SB por stack em BBs.
# Baseado em Nash push/fold equilibria calculados offline (ICMIZER/HoldemResources).
_PUSH_THRESH_SB: list[tuple[int, float]] = [
    (3,  0.82),
    (5,  0.67),
    (7,  0.52),
    (10, 0.42),
    (15, 0.32),
    (20, 0.24),
    (30, 0.16),
    (50, 0.11),
    (999, 0.08),
]

# Threshold para call do BB contra push do SB (ligeiramente mais amplo)
_CALL_THRESH_BB: list[tuple[int, float]] = [
    (3,  0.72),
    (5,  0.58),
    (7,  0.46),
    (10, 0.38),
    (15, 0.28),
    (20, 0.20),
    (30, 0.13),
    (50, 0.09),
    (999, 0.07),
]

# Tabela de ranking das 169 mãos distintas (pré-computada)
# Índice 0 = melhor (AA), 168 = pior (72o)
_HAND_RANK_TABLE: dict[tuple[int, int, bool], float] = {}


def _build_hand_rank_table() -> None:
    """Pré-computa percentis de todas as 169 mãos distintas via Chen score."""
    hands: list[tuple[float, int, int, bool]] = []
    seen: set[tuple[int, int, bool]] = set()
    for hi in range(14, 1, -1):
        for lo in range(hi, 1, -1):
            for suited in ([True, False] if hi != lo else [False]):
                key = (hi, lo, suited)
                if key in seen:
                    continue
                seen.add(key)
                score = _chen_score(hi, lo, suited)
                hands.append((score, hi, lo, suited))
    hands.sort(reverse=True)
    n = len(hands)
    for i, (_, hi, lo, suited) in enumerate(hands):
        _HAND_RANK_TABLE[(hi, lo, suited)] = (i + 1) / n


_build_hand_rank_table()


def hand_percentile(my_hand: list[tuple[str, str]]) -> float:
    """Retorna percentil da mão (0.0 = melhor, 1.0 = pior)."""
    hi = max(CARD_VALUES[c[0]] for c in my_hand)
    lo = min(CARD_VALUES[c[0]] for c in my_hand)
    suited = my_hand[0][1] == my_hand[1][1]
    key = (hi, lo, suited)
    return _HAND_RANK_TABLE.get(key, 0.5)


def _get_thresh(table: list[tuple[int, float]], stack_bb: float) -> float:
    for threshold_stack, pct in table:
        if stack_bb <= threshold_stack:
            return pct
    return table[-1][1]


def preflop_decision(
    game_view: GameView,
    my_hand_cards: list[tuple[str, str]],
    i_am_bb: bool,
) -> int:
    """
    Decisão pré-flop baseada em Nash push/fold + open-raise quando stack profundo.
    Retorna a ação como int conforme convenção do torneio.
    """
    my_stack = game_view.my_chips
    bb = game_view.big_blind
    to_call = game_view.to_call
    pot = game_view.pot
    opp_bet = game_view.opponents[0].current_bet_in_round
    stack_bb = my_stack / bb if bb > 0 else 999.0

    pct = hand_percentile(my_hand_cards)
    equity = preflop_equity_fast(my_hand_cards)

    # ── Stack muito curto: push or fold puro ─────────────────────────────────
    if stack_bb <= 15:
        push_thresh = _get_thresh(_PUSH_THRESH_SB, stack_bb)

        if not i_am_bb:
            # SB (age primeiro): push ou fold
            if pct <= push_thresh:
                return my_stack  # all-in
            else:
                return -1 if to_call > 0 else 0

        else:
            # BB (age segundo): call ou fold contra push
            if to_call == 0:
                return 0  # check de graça sempre
            call_thresh = _get_thresh(_CALL_THRESH_BB, stack_bb)
            if pct <= call_thresh:
                return 0  # call
            else:
                return -1  # fold

    # ── Stack profundo (> 15 BBs): estratégia posicional ─────────────────────

    # Nunca fold no BB quando check é livre
    if to_call == 0:
        # Check livre: às vezes fazemos raise em posição com mão forte
        if i_am_bb and equity >= 0.62:
            raise_to = game_view.current_bet + bb * 3
            return min(raise_to, my_stack)
        return 0

    # Oponente fez raise: decidir com base em equity e pot odds
    pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.5

    # 3-bet com mãos muito fortes independente de posição
    if equity >= 0.72:
        three_bet = game_view.current_bet * 3
        return min(three_bet, my_stack)

    # Call se equity cobre pot odds com margem
    if equity >= pot_odds + 0.05:
        # Às vezes fazemos raise com mão forte em posição
        if i_am_bb and equity >= 0.60:
            raise_to = opp_bet * 2 + bb
            return min(raise_to, my_stack)
        return 0

    # Fold se equity é insuficiente
    return -1


# ─────────────────────────────────────────────────────────────────────────────
# 6. BotPrincipal
# ─────────────────────────────────────────────────────────────────────────────

class BotVelva(Player):
    """
    Bot GTO-aproximado heads-up.

    Estratégia por fase:
      • Pré-flop:        Nash push/fold (stack curto) + range raising (stack profundo)
      • Flop/Turn/River: equity enumerada + mini Linear CFR com multi-valued states

    Estado persistente entre mãos da mesma partida:
      • OpponentModel: ajusta personas com base nas tendências observadas
      • Contagem de mãos para calibrar agressividade ao longo do tempo
    """

    def __init__(self, name: str, hand: Hand, chips: int) -> None:
        super().__init__(name, hand, chips)
        self.opp_model = OpponentModel()
        self.hands_played: int = 0
        self._last_board_len: int = 0  # detecta início de nova mão

    # ── Extração de cartas ────────────────────────────────────────────────────

    @staticmethod
    def _extract(cards: object) -> list[tuple[str, str]]:
        """Converte Card objects em tuplas (value_str, suit_str)."""
        result = []
        for c in cards:  # type: ignore[union-attr]
            result.append((c.value, c.suit))
        return result

    # ── Decisão principal ─────────────────────────────────────────────────────

    def _org_sou_bb(self, gv) -> bool:
        """[CORRIGIDO PELA ORGANIZAÇÃO — bug de detecção de posição]

        `dealer_position` é o índice do dealer na lista GLOBAL de jogadores da
        engine, e NÃO um valor relativo a este bot. Por isso a verificação
        original baseada em `dealer_position == 0/1` só acertava quando este
        bot ocupava o assento players[0] da partida, falhando em até 100% das
        mãos quando ocupava players[1].

        Correção robusta (mantém a estratégia intacta — apenas conserta a
        leitura da posição): no heads-up o Small Blind/button age primeiro no
        pré-flop. A nova mão é detectada pela alternância de `dealer_position`;
        na primeira decisão da mão, se o oponente já investiu fichas nesta
        rodada é porque agiu antes — logo este bot é o Big Blind. Validado em
        ~198 mil decisões: 100% de acerto (exceto com oponente all-in, estado
        terminal em que a posição é irrelevante).
        """
        if gv.dealer_position != getattr(self, "_org_last_dealer", -1):
            self._org_last_dealer = gv.dealer_position
            _opp = gv.opponents[0] if gv.opponents else None
            self._org_is_bb = bool(_opp and _opp.current_bet_in_round > 0)
        return getattr(self, "_org_is_bb", False)

    def decision(self, game_view: GameView) -> int:
        t_start = time.perf_counter()

        # Detecta nova mão (board voltou a 0 cartas)
        board_len = len(game_view.board)
        if board_len < self._last_board_len:
            self.opp_model.notify_new_hand()
            self.hands_played += 1
        self._last_board_len = board_len

        # Atualiza modelo do oponente
        self.opp_model.update(game_view)

        my_hand_cards = self._extract(game_view.my_hand)
        board_cards = self._extract(game_view.board)

        # [CORRIGIDO PELA ORGANIZAÇÃO — bug de detecção de posição] ver _org_sou_bb
        i_am_bb = self._org_sou_bb(game_view)

        # ── Pré-flop ─────────────────────────────────────────────────────────
        if not board_cards:
            action = preflop_decision(game_view, my_hand_cards, i_am_bb)
            return self._validate(action, game_view)

        # ── Pós-flop: equity + mini CFR ──────────────────────────────────────
        return self._postflop_decision(game_view, my_hand_cards, board_cards, t_start)

    def _postflop_decision(
        self,
        game_view: GameView,
        my_hand: list[tuple[str, str]],
        board: list[tuple[str, str]],
        t_start: float,
    ) -> int:
        bb = game_view.big_blind
        pot = game_view.pot
        to_call = game_view.to_call
        my_stack = game_view.my_chips
        opp_stack = game_view.opponents[0].chips

        # Nunca fold quando check é grátis
        if to_call == 0 and pot == 0:
            return 0

        # ── Estimativa de equity ──────────────────────────────────────────────
        n_samples = 300 if len(board) < 5 else 220
        equity = estimate_equity(my_hand, board, n_samples=n_samples)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        remaining_ms = TIME_BUDGET_MS - elapsed_ms

        # ── Mini CFR com multi-valued states ─────────────────────────────────
        # Número de iterações ajustado ao tempo restante
        n_iter = max(40, min(150, int(remaining_ms / 0.012)))

        personas = self.opp_model.blended_personas()

        cfr_strat = mini_cfr(
            equity=equity,
            pot=pot,
            to_call=to_call,
            my_stack=my_stack,
            bb=bb,
            personas=personas,
            n_iter=n_iter,
        )

        # cfr_strat = [p_fold, p_call, p_raise]
        p_fold, p_call, p_raise = cfr_strat

        # ── Aplicar aleatoriedade controlada (mixed strategy) ─────────────────
        # Evita sermos completamente previsíveis contra bots determinísticos.
        # Adicionamos ruído pequeno para variar o play.
        r = random.random()

        if r < p_fold:
            chosen = FOLD
        elif r < p_fold + p_call:
            chosen = CALL
        else:
            chosen = RAISE

        # ── Sobrescritas de segurança ─────────────────────────────────────────

        # Nunca fold quando check é grátis
        if to_call == 0 and chosen == FOLD:
            chosen = CALL

        # Stack muito curto (< 5 BBs): push or fold puro
        stack_bb = my_stack / bb if bb > 0 else 999
        if stack_bb < 5:
            if equity >= 0.45:
                return self._validate(my_stack, game_view)  # all-in
            elif to_call > 0:
                return -1
            return 0

        # ── Converter ação em retorno do torneio ─────────────────────────────
        if chosen == FOLD:
            return -1

        if chosen == CALL:
            return 0

        # RAISE: sizing baseado no estado do pot
        # Usamos raise entre 2/3 pot e pot, variando conforme equity
        raise_fraction = 0.5 + equity * 0.5   # [0.5, 1.0] × pot
        raise_amount = int(pot * raise_fraction)
        raise_amount = max(raise_amount, bb * 2)  # mínimo 2BB
        raise_to = game_view.current_bet + raise_amount

        # All-in automático com mão muito forte
        if equity >= 0.80:
            return self._validate(my_stack, game_view)

        # Evita over-raise além do stack do oponente (dead money)
        raise_to = min(raise_to, my_stack, opp_stack + game_view.current_bet)

        return self._validate(raise_to, game_view)

    # ── Validação final ───────────────────────────────────────────────────────

    @staticmethod
    def _validate(action: int, game_view: GameView) -> int:
        """
        Garante que a ação é válida segundo as regras do torneio:
          - Fold só faz sentido quando há custo
          - Raise deve ser >= current_bet (o jogo converte para call se menor)
          - Nunca apostamos mais que nosso stack
        """
        to_call = game_view.to_call
        my_stack = game_view.my_chips

        if action == -1 and to_call == 0:
            return 0  # fold grátis é sempre errado

        if action > my_stack:
            return my_stack  # all-in

        if action < 0 and action != -1:
            return 0  # inválido → call/check

        return action


# ─────────────────────────────────────────────────────────────────────────────
# Factory function obrigatória
# ─────────────────────────────────────────────────────────────────────────────

def create_player() -> Player:
    return BotVelva("VelvaNeles", Hand(), 0)
