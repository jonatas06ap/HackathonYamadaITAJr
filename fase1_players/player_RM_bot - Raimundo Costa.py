from __future__ import annotations

import sys
import random
import math
from pathlib import Path
from collections import Counter
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from players.player import Player
from game.game_view import GameView
from cards.cards import Hand


# ─────────────────────────────────────────────
#  Tabelas e constantes
# ─────────────────────────────────────────────

RANK_MAP = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}

# Força pré-flop heads-up (escalada de 0 a 1)
# Fonte: teoria GTO adaptada para HU
PREFLOP_STRENGTH = {
    # Pares
    ("A", "A"): 1.00, ("K", "K"): 0.97, ("Q", "Q"): 0.94,
    ("J", "J"): 0.91, ("10", "10"): 0.87, ("9", "9"): 0.82,
    ("8", "8"): 0.77, ("7", "7"): 0.72, ("6", "6"): 0.67,
    ("5", "5"): 0.62, ("4", "4"): 0.57, ("3", "3"): 0.52,
    ("2", "2"): 0.48,
    # Ases conectados (suited / offsuit)
    ("A", "K"): 0.87, ("A", "Q"): 0.84, ("A", "J"): 0.80,
    ("A", "10"): 0.77, ("A", "9"): 0.70, ("A", "8"): 0.67,
    ("A", "7"): 0.64, ("A", "6"): 0.62, ("A", "5"): 0.63,
    ("A", "4"): 0.60, ("A", "3"): 0.59, ("A", "2"): 0.58,
    # Broadway
    ("K", "Q"): 0.79, ("K", "J"): 0.76, ("K", "10"): 0.74,
    ("Q", "J"): 0.74, ("Q", "10"): 0.72, ("J", "10"): 0.70,
    # Conectores médios
    ("K", "9"): 0.65, ("K", "8"): 0.62, ("Q", "9"): 0.63,
    ("J", "9"): 0.65, ("10", "9"): 0.64, ("9", "8"): 0.61,
    ("8", "7"): 0.58, ("7", "6"): 0.55, ("6", "5"): 0.53,
    ("5", "4"): 0.51, ("4", "3"): 0.48, ("3", "2"): 0.45,
}

HAND_RANKS = {
    "high_card": 0, "one_pair": 1, "two_pair": 2, "three_of_a_kind": 3,
    "straight": 4, "flush": 5, "full_house": 6, "four_of_a_kind": 7,
    "straight_flush": 8,
}


# ─────────────────────────────────────────────
#  Avaliação de mão (7 cartas → melhor 5)
# ─────────────────────────────────────────────

def card_rank(card) -> int:
    return RANK_MAP[card.value]


def evaluate_5(cards) -> tuple:
    """Retorna (rank_index, tiebreakers) para 5 cartas."""
    values = sorted([card_rank(c) for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    is_flush = len(set(suits)) == 1
    is_straight = (values == list(range(values[0], values[0] - 5, -1)))
    # Wheel straight A-2-3-4-5
    if values == [14, 5, 4, 3, 2]:
        is_straight = True
        values = [5, 4, 3, 2, 1]

    counts = Counter(values)
    freq = sorted(counts.values(), reverse=True)
    groups = sorted(counts.keys(), key=lambda v: (counts[v], v), reverse=True)

    if is_straight and is_flush:
        return (8, values)
    if freq[0] == 4:
        return (7, groups)
    if freq[:2] == [3, 2]:
        return (6, groups)
    if is_flush:
        return (5, values)
    if is_straight:
        return (4, values)
    if freq[0] == 3:
        return (3, groups)
    if freq[:2] == [2, 2]:
        return (2, groups)
    if freq[0] == 2:
        return (1, groups)
    return (0, values)


def best_hand(cards) -> tuple:
    """Melhor mão possível com qualquer combinação de 5 das cartas dadas."""
    if len(cards) < 5:
        return evaluate_5(cards) if len(cards) == 5 else (0, [])
    return max(evaluate_5(list(combo)) for combo in combinations(cards, 5))


def hand_strength(my_cards, board) -> float:
    """
    Retorna força normalizada [0,1] da mão atual contra distribuição
    aleatória de mãos do oponente (Monte Carlo leve, 150 amostras).
    """
    all_cards = list(my_cards) + list(board)
    my_best = best_hand(all_cards)

    # Cartas "conhecidas" — não podem aparecer no deck oponente
    known = {(c.value, c.suit) for c in all_cards}

    # Construir deck virtual com objetos simples
    class FakeCard:
        def __init__(self, value, suit):
            self.value = value
            self.suit = suit

    deck = [
        FakeCard(v, s)
        for v in RANK_MAP
        for s in ("s", "h", "d", "c")
        if (v, s) not in known
    ]

    wins = 0
    ties = 0
    samples = 150
    cards_needed = 2  # mão do oponente
    board_needed = 5 - len(board)  # cartas faltantes na mesa

    for _ in range(samples):
        draw = random.sample(deck, cards_needed + board_needed)
        opp_hole = draw[:cards_needed]
        new_board = list(board) + draw[cards_needed:]
        opp_best = best_hand(opp_hole + new_board)
        my_best_full = best_hand(all_cards + draw[cards_needed:])
        if my_best_full > opp_best:
            wins += 1
        elif my_best_full == opp_best:
            ties += 1

    return (wins + ties * 0.5) / samples


# ─────────────────────────────────────────────
#  Calcular força pré-flop
# ─────────────────────────────────────────────

def preflop_strength(hand, suited: bool) -> float:
    v = sorted([card_rank(c) for c in hand], reverse=True)
    vals = [c.value for c in hand]
    vals_sorted = sorted(vals, key=lambda x: RANK_MAP[x], reverse=True)
    key = tuple(vals_sorted)
    base = PREFLOP_STRENGTH.get(key, None)
    if base is None:
        # Fallback: calcular via rank médio
        base = (sum(v) - 4) / (28 - 4)  # normalizado 2+2=4 até A+A=28
    bonus = 0.04 if suited else 0.0
    return min(1.0, base + bonus)


# ─────────────────────────────────────────────
#  Contagem de outs
# ─────────────────────────────────────────────

def count_outs(my_cards, board) -> int:
    """Estima outs para completar draws (flush draw, straight draw)."""
    all_cards = list(my_cards) + list(board)
    suits = [c.suit for c in all_cards]
    values = sorted(set(card_rank(c) for c in all_cards))

    outs = 0
    # Flush draw (4 do mesmo naipe)
    suit_counts = Counter(suits)
    if max(suit_counts.values()) == 4:
        outs += 9

    # Straight draw
    for start in range(2, 11):
        window = set(range(start, start + 5))
        have = window & set(values)
        if len(have) == 4:
            outs += 4 if outs == 0 else 2  # evitar dupla contagem parcial

    return outs


# ─────────────────────────────────────────────
#  Bot principal
# ─────────────────────────────────────────────

class ITABot(Player):
    """
    Bot competitivo para heads-up Texas Hold'em.

    Estratégias combinadas:
    • Força de mão (pré-flop table + Monte Carlo pós-flop)
    • Pot odds + implied odds
    • Posição (BB age por último pós-flop)
    • Leitura do oponente (agressividade adaptativa)
    • Short-stack push/fold
    • Bluff calibrado com frequência mista
    • Proteção de stack: evita calls ruins com stack curto
    """

    def __init__(self, name, hand, chips):
        super().__init__(name, hand, chips)
        self.hand_count = 0
        self.opp_raises_total = 0    # total de raises do oponente na partida
        self.opp_folds_total = 0     # quantas vezes ele foldou (inferido)
        self.last_opp_bet = 0
        self.my_vpip = 0             # vezes que entrei voluntariamente
        self.aggression_history = [] # % de raises do oponente por mão

    # ── Helpers ──────────────────────────────

    def _is_bb(self, gv: GameView) -> bool:
        """[CORRIGIDO PELA ORGANIZAÇÃO — bug de detecção de posição]

        `dealer_position` é o índice do dealer na lista GLOBAL de jogadores da
        engine, e NÃO um valor relativo a este bot. Por isso a verificação
        original `dealer_position == 0` só acertava quando este bot ocupava o
        assento players[0] da partida, falhando em até 100% das mãos quando
        ocupava players[1].

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

    def _suited(self, gv: GameView) -> bool:
        return gv.my_hand[0].suit == gv.my_hand[1].suit

    def _opp_aggression(self) -> float:
        """Fração de mãos em que oponente foi agressivo (0-1)."""
        if not self.aggression_history:
            return 0.5
        return sum(self.aggression_history[-20:]) / len(self.aggression_history[-20:])

    def _raise_size(self, gv: GameView, multiplier: float = 2.5) -> int:
        """Tamanho de raise padrão (retorna total apostado na rodada)."""
        target = int(gv.current_bet + gv.big_blind * multiplier)
        return min(target, gv.my_chips)

    def _allin(self, gv: GameView) -> int:
        return gv.my_chips

    # ── Decisão principal ─────────────────────

    def decision(self, game_view: GameView) -> int:
        gv = game_view
        self.hand_count += 1
        bb = gv.big_blind
        sb = gv.small_blind
        pot = gv.pot
        to_call = gv.to_call
        my_stack = gv.my_chips
        opp = gv.opponents[0]
        opp_stack = opp.chips
        board = gv.board
        is_bb = self._is_bb(gv)
        suited = self._suited(gv)

        # Atualizar leitura do oponente
        if opp.current_bet_in_round > bb:
            self.opp_raises_total += 1
            self.aggression_history.append(1)
        else:
            self.aggression_history.append(0)
        self.last_opp_bet = opp.current_bet_in_round

        opp_agg = self._opp_aggression()

        # ── 1. SHORT STACK: Push/Fold ────────────────
        if my_stack < bb * 8:
            return self._short_stack(gv, suited, is_bb, opp_agg)

        # ── 2. PRÉ-FLOP ─────────────────────────────
        if not board:
            return self._preflop(gv, suited, is_bb, opp_agg)

        # ── 3. PÓS-FLOP (Flop / Turn / River) ────────
        return self._postflop(gv, is_bb, opp_agg)

    # ── Short stack (< 8 BBs): push or fold ──

    def _short_stack(self, gv, suited, is_bb, opp_agg) -> int:
        strength = preflop_strength(gv.my_hand, suited)
        to_call = gv.to_call
        my_stack = gv.my_chips
        bb = gv.big_blind

        # GTO aproximado: push com top ~50% das mãos em SB, ~65% em BB
        threshold = 0.45 if not is_bb else 0.35
        if strength >= threshold:
            return my_stack  # all-in
        if to_call == 0:
            return 0  # check grátis
        return -1  # fold

    # ── Pré-flop ──────────────────────────────

    def _preflop(self, gv, suited, is_bb, opp_agg) -> int:
        to_call = gv.to_call
        bb = gv.big_blind
        my_stack = gv.my_chips
        strength = preflop_strength(gv.my_hand, suited)

        # Nunca fold de graça no BB
        if to_call == 0:
            if strength > 0.70:
                return self._raise_size(gv, 2.5)
            return 0  # check

        # Pot odds pré-flop
        pot_after_call = gv.pot + to_call
        pot_odds = to_call / pot_after_call

        # Ajuste por agressividade do oponente
        # Oponente passivo → ele tem mão fraca; expandir range de call/3bet
        # Oponente agressivo → tighten up
        adj = (0.5 - opp_agg) * 0.08  # [-0.04, +0.04]
        eff_strength = strength + adj

        # Re-raise (3-bet) com mãos muito fortes
        if eff_strength > 0.82:
            size = self._raise_size(gv, 3.0)
            return size

        # Call com equidade suficiente
        if eff_strength > pot_odds + 0.05:
            return 0  # call

        # Squeeze / bluff 3-bet ocasional em posição (BB) com conectores
        if is_bb and eff_strength > 0.55 and random.random() < 0.20:
            return self._raise_size(gv, 2.8)

        # Fold
        return -1

    # ── Pós-flop ──────────────────────────────

    def _postflop(self, gv, is_bb, opp_agg) -> int:
        to_call = gv.to_call
        bb = gv.big_blind
        pot = gv.pot
        my_stack = gv.my_chips
        board = gv.board
        opp = gv.opponents[0]

        # Calcular força da mão atual (Monte Carlo)
        equity = hand_strength(gv.my_hand, board)

        # Outs para draws (se board incompleto)
        outs = 0
        if len(board) < 5:
            outs = count_outs(gv.my_hand, board)
            cards_left = 52 - 2 - len(board)
            draw_equity = outs / cards_left if cards_left > 0 else 0
            equity = max(equity, equity + draw_equity * 0.5)

        equity = min(1.0, equity)

        # ── Check → Bet com equity alta ou bluff ──
        if to_call == 0:
            return self._bet_or_check(gv, equity, is_bb, opp_agg)

        # ── Precisa pagar algo ──
        return self._call_or_fold(gv, equity, is_bb, opp_agg, outs)

    def _bet_or_check(self, gv, equity, is_bb, opp_agg) -> int:
        bb = gv.big_blind
        pot = gv.pot
        my_stack = gv.my_chips

        # Value bet com mão forte
        if equity > 0.68:
            size = max(int(pot * 0.65), bb)
            size = min(size, my_stack)
            return gv.current_bet + size  # raise relativo ao bet atual

        # Semi-bluff com draw razoável
        if equity > 0.45 and random.random() < 0.35:
            size = max(int(pot * 0.45), bb)
            return min(gv.current_bet + size, my_stack)

        # Bluff puro calibrado: mais frequente fora de posição (SB pós-flop)
        bluff_freq = 0.18 if is_bb else 0.12
        # Reduz bluff contra oponente agressivo (ele vai pagar/re-raise)
        bluff_freq *= (1.2 - opp_agg)
        if random.random() < bluff_freq:
            size = max(int(gv.pot * 0.55), bb)
            return min(gv.current_bet + size, my_stack)

        return 0  # check

    def _call_or_fold(self, gv, equity, is_bb, opp_agg, outs) -> int:
        to_call = gv.to_call
        pot = gv.pot
        bb = gv.big_blind
        my_stack = gv.my_chips
        board = gv.board

        pot_total = pot + to_call
        pot_odds = to_call / pot_total if pot_total > 0 else 1.0

        # Implied odds: se temos draw forte, vale chamar um pouco além das pot odds
        implied_bonus = 0.0
        if outs >= 8:   # flush draw ou straight draw aberto
            implied_bonus = 0.06
        elif outs >= 4:
            implied_bonus = 0.03

        effective_equity = equity + implied_bonus

        # Raise para proteger / extrair valor
        if effective_equity > 0.70:
            # Oponente agressivo: re-raise; passivo: call e armadilha
            if opp_agg > 0.55:
                target = int(gv.current_bet * 2.5)
                return min(target, my_stack)
            # Slowplay ocasional com nuts
            if effective_equity > 0.88 and random.random() < 0.30:
                return 0  # call (trap)
            target = int(gv.current_bet * 2.2)
            return min(target, my_stack)

        # Call com odds favoráveis
        if effective_equity > pot_odds + 0.03:
            # Mas não chamar demais com stack curto
            stack_ratio = my_stack / (gv.pot + my_stack + 1)
            if to_call > my_stack * 0.4 and effective_equity < 0.55:
                return -1
            return 0  # call

        # Fold
        return -1


# ─────────────────────────────────────────────
#  Função obrigatória
# ─────────────────────────────────────────────

def create_player() -> Player:
    return ITABot("ITABot", Hand(), 0)
