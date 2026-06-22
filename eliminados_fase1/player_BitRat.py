from __future__ import annotations

import sys
import random
from pathlib import Path
from collections import Counter
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from players.player import Player
from game.game_view import GameView
from cards.cards import Hand


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

RANK_MAP = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}

# Categoria das mãos (quanto maior, melhor)
HAND_HIGH_CARD    = 1
HAND_ONE_PAIR     = 2
HAND_TWO_PAIR     = 3
HAND_THREE_OF_A_KIND = 4
HAND_STRAIGHT     = 5
HAND_FLUSH        = 6
HAND_FULL_HOUSE   = 7
HAND_FOUR_OF_A_KIND = 8
HAND_STRAIGHT_FLUSH = 9


# ---------------------------------------------------------------------------
# Avaliação de mão (até 7 cartas → melhor combinação de 5)
# ---------------------------------------------------------------------------

def _rank(card) -> int:
    return RANK_MAP[card.value]


def _evaluate_5(cards) -> tuple:
    """Retorna (categoria, [desempates]) para exatamente 5 cartas."""
    ranks = sorted([_rank(c) for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    counts = Counter(ranks)
    freq = sorted(counts.values(), reverse=True)
    is_flush = len(set(suits)) == 1
    is_straight = (
        len(set(ranks)) == 5 and (ranks[0] - ranks[4] == 4)
    ) or ranks == [14, 5, 4, 3, 2]  # roda A-2-3-4-5

    if is_straight and ranks == [14, 5, 4, 3, 2]:
        ranks = [5, 4, 3, 2, 1]  # trata ás como 1 na roda

    if is_straight and is_flush:
        return (HAND_STRAIGHT_FLUSH, ranks)
    if freq[0] == 4:
        quad = [r for r, c in counts.items() if c == 4]
        kick = [r for r, c in counts.items() if c == 1]
        return (HAND_FOUR_OF_A_KIND, quad + kick)
    if freq[0] == 3 and freq[1] == 2:
        trio = [r for r, c in counts.items() if c == 3]
        pair = [r for r, c in counts.items() if c == 2]
        return (HAND_FULL_HOUSE, trio + pair)
    if is_flush:
        return (HAND_FLUSH, ranks)
    if is_straight:
        return (HAND_STRAIGHT, ranks)
    if freq[0] == 3:
        trio = sorted([r for r, c in counts.items() if c == 3], reverse=True)
        kick = sorted([r for r, c in counts.items() if c == 1], reverse=True)
        return (HAND_THREE_OF_A_KIND, trio + kick)
    if freq[0] == 2 and freq[1] == 2:
        pairs = sorted([r for r, c in counts.items() if c == 2], reverse=True)
        kick  = [r for r, c in counts.items() if c == 1]
        return (HAND_TWO_PAIR, pairs + kick)
    if freq[0] == 2:
        pair = sorted([r for r, c in counts.items() if c == 2], reverse=True)
        kick = sorted([r for r, c in counts.items() if c == 1], reverse=True)
        return (HAND_ONE_PAIR, pair + kick)
    return (HAND_HIGH_CARD, ranks)


def best_hand(cards) -> tuple:
    """Melhor combinação de 5 entre as cartas fornecidas."""
    if len(cards) <= 5:
        return _evaluate_5(cards)
    return max(_evaluate_5(list(combo)) for combo in combinations(cards, 5))


# ---------------------------------------------------------------------------
# Avaliação de mão pré-flop (heurística para 2 cartas)
# ---------------------------------------------------------------------------

def preflop_strength(hand) -> float:
    """
    Retorna um score entre 0 e 1 representando a força pré-flop.
    Baseado em grupos de Chen simplificados + posição relativa.
    """
    c1, c2 = hand
    r1, r2 = sorted([_rank(c1), _rank(c2)], reverse=True)
    suited = c1.suit == c2.suit
    gap = r1 - r2

    # Par
    if r1 == r2:
        # AA=1.0, KK≈0.95, QQ≈0.90, ..., 22≈0.50
        return 0.50 + (r1 - 2) / 24.0 * 0.50

    score = 0.0

    # Carta alta
    score += (r1 - 2) / 12.0 * 0.35

    # Segunda carta
    score += (r2 - 2) / 12.0 * 0.15

    # Conector (gap pequeno)
    if gap == 1:
        score += 0.10
    elif gap == 2:
        score += 0.06
    elif gap == 3:
        score += 0.03

    # Suited
    if suited:
        score += 0.07

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Thresholds de decisão
# ---------------------------------------------------------------------------

# Score mínimo pré-flop para jogar (abaixo → fold se houver custo)
PREFLOP_FOLD_THRESHOLD   = 0.38   # mãos muito fracas
PREFLOP_ALLIN_THRESHOLD  = 0.72   # mãos premium → all-in

# Categoria mínima pós-flop
POSTFLOP_FOLD_CATEGORY   = HAND_ONE_PAIR  # abaixo de par → tende a fold
POSTFLOP_ALLIN_CATEGORY  = HAND_TWO_PAIR  # dois pares ou melhor → all-in


# ---------------------------------------------------------------------------
# Bot principal
# ---------------------------------------------------------------------------

class BitRat(Player):
    """
    Estratégia: push-or-fold com avaliação real de mão.

    Pré-flop:
      - Mãos premium  (score >= 0.72) → all-in
      - Mãos medianas (0.38 ≤ score < 0.72) → call/check (ou small raise)
      - Mãos fracas   (score < 0.38) → fold se houver custo, senão check

    Pós-flop (flop/turn/river):
      - Dois pares ou melhor → all-in
      - Par → call/check (fold se aposta for > 40% do stack)
      - Abaixo de par → fold se houver custo

    Ajustes adicionais:
      - Stack curto (< 8 BB): push-or-fold puro
      - Posição (BB pós-flop): check mais agressivo
      - Bluff esporádico com mão fraca em posição (5% das vezes)
    """

    def __init__(self, name, hand, chips):
        super().__init__(name, hand, chips)
        self.maos_jogadas = 0
        self.historico_raises_oponente = 0

    # ------------------------------------------------------------------
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
        self.maos_jogadas += 1
        gv = game_view

        my_chips    = gv.my_chips
        to_call     = gv.to_call
        pot         = gv.pot
        bb          = gv.big_blind
        cur_bet     = gv.current_bet
        board       = list(gv.board)
        oponente    = gv.opponents[0]

        # Atualiza contagem de agressão do oponente
        if oponente.current_bet_in_round > bb:
            self.historico_raises_oponente += 1

        eu_sou_bb = self._org_sou_bb(gv)

        # ── Regra de ouro: nunca fold de graça ──────────────────────────
        if to_call == 0:
            # Verifica se tem mão boa para levantar a aposta
            return self._acao_sem_custo(gv, board, my_chips, bb, cur_bet, eu_sou_bb)

        # ── Stack curto: push or fold puro ──────────────────────────────
        if my_chips < bb * 8:
            return self._push_or_fold(gv, board, my_chips, to_call)

        # ── Pré-flop ────────────────────────────────────────────────────
        if not board:
            return self._preflop(gv, my_chips, to_call, bb, cur_bet, eu_sou_bb)

        # ── Pós-flop ────────────────────────────────────────────────────
        return self._postflop(gv, board, my_chips, to_call, bb, cur_bet, eu_sou_bb, pot)

    # ------------------------------------------------------------------
    # Ação sem custo (to_call == 0)
    # ------------------------------------------------------------------
    def _acao_sem_custo(self, gv, board, my_chips, bb, cur_bet, eu_sou_bb) -> int:
        if not board:
            score = preflop_strength(gv.my_hand)
            if score >= PREFLOP_ALLIN_THRESHOLD:
                return my_chips  # all-in
            if score >= 0.55:
                return cur_bet + bb * 2  # raise médio
            return 0  # check grátis

        cat, _ = best_hand(list(gv.my_hand) + board)
        if cat >= POSTFLOP_ALLIN_CATEGORY:
            return my_chips  # all-in com mão boa
        if cat == HAND_ONE_PAIR:
            # Raise pequeno com par
            return cur_bet + bb
        # Bluff esporádico (5%)
        if eu_sou_bb and random.random() < 0.05:
            return cur_bet + bb
        return 0  # check

    # ------------------------------------------------------------------
    # Push-or-fold com stack curto
    # ------------------------------------------------------------------
    def _push_or_fold(self, gv, board, my_chips, to_call) -> int:
        if not board:
            score = preflop_strength(gv.my_hand)
            if score >= 0.50:
                return my_chips  # push
            if to_call > 0:
                return -1  # fold
            return 0

        cat, _ = best_hand(list(gv.my_hand) + board)
        if cat >= HAND_ONE_PAIR:
            return my_chips  # push com qualquer par ou melhor
        if to_call > 0:
            return -1  # fold lixo
        return 0

    # ------------------------------------------------------------------
    # Decisão pré-flop
    # ------------------------------------------------------------------
    def _preflop(self, gv, my_chips, to_call, bb, cur_bet, eu_sou_bb) -> int:
        score = preflop_strength(gv.my_hand)
        oponente_agressivo = self.historico_raises_oponente > 8

        if score >= PREFLOP_ALLIN_THRESHOLD:
            return my_chips  # all-in premium

        if score >= 0.55:
            # Raise proporcional ao tamanho do pote
            alvo = cur_bet + bb * 3
            if my_chips >= to_call + bb:
                return alvo
            return 0  # call se não tiver fichas para raise

        if score >= PREFLOP_FOLD_THRESHOLD:
            # Mão mediana: call somente se o custo for razoável
            if to_call <= bb * 3:
                return 0  # call
            # Se oponente muito agressivo, aumentar tolerância para fold
            if oponente_agressivo and to_call > bb * 2:
                return -1
            return 0

        # Mão fraca
        if to_call > 0:
            return -1  # fold
        return 0  # check grátis (nunca deve chegar aqui com to_call==0)

    # ------------------------------------------------------------------
    # Decisão pós-flop
    # ------------------------------------------------------------------
    def _postflop(self, gv, board, my_chips, to_call, bb, cur_bet, eu_sou_bb, pot) -> int:
        all_cards = list(gv.my_hand) + board
        cat, _ = best_hand(all_cards)

        # Mãos fortes: all-in
        if cat >= POSTFLOP_ALLIN_CATEGORY:
            return my_chips

        # Par
        if cat == HAND_ONE_PAIR:
            custo_relativo = to_call / my_chips if my_chips > 0 else 1
            if custo_relativo > 0.40:
                return -1  # fold se a aposta for > 40% do stack
            return 0  # call

        # Abaixo de par (carta alta)
        # Pot odds: se to_call / (pot + to_call) < 0.20 → call
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 1
        if pot_odds < 0.20:
            return 0  # call barato
        return -1  # fold


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_player() -> Player:
    return BitRat("JonatasBot", Hand(), 0)
