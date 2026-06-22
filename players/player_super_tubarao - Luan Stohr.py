"""
Super Tubarão — Tubarão com fixes cirúrgicos
==============================================

Tubarão é o ÚNICO bot do ecossistema que empata Camelo (50%, único 0% loss
em N=40). Sua força vem da DISCIPLINA EXTREMA: tight pré-flop, value bet
correto, sem mixed strategy. Camelo profila "tight" mas não consegue
explorar porque tight = pouco fold equity disponível.

Super Tubarão preserva 100% do estilo e adiciona 2 fixes cirúrgicos:

1. **POSITION FIX** — `eu_sou_bb = (dealer_position == 0)` do original
   assume bot sempre no índice 1. Errado 50% das partidas → 50% das
   decisões posicionais (3-bet light, steal, bluff em posição, blefe
   no flop) usam info invertida. Substituído por `_detect_position`
   via `opp.current_bet_in_round` no início da mão (robusto a ambas
   as ordens).

2. **HAND EVAL PROPER best-5-of-7** — o original conta frequência em
   TODAS as cartas (hand+board), tipo "if max_count==2 then pair".
   Isso confunde categorias em edge cases (ex: trips+pair na board
   conta como full house mas só se houver 2 pares; se houver só par
   no board e meu hand tem ou trinca de borda, eval erra). Substituído
   por `_eval_5`+`_eval_7` real do range_sym — calcula score exato
   considerando a melhor 5-card combo das 7 disponíveis.

DELIBERADAMENTE PRESERVADO:
- Pre-flop tier code é dead code (`not board` nunca True no engine —
  board tem 3 cartas no 1º bet_round). Tubarão acaba sendo TIGHT
  emergente — só joga pré-flop hands que JÁ conectam com flop.
  Isso é o que empata Camelo. Não tocar.
- Todos os thresholds, blefes, sizings. ZERO mixed strategy adicional.
- Atributos e nomenclatura (raises_do_oponente, etc.) — preserva
  identidade.

Hipótese: +5-10pp vs Camelo (54-60%), pequena melhora vs maioria.
Não vai esmagar — vai EMPATAR/VENCER decisivamente.
"""
from __future__ import annotations

import sys
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from players.player import Player
from game.game_view import GameView
from cards.cards import Hand, Card


# ═══════════════════════════════════════════════════════════════════════════════
# Hand eval proper best-5-of-7 (idêntico ao range_sym)
# ═══════════════════════════════════════════════════════════════════════════════

_RANK_MAP = {"2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, "9": 7,
             "10": 8, "J": 9, "Q": 10, "K": 11, "A": 12}
_SUIT_MAP = {"s": 0, "h": 1, "d": 2, "c": 3}
_BASE5 = 759375


def _card_to_int(card: Card) -> int:
    return _RANK_MAP[card.value] * 4 + _SUIT_MAP[card.suit]


def _eval_5(cards) -> int:
    """Avalia 5 cartas (ints rank*4+suit). Maior = melhor."""
    c0, c1, c2, c3, c4 = cards[0], cards[1], cards[2], cards[3], cards[4]
    r0, r1, r2, r3, r4 = c0 >> 2, c1 >> 2, c2 >> 2, c3 >> 2, c4 >> 2
    ranks = [r0, r1, r2, r3, r4]; ranks.sort(reverse=True)
    counts = [0] * 13
    counts[r0] += 1; counts[r1] += 1; counts[r2] += 1; counts[r3] += 1; counts[r4] += 1
    max_c = max(counts)
    flush = (c0 & 3) == (c1 & 3) == (c2 & 3) == (c3 & 3) == (c4 & 3)
    is_straight = False
    high = ranks[0]
    if max_c == 1:
        if ranks[0] - ranks[4] == 4:
            is_straight = True
        elif ranks[0] == 12 and ranks[1] == 3 and ranks[2] == 2 and ranks[3] == 1 and ranks[4] == 0:
            is_straight = True; high = 3
    if flush and is_straight: return 8 * _BASE5 + high
    if max_c == 4:
        quad = counts.index(4); kicker = next(r for r in ranks if r != quad)
        return 7 * _BASE5 + quad * 15 + kicker
    trip = -1; pair_a = -1; pair_b = -1
    for r in range(12, -1, -1):
        cnt = counts[r]
        if cnt == 3 and trip == -1: trip = r
        elif cnt == 2:
            if pair_a == -1: pair_a = r
            elif pair_b == -1: pair_b = r
    if trip >= 0 and pair_a >= 0: return 6 * _BASE5 + trip * 15 + pair_a
    if flush:
        s = 0
        for r in ranks: s = s * 15 + r
        return 5 * _BASE5 + s
    if is_straight: return 4 * _BASE5 + high
    if trip >= 0:
        k1 = -1; k2 = -1
        for r in ranks:
            if r == trip: continue
            if k1 == -1: k1 = r
            elif k2 == -1: k2 = r; break
        return 3 * _BASE5 + trip * 225 + k1 * 15 + k2
    if pair_a >= 0 and pair_b >= 0:
        kicker = next(r for r in ranks if r != pair_a and r != pair_b)
        return 2 * _BASE5 + pair_a * 225 + pair_b * 15 + kicker
    if pair_a >= 0:
        k1 = -1; k2 = -1; k3 = -1
        for r in ranks:
            if r == pair_a: continue
            if k1 == -1: k1 = r
            elif k2 == -1: k2 = r
            elif k3 == -1: k3 = r; break
        return 1 * _BASE5 + pair_a * 3375 + k1 * 225 + k2 * 15 + k3
    s = 0
    for r in ranks: s = s * 15 + r
    return s


def _eval_7_category(seven_cards) -> int:
    """Retorna apenas a categoria (rank 0-8) da melhor mão de 5 entre 7."""
    cards_int = [_card_to_int(c) for c in seven_cards]
    best_score = 0
    for combo in combinations(cards_int, 5):
        s = _eval_5(combo)
        if s > best_score: best_score = s
    return best_score // _BASE5  # categoria (0-8)


# ═══════════════════════════════════════════════════════════════════════════════
# Super Tubarão — Tubarão fixado
# ═══════════════════════════════════════════════════════════════════════════════

class SuperTubarao(Player):
    VALORES = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
               "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14}

    def __init__(self, name: str, hand: Hand, chips: int):
        super().__init__(name, hand, chips)
        self.maos_jogadas = 0
        self.raises_do_oponente = 0
        # FIX #1: posição robusta
        self._am_bb = None
        self._last_pot_seen = -1

    def _detect_position(self, gv: GameView):
        """No início da mão, opp.current_bet_in_round == 0 → eu sou SB
        (eu acto primeiro). Else eu sou BB (opp já investiu o blind/call)."""
        if self._am_bb is None or gv.pot < self._last_pot_seen:
            opp = gv.opponents[0]
            self._am_bb = (opp.current_bet_in_round > 0)
        self._last_pot_seen = gv.pot

    def _avaliar_forca(self, my_hand, board):
        """FIX #2: eval proper best-5-of-7. Mantém API (dict com rank+draw)."""
        todas_cartas = list(my_hand) + list(board)
        if not todas_cartas or len(todas_cartas) < 5:
            return {"rank": 0, "draw": None}

        # Categoria proper via best-5-of-7
        rank = _eval_7_category(todas_cartas)

        # Detecção de draws (mantém lógica original)
        valores = [self.VALORES[c.value] for c in todas_cartas]
        naipes = [c.suit for c in todas_cartas]
        max_naipe = max(Counter(naipes).values()) if naipes else 0
        is_flush_draw = (max_naipe == 4)

        valores_unicos = sorted(list(set(valores)))
        if 14 in valores_unicos:
            valores_unicos.insert(0, 1)
        max_seq = 1
        seq_atual = 1
        for i in range(1, len(valores_unicos)):
            if valores_unicos[i] == valores_unicos[i-1] + 1:
                seq_atual += 1
                max_seq = max(max_seq, seq_atual)
            else:
                seq_atual = 1
        is_straight_draw = (max_seq == 4 and rank < 4)  # só draw se não já é straight

        draw = None
        if is_flush_draw and rank < 5: draw = "flush_draw"
        elif is_straight_draw: draw = "straight_draw"

        return {"rank": rank, "draw": draw}

    def decision(self, game_view: GameView) -> int:
        # FIX #1: detectar posição robusta antes de tudo
        self._detect_position(game_view)
        self.maos_jogadas += 1

        # --- Lendo a Mesa (idêntico) ---
        to_call = game_view.to_call
        current_bet = game_view.current_bet
        pot = game_view.pot
        bb = game_view.big_blind
        meu_stack = game_view.my_chips
        board = game_view.board
        oponente = game_view.opponents[0]

        # Tracking de agressão (idêntico)
        if oponente.current_bet_in_round > bb:
            self.raises_do_oponente += 1

        agressividade = self.raises_do_oponente / max(1, self.maos_jogadas)
        oponente_maniaco = agressividade > 0.4

        # --- Posição (FIX #1) ---
        eu_sou_bb = (self._am_bb is True)
        em_posicao = eu_sou_bb  # BB age por último no pós-flop (engine convention)

        # --- Matemática do Pote (idêntico) ---
        pot_total = pot + to_call
        pot_odds = to_call / pot_total if pot_total > 0 else 0
        spr = meu_stack / max(1, pot)

        default = 0 if to_call == 0 else -1

        c1, c2 = game_view.my_hand
        v1, v2 = self.VALORES[c1.value], self.VALORES[c2.value]
        high_card = max(v1, v2)
        is_pocket_pair = (v1 == v2)
        is_suited = (c1.suit == c2.suit)

        # 1. MODO SOBREVIVÊNCIA (idêntico)
        if meu_stack <= bb * 8:
            if is_pocket_pair or high_card >= 10 or (v1 >= 8 and v2 >= 8):
                return meu_stack
            return default

        # 2. PRÉ-FLOP (dead code no engine — preservado intencionalmente)
        # `not board` nunca True; Tubarão pula direto pro pós-flop.
        if not board:
            if (is_pocket_pair and v1 >= 10) or (high_card == 14 and min(v1, v2) >= 12):
                return current_bet + (bb * 3)
            if is_pocket_pair or high_card >= 11 or (is_suited and high_card >= 9):
                if to_call > bb * 3 and oponente_maniaco and not is_pocket_pair:
                    return default
                if to_call <= bb * 2 and random.random() < 0.2:
                    return current_bet + int(bb * 2.5)
                return 0
            return default

        # 3. PÓS-FLOP (idêntico — usa novo _avaliar_forca)
        forca = self._avaliar_forca(game_view.my_hand, board)
        rank = forca["rank"]
        draw = forca["draw"]

        # 3.1. JOGO FORTE (idêntico)
        if rank >= 3:
            if to_call == 0:
                return current_bet + int(pot * 0.6)
            if spr < 2 or to_call > pot * 0.5:
                return meu_stack
            return 0

        # 3.2. JOGO MÉDIO (idêntico)
        if rank >= 1:
            if pot_odds < 0.33:
                return 0
            if to_call == 0 and em_posicao and random.random() < 0.4:
                return current_bet + int(pot * 0.4)
            return default

        # 3.3. PROJETOS (idêntico)
        if draw:
            equidade_estimada = 0.35 if draw == "flush_draw" else 0.18
            if pot_odds < equidade_estimada:
                return 0
            if to_call == 0 and random.random() < 0.5:
                return current_bet + int(pot * 0.5)

        # 3.4. AR PURO (idêntico)
        if to_call == 0:
            if em_posicao and not oponente_maniaco and random.random() < 0.25:
                return current_bet + int(pot * 0.33)
            return 0

        return default


def create_player() -> Player:
    return SuperTubarao("Super_Tubarao", Hand(), 0)
