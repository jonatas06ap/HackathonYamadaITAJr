"""
Pinguim Rei v4 — Thresholds dinâmicos por nº oponentes
=======================================================

v3 (Pistache+Gordo fusion) funcionou em 4h (25% vs 15% Gordo, +10pp), mas
regrediu em 6h (12.5% vs 27.5%). Causa: range de raise (24%) muito amplo
em multi-way → showdown WR caiu de 43% (Gordo) pra 40% (v3). Apostou muito
e perdeu no showdown porque opps de 6h tinham range mais forte.

v4 adiciona THRESHOLDS DINÂMICOS por `n_active_opps`:

  n_factor = (n_active_opps - 1)
  PREMIUM = 0.60 + 0.025 * n_factor    # 6h: 0.70, 4h: 0.65, HU: 0.60
  PLAYABLE = 0.40 + 0.025 * n_factor   # 6h: 0.50, 4h: 0.45, HU: 0.40
  STRONG_POSTFLOP = 0.70 + 0.015 * n_factor

Em 6h: mid pair (hs=0.42) cai abaixo de PLAYABLE → fold (antes era playable).
Em 4h/HU: behavior idêntico ao v3 (que funcionou).

Mantém todas as features do v3:
- M1: fix de posição multi-way via current_bet - to_call
- Eval refinado pós-flop (PinguimGordo style)
- Branch pré-flop ativo (Pistache style: pot ≤ 3bb e current_bet ≤ 2bb)
- Sizing polarizado, mixed strategy borderline, tracker VPIP
- Multi-way bluff decay

Hipótese: WR 6h ≥ 22% (vs 12.5% v3), WR 4h ≥ 22% (vs 25% v3 — pequena regressão ok).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from collections import Counter
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from players.player import Player
from game.game_view import GameView
from cards.cards import Hand


VALORES = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 11, "Q": 12, "K": 13, "A": 14,
}


def cv(card) -> int:
    return VALORES[card.value]


# ─────────────────────────────────────────────────────────────
#  Avaliador best-5-of-7 (do PinguimGordo)
# ─────────────────────────────────────────────────────────────
def _eval5(cards) -> tuple:
    vals = sorted([cv(c) for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    val_counts = Counter(vals)
    counts = sorted(val_counts.values(), reverse=True)

    is_flush = max(Counter(suits).values()) == 5

    uv = sorted(set(vals))
    if 14 in uv: uv = [1] + uv
    is_straight = False
    straight_high = 0
    uv_desc = sorted(set(uv), reverse=True)
    for i in range(len(uv_desc) - 4):
        window = uv_desc[i:i + 5]
        if window[0] - window[4] == 4:
            is_straight = True
            straight_high = window[0]
            break

    if is_flush and is_straight: return (8, straight_high)
    if counts[0] == 4:
        q = max(v for v, c in val_counts.items() if c == 4)
        k = max(v for v in vals if v != q)
        return (7, q, k)
    if counts[0] == 3 and counts[1] == 2:
        t = max(v for v, c in val_counts.items() if c == 3)
        p = max(v for v, c in val_counts.items() if c == 2)
        return (6, t, p)
    if is_flush: return (5,) + tuple(vals[:5])
    if is_straight: return (4, straight_high)
    if counts[0] == 3:
        t = max(v for v, c in val_counts.items() if c == 3)
        ks = sorted([v for v in vals if v != t], reverse=True)[:2]
        return (3, t) + tuple(ks)
    if counts[0] == 2 and counts[1] == 2:
        pairs = sorted([v for v, c in val_counts.items() if c == 2], reverse=True)
        k = max(v for v in vals if v != pairs[0] and v != pairs[1])
        return (2, pairs[0], pairs[1], k)
    if counts[0] == 2:
        p = max(v for v, c in val_counts.items() if c == 2)
        ks = sorted([v for v in vals if v != p], reverse=True)[:3]
        return (1, p) + tuple(ks)
    return (0,) + tuple(vals[:5])


def best_hand(hole, board) -> tuple:
    all_cards = list(hole) + list(board)
    if len(all_cards) < 5:
        return _eval5(all_cards + all_cards)
    return max(_eval5(list(combo)) for combo in combinations(all_cards, 5))


# ─────────────────────────────────────────────────────────────
#  Chen formula pré-flop (do Baunilha/Pistache)
# ─────────────────────────────────────────────────────────────
def chen_preflop(hole) -> float:
    v1, v2 = cv(hole[0]), cv(hole[1])
    high, low = max(v1, v2), min(v1, v2)
    score = high * 2
    if high == low:
        score *= 2.5
    if hole[0].suit == hole[1].suit:
        score += 4
    gap = high - low
    if gap == 1: score -= 1
    elif gap == 2: score -= 2
    elif gap >= 3: score -= 4
    return min(score / 45.0, 1.0)


# ─────────────────────────────────────────────────────────────
#  Eval pós-flop refinado (do PinguimGordo)
# ─────────────────────────────────────────────────────────────
def evaluate_relative_strength(hole, board) -> dict:
    rank = best_hand(hole, board)
    cat = rank[0]

    board_vals = [cv(c) for c in board]
    max_board = max(board_vals) if board_vals else 0
    hole_vals = [cv(c) for c in hole]

    is_overpair = cat == 1 and rank[1] > max_board and rank[1] in hole_vals
    is_top_pair = cat == 1 and rank[1] == max_board and rank[1] in hole_vals

    suits_all = [c.suit for c in list(hole) + list(board)]
    max_suit_count = max(Counter(suits_all).values()) if suits_all else 0
    flush_draw = max_suit_count == 4

    board_suits = [c.suit for c in board]
    monotone_board = max(Counter(board_suits).values()) >= 3 if board_suits else False

    hs = 0.30
    if cat >= 6: hs = 0.95
    elif cat == 5: hs = 0.85
    elif cat == 4: hs = 0.78
    elif cat == 3: hs = 0.72
    elif cat == 2: hs = 0.62
    elif is_overpair: hs = 0.65
    elif is_top_pair: hs = 0.55
    elif cat == 1 and rank[1] in hole_vals: hs = 0.42
    elif cat == 1: hs = 0.30  # par no board apenas

    if flush_draw: hs += 0.20
    if monotone_board and cat < 5: hs -= 0.15

    return {"hs": min(0.98, max(0.05, hs)), "cat": cat, "flush_draw": flush_draw}


# ─────────────────────────────────────────────────────────────
#  Pinguim Rei v3
# ─────────────────────────────────────────────────────────────
class PinguimRei(Player):
    """Pistache-skeleton + eval refinado do PinguimGordo + fix de posição."""

    _PUSH_FOLD_BB = 8
    # Thresholds BASE (HU). Ajustados dinamicamente por n_active_opps em _thresholds().
    _PREMIUM_HS_BASE = 0.60
    _PLAYABLE_HS_BASE = 0.40
    _STRONG_POSTFLOP_HS_BASE = 0.70

    def _thresholds(self, n_active_opps: int):
        """Thresholds escalam com nº de opps ativos (tighter em multi-way)."""
        n_factor = max(0, n_active_opps - 1)
        premium = self._PREMIUM_HS_BASE + 0.025 * n_factor
        playable = self._PLAYABLE_HS_BASE + 0.025 * n_factor
        strong_pf = self._STRONG_POSTFLOP_HS_BASE + 0.015 * n_factor
        return premium, playable, strong_pf

    def __init__(self, name, hand, chips):
        super().__init__(name, hand, chips)
        # Tracker
        self.hands_played = 0
        self.opp_folds = 0
        self.opp_raises = 0
        self.opp_vpip_hands = 0
        self.opp_faced_bet = 0
        self.opp_fold_to_bet = 0
        # Estado da mão
        self._we_bet_this_hand = False
        self._opp_entered_pot = False
        self._last_we_bet_action = False
        self._board_len_last = -1
        self._hand_idx = 0
        # M1: detecção de posição (lock por mão)
        self._am_bb = False
        self._am_sb = False
        self._am_button = False
        self._position_locked = False
        self._last_pot_seen = -1

    def _is_new_hand(self, gv: GameView) -> bool:
        if self._last_pot_seen < 0:
            return True
        return gv.pot < self._last_pot_seen

    def _detect_position(self, gv: GameView):
        """M1: BB/SB/non-blind via my_invested = current_bet - to_call.

        Lock por mão. Reset por queda de pot (nova mão).
        Em 4+ jogadores, `dealer_position` é índice absoluto na list de players
        (incluindo o próprio bot), inútil para position relativa. Esse fix
        substitui a heurística buggy.
        """
        if self._is_new_hand(gv):
            self._position_locked = False

        # Lock só faz sentido pré-flop (pot baixo, ainda no início)
        if not self._position_locked and gv.pot <= gv.big_blind * 4:
            my_invested = gv.current_bet - gv.to_call
            if my_invested == gv.big_blind:
                self._am_bb = True
                self._am_sb = False
                self._am_button = False
            elif my_invested == gv.small_blind:
                self._am_bb = False
                self._am_sb = True
                self._am_button = False
            else:
                self._am_bb = False
                self._am_sb = False
                self._am_button = True
            self._position_locked = True

        self._last_pot_seen = gv.pot

    def _opp_vpip_rate(self) -> float:
        if self.hands_played <= 0: return 0.30
        return (self.opp_vpip_hands + 0.30 * 6) / (self.hands_played + 6)

    def _opp_fold_to_bet_rate(self) -> float:
        return (self.opp_fold_to_bet + 0.40 * 6) / (self.opp_faced_bet + 6)

    def _opp_is_passive_basic(self) -> bool:
        return (self.opp_raises / max(1, self.hands_played)) < 0.10

    def decision(self, game_view: GameView) -> int:
        try:
            return self._decide(game_view)
        except Exception:
            return 0

    def _decide(self, gv: GameView) -> int:
        self._detect_position(gv)

        opp = gv.opponents[0]
        bb = gv.big_blind
        pot = gv.pot
        to_call = gv.to_call
        my_chips = gv.my_chips
        board_len = len(gv.board)
        current_bet = gv.current_bet

        # Detecta nova mão (engine: board=3 sempre, então usa transição)
        is_new_hand_marker = (board_len == 3 and pot <= 3 * bb
                               and self._board_len_last != 3)
        if is_new_hand_marker or board_len == 0:
            if self.hands_played >= 1:
                if self._opp_entered_pot: self.opp_vpip_hands += 1
                if self._we_bet_this_hand:
                    self.opp_faced_bet += 1
                    if not opp.is_active: self.opp_fold_to_bet += 1
            self.hands_played += 1
            self._hand_idx += 1
            self._we_bet_this_hand = False
            self._opp_entered_pot = False
            self._last_we_bet_action = False

        self._board_len_last = board_len

        # Tracker leve
        if opp.current_bet_in_round > bb * 2:
            self.opp_raises += 1
        if opp.current_bet_in_round > 0:
            self._opp_entered_pot = True

        # M1: position (substitui `eu_sou_bb = dealer_position == 0`)
        eu_sou_bb = self._am_bb
        em_posicao = self._am_button  # non-blind ≈ IP em multi-way

        # Equity / stats
        n_active_opps = max(1, sum(1 for o in gv.opponents if o.is_active))
        spr = my_chips / pot if pot > 0 else 100
        pot_total_futuro = pot + to_call
        pot_odds = to_call / pot_total_futuro if pot_total_futuro > 0 else 0

        # Eval: pre-flop usa Chen (do Pistache), pós-flop usa eval refinado do Gordo
        if board_len == 0:
            hs = chen_preflop(gv.my_hand)
            flush_draw = False
        else:
            analise = evaluate_relative_strength(gv.my_hand, gv.board)
            hs = analise["hs"]
            flush_draw = analise["flush_draw"]

        # Opp classification
        vpip = self._opp_vpip_rate()
        fold_to_bet = self._opp_fold_to_bet_rate()
        sample_ok = self.hands_played >= 10
        opp_is_tight = sample_ok and vpip < 0.30
        opp_is_station = sample_ok and fold_to_bet < 0.20

        seed_tuple = (self._hand_idx, board_len)
        rng_val = (abs(hash(seed_tuple)) % 1_000_003) / 1_000_003.0

        my_round_inv = current_bet - to_call
        all_in_target = my_round_inv + my_chips

        def cap(target):
            if target <= current_bet: return 0
            return min(target, all_in_target)

        def mark_bet(target):
            if target > current_bet:
                self._we_bet_this_hand = True
                self._last_we_bet_action = True
            return target

        # Thresholds dinâmicos (v4)
        premium_th, playable_th, strong_pf_th = self._thresholds(n_active_opps)
        borderline_low = playable_th
        borderline_high = premium_th

        # ── A. Push/Fold ────────────────────────────────────────────
        if my_chips < bb * self._PUSH_FOLD_BB:
            if hs > 0.40 or (em_posicao and to_call == 0):
                return mark_bet(all_in_target)
            return 0 if to_call == 0 else -1

        # ── B. Pré-flop (Pistache style: board_len==0 OR pot pequeno + min raise) ─
        is_preflop_phase = (
            board_len == 0
            or (board_len == 3 and pot <= 3 * bb and current_bet <= 2 * bb)
        )
        if is_preflop_phase:
            # Premium → raise sizing polarizado
            if hs > premium_th:
                if rng_val < 0.30:
                    size = max(int(bb * 2.2), int(pot * 0.6))
                elif rng_val < 0.80:
                    size = max(int(bb * 3), int(pot * 0.85))
                else:
                    size = max(int(bb * 4.5), int(pot * 1.3))
                return mark_bet(cap(current_bet + size))

            # Jogável: call ou borderline 3-bet light
            if hs > playable_th:
                if borderline_low <= hs < borderline_high:
                    if opp_is_tight and rng_val < 0.35:
                        size = max(int(bb * 2.5), int(pot * 0.7))
                        return mark_bet(cap(current_bet + size))
                return 0

            # Steal vs tight/passivo + posição
            if em_posicao and (opp_is_tight or self._opp_is_passive_basic()) \
                    and to_call <= bb and rng_val < (0.55 if opp_is_tight else 0.30):
                size = max(int(bb * 2.2), int(pot * 0.65))
                return mark_bet(cap(current_bet + size))
            return 0 if to_call == 0 else -1

        # ── C. Pós-flop ─────────────────────────────────────────────
        is_river = (board_len == 5)

        # C.1. Mão muito forte: value bet polarizado
        if hs > strong_pf_th:
            if spr < 2:
                return mark_bet(all_in_target)
            if rng_val < 0.25:
                size = max(int(bb * 2), int(pot * 0.35))    # block
            elif rng_val < 0.75:
                size = max(int(bb * 3), int(pot * 0.65))    # value médio
            else:
                size = max(int(bb * 4), int(pot * 1.20))    # overbet
            return mark_bet(cap(current_bet + size))

        # C.2. Mão mediana — borderline mix com penalidade multi-way
        if borderline_low <= hs < borderline_high:
            if to_call == 0:
                # Não pode foldar. Multi-way: c-bet menos frequente
                cbet_threshold = 0.50 - 0.10 * (n_active_opps - 1)
                cbet_threshold = max(0.20, cbet_threshold)
                if rng_val < cbet_threshold and not opp_is_station:
                    size = max(int(bb * 2), int(pot * 0.5))
                    return mark_bet(cap(current_bet + size))
                return 0
            # Há to_call. Pot odds
            if hs > pot_odds + 0.10:
                if rng_val < 0.20 and not opp_is_station:
                    size = max(int(bb * 2), int(pot * 0.55))
                    return mark_bet(cap(current_bet + size))
                return 0
            # EV ambíguo
            if rng_val < 0.30: return -1
            return 0 if to_call <= pot * 0.3 else -1

        # C.3. Hand strength baixo
        if to_call == 0:
            # Bluff balancing river (Pistache style)
            if is_river and 0.15 <= hs < 0.35 and rng_val < 0.30:
                size = max(int(bb * 2), int(pot * 0.3))
                return mark_bet(cap(current_bet + size))
            if opp_is_station:
                return 0
            # Bluff posicional com decay multi-way
            base_bluff_p = 0.20 if opp_is_tight else 0.10
            bluff_p = base_bluff_p * max(0.2, 1 - 0.2 * (n_active_opps - 1))
            if em_posicao and opp.current_bet_in_round == 0 and rng_val < bluff_p:
                size = max(int(bb * 2), int(pot * 0.40))
                return mark_bet(cap(current_bet + size))
            return 0

        return -1


def create_player() -> Player:
    return PinguimRei("Pinguim_Rei", Hand(), 0)
