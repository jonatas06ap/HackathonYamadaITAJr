from __future__ import annotations
import random
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cards.cards import Card, Hand
from cards.sequences import RANK_CARTA_ALTA, RANK_DOIS_PARES, RANK_QUADRA, RANK_STRAIGHT, RANK_TRINCA, RANK_UM_PAR, VALORES, avaliar_cinco_cartas, score_cinco_cartas
from game.game_view import GameView
from players.player import Player
_VALUES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
_SUITS = ['s', 'h', 'd', 'c']
_FULL_DECK = [Card(v, s) for v in _VALUES for s in _SUITS]

def _card_key(c: Card) -> tuple[str, str]:
    return (c.value, c.suit)

def _best_score(cards: list[Card]) -> int:
    n = len(cards)
    if n == 5:
        return score_cinco_cartas(cards)
    best = 0
    for combo in combinations(cards, 5):
        s = score_cinco_cartas(list(combo))
        if s > best:
            best = s
    return best

def equity_monte_carlo(my_hand, board, max_sims=55, rng=None):
    if rng is None:
        rng = random
    visible = set((_card_key(c) for c in my_hand)) | set((_card_key(c) for c in board))
    remaining = [c for c in _FULL_DECK if _card_key(c) not in visible]
    needed_board = 5 - len(board)
    sample_n = needed_board + 2
    wins = 0
    ties = 0
    n_done = 0
    for _ in range(max_sims):
        sampled = rng.sample(remaining, sample_n)
        opp_cards = sampled[:2]
        extra_board = sampled[2:]
        full_board = list(board) + extra_board
        my_s = _best_score(list(my_hand) + full_board)
        opp_s = _best_score(opp_cards + full_board)
        if my_s > opp_s:
            wins += 1
        elif my_s == opp_s:
            ties += 1
        n_done += 1
    if n_done == 0:
        return (0.5, 0)
    return ((wins + 0.5 * ties) / n_done, n_done)

class NemesisBot(Player):

    def __init__(self, name, hand, chips):
        super().__init__(name, hand, chips)
        self._rng = random.Random()
        self._sims_by_n = {3: 55, 4: 55, 5: 55}
        self._opp_folds = 0
        self._opp_calls = 0
        self._opp_raises = 0
        self._opp_hand_actions = 0
        self._opp_river_barrels = 0
        self._last_opp_bet_in_round = 0
        self._last_pot = 0
        self._last_board_n = 0
        self._opp_raised_this_round = False

    def _best_eval(self, cards):
        n = len(cards)
        if n < 5:
            vals = sorted([VALORES[c.value] for c in cards], reverse=True)
            return (RANK_CARTA_ALTA, vals + [0] * (5 - n))
        best = (-1, [])
        for combo in combinations(cards, 5):
            r, t = avaliar_cinco_cartas(list(combo))
            if r > best[0] or (r == best[0] and t > best[1]):
                best = (r, t)
        return best

    def _hand_strength(self, my_hand, board):
        all_cards = list(my_hand) + list(board)
        rank, tie = self._best_eval(all_cards)
        info = {'rank': rank, 'tie': tie, 'top_pair': False, 'overpair': False, 'paired_board': False, 'flush_draw': False, 'trips_on_board': False, 'second_pair': False, 'weak_pair': False}
        board_vals = [VALORES[c.value] for c in board]
        bs_count = Counter((c.suit for c in board))
        bv_count = Counter(board_vals)
        info['paired_board'] = any((c >= 2 for c in bv_count.values()))
        info['trips_on_board'] = any((c >= 3 for c in bv_count.values()))
        my_vals = [VALORES[c.value] for c in my_hand]
        if board_vals:
            sorted_board = sorted(set(board_vals), reverse=True)
            top_board = sorted_board[0]
            info['top_pair'] = any((v == top_board for v in my_vals))
            if len(sorted_board) >= 2 and rank == RANK_UM_PAR:
                second = sorted_board[1]
                if any((v == second for v in my_vals)):
                    info['second_pair'] = True
            if rank == RANK_UM_PAR and (not info['top_pair']):
                info['weak_pair'] = True
        if len(my_hand) >= 2 and my_hand[0].value == my_hand[1].value:
            if board_vals and my_vals[0] > max(board_vals):
                info['overpair'] = True
        my_suits = [c.suit for c in my_hand]
        for suit, cnt_b in bs_count.items():
            total = cnt_b + sum((1 for s in my_suits if s == suit))
            if total == 4 and len(board) < 5:
                info['flush_draw'] = True
                break
        return info

    def _board_texture(self, board):
        if not board:
            return {'wet_score': 0.0, 'suited': False, 'connected': False, 'paired': False, 'ace_high': False}
        suits = Counter((c.suit for c in board))
        vals = sorted([VALORES[c.value] for c in board])
        info = {'wet_score': 0.0, 'suited': max(suits.values()) >= 3, 'connected': False, 'paired': any((v >= 2 for v in Counter(vals).values())), 'ace_high': max(vals) == 14}
        if len(vals) >= 3:
            info['connected'] = max(vals) - min(vals) <= 4
        wet = 0.0
        if info['suited']:
            wet += 0.5
        if info['connected']:
            wet += 0.3
        if max(suits.values()) >= 2:
            wet += 0.2
        info['wet_score'] = min(wet, 1.0)
        return info

    def _my_stack_bbs(self, gv):
        return gv.my_chips // max(1, gv.big_blind)

    def _opp_all_in(self, gv):
        return gv.opponents[0].chips == 0

    def _am_first_in_round(self, gv):
        return gv.opponents[0].current_bet_in_round == 0

    def _bet_amount(self, gv, mult):
        cb = gv.current_bet
        jitter = 1.0 + (self._rng.random() - 0.5) * 0.14
        target = int(cb * mult * jitter)
        floor = cb + gv.big_blind
        return max(target, floor)

    def _all_in(self, gv):
        return gv.my_chips + gv.current_bet

    def _equity(self, gv):
        n = self._sims_by_n.get(len(gv.board), 55)
        eq, _ = equity_monte_carlo(gv.my_hand, gv.board, n, self._rng)
        return eq

    def _track_opp(self, gv):
        opp = gv.opponents[0]
        board_n = len(gv.board)
        if gv.pot < self._last_pot - 1:
            self._last_opp_bet_in_round = 0
            self._opp_raised_this_round = False
        if board_n != self._last_board_n:
            self._last_opp_bet_in_round = 0
            self._opp_raised_this_round = False
        if opp.current_bet_in_round > self._last_opp_bet_in_round:
            delta = opp.current_bet_in_round - self._last_opp_bet_in_round
            if opp.current_bet_in_round > gv.big_blind and delta > gv.big_blind:
                self._opp_raises += 1
                self._opp_raised_this_round = True
                if board_n == 5:
                    self._opp_river_barrels += 1
            else:
                self._opp_calls += 1
            self._opp_hand_actions += 1
        if not opp.is_active and opp.chips > 0:
            self._opp_folds += 1
        self._last_opp_bet_in_round = opp.current_bet_in_round
        self._last_pot = gv.pot
        self._last_board_n = board_n

    def _opp_profile(self):
        total = self._opp_calls + self._opp_raises + self._opp_folds
        if total < 5:
            return {'bucket': 'unknown', 'aggr': 0.5, 'fold_freq': 0.3, 'call_freq': 0.5}
        fold_freq = self._opp_folds / total
        aggr = self._opp_raises / max(1, self._opp_raises + self._opp_calls)
        call_freq = self._opp_calls / max(1, self._opp_calls + self._opp_raises)
        if fold_freq > 0.5:
            bucket = 'passive_folder'
        elif aggr > 0.45:
            bucket = 'aggressive'
        elif call_freq > 0.6 and fold_freq < 0.2:
            bucket = 'calling_station'
        else:
            bucket = 'balanced'
        return {'bucket': bucket, 'aggr': aggr, 'fold_freq': fold_freq, 'call_freq': call_freq}

    def decision(self, gv):
        try:
            self._track_opp(gv)
            return self._decide(gv)
        except Exception:
            return 0

    def _decide(self, gv):
        my_hand = gv.my_hand
        board = gv.board
        to_call = gv.to_call
        pot = gv.pot
        bb = gv.big_blind
        ip = not self._am_first_in_round(gv)
        board_info = self._board_texture(board)
        prof = self._opp_profile()
        bucket = prof['bucket']
        on_river = len(board) == 5
        bluff_freq = 0.05
        value_mult_bump = 0.0
        bluff_size_mult = 1.5
        catch_buffer_bump = 0.0
        top_pair_threshold_bump = 0
        if bucket == 'passive_folder':
            bluff_freq = 0.32
            value_mult_bump = -0.1
            bluff_size_mult = 1.4
            catch_buffer_bump = 0.0
        elif bucket == 'calling_station':
            bluff_freq = 0.0
            value_mult_bump = 0.3
            catch_buffer_bump = 0.0
        elif bucket == 'aggressive':
            bluff_freq = 0.0
            value_mult_bump = 0.05
            catch_buffer_bump = -0.04
            top_pair_threshold_bump = +3
        if to_call == 0:
            info = self._hand_strength(my_hand, board)
            if info['rank'] >= RANK_TRINCA:
                return self._bet_amount(gv, 2.0 + value_mult_bump)
            if info['rank'] == RANK_DOIS_PARES:
                return self._bet_amount(gv, 1.7 + value_mult_bump)
            if info['overpair'] or info['top_pair']:
                return self._bet_amount(gv, 1.6 + value_mult_bump)
            eq = self._equity(gv)
            if eq >= 0.78:
                return self._bet_amount(gv, 1.8 + value_mult_bump)
            if eq >= 0.65 and bucket != 'calling_station':
                return self._bet_amount(gv, 1.5 + value_mult_bump)
            if self._rng.random() < bluff_freq and pot <= 10 * bb:
                return self._bet_amount(gv, bluff_size_mult)
            return 0
        if self._opp_all_in(gv):
            eq = self._equity(gv)
            pot_odds = to_call / max(1, pot + to_call)
            margin = 0.02 + catch_buffer_bump
            if eq > pot_odds + margin or to_call <= 2 * bb:
                return 0
            return -1
        my_bbs = self._my_stack_bbs(gv)
        if my_bbs <= 10:
            info = self._hand_strength(my_hand, board)
            eq = self._equity(gv)
            if info['rank'] >= RANK_UM_PAR or info['flush_draw'] or eq >= 0.42:
                return self._all_in(gv)
            my_vals = sorted([VALORES[c.value] for c in my_hand], reverse=True)
            if my_vals[0] >= 13 or (my_vals[0] >= 10 and my_vals[1] >= 10):
                return self._all_in(gv)
            return -1
        info = self._hand_strength(my_hand, board)
        rank = info['rank']
        call_in_bb = to_call / bb if bb > 0 else 0
        pot_odds = to_call / max(1, pot + to_call)
        _eq = [None]

        def get_eq():
            if _eq[0] is None:
                _eq[0] = self._equity(gv)
            return _eq[0]
        if rank >= RANK_QUADRA:
            return self._bet_amount(gv, 3.0)
        if rank >= RANK_STRAIGHT:
            return self._bet_amount(gv, 2.5 + value_mult_bump)
        if rank == RANK_TRINCA:
            if info['trips_on_board']:
                if call_in_bb <= 4:
                    return 0
                if get_eq() > pot_odds + 0.1 + catch_buffer_bump:
                    return 0
                return -1
            if bucket == 'aggressive' and self._opp_raised_this_round and (call_in_bb <= 12):
                return self._bet_amount(gv, 2.5 + value_mult_bump)
            return self._bet_amount(gv, 2.2 + value_mult_bump)
        if rank == RANK_DOIS_PARES:
            if info['paired_board']:
                if call_in_bb <= 6:
                    return 0
                if get_eq() > pot_odds + 0.05 + catch_buffer_bump:
                    return 0
                return -1
            if bucket == 'aggressive' and self._opp_raised_this_round and (call_in_bb <= 10):
                return self._bet_amount(gv, 2.2 + value_mult_bump)
            return self._bet_amount(gv, 1.8 + value_mult_bump)
        if rank == RANK_UM_PAR:
            if info['overpair']:
                return self._bet_amount(gv, 1.8 + value_mult_bump)
            if info['top_pair']:
                threshold_bb = 10 if ip else 7
                threshold_bb += top_pair_threshold_bump
                if bucket == 'aggressive' and self._opp_raised_this_round and (call_in_bb <= 8) and (pot <= 20 * bb):
                    return self._bet_amount(gv, 2.0 + value_mult_bump)
                if call_in_bb <= threshold_bb:
                    return 0
                if get_eq() > pot_odds + 0.05 + catch_buffer_bump:
                    return 0
                return -1
            if on_river and bucket == 'aggressive':
                if get_eq() > pot_odds - 0.03:
                    return 0
                return -1
            if info['paired_board']:
                if call_in_bb <= 2:
                    return 0
                if get_eq() > pot_odds + 0.1 + catch_buffer_bump:
                    return 0
                return -1
            threshold_bb = 5 if ip else 4
            if bucket == 'aggressive':
                threshold_bb += 2
            if call_in_bb <= threshold_bb:
                return 0
            if get_eq() > pot_odds + 0.07 + catch_buffer_bump:
                return 0
            return -1
        if info['flush_draw']:
            threshold_bb = 7 if ip else 5
            if call_in_bb <= threshold_bb:
                return 0
            if get_eq() > pot_odds + 0.05 + catch_buffer_bump:
                return 0
            return -1
        threshold_bb = 1.8 if ip else 1.2
        if call_in_bb <= threshold_bb:
            return 0
        eq = get_eq()
        buffer = 0.05 if call_in_bb <= 4 else 0.1 if call_in_bb <= 8 else 0.15
        buffer += catch_buffer_bump
        if on_river and bucket == 'aggressive' and (call_in_bb <= 6):
            my_vals = sorted([VALORES[c.value] for c in my_hand], reverse=True)
            if my_vals[0] >= 13 and eq > pot_odds - 0.05:
                return 0
        if eq > pot_odds + buffer:
            return 0
        return -1

def create_player() -> Player:
    return NemesisBot('NEMESIS', Hand(), 0)
