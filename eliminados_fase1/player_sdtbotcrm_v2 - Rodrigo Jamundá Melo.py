from __future__ import annotations

import sys
import random
from pathlib import Path

# Adiciona src/ ao path para importar a engine do jogo
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from players.player import Player
from game.game_view import GameView
from cards.cards import Hand

VALORES = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14}

class SdTBotCRM_v2(Player):
    """
    SdTBotCRM_v2 (Warmup Edition): Hybrid bot with a profiling phase.

    Strategic Approach:
    1. Profiling Phase: Plays the first N hands using ONLY the Base strategy.
       It collects regret data during this time but doesn't act on it.
    2. Adaptive Phase: Transitions to a blend of Base + Learned strategies.
    3. Stability: Uses weighted updates and regret decay to avoid noise.
    """

    def __init__(self, name, hand, chips):
        super().__init__(name, hand, chips)
        self.regrets = {}
        self.state_counts = {}
        self.actions = [-1, 0, 1]

        # Hyperparameters
        self.epsilon = 0.05
        self.learning_rate = 0.005
        self.alpha = 0.8
        self.warmup_period = 50  # Number of hands to play exclusively as Base
        self.decay_factor = 0.98 # Slightly higher decay to stay fresh
        self.trust_threshold = 3  # Minimum samples per state to trust learning

        # State tracking
        self.total_hands = 0
        self.hand_history = []
        self.chips_at_start = chips
        self.last_board_size = -1

    def evaluate_hand(self, hand, board) -> int:
        cards = list(hand) + list(board)
        if not cards: return 0
        counts = {}
        for c in cards:
            v = c.value
            counts[v] = counts.get(v, 0) + 1
        suits = [c.suit for c in cards]
        flush_suit = None
        for s in set(suits):
            if suits.count(s) >= 5:
                flush_suit = s
                break
        all_vals = sorted([VALORES.get(c.value, 0) for c in cards], reverse=True)
        unique_vals = sorted(list(set(all_vals)), reverse=True)
        consecutive = 1
        max_consecutive = 1
        if len(unique_vals) > 1:
            for i in range(len(unique_vals) - 1):
                if unique_vals[i] == unique_vals[i+1] + 1:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 1
        has_low_straight = "A" in counts and all(v in counts for v in ["2", "3", "4", "5"])
        is_straight = max_consecutive >= 5 or has_low_straight
        freqs = sorted(counts.values(), reverse=True)
        if not freqs: return 0
        if flush_suit:
            suit_cards = sorted([VALORES.get(c.value, 0) for c in cards if c.suit == flush_suit], reverse=True)
            sc_unique = sorted(list(set(suit_cards)), reverse=True)
            sc_consecutive = 1
            sc_max = 1
            if len(sc_unique) > 1:
                for i in range(len(sc_unique) - 1):
                    if sc_unique[i] == sc_unique[i+1] + 1:
                        sc_consecutive += 1
                        sc_max = max(sc_max, sc_consecutive)
                    else:
                        sc_consecutive = 1
            sc_has_low = "A" in [c.value for c in cards if c.suit == flush_suit] and \
                         all(v in [c.value for c in cards if c.suit == flush_suit] for v in ["2", "3", "4", "5"])
            if sc_max >= 5 or sc_has_low: return 8
        if freqs[0] == 4: return 7
        if len(freqs) >= 2 and freqs[0] == 3 and freqs[1] >= 2: return 6
        if flush_suit: return 5
        if is_straight: return 4
        if freqs[0] == 3: return 3
        if len(freqs) >= 2 and freqs[0] == 2 and freqs[1] == 2: return 2
        if freqs[0] == 2: return 1
        return 0

    def get_bucket(self, rank: int) -> int:
        if rank >= 6: return 3
        if rank >= 3: return 2
        if rank >= 1: return 1
        return 0

    def get_base_action(self, bucket: int, is_facing_bet: bool) -> int:
        if not is_facing_bet:
            return 1 if bucket == 3 else 0
        if bucket == 3: return 1
        if bucket == 2: return 0
        if bucket == 1: return 0
        return -1

    def _get_blended_probs(self, state):
        bucket, street, is_facing_bet = state
        base_action = self.get_base_action(bucket, is_facing_bet)
        base_probs = {a: 0.0 for a in self.actions}
        base_probs[base_action] = 1.0

        if state not in self.regrets:
            self.regrets[state] = {a: 0.0 for a in self.actions}

        if self.state_counts.get(state, 0) < self.trust_threshold:
            return base_probs

        regrets = self.regrets[state]
        positive_regrets = {a: max(0, v) for a, v in regrets.items()}
        sum_pos_regret = sum(positive_regrets.values())

        if sum_pos_regret > 0:
            learned_probs = {a: positive_regrets[a] / sum_pos_regret for a in self.actions}
        else:
            learned_probs = base_probs

        final_probs = {}
        for a in self.actions:
            final_probs[a] = (self.alpha * base_probs[a]) + ((1 - self.alpha) * learned_probs[a])

        return final_probs

    def _update_regrets(self, final_outcome):
        capped_outcome = max(-1000, min(1000, final_outcome))
        weighted_outcome = capped_outcome * self.learning_rate

        for state, action_taken in self.hand_history:
            if state in self.regrets:
                for a in self.regrets[state]:
                    self.regrets[state][a] *= self.decay_factor

            current_regrets = self.regrets[state]
            u_taken = weighted_outcome
            for a in self.actions:
                if a == action_taken:
                    u_alt = u_taken
                elif a == -1:
                    u_alt = 0 if final_outcome < 0 else -abs(weighted_outcome) * 0.1
                else:
                    u_alt = -0.01 if final_outcome > 0 else 0
                current_regrets[a] += (u_alt - u_taken)

    def decision(self, game_view: GameView) -> int:
        # Hand end detection
        if game_view.board == () and self.last_board_size != -1:
            final_outcome = game_view.my_chips - self.chips_at_start
            self._update_regrets(final_outcome)
            self.hand_history = []
            self.chips_at_start = game_view.my_chips

        self.last_board_size = len(game_view.board)

        # Increment total hands played in the match
        if game_view.board == ():
            self.total_hands += 1

        rank = self.evaluate_hand(game_view.my_hand, game_view.board)
        bucket = self.get_bucket(rank)
        street = len(game_view.board)
        is_facing_bet = (game_view.to_call > 0)
        state = (bucket, street, is_facing_bet)
        self.state_counts[state] = self.state_counts.get(state, 0) + 1

        # --- STRATEGY SWITCH ---
        if self.total_hands <= self.warmup_period:
            # Warmup phase: Play strictly as Base, but still track history for learning
            chosen_action = self.get_base_action(bucket, is_facing_bet)
        else:
            # Adaptive phase: Blend Base and Learned
            if random.random() < self.epsilon:
                chosen_action = random.choice(self.actions)
            else:
                probs = self._get_blended_probs(state)
                r = random.random()
                cumulative = 0
                chosen_action = 0
                for a in self.actions:
                    cumulative += probs[a]
                    if r <= cumulative:
                        chosen_action = a
                        break

        self.hand_history.append((state, chosen_action))

        if chosen_action == -1:
            return -1
        elif chosen_action == 0:
            return 0
        else:
            multiplier = 4 if bucket == 3 else (2 if bucket == 2 else 1)
            return game_view.current_bet + game_view.big_blind * multiplier

def create_player() -> Player:
    return SdTBotCRM_v2("SdTBotCRM_v2", Hand(), 0)
