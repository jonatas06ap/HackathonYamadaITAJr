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

class OpponentProfile:
    """Tracks basic HUD stats for a single opponent."""
    def __init__(self):
        self.hands_played = 0
        self.vpip_count = 0
        self.pfr_count = 0
        self.aggr_actions = 0
        self.call_actions = 0

    @property
    def vpip(self) -> float:
        return self.vpip_count / self.hands_played if self.hands_played > 0 else 0.5

    @property
    def pfr(self) -> float:
        return self.pfr_count / self.hands_played if self.hands_played > 0 else 0.1

    @property
    def aggression_factor(self) -> float:
        return self.aggr_actions / max(1, self.call_actions)

class SdTBotChen_v1(Player):
    """
    SdTBotChen_v1: Advanced hybrid bot.
    Features: Chen Pre-flop, Texture-based post-flop, Opponent Profiling,
    and SAGE endgame logic for short-stack push/fold.
    """

    def __init__(self, name, hand, chips):
        super().__init__(name, hand, chips)
        self.regrets = {}
        self.state_counts = {}
        self.actions = [-1, 0, 1]

        self.opponent_profile = OpponentProfile()

        self.epsilon = 0.05
        self.learning_rate = 0.005
        self.alpha = 0.8
        self.warmup_period = 50
        self.decay_factor = 0.98
        self.trust_threshold = 3

        self.total_hands = 0
        self.hand_history = []
        self.chips_at_start = chips
        self.last_board_size = -1

    def calculate_chen_score(self, hand) -> float:
        c1, c2 = hand
        v1, v2 = VALORES[c1.value], VALORES[c2.value]
        high_val = max(v1, v2)
        low_val = min(v1, v2)

        score = 0
        if high_val == 14: score = 10
        elif high_val == 13: score = 8
        elif high_val == 12: score = 7
        elif high_val == 11: score = 6
        else: score = high_val / 2.0

        if c1.value == c2.value:
            score *= 2
            if score < 5: score = 5

        if c1.suit == c2.suit:
            score += 2

        gap = high_val - low_val - 1
        if gap == 0: penalty = 0
        elif gap == 1: penalty = -1
        elif gap == 2: penalty = -2
        elif gap == 3: penalty = -4
        else: penalty = -5
        score += penalty

        if gap <= 1 and high_val < 12:
            score += 1

        return float(int(score + 0.5)) if score % 1 != 0 else score

    def calculate_sage_pi(self, hand) -> int:
        """Calculates the SAGE Power Index (PI)."""
        c1, c2 = hand
        # SAGE values: 2-10 = face, J=11, Q=12, K=13, A=15
        val_map = VALORES.copy()
        val_map["A"] = 15

        v1, v2 = val_map[c1.value], val_map[c2.value]
        high = max(v1, v2)
        low = min(v1, v2)

        if c1.value == c2.value:
            # Pocket pair: 2 * one card + 22
            return (high * 2) + 22

        # Normal hand: (Max * 2) + Min + (2 if suited)
        score = (high * 2) + low
        if c1.suit == c2.suit:
            score += 2
        return score

    def get_sage_decision(self, hand, game_view) -> int:
        """SAGE Push/Fold logic for short stacks (R <= 7)."""
        pi = self.calculate_sage_pi(hand)

        # Effective stack in BBs
        opp_chips = game_view.opponents[0].chips
        eff_stack = min(game_view.my_chips, opp_chips)
        r = eff_stack / game_view.big_blind

        # [CORRIGIDO PELA ORGANIZAÇÃO — bug de detecção de posição] ver _org_sou_bb.
        # (O comentário original sobre dealer_position == 0/1 estava incorreto.)
        am_i_sb = not self._org_sou_bb(game_view)

        if am_i_sb:
            # SB Push Decision
            # Simplified SAGE thresholds for R=1 to 7
            # PI requirements for SB Push increase as R increases
            thresholds = {1: 17, 2: 21, 3: 22, 4: 23, 5: 24, 6: 25, 7: 26}
            required_pi = thresholds.get(int(r), 26 if r > 7 else 17)
            return 1 if pi >= required_pi else -1
        else:
            # BB Defense Decision
            thresholds = {1: 0, 2: 17, 3: 24, 4: 26, 5: 28, 6: 29, 7: 30}
            required_pi = thresholds.get(int(r), 30 if r > 7 else 0)
            return 0 if pi >= required_pi else -1

    def get_preflop_bucket(self, hand) -> int:
        score = self.calculate_chen_score(hand)
        if score >= 12: return 3
        if score >= 8: return 2
        if score >= 5: return 1
        return 0

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

    def analyze_board_texture(self, board) -> str:
        if not board: return "dry"
        cards = list(board)
        suits = [c.suit for c in cards]
        ranks = [c.value for c in cards]
        for s in set(suits):
            if suits.count(s) >= 3: return "monotone"
        if len(set(ranks)) < len(ranks): return "paired"
        val_ranks = sorted([VALORES[r] for r in ranks])
        is_connected = False
        for i in range(len(val_ranks)-1):
            if val_ranks[i+1] - val_ranks[i] <= 2:
                is_connected = True
                break
        if is_connected: return "wet"
        return "dry"

    def get_bet_amount(self, pot, current_bet, texture, bucket) -> int:
        if texture == "wet": perc = 0.65
        elif texture == "monotone": perc = 0.40
        elif texture == "paired": perc = 0.30
        else: perc = 0.30
        raise_val = current_bet + int(pot * perc)
        if bucket == 3: raise_val += int(pot * 0.2)
        elif bucket == 1: raise_val = current_bet + int(pot * 0.15)
        return raise_val

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
        adj_alpha = self.alpha
        if self.opponent_profile.aggression_factor > 3.0: adj_alpha = 0.9
        elif self.opponent_profile.aggression_factor < 1.0: adj_alpha = 0.6
        final_probs = {a: (adj_alpha * base_probs[a]) + ((1 - adj_alpha) * learned_probs[a]) for a in self.actions}
        return final_probs

    def _update_regrets(self, final_outcome):
        capped_outcome = max(-1000, min(1000, final_outcome))
        weighted_outcome = capped_outcome * self.learning_rate
        for state, action_taken in self.hand_history:
            if state in self.regrets:
                for a in self.regrets[state]: self.regrets[state][a] *= self.decay_factor
            current_regrets = self.regrets[state]
            u_taken = weighted_outcome
            for a in self.actions:
                if a == action_taken: u_alt = u_taken
                elif a == -1: u_alt = 0 if final_outcome < 0 else -abs(weighted_outcome) * 0.1
                else: u_alt = -0.01 if final_outcome > 0 else 0
                current_regrets[a] += (u_alt - u_taken)

    def _org_sou_bb(self, gv) -> bool:
        """[CORRIGIDO PELA ORGANIZAÇÃO — bug de detecção de posição]

        `dealer_position` é o índice do dealer na lista GLOBAL de jogadores da
        engine, e NÃO um valor relativo a este bot. Por isso a verificação
        original baseada em `dealer_position == 1` (am_i_sb) só acertava quando
        este bot ocupava o assento players[0] da partida, falhando em até 100%
        das mãos quando ocupava players[1].

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
        opponent = game_view.opponents[0]
        if game_view.board == ():
            if opponent.current_bet_in_round > game_view.big_blind:
                self.opponent_profile.vpip_count += 1
            if opponent.current_bet_in_round > game_view.big_blind:
                self.opponent_profile.pfr_count += 1
            self.opponent_profile.hands_played += 1
        if game_view.to_call > 0:
            self.opponent_profile.aggr_actions += 1

        if game_view.board == () and self.last_board_size != -1:
            final_outcome = game_view.my_chips - self.chips_at_start
            self._update_regrets(final_outcome)
            self.hand_history = []
            self.chips_at_start = game_view.my_chips
        self.last_board_size = len(game_view.board)

        if game_view.board == ():
            if self.hand_history == []:
                self.total_hands += 1

            # --- SAGE ENDGAME LOGIC ---
            opp_chips = opponent.chips
            eff_stack = min(game_view.my_chips, opp_chips)
            r = eff_stack / game_view.big_blind
            if r <= 7:
                # Short stack: use SAGE Push/Fold
                sage_action = self.get_sage_decision(game_view.my_hand, game_view)
                # Only use SAGE for pre-flop all-ins (1 = Raise/Push, -1 = Fold)
                # If SAGE says call (0), we let the standard logic handle it.
                if sage_action != 0:
                    # We still record the state for learning, but return SAGE action
                    bucket = self.get_preflop_bucket(game_view.my_hand)
                    state = (bucket, 0, game_view.to_call > 0)
                    self.state_counts[state] = self.state_counts.get(state, 0) + 1
                    self.hand_history.append((state, sage_action))
                    return sage_action

        # Standard Logic
        if not game_view.board:
            bucket = self.get_preflop_bucket(game_view.my_hand)
        else:
            rank = self.evaluate_hand(game_view.my_hand, game_view.board)
            bucket = self.get_bucket(rank)

        street = len(game_view.board)
        is_facing_bet = (game_view.to_call > 0)
        state = (bucket, street, is_facing_bet)
        self.state_counts[state] = self.state_counts.get(state, 0) + 1

        if self.total_hands <= self.warmup_period:
            chosen_action = self.get_base_action(bucket, is_facing_bet)
        else:
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
        if chosen_action == -1: return -1
        elif chosen_action == 0: return 0
        else:
            if game_view.board:
                texture = self.analyze_board_texture(game_view.board)
                return self.get_bet_amount(game_view.pot, game_view.current_bet, texture, bucket)
            multiplier = 4 if bucket == 3 else (2 if bucket == 2 else 1)
            return game_view.current_bet + game_view.big_blind * multiplier

def create_player() -> Player:
    return SdTBotChen_v1("SdTBotChen_v1", Hand(), 0)
