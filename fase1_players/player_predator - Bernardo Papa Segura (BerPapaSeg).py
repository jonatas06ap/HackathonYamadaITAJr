# players/player_predator.py
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter
from itertools import combinations
import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from players.player import Player
from game.game_view import GameView
from cards.cards import Hand

VALORES = { "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, 
            "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14 }

# --- Aproveitamos o avaliador rápido do Pinguim para economizar ms ---
def _eval5(cards) -> tuple:
    vals = sorted([VALORES[c.value] for c in cards], reverse=True)
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

class AntiPinguimBot(Player):
    def __init__(self, name, hand, chips):
        super().__init__(name, hand, chips)
        self.hands_played = 0
        self.last_pot = 0
        self.chameleon_mode = True # Modo inicial: criar falsa imagem "Tight"

    def decision(self, game_view: GameView) -> int:
        # Detecta nova mão para atualizar contador
        if game_view.pot < self.last_pot or game_view.pot == game_view.big_blind * 1.5:
            self.hands_played += 1
        self.last_pot = game_view.pot

        # META-GAME: Mudar de personalidade após 15 mãos. 
        # O Pinguim já nos classificou como 'opp_is_tight' (vpip < 0.30)
        if self.hands_played > 15:
            self.chameleon_mode = False

        my_chips = game_view.my_chips
        bb = game_view.big_blind
        to_call = game_view.to_call
        current_bet = game_view.current_bet
        pot = game_view.pot
        board = game_view.board

        # 1. PUSH/FOLD (Sobrevivência)
        if my_chips < 12 * bb:
            v1, v2 = VALORES[game_view.my_hand[0].value], VALORES[game_view.my_hand[1].value]
            if v1 >= 10 or v2 >= 10 or v1 == v2:
                return current_bet + my_chips
            return 0 if to_call == 0 else -1

        # 2. PRÉ-FLOP
        if not board:
            v1, v2 = VALORES[game_view.my_hand[0].value], VALORES[game_view.my_hand[1].value]
            is_pair = (v1 == v2)
            high_card = max(v1, v2)
            
            # Se estamos construindo a imagem de "Tight" (primeiras 15 mãos)
            if self.chameleon_mode:
                if high_card >= 13 or is_pair:
                    return current_bet + (bb * 3)
                return 0 if to_call == 0 else -1
            
            # Modo Agressivo Expandido (O Pinguim vai foldar demais achando que somos Tight)
            else:
                # Premium (AA, KK, QQ, AK)
                if high_card >= 13 or (is_pair and v1 >= 8):
                    return current_bet + (bb * 4) # Aumentamos o size para punir o call dele
                
                # Mãos marginais com posição (Steal)
                my_invested = current_bet - to_call
                eu_sou_sb = (my_invested == game_view.small_blind)
                if eu_sou_sb and to_call == 0 and random.random() < 0.60:
                    return current_bet + (bb * 2.5) # Raise para roubar
                
                # Call para ver o flop com conectadas
                if abs(v1 - v2) <= 2 and to_call <= bb * 2:
                    return 0
                    
                return 0 if to_call == 0 else -1

        # 3. PÓS-FLOP (Explorando o cálculo linear do Pinguim)
        rank = best_hand(game_view.my_hand, board)
        cat = rank[0] # 0=High card, 1=Par, 2=Dois pares, 3=Trinca...

        # Temos uma mão forte real (Dois pares+)
        if cat >= 2:
            # Pinguim paga se o HS dele for bom. Vamos cobrar caro.
            target_bet = current_bet + int(pot * 0.8)
            return min(my_chips, target_bet)

        # Temos Top Pair ou Overpair
        board_vals = sorted([VALORES[c.value] for c in board], reverse=True)
        hand_high = max(VALORES[game_view.my_hand[0].value], VALORES[game_view.my_hand[1].value])
        
        is_top_pair = (cat == 1 and rank[1] == board_vals[0] and rank[1] == hand_high)
        
        if is_top_pair:
            if to_call > pot * 0.8: 
                # Pinguim apostou forte. Ele tem `hs > strong_pf_th`. Damos fold e fugimos.
                return -1 
            target_bet = current_bet + int(pot * 0.5)
            return target_bet if to_call == 0 else 0

        # BLEFE TÁTICO: O Pinguim folda se `hs < pot_odds + 0.10`. 
        # Se dermos um overbet gigante (pote * 1.2), o pot odds dele fica terrível.
        # Ele vai dar insta-fold de mãos como 2º e 3º pares, que normalmente ele daria call num size menor.
        if to_call == 0 and cat == 0 and not self.chameleon_mode:
            # Blefamos 25% das vezes se a mesa estiver seca (sem muitas cartas altas repetidas)
            if random.random() < 0.25:
                target_bet = current_bet + int(pot * 1.2) # OVERBET
                return min(my_chips, target_bet)
                
        # Draw check/call barato
        if to_call > 0 and to_call < pot * 0.25:
            return 0
            
        return 0 if to_call == 0 else -1

def create_player() -> Player:
    return AntiPinguimBot("PinguimHunter", Hand(), 0)