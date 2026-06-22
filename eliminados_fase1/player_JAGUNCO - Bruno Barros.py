from __future__ import annotations

import sys
from pathlib import Path

# Adiciona src/ ao path para importar a engine do jogo
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from players.player import Player
from game.game_view import GameView
from cards.cards import Hand


class MeuBot(Player):
    def decision(self, game_view: GameView) -> int:
        # Inicializaando todas as informações disponiveis
        my_hand = list(game_view.my_hand)
        my_chips = game_view.my_chips
        board = list(game_view.board)
        l_board = len(board)
        rodada = l_board - 3 if l_board > 0 else 0
        pot = game_view.pot
        to_call = game_view.to_call
        current_bet = game_view.current_bet
        big_blind = game_view.big_blind
        small_blind = game_view.small_blind
        oponnent = game_view.opponents[0]

        # Importando utilidades relevantes
        from cards.sequences import score_cinco_cartas
        from cards.cards import Card
        import random
        import itertools
        import numpy as np

        # Gerando o conjunto de cartas restantes no baralho
        values = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        suits = ["s", "h", "d", "c"]
        cards = [Card(v, s) for v in values for s in suits]
        info = my_hand + board
        deck = [card for card in cards if card not in info]

        # Definindo funcoes uteis
        def deal_card(deck: list[Card]) -> Card:
            return deck.pop(random.randrange(len(deck)))

        def deal_bunch(deck: list[Card], n: int = 2) -> list[Card]:
            return [deal_card(deck) for i in range(n)]

        def add_card(deck: list[Card], card: Card) -> None:
            deck.append(card)

        def add_bunch(deck: list[Card], cards: list[Card]) -> None:
            while len(cards) >= 1:
                deck.append(cards.pop())

        def hand_score(hand: list[Card]) -> int:
            if len(hand) < 5:
                return 0
            melhor = 0
            for cinco in itertools.combinations(hand, 5):
                s = score_cinco_cartas(list(cinco))
                if s > melhor:
                    melhor = s
            return melhor

        def chance(board: list[Card], hand: list[Card], iteracoes: int = 20) -> float:
            _board: list[Card] = []
            chance = 0.0
            for i in range(iteracoes):
                _board = deal_bunch(deck=deck, n=5 - len(_board))
                _enemy_hand = deal_bunch(deck)
                if hand_score(hand + _board + board) >= hand_score(
                    _enemy_hand + _board + board
                ):
                    chance += 1
                add_bunch(deck=deck, cards=_enemy_hand + _board)
            chance = chance / iteracoes
            return chance

        def preco_optimal_proporcional(p0: float, pot: int, chips: int) -> float:
            g = pot / chips
            return (2 * p0 - 1) * (g + 1) - g * p0

        def preco_maximo_proporcional(p0: float, pot: int, chips: int) -> float:
            if p0 == 0.5:
                return 0
            c = 0.1
            sup = 1
            inf = -1
            g = pot / chips
            while sup - inf > 0.01:
                if c == 0.0:
                    c += 0.1
                    continue
                p = np.log(1 / (1 - c)) / np.log((1 + g + c) / (1 - c))
                if p >= p0:
                    sup = c
                    c = (inf + c) / 2
                else:
                    inf = c
                    c = (sup + c) / 2
            return c

        # Calculo da chance vigente de ganhar edo custo
        odd = chance(board=board, hand=my_hand)
        custo_maximo = int(
            preco_maximo_proporcional(p0=odd, pot=pot, chips=my_chips) * my_chips
        )

        custo_ideal = int(
            preco_maximo_proporcional(p0=odd, pot=pot, chips=my_chips) * my_chips
        )

        if to_call > custo_maximo:
            return -1
        else:
            return max(current_bet + custo_ideal, 0)


def create_player() -> Player:
    return MeuBot("Jagas", Hand(), 0)
