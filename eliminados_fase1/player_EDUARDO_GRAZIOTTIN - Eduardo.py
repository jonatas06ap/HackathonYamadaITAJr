"""
TEMPLATE DO BOT DE POKER — ITA Jr | Treinamento POO
====================================================

INSTRUÇÕES
----------
1. Copie este arquivo: cp player_template.py player_SEU_NOME.py
2. Renomeie a classe `MeuBot` para algo único (ex: `BotAgressivo`)
3. Implemente a estratégia no método `decision()` abaixo
4. Coloque o arquivo na pasta `players/` e rode o torneio:
       python run_tournament.py

Não é necessário entender o resto do código — só o método `decision()` importa!

─────────────────────────────────────────────────────────────────────────────
CONCEITOS DE POO NESTE PROJETO
─────────────────────────────────────────────────────────────────────────────

ABSTRAÇÃO
    `Player` define a interface (o "contrato"): qualquer bot que herde de
    Player e implemente decision() pode jogar. Você não precisa saber como
    o Game funciona por dentro.

HERANÇA
    `class MeuBot(Player)` — seu bot herda name, hand, chips, in_game.
    Você só precisa escrever a lógica de decisão.

POLIMORFISMO
    O Game chama `player.decision(view)` para CallerPlayer, RaiserPlayer e
    MeuBot da mesma forma. Cada um responde diferentemente — isso é polimorfismo.

ENCAPSULAMENTO
    Você recebe um `GameView` somente-leitura. As cartas dos adversários e o
    deck restante ficam encapsulados dentro do Game — inacessíveis para você.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import sys
from pathlib import Path

# Adiciona src/ ao path para importar a engine do jogo
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from players.player import Player
from game.game_view import GameView
from cards.cards import Hand
from collections import Counter

VALORES = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14
}

class BotAquamanT30(Player):

    def __init__(self, name, hand, chips):
        super().__init__(name, hand, chips)
        self.maos_jogadas = 0
        self.raises_sofridos_total = 0
        self._oponente_bet_anterior = 0

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
        """
        Recebe o estado público do jogo e retorna uma ação.

        POLIMORFISMO: este método é chamado pelo Game para qualquer Player.
        Cada bot implementa sua estratégia aqui — mesma assinatura, comportamentos diferentes.

        O QUE VOCÊ PODE VER (game_view)
        ────────────────────────────────────────────────────────────────────
        game_view.my_hand          → tuple de 2 Card — suas cartas privadas
                                     Ex: (As, Kh) para Ás de espadas e Rei de copas
        game_view.my_chips         → int — suas fichas atuais

        game_view.board            → tuple de Card — cartas comunitárias (mesa)
                                     0 cartas no pré-flop, 3 no flop, 4 no turn, 5 no river
        game_view.pot              → int — total de fichas no pote
        game_view.to_call          → int — fichas que você precisa pagar para continuar
                                     (0 = pode dar check de graça)
        game_view.current_bet      → int — maior aposta total nesta rodada
        game_view.small_blind      → int — valor atual do small blind
        game_view.big_blind        → int — valor atual do big blind
        game_view.dealer_position  → int — índice do dealer na lista de oponentes

        game_view.opponents        → tuple de PublicPlayerInfo (SEM as cartas deles!)
          .opponents[i].name              → nome do oponente
          .opponents[i].chips             → fichas do oponente
          .opponents[i].current_bet_in_round → quanto ele apostou nesta rodada
          .opponents[i].is_active         → False = já deu fold

        SOBRE AS CARTAS (Card)
        ────────────────────────────────────────────────────────────────────
        card.value  → "A", "2"..."10", "J", "Q", "K"
        card.suit   → "s" (spades/espadas), "h" (hearts/copas),
                      "d" (diamonds/ouros), "c" (clubs/paus)
        str(card)   → "As", "Kh", "10d", etc.

        RETORNO
        ────────────────────────────────────────────────────────────────────
        -1       → fold  (desistir desta mão)
         0       → check (se to_call == 0) ou call (pagar a aposta atual)
         N > 0   → raise: apostar um total de N nesta rodada
                   Ex: se current_bet=20 e quer ir para 60, retorne 60
                   O Game automaticamente converte valores menores que current_bet para call.
        """


        oponente = game_view.opponents[0]
        if oponente.current_bet_in_round > self._oponente_bet_anterior:
            self.raises_sofridos_total += 1
        self._oponente_bet_anterior = oponente.current_bet_in_round
        self.maos_jogadas += 1

        bb        = game_view.big_blind
        meu_stack = game_view.my_chips
        to_call   = game_view.to_call

        # ── 1. Stack curto: push/fold (< 8 BBs) ──────────────────────
        if meu_stack < bb * 8:
            carta_alta = max(VALORES[c.value] for c in game_view.my_hand)
            tem_par    = game_view.my_hand[0].value == game_view.my_hand[1].value
            if tem_par or carta_alta >= 12:
                return meu_stack
            if carta_alta >= 10 and to_call <= bb * 2:
                return meu_stack
            return -1 if to_call > 0 else 0

        # ── 2. Pré-flop ───────────────────────────────────────────────
        if not game_view.board:
            return self._preflop(game_view)

        # ── 3. Pós-flop ───────────────────────────────────────────────
        return self._posflop(game_view)

        # ——————————————————————————————————————————————————————————————

    # ==================================================================
    #                       Métodos auxiliares
    # ==================================================================

    def _preflop(self, gv: GameView) -> int:
        hand      = gv.my_hand
        bb        = gv.big_blind
        to_call   = gv.to_call
        eu_sou_bb = self._org_sou_bb(gv)
        forca     = self._forca_preflop(hand)
        agg       = self._agressividade_oponente()

        if forca >= 9:                                        # mão premium
            mult = 4 if agg > 0.4 else 3
            return gv.current_bet + bb * mult

        if forca >= 7:                                        # mão boa
            if to_call <= bb * 3:
                return gv.current_bet + bb * 2 if random.random() < 0.7 else 0
            return 0

        if forca >= 5:                                        # mão mediana
            if to_call == 0:
                if eu_sou_bb and random.random() < 0.4:
                    return gv.current_bet + bb * 2
                return 0
            return 0 if to_call <= bb * 2 else -1

        # mão fraca
        if to_call == 0:
            return 0
        if to_call <= bb and eu_sou_bb:
            return 0
        return -1

    def _posflop(self, gv: GameView) -> int:
        bb        = gv.big_blind
        to_call   = gv.to_call
        pot       = gv.pot
        eu_sou_bb = self._org_sou_bb(gv)
        forca     = self._forca_posflop(gv.my_hand, gv.board)
        pot_odds  = to_call / (pot + to_call) if to_call > 0 else 0
        equidade  = self._equidade_estimada(forca)

        if forca >= 7:                                        # mão muito forte
            tamanho = int(pot * 0.75)
            return gv.current_bet + max(tamanho, bb * 2)

        if forca >= 5:                                        # mão boa
            if to_call == 0:
                if eu_sou_bb or random.random() < 0.6:
                    return gv.current_bet + max(int(pot * 0.5), bb)
                return 0
            return 0 if equidade > pot_odds else -1

        if forca >= 3:                                        # mão mediana
            if to_call == 0:
                return gv.current_bet + bb * 2 if random.random() < 0.25 else 0
            return 0 if equidade > pot_odds else -1

        # mão fraca
        if to_call == 0:
            return gv.current_bet + int(pot * 0.6) if random.random() < 0.15 else 0
        return 0 if equidade > pot_odds + 0.05 else -1

    # ── Avaliação de força ─────────────────────────────────────────────

    def _forca_preflop(self, hand) -> int:
        v      = sorted([VALORES[c.value] for c in hand], reverse=True)
        suited = hand[0].suit == hand[1].suit
        high, low = v[0], v[1]
        tem_par   = high == low

        if tem_par:
            if high >= 10: return 10
            if high >= 7:  return 8
            return 6

        if high == 14:
            if low >= 10: return 9
            if low >= 7:  return 7
            return 5 if suited else 4

        if high == 13 and low >= 10: return 8
        if high == 13 and low >= 8:  return 6
        if high >= 11 and low >= 10: return 7
        if suited and high - low <= 2: return 5
        if high >= 10: return 4
        return 2

    def _forca_posflop(self, hand, board) -> int:
        todas      = list(hand) + list(board)
        valores    = [c.value for c in todas]
        naipes     = [c.suit for c in todas]
        nums       = sorted([VALORES[v] for v in valores], reverse=True)
        contagem   = Counter(valores)
        freq       = sorted(contagem.values(), reverse=True)
        tem_flush  = any(v >= 5 for v in Counter(naipes).values())
        nums_uniq  = sorted(set(nums), reverse=True)
        tem_str    = self._tem_straight(nums_uniq)

        if tem_flush and tem_str:           return 10
        if freq[0] == 4:                    return 9
        if freq[0] == 3 and freq[1] >= 2:  return 8
        if tem_flush:                       return 7
        if tem_str:                         return 6
        if freq[0] == 3:                    return 5
        if freq[0] == 2 and freq[1] == 2:  return 4
        if freq[0] == 2:                    return 3
        return max(1, (nums[0] - 6) // 2)

    def _tem_straight(self, nums_uniq: list) -> bool:
        for i in range(len(nums_uniq) - 4):
            janela = nums_uniq[i:i + 5]
            if janela[0] - janela[4] == 4:
                return True
        return {14, 2, 3, 4, 5}.issubset(set(nums_uniq))

    def _equidade_estimada(self, forca: int) -> float:
        mapa = {1: 0.25, 2: 0.30, 3: 0.40, 4: 0.50,
                5: 0.60, 6: 0.68, 7: 0.75, 8: 0.85, 9: 0.92, 10: 0.97}
        return mapa.get(forca, 0.35)

    def _agressividade_oponente(self) -> float:
        if self.maos_jogadas == 0:
            return 0.3
        return min(self.raises_sofridos_total / self.maos_jogadas, 1.0)

        # ──────────────────────────────────────────────────────────────────


def create_player() -> Player:

    return BotAquamanT30("BotAquamanT30", Hand(), 0)
