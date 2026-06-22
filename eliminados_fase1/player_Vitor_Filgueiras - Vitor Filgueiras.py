from __future__ import annotations

import sys
from pathlib import Path

# Ajuste obrigatório do path para garantir que o interpretador encontre os módulos do torneio
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from players.player import Player
from game.game_view import GameView
from cards.cards import Hand

# Mapeamento estrito dos valores das cartas de acordo com as regras do torneio
VALORES: dict[str, int] = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14
}

class MeuBot(Player):
    def __init__(self, name: str, hand: Hand, chips: int) -> None:
        """
        Construtor do Bot. Inicializa a classe base Player do torneio.
        """
        super().__init__(name, hand, chips)

    def tem_par_na_mao(self, hand: tuple) -> bool:
        """
        Método auxiliar para identificar se o bot recebeu um par de início.
        """
        if len(hand) >= 2:
            return hand[0].value == hand[1].value
        return False

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
        Método principal de tomada de decisão.
        Executado em cada rodada de apostas com timeout de 50ms.
        """
        try:
            # --- 1. Extração e Isolamento de Variáveis do Cenário ---
            to_call: int = game_view.to_call
            my_chips: int = game_view.my_chips
            bb: int = game_view.big_blind
            current_bet: int = game_view.current_bet
            
            # Identifica se somos o Big Blind (posição 0 do dealer no heads-up)
            eu_sou_bb: bool = self._org_sou_bb(game_view)

            # --- 2. Modo de Sobrevivência Crítico (Push or Fold) ---
            # Se nosso stack cair para menos de 5 Big Blinds, os blinds que dobram 
            # a cada 50 mãos vão nos engolir. Entramos em modo All-in ou Fold.
            if my_chips < (bb * 5):
                if game_view.my_hand:
                    carta_alta_survival = max(VALORES[c.value] for c in game_view.my_hand)
                    # Se tivermos um 10 ou mais, ou qualquer par, vamos All-in
                    if carta_alta_survival >= 10 or self.tem_par_na_mao(game_view.my_hand):
                        return my_chips
                # Se a mão for lixo, dá check se for de graça, senão desiste
                return 0 if to_call == 0 else -1

            # --- 3. Estratégia Base: Pré-Flop (Mesa / Board Vazia) ---
            if not game_view.board:
                if not game_view.my_hand:
                    return 0 # Salvaguarda caso a mão venha vazia por erro do motor
                
                # Calcula a maior carta da nossa mão
                carta_alta: int = max(VALORES[c.value] for c in game_view.my_hand)
                
                # Condição de Agressividade (Carta Alta >= 11: J, Q, K, A)
                if carta_alta >= 11:
                    # Regra de cálculo de raise exigida pelo torneio: aposta atual + incremento
                    raise_alvo = current_bet + (bb * 2)
                    
                    # Proteção de Stack: se o raise for maior que nossas fichas, vamos All-in
                    if raise_alvo >= my_chips:
                        return my_chips
                    
                    # Garante que temos fichas suficientes para pagar o custo atual + o aumento
                    if my_chips > to_call:
                        return raise_alvo
                    return 0 # Se não puder dar raise por falta de fichas, apenas dá Call

                # Condição de Desistência para cartas muito baixas (<= 6)
                if carta_alta <= 6:
                    # REGRA DE DEFESA ABSOLUTA: Nunca dar fold no Big Blind se o custo for zero
                    if to_call > 0:
                        return -1 # Fold com segurança
                
                # Para cartas médias (7, 8, 9, 10), jogamos de forma passiva (Check/Call)
                return 0

            # --- 4. Estratégia Base: Pós-Flop (Flop, Turn, River) ---
            else:
                # Se a ação chegou até nós sem apostas do adversário (to_call == 0)
                # nós damos Check para ver a próxima carta ou o Showdown de graça.
                if to_call == 0:
                    return 0
                
                # Se o oponente apostou e estamos no pós-flop, como nossa estratégia 
                # é focada puramente em Carta Alta pré-flop, optamos por não arriscar
                # o stack de 5.000 fichas sem uma lógica pós-flop complexa. Damos Fold.
                return -1

        except Exception:
            # Em caso de qualquer erro inesperado de execução, retorna 0 (Call automático)
            # para evitar punições ou alertas severos no console do torneio.
            return 0

def create_player() -> Player:
    """
    Função obrigatória requisitada pelo gerenciador do campeonato 
    para instanciar o seu bot na tabela Round-Robin.
    """
    return MeuBot("Vitor_Filgueiras", Hand(), 0)