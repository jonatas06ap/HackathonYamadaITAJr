from __future__ import annotations

import sys
import random
from pathlib import Path
from collections import deque
from itertools import combinations
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from players.player import Player
from game.game_view import GameView
from cards.cards import Hand


# ==============================================================================
#  LOOKUP TABLE DE AVALIAÇÃO DE MÃO
#  eval_ranks_flush é decorado com lru_cache — na prática só existem ~10k
#  padrões únicos (rank_tuple, is_flush), então o cache satura rápido e
#  chamadas subsequentes são O(1) dict lookup.
# ==============================================================================

RANK_MAP = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14
}

# Bitmasks para cada straight possível (52-bit deck, bit i = rank i+2)
# Pré-computado uma vez no import
_STRAIGHT_MASKS: list[tuple[int, int]] = []
for _high in range(14, 5, -1):          # A-high=14 down to 6-high
    _mask = 0
    for _r in range(_high, _high - 5, -1):
        _mask |= (1 << (_r - 2))
    _STRAIGHT_MASKS.append((_mask, _high))
_STRAIGHT_MASKS.append(((1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3), 5))  # wheel


@lru_cache(maxsize=None)
def _eval_ranks_flush(ranks: tuple, is_flush: bool) -> tuple:
    """
    Avalia 5 cartas representadas como (ranks_sorted_desc, is_flush).
    Retorna score comparável como tuple de ints.
    lru_cache garante que padrões repetidos custam O(1).
    """
    r = ranks  # já sorted desc

    # Straight via bitmask — O(9) comparações fixas
    rmask = 0
    for x in r:
        rmask |= (1 << (x - 2))
    straight = False
    st_high = 0
    for mask, high in _STRAIGHT_MASKS:
        if (rmask & mask) == mask:
            straight = True
            st_high = high
            break

    if is_flush and straight:
        return (8, st_high, 0, 0, 0, 0)

    # Contagem de grupos
    cnt: dict[int, int] = {}
    for x in r:
        cnt[x] = cnt.get(x, 0) + 1
    groups = sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)
    c0 = groups[0][1]

    if c0 == 4:
        return (7, groups[0][0], groups[1][0], 0, 0, 0)
    if c0 == 3 and len(groups) > 1 and groups[1][1] == 2:
        return (6, groups[0][0], groups[1][0], 0, 0, 0)
    if is_flush:
        return (5, r[0], r[1], r[2], r[3], r[4])
    if straight:
        return (4, st_high, 0, 0, 0, 0)
    if c0 == 3:
        k = [g[0] for g in groups[1:]]
        return (3, groups[0][0], k[0] if k else 0, k[1] if len(k) > 1 else 0, 0, 0)
    if c0 == 2 and len(groups) > 1 and groups[1][1] == 2:
        k = [g[0] for g in groups[2:]]
        return (2, groups[0][0], groups[1][0], k[0] if k else 0, 0, 0)
    if c0 == 2:
        k = [g[0] for g in groups[1:]]
        return (1, groups[0][0], k[0] if k else 0, k[1] if len(k) > 1 else 0,
                k[2] if len(k) > 2 else 0, 0)
    return (0, r[0], r[1], r[2], r[3], r[4])


def _evaluate_5(cards) -> tuple:
    """Wrapper que extrai ranks/suit e chama o cache."""
    r = tuple(sorted((RANK_MAP[c.value] for c in cards), reverse=True))
    fl = (cards[0].suit == cards[1].suit == cards[2].suit ==
          cards[3].suit == cards[4].suit)
    return _eval_ranks_flush(r, fl)


def best_hand_score(hole, board) -> tuple:
    """Melhor combinação de 5 em (hole + board). Usa cache interno."""
    all_cards = list(hole) + list(board)
    n = len(all_cards)
    if n < 5:
        r = tuple(sorted((RANK_MAP[c.value] for c in all_cards), reverse=True))
        return (0,) + r + (0,) * (5 - len(r))
    best = None
    for combo in combinations(all_cards, 5):
        s = _evaluate_5(combo)
        if best is None or s > best:
            best = s
    return best


# ==============================================================================
#  MONTE CARLO COM TRIALS ADAPTATIVOS + TIME GUARD
# ==============================================================================

# Classe leve para cartas simuladas (sem overhead de __dict__)
class _Card:
    __slots__ = ("value", "suit")

    def __init__(self, value: str, suit: str):
        self.value = value
        self.suit = suit

    def __str__(self):
        return self.value + self.suit


# Deck completo pré-construído (evita recriar a cada chamada)
_FULL_DECK: list[tuple[str, str]] = [
    (r, s)
    for r in ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
    for s in ("s", "h", "d", "c")
]


def hand_percentile(hole, board) -> float:
    """
    Estima equity via Monte Carlo com:
    - Trials adaptativos por rua (menos no river onde equity converge mais rápido)
    - Time guard: interrompe se estiver demorando (fallback seguro)
    - Cache de avaliação (lru_cache em _eval_ranks_flush)

    Tempos medidos (servidor do torneio pode variar):
      flop  80 trials → avg ~4ms, max ~11ms
      turn  70 trials → avg ~4ms, max ~7ms
      river 60 trials → avg ~3ms, max ~6ms
    """
    import time as _time

    board_len = len(board)
    if board_len <= 3:
        max_trials = 80
    elif board_len == 4:
        max_trials = 70
    else:
        max_trials = 60

    known = set()
    for c in hole:
        known.add(c.value + c.suit)
    for c in board:
        known.add(c.value + c.suit)

    deck = [pair for pair in _FULL_DECK if pair[0] + pair[1] not in known]
    cards_needed = 5 - board_len

    wins = 0.0
    trials_done = 0
    deadline = _time.perf_counter() + 0.035  # 35ms — margem de 15ms para o resto da função

    for _ in range(max_trials):
        # Time guard: para antes de estorar o limite
        if trials_done % 20 == 0 and _time.perf_counter() > deadline:
            break

        sample = random.sample(deck, 2 + cards_needed)
        opp = [_Card(*sample[0]), _Card(*sample[1])]
        extra = [_Card(*x) for x in sample[2:]]
        full_board = list(board) + extra

        my_s = best_hand_score(hole, full_board)
        op_s = best_hand_score(opp, full_board)

        if my_s > op_s:
            wins += 1.0
        elif my_s == op_s:
            wins += 0.5

        trials_done += 1

    return wins / max(1, trials_done)


# ==============================================================================
#  FORÇA PRÉ-FLOP (Chen formula ajustada para heads-up)
# ==============================================================================

def chen_score(c1, c2) -> float:
    r1, r2 = RANK_MAP[c1.value], RANK_MAP[c2.value]
    if r1 < r2:
        r1, r2 = r2, r1
    suited = c1.suit == c2.suit
    gap = r1 - r2

    score = {14: 10, 13: 8, 12: 7, 11: 6}.get(r1, r1 / 2)

    if gap == 0:
        score = max(score * 2, 5)
    if suited:
        score += 2
    if gap == 1:
        score += 1
    elif gap == 2:
        score -= 1
    elif gap == 3:
        score -= 2
    elif gap > 3:
        score -= (gap - 3 + 4)
    if gap != 0 and r2 >= 7:
        score += 1  # conectores médios valem mais heads-up

    return max(0.0, score)


# ==============================================================================
#  ANÁLISE DE BOARD TEXTURE
# ==============================================================================

def board_texture(board) -> dict:
    if not board:
        return {"wet": False, "paired": False}
    suits_b = [c.suit for c in board]
    ranks_b = [RANK_MAP[c.value] for c in board]
    flush_draw = max(
        suits_b.count(s) for s in set(suits_b)
    ) >= 2
    paired = len(set(ranks_b)) < len(ranks_b)
    straight_draw = False
    if len(ranks_b) >= 3:
        r_sorted = sorted(set(ranks_b))
        for i in range(len(r_sorted) - 2):
            if r_sorted[i + 2] - r_sorted[i] <= 4:
                straight_draw = True
                break
    return {
        "wet": flush_draw or straight_draw,
        "paired": paired,
    }


# ==============================================================================
#  NASH PUSH/FOLD (heads-up)
# ==============================================================================

def _nash_push_min_chen(stack_bb: float) -> float:
    """Chen mínimo para push all-in dado stack em BBs."""
    if stack_bb <= 3:  return 2
    if stack_bb <= 4:  return 3
    if stack_bb <= 5:  return 4
    if stack_bb <= 6:  return 5
    if stack_bb <= 8:  return 6
    if stack_bb <= 10: return 7
    if stack_bb <= 12: return 8
    if stack_bb <= 15: return 9
    if stack_bb <= 20: return 11
    return 13  # stack grande: só premium


def _nash_call_min_chen(stack_bb: float) -> float:
    """Chen mínimo para chamar push adversário."""
    if stack_bb <= 3:  return 3
    if stack_bb <= 5:  return 5
    if stack_bb <= 8:  return 7
    if stack_bb <= 12: return 9
    if stack_bb <= 20: return 11
    return 13


def handle_short_stack(gv: GameView, chen: float, equity: float) -> int | None:
    """Push/fold de Nash para stack ≤ 15 BBs. Retorna None se não aplicável."""
    bb = gv.big_blind
    stack_bb = gv.my_chips / max(1, bb)
    if stack_bb > 15:
        return None

    has_pair = gv.my_hand[0].value == gv.my_hand[1].value
    opp = gv.opponents[0]

    # Oponente já empurrou?
    opp_pushed = (gv.to_call >= opp.chips * 0.75 or
                  gv.to_call >= gv.my_chips * 0.65)

    if opp_pushed:
        call_thresh = _nash_call_min_chen(stack_bb)
        if chen >= call_thresh or has_pair:
            return gv.my_chips
        return -1

    push_thresh = _nash_push_min_chen(stack_bb)
    if chen >= push_thresh or has_pair:
        return gv.my_chips

    # Stack crítico (< 4 BBs): push qualquer coisa ou morre nos blinds
    if stack_bb < 4:
        if gv.to_call == 0:
            return gv.my_chips
        if equity > 0.33:
            return gv.my_chips
        return -1

    if gv.to_call == 0:
        return 0
    if gv.to_call <= bb:
        return 0
    return -1


def handle_danger_zone(gv: GameView, chen: float, equity: float) -> int | None:
    """Proteção extra quando 15-25 BBs e oponente empurra all-in."""
    bb = gv.big_blind
    stack_bb = gv.my_chips / max(1, bb)
    if stack_bb > 25:
        return None
    opp = gv.opponents[0]
    opp_committed = gv.to_call >= min(gv.my_chips, opp.chips) * 0.55
    if not opp_committed:
        return None
    required = max(0.50, 0.65 - stack_bb * 0.01)
    if equity >= required or chen >= 12:
        return 0  # call
    return -1


# ==============================================================================
#  PROFILER DE OPONENTE
# ==============================================================================

class OpponentProfile:
    """
    Classifica o oponente com janela deslizante de 30 mãos.
    Detecta mudanças de estratégia comparando metades da janela.
    """

    def __init__(self):
        self.maos = 0
        self.recent_agg   = deque(maxlen=30)  # 1=raise, 0=não
        self.recent_allin = deque(maxlen=30)  # 1=all-in, 0=não
        self.total_raises = 0
        self.total_allins = 0
        self._hand_raised = False
        self._hand_allin  = False
        self._last_bet    = 0

    def new_hand(self):
        if self.maos > 0:
            self.recent_agg.append(1 if self._hand_raised else 0)
            self.recent_allin.append(1 if self._hand_allin else 0)
        self.maos += 1
        self._hand_raised = False
        self._hand_allin  = False
        self._last_bet    = 0

    def update(self, opp, bb: int, board_len: int):
        bet = opp.current_bet_in_round
        if bet > self._last_bet:
            self._hand_raised = True
            self.total_raises += 1
            if bet >= opp.chips or opp.chips == 0:
                self._hand_allin = True
                self.total_allins += 1
        self._last_bet = bet

    # Métricas
    @property
    def _recent_agg_rate(self) -> float:
        if len(self.recent_agg) < 5:
            return 0.5
        return sum(self.recent_agg) / len(self.recent_agg)

    @property
    def _recent_allin_rate(self) -> float:
        if len(self.recent_allin) < 5:
            return 0.0
        return sum(self.recent_allin) / len(self.recent_allin)

    @property
    def _global_agg_rate(self) -> float:
        return self.total_raises / max(1, self.maos)

    def _changed_strategy(self) -> bool:
        if len(self.recent_agg) < 20:
            return False
        lst = list(self.recent_agg)
        old_avg = sum(lst[:10]) / 10
        new_avg = sum(lst[10:]) / max(1, len(lst[10:]))
        return abs(new_avg - old_avg) > 0.30

    def classify(self) -> str:
        if self.maos < 6:
            return "unknown"
        changed = self._changed_strategy()
        # Recente pesa mais se houve mudança de estratégia
        agg = (self._recent_agg_rate if changed
               else 0.55 * self._recent_agg_rate + 0.45 * self._global_agg_rate)
        allin = (self._recent_allin_rate if changed
                 else 0.55 * self._recent_allin_rate + 0.45 *
                 (self.total_allins / max(1, self.maos)))

        if allin > 0.20:
            return "maniac"
        if agg > 0.65:
            return "aggro"
        if agg < 0.20:
            return "passive"
        return "balanced"


# ==============================================================================
#  BOT PRINCIPAL
# ==============================================================================

class AdaptiveMesgus(Player):

    def __init__(self, name, hand, chips):
        super().__init__(name, hand, chips)
        self.profile = OpponentProfile()

        # Estado persistente entre mãos
        self._hand_number     = 0
        self._street_raises   = 0

        # Snapshots para detecção de nova mão
        self._snap_board_len  = -1   # tamanho do board na última decisão
        self._snap_my_chips   = -1
        self._snap_opp_chips  = -1
        self._snap_pot        = -1

    # ------------------------------------------------------------------
    # Detecção de nova mão — CORRIGIDA
    # ------------------------------------------------------------------

    def _detect_new_hand(self, gv: GameView) -> bool:
        """
        Uma nova mão começa quando:
          1. É a primeira decisão da partida (snap_board_len == -1), OU
          2. O board voltou para 0 cartas depois de ter tido cartas, OU
          3. O pot voltou para um valor baixo (< 3× BB) com board em 0
             → cobre o caso de fold pré-flop onde o board nunca cresceu
             → NÃO confunde com pagamento de blind (pot fica em bb na 1ª ação)

        NÃO depende apenas de mudança de fichas, evitando falsos positivos
        causados pelo pagamento automático dos blinds.
        """
        board_len = len(gv.board)

        if self._snap_board_len == -1:
            return True  # primeira decisão

        board_reset = (board_len == 0 and self._snap_board_len > 0)
        if board_reset:
            return True

        # Board continua em 0 mas pot foi resetado para valor pequeno
        # (indica nova mão pré-flop mesmo sem board anterior)
        if (board_len == 0
                and self._snap_board_len == 0
                and self._snap_pot > gv.big_blind * 10
                and gv.pot <= gv.big_blind * 3):
            return True

        return False

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def _pot_odds(self, gv: GameView) -> float:
        total = gv.pot + gv.to_call
        return gv.to_call / total if total > 0 else 0.0

    def _raise_to(self, gv: GameView, mult: float) -> int:
        target = int(gv.current_bet + gv.big_blind * mult)
        return min(gv.my_chips, max(target, gv.current_bet + 1))

    # ------------------------------------------------------------------
    # Estratégias por perfil
    # ------------------------------------------------------------------

    def _vs_maniac(self, gv, equity, preflop, chen, tex):
        bb = gv.big_blind
        if preflop:
            if chen >= 9 or gv.my_hand[0].value == gv.my_hand[1].value:
                if gv.to_call > bb * 4:
                    return min(gv.my_chips, gv.current_bet * 3)
                return 0  # trap
            if gv.to_call <= bb * 2:
                return 0
            return -1
        else:
            if equity > 0.70:
                if gv.to_call == 0:
                    pct = 0.50 if tex["wet"] else 0.75
                    bet = int(gv.pot * pct)
                    return max(gv.current_bet + 1, gv.current_bet + bet)
                return min(gv.my_chips, int(gv.pot * 1.5))
            if equity > 0.50:
                return 0 if equity > self._pot_odds(gv) + 0.05 else -1
            return -1 if gv.to_call > 0 else 0

    def _vs_passive(self, gv, equity, preflop, chen, tex):
        bb = gv.big_blind
        if preflop:
            if chen >= 7:  return self._raise_to(gv, 3)
            if chen >= 4:  return self._raise_to(gv, 2)
            if gv.to_call <= bb: return 0
            return -1
        else:
            if equity > 0.60:
                pct = 0.70 if not tex["wet"] else 0.55
                bet = int(gv.pot * pct)
                return max(gv.current_bet + 1, gv.current_bet + bet)
            if equity > 0.45:
                if gv.to_call == 0:
                    bet = int(gv.pot * 0.45)
                    return max(gv.current_bet + 1, gv.current_bet + bet)
                return 0 if equity > self._pot_odds(gv) else -1
            if gv.to_call == 0: return 0
            if gv.to_call <= bb * 2: return 0
            return -1

    def _vs_aggro(self, gv, equity, preflop, chen, tex):
        bb = gv.big_blind
        if preflop:
            if chen >= 10:
                if gv.to_call > bb * 2:
                    return min(gv.my_chips, gv.current_bet * 3)
                return self._raise_to(gv, 3)
            if chen >= 7:
                return 0 if gv.to_call <= bb * 5 else -1
            return -1 if gv.to_call > bb * 2 else 0
        else:
            if equity > 0.65:
                if gv.to_call > 0:
                    return min(gv.my_chips, int(gv.pot * 1.2))
                return self._raise_to(gv, 2) if not tex["wet"] else 0
            if equity > 0.48:
                return 0 if equity > self._pot_odds(gv) + 0.03 else -1
            return -1 if gv.to_call > 0 else 0

    def _gto_base(self, gv, equity, preflop, chen, tex):
        bb = gv.big_blind
        eu_sou_bb = self._org_sou_bb(gv)

        if preflop:
            if chen >= 12:
                if gv.to_call > bb * 4 and self._street_raises < 2:
                    return min(gv.my_chips, gv.current_bet * 3)
                return self._raise_to(gv, 3)

            if chen >= 8:
                if eu_sou_bb and gv.to_call == 0:
                    return self._raise_to(gv, 2)
                if gv.to_call <= bb * 4:
                    return self._raise_to(gv, 2)
                return 0 if gv.to_call <= bb * 7 else -1

            if chen >= 5:
                if gv.to_call == 0:
                    return self._raise_to(gv, 2.5) if random.random() < 0.55 else 0
                return 0 if gv.to_call <= bb * 2 else -1

            else:  # lixo
                if gv.to_call == 0:
                    if not eu_sou_bb and random.random() < 0.45:
                        return self._raise_to(gv, 2.5)
                    return 0
                return -1 if gv.to_call > bb else 0

        else:
            po = self._pot_odds(gv)

            if equity > 0.72:
                if gv.to_call == 0:
                    pct = 0.55 if tex["wet"] else 0.80
                    bet = int(gv.pot * pct)
                    return max(gv.current_bet + 1, gv.current_bet + bet)
                if self._street_raises < 2:
                    return min(gv.my_chips, int(gv.pot * 1.4))
                return 0

            if equity > 0.55:
                if gv.to_call == 0:
                    pct = 0.40 if tex["wet"] else 0.60
                    bet = int(gv.pot * pct)
                    return max(gv.current_bet + 1, gv.current_bet + bet)
                return 0 if equity > po + 0.05 else -1

            if equity > 0.42:
                if gv.to_call == 0:
                    can_bluff = not tex["wet"] and self._street_raises == 0
                    if can_bluff and random.random() < 0.30:
                        return self._raise_to(gv, 1.5)
                    return 0
                return 0 if equity > po else -1

            if equity > 0.30:
                if gv.to_call == 0: return 0
                return 0 if gv.to_call <= bb and equity > po else -1

            return 0 if gv.to_call == 0 else -1

    # ------------------------------------------------------------------
    # Ponto de entrada
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
        gv = game_view
        bb = gv.big_blind
        opp = gv.opponents[0]
        board_len = len(gv.board)

        # --- Detectar nova mão ---
        if self._detect_new_hand(gv):
            self.profile.new_hand()
            self._hand_number += 1
            self._street_raises = 0

        # --- Detectar nova rua (reset de raises) ---
        if (self._snap_board_len >= 0
                and board_len > self._snap_board_len):
            self._street_raises = 0

        # --- Atualizar snapshots ---
        self._snap_board_len = board_len
        self._snap_my_chips  = gv.my_chips
        self._snap_opp_chips = opp.chips
        self._snap_pot       = gv.pot

        # --- Atualizar perfil ---
        self.profile.update(opp, bb, board_len)

        # --- Avaliar mão ---
        preflop = (board_len == 0)
        chen = chen_score(gv.my_hand[0], gv.my_hand[1])
        has_pair = gv.my_hand[0].value == gv.my_hand[1].value

        if preflop:
            equity = min(1.0, chen / 20 * 0.65 + 0.18)
            if has_pair:
                equity = min(1.0, equity + 0.15)
        else:
            equity = hand_percentile(gv.my_hand, gv.board)

        tex = board_texture(gv.board)
        stack_bb = gv.my_chips / max(1, bb)

        # --- PRIORIDADE 1: Stack curto (≤ 15 BBs) → Nash ---
        action = handle_short_stack(gv, chen, equity)

        # --- PRIORIDADE 2: Zona de perigo (15-25 BBs + all-in adversário) ---
        if action is None:
            action = handle_danger_zone(gv, chen, equity)

        # --- PRIORIDADE 3: Estratégia adaptativa ---
        if action is None:
            perfil = self.profile.classify()
            if perfil == "maniac":
                action = self._vs_maniac(gv, equity, preflop, chen, tex)
            elif perfil == "passive":
                action = self._vs_passive(gv, equity, preflop, chen, tex)
            elif perfil == "aggro":
                action = self._vs_aggro(gv, equity, preflop, chen, tex)
            else:
                action = self._gto_base(gv, equity, preflop, chen, tex)

        # --- Sanity checks ---
        if action == -1 and gv.to_call == 0:
            action = 0  # nunca fold de graça

        if action > gv.my_chips:
            action = gv.my_chips

        if action > gv.current_bet:
            self._street_raises += 1

        return action


def create_player() -> Player:
    return AdaptiveMesgus("AdaptiveMesgus", Hand(), 0)