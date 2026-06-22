"""
versao_14 — anti-v13 dedicado (alvo único: v13, meta >= 70%).

Fatos que guiaram o design (medidos em analyze_v14.py, 300 partidas):

  - O jogo é decidido nas últimas ~40 mãos (blinds dobram a cada 50):
    lead de fichas na mão 200 prediz só ~51% — early game é quase ruído.
  - Estratégias agressivas demais disparam o radar de maníaco do v13
    (raises/mão > 0.80 → ele paga o dobro); passividade total envenena
    as flags dele mas sangra mais do que rende (testado: 45%).

Dois mecanismos validados (62-63% vs v13, 1000 partidas/config):

  1. BLITZ: raise pot-size em ~75% dos spots, com orçamento próprio logo
     abaixo do radar (BLITZ_RATE_CAP 0.75 < 0.80). Contra raise pot-size
     o v13 só paga com hs > ~0.65-0.75 → ~80% de folds.
  2. Endgame range-aware (eff <= RANGE_BB): conhecemos os thresholds
     exatos de shove (hs >= 0.54/0.49) e call (pot_odds + disc replicado)
     do v13 → amostramos o range dele e decidimos call/shove de all-in
     por equity logística real (RANGE_K) com margens conservadoras,
     em vez de heurística cega.

  Falhou no sweep: bifásico passivo→shove (44-47%), blitz 0.8 pot fora
  do radar de big-raise (56-59%), cap de range no limp curto (neutro),
  call de pedágio por equity no endgame (-5pp; o pedágio se repaga toda
  street), margens maiores que c12/s15, RANGE_BB 13-14, k != 4-5.

Fatos da engine (lidos no código, confirmados em logs):

  1. O flop é virado ANTES da 1ª rodada de apostas — board nunca tem 0
     cartas. Há 4 rodadas: R1 (board=3, blinds), R2 (board=3), turn, river.

  2. `current_bet` NÃO reseta entre streets (só entre mãos), e o
     `invested[]` de cada rodada começa em 0 — inclusive os blinds não
     contam. Resultado: AMBOS os jogadores pagam o `current_bet` inteiro
     de novo A CADA street ("pedágio"). Não existe check: toda decisão é
     pagar / foldar / raisar. Um raise para X vira pedágio de X por
     street para os dois até o fim da mão.

  3. Heads-up: o dealer posta o BB e age por último em toda rodada; o SB
     age primeiro. Na 1ª decisão da mão: pot == sb+bb → somos SB.

  4. v8 (alvo): PFR do modelo de oponente nunca incrementa (bug) → diante
     de raises ele assume range estreito no Monte Carlo e overfolda mãos
     médias contra raises grandes. Com lixo, folda ~95% dos pedágios.
     Não tem regime de shove no endgame (blinds dobram a cada 50 mãos).

Estratégia v11:

  - hs ≈ P(à frente vs mão aleatória) por avaliação exata + classificação
    fina (overpair / top pair+kicker / par de board / flush de board...)
    com bônus de draw e descontos de textura.
  - Spots de pedágio: paga por pot odds (+taxa de streets futuros), raise
    moderado por valor (multiplica pedágios futuros), bluff-raise >= pot
    adaptativo contra quem folda.
  - Spots de agressão real (current_bet > nível que já igualamos): range
    do oponente é forte → desconto na hs, fold de mãos médias.
  - Push/fold em stack curto (em BBs) — domina o endgame de blinds altos.
  - Trackers: fold-to-raise (inferido pela progressão de streets) e
    frequência de raise do oponente → adapta blefe e calldowns.
  - Decisão <1ms — zero risco do timeout de 50ms.
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from players.player import Player
from game.game_view import GameView
from cards.cards import Hand

VAL = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}

_ALL_CARDS = [(v, s) for v in VAL for s in "shdc"]


class _C:
    """Carta leve (só .value/.suit) para avaliar mãos hipotéticas do opp."""
    __slots__ = ("value", "suit")

    def __init__(self, value: str, suit: str) -> None:
        self.value = value
        self.suit = suit


def _cv(c) -> int:
    return VAL[c.value]


# ─── Avaliador de 5 cartas (tuple comparável) ─────────────────────────────

def _eval5(cards) -> tuple:
    vals = sorted((_cv(c) for c in cards), reverse=True)
    suits = [c.suit for c in cards]
    vc = Counter(vals)
    counts = sorted(vc.values(), reverse=True)
    is_flush = len(set(suits)) == 1

    uv = sorted(set(vals))
    if 14 in uv:
        uv = [1] + uv
    is_straight = False
    straight_high = 0
    for i in range(len(uv) - 5, -1, -1):
        if uv[i + 4] - uv[i] == 4:
            is_straight = True
            straight_high = uv[i + 4]
            break

    if is_flush and is_straight:
        return (8, straight_high)
    if counts[0] == 4:
        q = max(v for v, n in vc.items() if n == 4)
        k = max(v for v in vals if v != q)
        return (7, q, k)
    if counts[0] == 3 and counts[1] >= 2:
        t = max(v for v, n in vc.items() if n == 3)
        p = max(v for v, n in vc.items() if n >= 2 and v != t)
        return (6, t, p)
    if is_flush:
        return (5,) + tuple(vals)
    if is_straight:
        return (4, straight_high)
    if counts[0] == 3:
        t = max(v for v, n in vc.items() if n == 3)
        ks = [v for v in vals if v != t][:2]
        return (3, t) + tuple(ks)
    if counts[0] == 2 and counts[1] == 2:
        ps = sorted((v for v, n in vc.items() if n == 2), reverse=True)
        k = max(v for v in vals if v not in ps[:2])
        return (2, ps[0], ps[1], k)
    if counts[0] == 2:
        p = max(v for v, n in vc.items() if n == 2)
        ks = [v for v in vals if v != p][:3]
        return (1, p) + tuple(ks)
    return (0,) + tuple(vals)


def _best_hand(cards: list) -> tuple:
    if len(cards) == 5:
        return _eval5(cards)
    return max(_eval5(list(c)) for c in combinations(cards, 5))


# ─── Draws ────────────────────────────────────────────────────────────────

def _flush_draw(hole, board) -> bool:
    sc = Counter(c.suit for c in list(hole) + list(board))
    for s, n in sc.items():
        if n == 4 and any(c.suit == s for c in hole):
            return True
    return False


def _straight_draws(cards) -> tuple[bool, bool]:
    vals = {_cv(c) for c in cards}
    if 14 in vals:
        vals = vals | {1}
    oesd = gut = False
    for lo in range(1, 11):
        present = [k for k in range(5) if (lo + k) in vals]
        if len(present) == 4:
            if 0 in present and 4 in present:
                gut = True
            else:
                oesd = True
    return oesd, gut


# ─── Hand strength heurística (≈ P(à frente) vs mão aleatória) ───────────

def _strength(hole, board) -> tuple[float, float, bool, bool]:
    """Retorna (hs_total, hs_made, strong_draw, any_draw)."""
    hole = list(hole)
    board = list(board)
    allc = hole + board
    best = _best_hand(allc)
    cat = best[0]

    hv = sorted((_cv(c) for c in hole), reverse=True)
    bv = sorted((_cv(c) for c in board), reverse=True)
    top_b = bv[0]
    bcount = Counter(bv)
    hcount = Counter(hv)
    pocket = hv[0] == hv[1]
    ctc = 5 - len(board)

    hs = 0.30
    if cat == 8:
        hs = 0.99
    elif cat == 7:
        q = best[1]
        hs = 0.90 if bcount.get(q, 0) == 4 else 0.985
    elif cat == 6:
        t, p = best[1], best[2]
        board_made = bcount.get(t, 0) >= 3 and bcount.get(p, 0) >= 2
        hs = 0.62 if board_made else 0.965
    elif cat == 5:
        sc = Counter(c.suit for c in allc)
        fsuit = max(sc, key=sc.get)
        mine = [_cv(c) for c in hole if c.suit == fsuit]
        if not mine:
            hs = 0.55
        else:
            hi = max(mine)
            hs = {14: 0.95, 13: 0.91, 12: 0.87, 11: 0.82, 10: 0.78}.get(hi, 0.72)
            if sum(1 for c in board if c.suit == fsuit) >= 4:
                hs -= 0.07
    elif cat == 4:
        hs = 0.90
        if len(board) == 5 and _eval5(board)[0] == 4 and _eval5(board) >= best:
            hs = 0.55
        else:
            buv = sorted(set(bv))
            if 14 in buv:
                buv = [1] + buv
            for i in range(len(buv) - 4, -1, -1):
                if buv[i + 3] - buv[i] <= 4:
                    hs = 0.82
                    break
    elif cat == 3:
        t = best[1]
        if pocket and hv[0] == t:
            hs = 0.93
        elif bcount.get(t, 0) == 2:
            kick = max((v for v in hv if v != t), default=0)
            hs = 0.80 + (0.05 if kick >= 12 else 0.0)
        else:
            hs = min(0.62, 0.38 + 0.022 * (hv[0] - 7))
    elif cat == 2:
        p1, p2 = best[1], best[2]
        live = [p for p in (p1, p2) if hcount.get(p, 0) >= 1 and bcount.get(p, 0) <= 1]
        if len(live) == 2:
            hs = 0.86 + (0.03 if p1 >= top_b else 0.0)
        elif len(live) == 1:
            p = live[0]
            if pocket and p == hv[0]:
                hs = 0.70 if p > top_b else 0.58
            elif p == top_b:
                kick = max((v for v in hv if v != p), default=0)
                hs = 0.70 + (0.04 if kick >= 13 else 0.0)
            else:
                hs = 0.58
        else:
            hs = min(0.55, 0.32 + 0.020 * (hv[0] - 7))
    elif cat == 1:
        p = best[1]
        if bcount.get(p, 0) >= 2:
            hs = min(0.46, 0.28 + 0.018 * (hv[0] - 7) + 0.008 * (hv[1] - 7))
        elif pocket:
            if p > top_b:
                hs = 0.72 + 0.004 * max(0, p - 10)
            else:
                above = sum(1 for v in set(bv) if v > p)
                hs = 0.60 if above == 1 else (0.52 if above == 2 else 0.46)
        else:
            kick = max((v for v in hv if v != p), default=0)
            if p == top_b:
                hs = 0.60 + (0.08 if kick >= 13 else 0.05 if kick >= 11 else 0.02 if kick >= 9 else 0.0)
            else:
                above = sum(1 for v in set(bv) if v > p)
                hs = 0.54 if above == 1 else 0.47
    else:
        hs = 0.16 + 0.022 * (hv[0] - 7) + 0.010 * (hv[1] - 7)

    # Desconto de textura: board monotone sem nossa cor.
    if 1 <= cat <= 2:
        bs = Counter(c.suit for c in board)
        if bs:
            ms, mn = bs.most_common(1)[0]
            if mn >= 3 and not any(c.suit == ms for c in hole):
                hs -= 0.05
            if mn >= 4 and not any(c.suit == ms for c in hole):
                hs -= 0.08

    hs_made = max(0.05, min(0.99, hs))

    fdraw = _flush_draw(hole, board) if ctc > 0 and cat < 5 else False
    oesd, gut = _straight_draws(allc) if ctc > 0 and cat < 4 else (False, False)
    overcards = (not pocket) and hv[1] > top_b and cat == 0

    bonus = 0.0
    if cat <= 1:
        if fdraw:
            bonus += 0.16 if ctc == 2 else 0.09
        if oesd:
            bonus += 0.12 if ctc == 2 else 0.07
        elif gut:
            bonus += 0.05 if ctc == 2 else 0.03
        if overcards:
            bonus += 0.05 if ctc == 2 else 0.02
        bonus = min(bonus, 0.26)
    elif cat == 2 and fdraw:
        bonus = 0.04

    hs_total = min(0.97, hs_made + bonus)
    strong_draw = fdraw or oesd
    any_draw = strong_draw or gut
    return hs_total, hs_made, strong_draw, any_draw


# ─── Player ───────────────────────────────────────────────────────────────

class Versao14(Player):

    SAFE = True               # captura exceções (produção)

    # Tunables.
    SHORT_BB = 9.0            # regime push/fold abaixo disso (stack efetivo em BB)
    SHOVE_HS = 0.54           # hs mínimo para shove no regime curto
    R1_OPEN_RAISE_HS = 0.62   # raise por valor na R1
    R1_SB_CALL_HS = 0.44      # SB paga pedágio R1 (mais tight que v11: +2pp vs v1/v11)
    R1_STEAL_FREQ = 0.16      # steal raise R1 com lixo (escala c/ fold-to-raise)
    VALUE_RAISE_HS = 0.72     # raise por valor em pedágio
    STRONG_HS = 0.60          # raise pequeno ocasional
    TOLL_TAX = {3: 0.05, 4: 0.03, 5: 0.0}  # taxa por streets futuros
    BLUFF_FREQ = {3: 0.10, 4: 0.08, 5: 0.05}
    VS_DISC_MULT = 2.00       # multiplicador do desconto em _vs_raise (anti v1/v11/v12)
    ALLIN_DISC_ADD = 0.04     # desconto extra ao defender call que nos deixa all-in
    # ── Novos knobs anti-v13 ──
    RERAISED_EXTRA_DISC = 0.15  # desconto extra vs raise do opp após nosso raise
    RARE_SHOVE_FREQ = 0.0      # shove raro em pote inflado (0 = off)
    RARE_SHOVE_POTBB = 8.0     # pot mínimo em bb para o rare shove
    RARE_SHOVE_LO = 0.50       # hs mínimo do rare shove
    VRAISE_FREQ = 0.72         # freq do raise por valor em pedágio
    # ── Modo blitz: raise pot-size quase toda mão, sob o radar de maníaco
    # do v13 (rate < 0.80 → ele só paga com hs > ~0.75) ──
    BLITZ_FREQ = 0.75          # freq de blitz por spot (0 = off)
    BLITZ_POT = 1.0            # tamanho do blitz em frações do pot
    BLITZ_RATE_CAP = 0.75      # teto de raises/mão (radar de maníaco = 0.80)
    BLITZ_CALLED_PEN = 0.10    # pen extra em pedágio após blitz pago (range 0.75+)
    BLITZ_CHIP_MULT = 1.8      # só blitza se stack >= custo × isto
    BLITZ_UNIFORM = False      # True: blitza também as mãos de valor (range opaco)
    OPEN_DEFEND_MULT = 1.0     # mult do disc vs open do opp em s3 (range c/ ~40% bluff)
    # ── Estratégia bifásica: envenenar flags do v13 e dominar o endgame ──
    # O jogo se decide nas últimas ~40 mãos (lead na mão 200 → WR 51%).
    # A defesa de all-in do v13: +0.10 se big_rate<=0.05, +0.05 se passivo.
    PASSIVE_PREP = False       # fase 1: zero raises (envenena flags do alvo)
    PHASE2_BB = 0.0            # entra na fase 2 (shove war) com eff <= isto (0 = off)
    P2_SHOVE_HS = 0.40         # hs mínimo de shove na fase 2
    P2_CALL_HS = 0.62          # hs mínimo p/ pagar all-in dele na fase 2
    # ── Endgame range-aware: equity real vs range conhecido do v13 ──
    RANGE_BB = 11.0            # liga o módulo com eff <= isto (0 = off)
    RANGE_K = 4.0              # inclinação do logístico hs->equity
    RANGE_T_SHORT = 0.51       # piso do range de shove curto do v13
    RANGE_T_DEEP = 0.72        # piso do range de value-shove deep do v13
    RANGE_CALL_MARGIN = 0.12   # margem sobre pot odds no call de all-in
    RANGE_SHOVE_MARGIN = 1.5   # margem (em bb) do EV de shove vs fold
    RANGE_SAMPLES = 120        # mãos amostradas do range
    RANGE_USE_CAP = True       # capa o range dele em 0.54 quando ele limpou curto
    RANGE_CAP_HS = 0.54        # teto do range de quem não shovou no regime curto
    RIVER_VRAISE_FREQ = 0.55  # freq de raise por valor no river (pedágio)
    RERAISE_VALUE_HS = 0.84   # raise por valor vs agressão real
    # Penalidade de range (opp que continua é mais forte que aleatório).
    PEN_S4 = 0.07
    PEN_S5 = 0.05
    PEN_L25 = 0.05            # nível de aposta >= 2.5bb
    PEN_L5 = 0.05             # >= 5bb
    PEN_L10 = 0.06            # >= 10bb
    PEN_EARLY = 0.09          # boost vs opp fit-or-fold (street >= 4)
    PEN_EARLY_R = 0.04        # extra no river
    TRAP_FREQ = 0.30          # call-trap com monster vs raise
    R1_OPEN_MULT = 2.4        # tamanho do raise R1 (em bb; < breakpoint 2.5bb dos alvos)
    VRAISE_POT = 0.40         # fração do pot no raise por valor em pedágio
    STEAL_EARLY_MULT = 1.0    # multiplicador de steal vs opp que folda cedo

    def __init__(self, name, hand, chips):
        super().__init__(name, hand, chips)
        self._rng = random.Random()

        # Opponent model (persistente na partida).
        self._hands = 0
        self._opp_faced_raise = 0   # nossos raises que opp teve de responder
        self._opp_fold_raise = 0    # ...e foldou
        self._opp_raises = 0        # raises do opp observados
        self._opp_big_raises = 0    # raises grandes/all-in do opp
        self._opp_early_folds = 0   # mãos em que opp foldou antes do river
        self._decisions = 0

        self._my_big_raises = 0    # nossos raises grandes/all-in (radar do alvo)
        self._my_raise_events = 0  # nossos raises totais (radar de maníaco do alvo)

        # Estado por mão.
        self._prep = False
        self._we_raised_hand = False
        self._opp_called_our_raise = False
        self._last_dealer = None
        self._am_sb = False
        self._first_done = False
        self._matched = 0          # current_bet que já igualamos nesta mão
        self._pending_raise = None  # (street, was_river, was_allin)
        self._opp_raised_hand = False  # opp foi o agressor nesta mão
        self._we_folded_hand = False
        self._saw_allin_hand = False
        self._last_street_seen = 3

    # ═══ Tracking ═════════════════════════════════════════════════════════

    def _update(self, gv: GameView) -> None:
        self._decisions += 1
        new_hand = self._last_dealer is None or gv.dealer_position != self._last_dealer
        if new_hand:
            if self._pending_raise is not None:
                street, was_river, was_allin = self._pending_raise
                if not was_river and not was_allin:
                    self._opp_faced_raise += 1
                    self._opp_fold_raise += 1
                self._pending_raise = None
            # Mão anterior terminou antes do river, sem fold nosso e sem
            # all-in → opp foldou cedo (fit-or-fold).
            if (self._hands > 0 and not self._we_folded_hand
                    and not self._saw_allin_hand and self._last_street_seen < 5):
                self._opp_early_folds += 1
            self._hands += 1
            self._first_done = False
            self._we_raised_hand = False
            self._opp_called_our_raise = False
            self._opp_raised_hand = False
            self._we_folded_hand = False
            self._saw_allin_hand = False
            self._matched = gv.big_blind   # blinds são o nível base
            # SB age primeiro: na 1ª decisão da mão o pot ainda é só blinds.
            self._am_sb = gv.pot <= gv.small_blind + gv.big_blind
            self._last_dealer = gv.dealer_position
        else:
            if self._pending_raise is not None:
                # Recebemos nova decisão na mesma mão → opp pagou/raisou.
                self._opp_faced_raise += 1
                self._pending_raise = None
                if gv.current_bet <= self._matched:
                    self._opp_called_our_raise = True

        self._last_street_seen = len(gv.board)
        if gv.to_call >= gv.my_chips or (gv.opponents and gv.opponents[0].chips == 0):
            self._saw_allin_hand = True

        # Agressão real do opp: current_bet acima do que já igualamos.
        if gv.current_bet > self._matched:
            self._opp_raises += 1
            self._opp_raised_hand = True
            inc = gv.current_bet - self._matched
            pot_before = max(1, gv.pot - gv.to_call)
            if gv.to_call >= gv.my_chips or inc >= 0.9 * pot_before:
                self._opp_big_raises += 1

    def _fold_to_raise(self) -> float:
        return (self._opp_fold_raise + 1.6) / (self._opp_faced_raise + 4.0)

    def _ftr_scale(self) -> float:
        f = self._fold_to_raise()
        return max(0.10, min(1.8, (f - 0.10) / 0.30))

    def _opp_aggro_rate(self) -> float:
        return self._opp_raises / max(6.0, self._hands)

    def _opp_maniac(self) -> bool:
        return self._hands >= 14 and self._opp_aggro_rate() > 0.80

    def _opp_passive(self) -> bool:
        return self._hands >= 14 and self._opp_aggro_rate() < 0.15

    def _range_penalty(self, street: int, level_bb: float) -> float:
        """Quanto o range do opp que continuou está acima de 'mão aleatória'.

        O opp foldou lixo nas streets anteriores; quanto mais alto o nível
        de aposta da mão (current_bet/bb) e mais avançada a street, mais
        forte o range dele — nossa hs vs aleatório superestima.
        """
        pen = 0.0
        if street >= 4:
            pen += self.PEN_S4
        if street == 5:
            pen += self.PEN_S5
        # Opp fit-or-fold (folda cedo com frequência): range que continua
        # até turn/river é forte mesmo sem raise.
        if self._hands >= 16 and street >= 4:
            early_rate = self._opp_early_folds / self._hands
            if early_rate > 0.45:
                pen += self.PEN_EARLY
                if street == 5:
                    pen += self.PEN_EARLY_R
        lv = 0.0
        if level_bb >= 2.5:
            lv += self.PEN_L25
        if level_bb >= 5.0:
            lv += self.PEN_L5
        if level_bb >= 10.0:
            lv += self.PEN_L10
        # Pote raisado PELO OPONENTE: range dele é muito mais forte.
        if self._opp_raised_hand:
            lv *= 1.7
            if self._opp_passive():
                lv *= 1.3
        pen += lv
        if self._opp_maniac():
            pen *= 0.5
        return pen

    # ═══ Helpers ══════════════════════════════════════════════════════════

    def _cap(self, gv: GameView, target: int) -> int:
        invested = gv.current_bet - gv.to_call
        max_total = invested + gv.my_chips
        return min(target, max_total)

    def _shove_total(self, gv: GameView) -> int:
        return (gv.current_bet - gv.to_call) + gv.my_chips

    def _do_call(self, gv: GameView) -> int:
        self._matched = gv.current_bet
        return 0

    def _do_raise(self, gv: GameView, target: int) -> int:
        target = self._cap(gv, target)
        if target <= gv.current_bet:
            return self._do_call(gv)
        opp = gv.opponents[0]
        invested = gv.current_bet - gv.to_call
        delta = target - invested
        was_allin = (delta >= gv.my_chips) or (target - gv.current_bet >= opp.chips)
        self._pending_raise = (len(gv.board), len(gv.board) == 5, was_allin)
        self._matched = target
        self._we_raised_hand = True
        self._my_raise_events += 1
        pot_before = max(1, gv.pot - gv.to_call)
        if was_allin or (target - gv.current_bet) >= 0.9 * pot_before:
            self._my_big_raises += 1
        return target

    # ═══ Decisão ══════════════════════════════════════════════════════════

    def decision(self, gv: GameView) -> int:
        try:
            act = self._decide(gv)
        except Exception:
            if not self.SAFE:
                raise
            return 0
        if act == -1:
            self._we_folded_hand = True
        return act

    def _decide(self, gv: GameView) -> int:
        self._update(gv)

        pot = max(1, gv.pot)
        to_call = gv.to_call
        bb = max(1, gv.big_blind)
        opp = gv.opponents[0]
        street = len(gv.board)
        first_action = not self._first_done
        self._first_done = True

        hs_total, hs_made, strong_draw, any_draw = _strength(gv.my_hand, gv.board)
        pot_odds = to_call / (pot + to_call) if to_call > 0 else 0.0
        eff = min(gv.my_chips, opp.chips + to_call)
        eff_bb = eff / bb
        spr = gv.my_chips / pot
        rng = self._rng.random()

        is_aggro = gv.current_bet > self._matched and not (
            first_action and gv.current_bet <= bb
        )

        phase2 = self.PHASE2_BB > 0 and eff_bb <= self.PHASE2_BB
        self._prep = self.PASSIVE_PREP

        # ── 1. Call nos deixa all-in ───────────────────────────────────────
        if to_call >= gv.my_chips:
            if self.RANGE_BB > 0:
                r = self._range_call_allin(gv, hs_total, pot_odds)
                if r != -2:
                    return r
            if phase2:
                if pot_odds <= 0.12 or hs_total >= self.P2_CALL_HS:
                    return self._do_call(gv)
                return -1
            wide = eff_bb <= self.SHORT_BB or (pot / bb) >= 10
            disc = 0.04 if wide else 0.11
            # Frequência de raises grandes do opp: shover raro = só valor.
            big_rate = self._opp_big_raises / max(8.0, self._hands)
            if big_rate <= 0.05:
                disc += 0.10
            elif big_rate >= 0.20:
                disc -= 0.05
            if self._opp_maniac():
                disc -= 0.04
            if self._opp_passive():
                disc += 0.05
            disc += self.ALLIN_DISC_ADD
            if pot_odds <= 0.12:
                return self._do_call(gv)
            return self._do_call(gv) if hs_total > pot_odds + disc else -1

        # ── 2a. Fase 2: shove war com as flags do alvo envenenadas ────────
        # Sem shove, cai na lógica normal (pens corretas, sem raises).
        if phase2 and hs_total >= self.P2_SHOVE_HS:
            return self._do_raise(gv, self._shove_total(gv))

        # ── 2b. Endgame range-aware: shove por EV vs range de call do v13 ─
        if self.RANGE_BB > 0 and eff_bb <= self.RANGE_BB:
            # Ele agiu antes de nós nesta street sem shovar e está no regime
            # curto → range dele está capado (teria shovado hs >= 0.54).
            cap = 2.0
            if (self.RANGE_USE_CAP and not self._am_sb
                    and gv.current_bet <= bb and not is_aggro
                    and eff_bb <= self.SHORT_BB + 1):
                cap = self.RANGE_CAP_HS
            ev = self._range_shove_ev(gv, hs_total, bb, cap_hs=cap)
            if ev is not None and ev > self.RANGE_SHOVE_MARGIN * bb:
                return self._do_raise(gv, self._shove_total(gv))
            if eff_bb <= self.SHORT_BB:
                return self._do_call(gv) if hs_total > pot_odds + 0.04 else -1
            # eff entre SHORT_BB e RANGE_BB sem shove: lógica normal abaixo.

        # ── 2. Stack curto: push/fold ──────────────────────────────────────
        if eff_bb <= self.SHORT_BB:
            if hs_total >= self.SHOVE_HS:
                return self._do_raise(gv, self._shove_total(gv))
            if hs_total >= self.SHOVE_HS - 0.05 and rng < 0.40:
                return self._do_raise(gv, self._shove_total(gv))
            return self._do_call(gv) if hs_total > pot_odds + 0.04 else -1

        # ── 3. R1, primeira ação, sem raise à frente ───────────────────────
        if first_action and street == 3 and gv.current_bet <= bb:
            return self._r1_open(gv, hs_total, hs_made, strong_draw, pot_odds, bb)

        # ── 4. Agressão real do oponente ───────────────────────────────────
        if is_aggro:
            return self._vs_raise(gv, hs_total, hs_made, strong_draw, street,
                                  pot_odds, pot, rng, spr)

        # ── 5. Pedágio (nível já igualado, nova street) ────────────────────
        return self._toll(gv, hs_total, hs_made, strong_draw, street,
                          pot_odds, pot, rng, spr, bb)

    # ═══ Endgame range-aware ══════════════════════════════════════════════

    def _sample_opp_hs(self, gv) -> list[float]:
        """hs_total de mãos aleatórias do opp consistentes com o board."""
        seen = {(c.value, c.suit) for c in list(gv.my_hand) + list(gv.board)}
        deck = [_C(v, s) for (v, s) in _ALL_CARDS if (v, s) not in seen]
        board = list(gv.board)
        out = []
        for _ in range(self.RANGE_SAMPLES):
            a, b = self._rng.sample(deck, 2)
            out.append(_strength((a, b), board)[0])
        return out

    def _eq_vs(self, hs_us: float, hs_opp: float) -> float:
        from math import exp
        return 1.0 / (1.0 + exp(-self.RANGE_K * (hs_us - hs_opp)))

    def _range_call_allin(self, gv, hs_total, pot_odds) -> int:
        """Call de all-in por equity real vs o range de shove do v13."""
        eff_bb = min(gv.my_chips, gv.opponents[0].chips + gv.to_call) / max(1, gv.big_blind)
        floor = self.RANGE_T_SHORT if eff_bb <= self.SHORT_BB + 1 else self.RANGE_T_DEEP
        sample = [h for h in self._sample_opp_hs(gv) if h >= floor]
        if len(sample) < 8:
            return -2   # amostra insuficiente: usa lógica antiga
        eq = sum(self._eq_vs(hs_total, h) for h in sample) / len(sample)
        if pot_odds <= 0.12 or eq > pot_odds + self.RANGE_CALL_MARGIN:
            return self._do_call(gv)
        return -1

    def _range_shove_ev(self, gv, hs_total, bb, cap_hs: float = 2.0) -> float | None:
        """EV (em chips) de shovar agora, dado o threshold de call do v13."""
        opp = gv.opponents[0]
        pot0 = gv.pot
        my_more = gv.my_chips
        call_amt = min(opp.chips, my_more + gv.to_call)
        opp_odds = call_amt / max(1, pot0 + my_more + call_amt)
        # Disc do v13 ao defender all-in (réplica do código dele).
        rate = self._my_raise_events / max(6.0, float(self._hands))
        big_rate = self._my_big_raises / max(8.0, float(self._hands))
        disc = 0.04
        if big_rate <= 0.05:
            disc += 0.10
        elif big_rate >= 0.20:
            disc -= 0.05
        if self._hands >= 14 and rate > 0.80:
            disc -= 0.04
        if self._hands >= 14 and rate < 0.15:
            disc += 0.05
        disc += 0.04   # ALLIN_DISC_ADD do v13
        t_call = opp_odds + disc
        hs_opps = self._sample_opp_hs(gv)
        if cap_hs < 2.0:
            # Range condicionado: mãos acima do cap teriam shovado antes.
            capped = [h for h in hs_opps if h <= cap_hs]
            if len(capped) >= 8:
                hs_opps = capped
        callers = [h for h in hs_opps if h > t_call]
        fold_p = 1.0 - len(callers) / max(1, len(hs_opps))
        eq = (sum(self._eq_vs(hs_total, h) for h in callers) / len(callers)
              if callers else 0.5)
        return (fold_p * pot0
                + (1.0 - fold_p) * (eq * (pot0 + my_more + call_amt) - my_more))

    def _try_blitz(self, gv, pot, rng) -> int | None:
        """Raise pot-size sob o radar de maníaco do v13 (paga só hs>~0.75)."""
        if self.BLITZ_FREQ <= 0.0 or rng >= self.BLITZ_FREQ:
            return None
        if self._my_raise_events / max(6.0, float(self._hands)) >= self.BLITZ_RATE_CAP:
            return None
        if self._opp_called_our_raise:
            return None   # blitz já pago nesta mão: range dele é 0.75+
        target = gv.current_bet + int(self.BLITZ_POT * pot)
        invested = gv.current_bet - gv.to_call
        cost = target - invested
        if gv.my_chips < cost * self.BLITZ_CHIP_MULT:
            return None   # sem stack para blitzar mantendo fold-ability
        return self._do_raise(gv, target)

    # ── R1 open ────────────────────────────────────────────────────────────

    def _r1_open(self, gv, hs_total, hs_made, strong_draw, pot_odds, bb):
        rng = self._rng.random()
        raise_target = int(self.R1_OPEN_MULT * bb)

        if self._prep:
            # Fase passiva: nunca raisa — só call/fold pelo threshold.
            if self._am_sb:
                return self._do_call(gv) if hs_total >= self.R1_SB_CALL_HS else -1
            return self._do_call(gv) if hs_total >= self.R1_SB_CALL_HS - 0.05 else -1

        blitz = self._try_blitz(gv, max(1, gv.pot), self._rng.random())
        if blitz is not None:
            return blitz

        if hs_total >= self.R1_OPEN_RAISE_HS:
            if rng < 0.80:
                return self._do_raise(gv, raise_target)
            return self._do_call(gv)
        if strong_draw and rng < 0.45:
            return self._do_raise(gv, raise_target)
        steal = self.R1_STEAL_FREQ * self._ftr_scale()
        if self._hands >= 16 and self._opp_early_folds / self._hands > 0.55:
            steal *= self.STEAL_EARLY_MULT
        if rng < steal:
            return self._do_raise(gv, raise_target)

        if self._am_sb:
            return self._do_call(gv) if hs_total >= self.R1_SB_CALL_HS else -1
        # BB: pot já tem o call do SB → odds melhores.
        return self._do_call(gv) if hs_total >= self.R1_SB_CALL_HS - 0.05 else -1

    # ── Enfrentando raise real ─────────────────────────────────────────────

    def _vs_raise(self, gv, hs_total, hs_made, strong_draw, street,
                  pot_odds, pot, rng, spr):
        to_call = gv.to_call
        inc = gv.current_bet - self._matched
        rel = inc / max(1, pot - to_call)

        # Re-raise por valor.
        if hs_made >= self.RERAISE_VALUE_HS:
            if self._prep:
                return self._do_call(gv)   # fase passiva: trap sempre
            if rng < self.TRAP_FREQ:
                return self._do_call(gv)   # trap: deixa ele pagar pedágios
            if spr <= 2.2:
                return self._do_raise(gv, self._shove_total(gv))
            return self._do_raise(gv, gv.current_bet + int(0.9 * pot))

        disc = 0.05 + 0.11 * min(1.5, rel)
        disc += 0.6 * self._range_penalty(street, gv.current_bet / max(1, gv.big_blind))
        disc *= self.VS_DISC_MULT
        if self._we_raised_hand:
            disc += self.RERAISED_EXTRA_DISC
        elif street == 3 and gv.current_bet <= 4 * gv.big_blind:
            # Open do opp em s3: o steal adaptativo dele infla o bluff share.
            disc *= self.OPEN_DEFEND_MULT
        if self._opp_maniac():
            disc *= 0.45
        if self._opp_passive():
            disc *= 1.4

        if hs_total - disc > pot_odds:
            return self._do_call(gv)

        if strong_draw and street < 5 and pot_odds < 0.28:
            return self._do_call(gv)

        # Semi-bluff re-raise ocasional com draw forte.
        if strong_draw and street == 3 and rng < 0.16 and not self._prep:
            return self._do_raise(gv, gv.current_bet + int(0.9 * pot))

        return -1

    # ── Pedágio ────────────────────────────────────────────────────────────

    def _toll(self, gv, hs_total, hs_made, strong_draw, street,
              pot_odds, pot, rng, spr, bb):
        to_call = gv.to_call

        if self._prep:
            # Fase passiva: nunca raisa — paga/folda por odds.
            tax = self.TOLL_TAX.get(street, 0.0)
            pen = self._range_penalty(street, gv.current_bet / bb)
            if hs_total > pot_odds + tax + pen:
                return self._do_call(gv)
            if strong_draw and street < 5 and pot_odds < 0.30:
                return self._do_call(gv)
            if street == 3 and hs_total > pot_odds - 0.02 and to_call <= bb:
                return self._do_call(gv)
            return -1

        # Shove raro em pote inflado: o alvo defende all-in +0.10 mais
        # tight quando nosso big_rate <= 0.05 — manter-se sob o radar.
        if (self.RARE_SHOVE_FREQ > 0.0
                and pot >= self.RARE_SHOVE_POTBB * bb
                and gv.my_chips > gv.to_call
                and self._my_big_raises <= 0.05 * max(8.0, self._hands)
                and hs_total >= self.RARE_SHOVE_LO
                and rng < self.RARE_SHOVE_FREQ):
            return self._do_raise(gv, self._shove_total(gv))

        if self.BLITZ_UNIFORM or hs_made < self.VALUE_RAISE_HS:
            blitz = self._try_blitz(gv, pot, self._rng.random())
            if blitz is not None:
                return blitz

        # Raise por valor: multiplica pedágios futuros do oponente.
        if hs_made >= self.VALUE_RAISE_HS:
            if spr <= 1.8:
                return self._do_raise(gv, self._shove_total(gv))
            if street == 5 and rng < self.RIVER_VRAISE_FREQ:
                # River: única street, polariza maior.
                return self._do_raise(gv, gv.current_bet + int(1.1 * pot))
            if rng < self.VRAISE_FREQ:
                target = max(int(gv.current_bet * 2.2),
                             gv.current_bet + int(self.VRAISE_POT * pot))
                return self._do_raise(gv, target)
            return self._do_call(gv)

        if hs_made >= self.STRONG_HS:
            if rng < 0.30:
                target = max(int(gv.current_bet * 2.0),
                             gv.current_bet + int(0.45 * pot))
                return self._do_raise(gv, target)
            return self._do_call(gv)

        # Semi-bluff com draw forte.
        if strong_draw and street < 5 and rng < 0.28:
            return self._do_raise(gv, gv.current_bet + int(0.9 * pot))

        # Bluff-raise adaptativo (>= pot → gatilho de overfold do v8).
        if hs_total < 0.40 and rng < self.BLUFF_FREQ.get(street, 0.10) * self._ftr_scale():
            return self._do_raise(gv, gv.current_bet + int(1.0 * pot))

        # Call por pot odds + taxa de pedágios futuros + range do opp.
        tax = self.TOLL_TAX.get(street, 0.0)
        pen = self._range_penalty(street, gv.current_bet / bb)
        if self._opp_called_our_raise:
            pen += self.BLITZ_CALLED_PEN   # ele pagou nosso raise: range forte
        if hs_total > pot_odds + tax + pen:
            return self._do_call(gv)
        if strong_draw and street < 5 and pot_odds < 0.30:
            return self._do_call(gv)
        if street == 3 and not strong_draw and hs_total > pot_odds - 0.02 and to_call <= bb:
            return self._do_call(gv)   # pedágio barato com equity marginal
        return -1


def create_player() -> Player:
    return Versao14("versao_14", Hand(), 0)
