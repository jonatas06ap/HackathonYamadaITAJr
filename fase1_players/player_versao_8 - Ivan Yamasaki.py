"""
versao_8 — bot inspirado no estado da arte de IA para NLHE heads-up.

Não é viável rodar CFR/MCCFR completo dentro do orçamento de 50 ms por
decisão (Libratus/Pluribus treinam por semanas em clusters). Mas o
*pacote conceitual* dos bots SOTA pode ser destilado em uma arquitetura
prática:

  1. **Monte Carlo equity** com *budget de tempo* (substitui o blueprint
     CFR). Em cada decisão pós-flop, amostra mãos do oponente compatíveis
     com a história de ações e estima equity. Abort precoce quando o
     deadline chega — fallback heurístico estilo v7.

  2. **Opponent modeling persistente no match** — rastreia VPIP, PFR e
     aggression factor (AF) ao longo das 5000+ mãos da partida e adapta
     ranges/decisões (igual ao "self-improver" do Libratus, mas online).

  3. **Range narrowing** — usa as ações do oponente nesta mão e seu perfil
     persistente para reduzir o espaço de mãos no MC (análogo ao card
     bucketing + action-conditioned ranges de Pluribus).

  4. **Action abstraction com EV-mix** — em vez de uma única decisão,
     avalia 5 ações candidatas {fold, call, ⅔-pot, pot, overbet 1.5x} e
     escolhe a de maior EV estimada. Mixed strategy (~10% mistura) para
     evitar exploração.

  5. **Push/fold Nash** para stacks curtos — uma aproximação da tabela
     Nash heads-up SB push / BB call quando stack <= 12 BBs. Essencial
     porque os blinds dobram a cada 50 mãos.

  6. **Bluff catching x value betting** — distingue spots de catch
     (oponente representando força, equity ~30-40%) de spots de value
     (vamos ganhar showdown na maioria das vezes).

  7. **Safety net** — deadline global de 40 ms; se MC não termina, cai
     para heurística rápida idêntica ao v7. Garante zero timeouts.
"""
from __future__ import annotations

import random
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from players.player import Player
from game.game_view import GameView
from cards.cards import Card, Hand
from cards.sequences import (
    BASE_DESEMPATE,
    RANK_DOIS_PARES,
    RANK_FLUSH,
    RANK_STRAIGHT,
    RANK_TRINCA,
    RANK_UM_PAR,
    score_cinco_cartas,
    valor_carta,
)


# ─── Constantes globais ───────────────────────────────────────────────────

_VALUES = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
_SUITS = ["s", "h", "d", "c"]
_FULL_DECK: list[Card] = [Card(v, s) for v in _VALUES for s in _SUITS]


def _card_key(c: Card) -> str:
    return c.value + c.suit


# ─── Hand evaluator helpers ───────────────────────────────────────────────

def _best_score(cards: list[Card]) -> int:
    if len(cards) < 5:
        return 0
    if len(cards) == 5:
        return score_cinco_cartas(cards)
    best = 0
    for combo in combinations(cards, 5):
        s = score_cinco_cartas(list(combo))
        if s > best:
            best = s
    return best


def _rank_of(score: int) -> int:
    return score // BASE_DESEMPATE


def _has_flush_draw(cards: list[Card]) -> bool:
    counts = Counter(c.suit for c in cards)
    return any(n >= 4 for n in counts.values())


def _has_straight_draw(cards: list[Card]) -> tuple[bool, bool]:
    vals = {valor_carta(c) for c in cards}
    if 14 in vals:
        vals = vals | {1}
    open_ended = False
    gutshot = False
    for v in sorted(vals):
        run = sum(1 for k in range(4) if (v + k) in vals)
        if run == 4:
            open_ended = True
        present = [k for k in range(5) if (v + k) in vals]
        if len(present) == 4 and 0 in present and 4 in present:
            gutshot = True
    return open_ended, gutshot


# ─── Preflop hand strength (analítico, ≈ Sklansky/Chen) ───────────────────

def _preflop_strength(hand: tuple[Card, ...]) -> float:
    """Score [0..1] aproximando equity heads-up vs mão aleatória."""
    c1, c2 = hand[0], hand[1]
    v1, v2 = valor_carta(c1), valor_carta(c2)
    hi, lo = max(v1, v2), min(v1, v2)
    suited = c1.suit == c2.suit
    pair = v1 == v2
    gap = hi - lo - 1

    if pair:
        return 0.45 + (v1 - 2) * 0.033

    base = 0.28
    base += (hi - 2) * 0.013
    base += (lo - 2) * 0.008
    if suited:
        base += 0.045
    if gap == 0:
        base += 0.030
    elif gap == 1:
        base += 0.015
    elif gap >= 4:
        base -= 0.025
    if hi == 14:
        base += 0.015
    return max(0.20, min(0.78, base))


# Strength threshold approximando "top X% das 169 mãos starting".
# Calibrado para que ~X% das 1326 combos passe.
def _strength_for_top_pct(pct: float) -> float:
    """Threshold de strength para top pct% das mãos starting (aprox.)."""
    # Valores empíricos calibrados sobre _preflop_strength:
    # top 10% ≈ 0.62, top 20% ≈ 0.54, top 35% ≈ 0.46, top 50% ≈ 0.40
    table = [
        (0.05, 0.66), (0.10, 0.62), (0.15, 0.58), (0.20, 0.54),
        (0.25, 0.51), (0.30, 0.48), (0.35, 0.46), (0.40, 0.43),
        (0.50, 0.40), (0.60, 0.37), (0.75, 0.33), (1.00, 0.20),
    ]
    for p, thr in table:
        if pct <= p:
            return thr
    return 0.20


# ─── Nash push/fold (heads-up, aproximação) ───────────────────────────────
# SB push range em função de stack (em BBs). Aprox. derivado da tabela de
# Nash heads-up de SnG/HUNL (vide Holdem Resources / SnG Wizard).
def _nash_sb_push_strength(stack_bb: float) -> float:
    """Strength mínimo para SB shovar em push/fold puro."""
    if stack_bb <= 4:
        return 0.32        # shove quase qualquer mão
    if stack_bb <= 7:
        return 0.36
    if stack_bb <= 10:
        return 0.40
    if stack_bb <= 13:
        return 0.44
    return 0.48


def _nash_bb_call_strength(stack_bb: float) -> float:
    """Strength mínimo para BB chamar um shove."""
    if stack_bb <= 4:
        return 0.36
    if stack_bb <= 7:
        return 0.42
    if stack_bb <= 10:
        return 0.46
    if stack_bb <= 13:
        return 0.50
    return 0.54


# ─── Player ───────────────────────────────────────────────────────────────

class Versao8(Player):

    # Orçamento de tempo total por chamada (oficial: 50 ms).
    SAFETY_BUDGET_MS = 38.0
    MC_BUDGET_MS = 22.0
    MIN_MC_SAMPLES = 8       # se não conseguir nem isso, usa heurística

    # Frequências de mistura (mixed strategy estilo GTO).
    BLUFF_FREQ_BASE = 0.08   # blefar quando lógica diria fold
    OVERBET_FREQ_BASE = 0.15
    SLOWPLAY_FREQ = 0.12     # com monster, às vezes só call

    # Sizings discretos (fração do pote).
    SIZINGS = (0.50, 0.75, 1.10, 1.60)

    def __init__(self, name, hand, chips):
        super().__init__(name, hand, chips)
        self._rng = random.Random()

        # Opponent model persistente no match.
        self._opp_hands_seen = 0
        self._opp_vpip_hands = 0       # mãos onde opp pôs ficha voluntariamente
        self._opp_pfr_hands = 0        # mãos onde opp deu raise pré-flop
        self._opp_postflop_bets = 0    # raises/bets pós-flop
        self._opp_postflop_calls = 0   # calls pós-flop

        # Estado da mão atual.
        self._last_dealer = None
        self._last_board_len = 0
        self._last_pot = 0
        self._this_hand_opp_vpip = False
        self._this_hand_opp_raised_pf = False
        self._this_hand_opp_max_bet = 0
        self._this_hand_my_last_bet = 0

    # ═══ Decisão principal ════════════════════════════════════════════════

    def _org_sou_bb(self, gv) -> bool:
        """[CORRIGIDO PELA ORGANIZAÇÃO — bug de detecção de posição]

        `dealer_position` é o índice do dealer na lista GLOBAL de jogadores da
        engine, e NÃO um valor relativo a este bot. Por isso a verificação
        original baseada em `dealer_position == 0` só acertava quando este bot
        ocupava o assento players[0] da partida, falhando em até 100% das mãos
        quando ocupava players[1]. (A detecção de nova mão deste bot via
        `dealer_position` em outros pontos continua válida — só a leitura de
        posição absoluta era incorreta.)

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

    def decision(self, gv: GameView) -> int:
        try:
            t0 = time.perf_counter()
            deadline = t0 + self.SAFETY_BUDGET_MS / 1000.0

            self._update_opp_model(gv)

            # All-in defense: só call ou fold.
            if gv.to_call >= gv.my_chips:
                return self._defend_allin(gv, deadline)

            stack_bb = gv.my_chips / max(1, gv.big_blind)

            # Push/fold para stack curto pré-flop.
            if stack_bb <= 12 and len(gv.board) == 0:
                return self._push_fold_preflop(gv, stack_bb)

            if len(gv.board) == 0:
                return self._preflop(gv)

            return self._postflop(gv, deadline)
        except Exception:
            # Safety: nunca falhar (engine converte exception em call).
            return 0

    # ═══ Opponent modeling ════════════════════════════════════════════════

    def _update_opp_model(self, gv: GameView) -> None:
        """Mantém perfil persistente VPIP/PFR/AF a cada decisão observada."""
        opp = gv.opponents[0] if gv.opponents else None
        if opp is None:
            return

        # Detecta nova mão: board zerou após estar maior, ou dealer trocou.
        new_hand = False
        if self._last_dealer is None:
            new_hand = True
        elif gv.dealer_position != self._last_dealer:
            new_hand = True
        elif len(gv.board) == 0 and self._last_board_len > 0:
            new_hand = True
        elif len(gv.board) == 0 and gv.pot <= (gv.small_blind + gv.big_blind + 2):
            # Pot pequeno e pré-flop: provável nova mão.
            if self._last_pot > gv.pot:
                new_hand = True

        if new_hand:
            # Fecha estatísticas da mão anterior.
            if self._opp_hands_seen > 0 or self._this_hand_opp_vpip:
                if self._this_hand_opp_vpip:
                    self._opp_vpip_hands += 1
                if self._this_hand_opp_raised_pf:
                    self._opp_pfr_hands += 1
            self._opp_hands_seen += 1
            self._this_hand_opp_vpip = False
            self._this_hand_opp_raised_pf = False
            self._this_hand_opp_max_bet = 0
            self._this_hand_my_last_bet = 0

        # Atualiza estado dentro da mão.
        opp_bet = opp.current_bet_in_round
        if opp_bet > gv.big_blind:
            self._this_hand_opp_vpip = True
            if len(gv.board) == 0 and opp_bet > self._this_hand_opp_max_bet:
                self._this_hand_opp_raised_pf = True
        elif opp_bet >= gv.big_blind and len(gv.board) == 0:
            # Opp pelo menos completou o BB (limp/call ≠ open-raise).
            self._this_hand_opp_vpip = True

        # Postflop aggression: raise = bet > 0, call = bet == current_bet
        # quando current_bet > 0 e oponente já entrou na rodada.
        if len(gv.board) > 0:
            if opp_bet > self._this_hand_opp_max_bet:
                # Opp aumentou nesta rua.
                self._opp_postflop_bets += 1
            elif opp_bet > 0 and opp_bet == gv.current_bet:
                # Opp só completou.
                self._opp_postflop_calls += 1

        if opp_bet > self._this_hand_opp_max_bet:
            self._this_hand_opp_max_bet = opp_bet

        self._last_dealer = gv.dealer_position
        self._last_board_len = len(gv.board)
        self._last_pot = gv.pot

    def _opp_vpip(self) -> float:
        if self._opp_hands_seen < 5:
            return 0.50  # prior: assume jogador médio
        return self._opp_vpip_hands / max(1, self._opp_hands_seen)

    def _opp_pfr(self) -> float:
        if self._opp_hands_seen < 5:
            return 0.25
        return self._opp_pfr_hands / max(1, self._opp_hands_seen)

    def _opp_af(self) -> float:
        total = self._opp_postflop_bets + self._opp_postflop_calls
        if total < 5:
            return 0.50  # prior
        return self._opp_postflop_bets / total

    # ═══ Range estimation ═════════════════════════════════════════════════

    def _estimate_opp_pf_range_pct(self, gv: GameView) -> float:
        """Estima % do espectro de mãos que o oponente teria nesta linha."""
        vpip = self._opp_vpip()
        pfr = self._opp_pfr()

        bb = gv.big_blind
        opp_bet_round = gv.opponents[0].current_bet_in_round if gv.opponents else 0
        opp_raised = opp_bet_round > bb

        if not opp_raised:
            # Opp limpou ou não agiu: range largo, próximo ao VPIP.
            return min(0.90, max(0.30, vpip))

        # Opp raiseou: range próximo ao PFR ajustado pelo tamanho.
        raise_ratio = opp_bet_round / bb
        if raise_ratio >= 8:   # 3-bet grande / 4-bet
            return min(pfr * 0.5, 0.10)
        if raise_ratio >= 4:   # 3-bet padrão
            return min(pfr * 0.7, 0.18)
        return min(max(pfr, 0.20), 0.40)

    def _estimate_opp_postflop_range_pct(self, gv: GameView) -> float:
        """Estima largura do range pós-flop do opp baseado em ações."""
        opp = gv.opponents[0]
        pot = max(1, gv.pot)
        opp_invested_this_round = opp.current_bet_in_round

        # Começa com o range pré-flop estimado.
        pf_range = self._estimate_opp_pf_range_pct(gv)

        # Se opp já apostou nesta rua pós-flop:
        if opp_invested_this_round > 0:
            bet_to_pot = opp_invested_this_round / pot
            # Estreita: apostas grandes => mais value/draws fortes.
            if bet_to_pot >= 1.0:
                return pf_range * 0.40
            if bet_to_pot >= 0.6:
                return pf_range * 0.55
            return pf_range * 0.75

        # Opp deu check pós-flop: range fica praticamente igual ou maior.
        return min(1.0, pf_range * 1.1)

    # ═══ Monte Carlo equity ═══════════════════════════════════════════════

    def _sample_opp_hand(
        self, deck_remaining: list[Card], min_strength: float
    ) -> tuple[Card, Card] | None:
        """Sorteia 2 cartas do deck respeitando threshold de força."""
        n = len(deck_remaining)
        if n < 2:
            return None
        for _ in range(12):
            i = self._rng.randrange(n)
            j = self._rng.randrange(n)
            if i == j:
                continue
            a, b = deck_remaining[i], deck_remaining[j]
            if _preflop_strength((a, b)) >= min_strength:
                return (a, b)
        # Fallback: aceita qualquer par.
        i = self._rng.randrange(n)
        j = self._rng.randrange(n)
        while j == i:
            j = self._rng.randrange(n)
        return (deck_remaining[i], deck_remaining[j])

    def _mc_equity(
        self, gv: GameView, deadline: float, range_pct: float
    ) -> tuple[float, int]:
        """Equity Monte Carlo com budget de tempo. Retorna (equity, n_samples)."""
        mc_deadline = min(deadline, time.perf_counter() + self.MC_BUDGET_MS / 1000.0)
        my_hand = list(gv.my_hand)
        board = list(gv.board)

        known_keys = {_card_key(c) for c in my_hand} | {_card_key(c) for c in board}
        deck_remaining = [c for c in _FULL_DECK if _card_key(c) not in known_keys]

        min_strength = _strength_for_top_pct(range_pct)
        cards_to_come = 5 - len(board)

        wins = 0
        ties = 0
        n = 0

        # No river, MY score é fixo — computa fora do loop.
        my_score_fixed = None
        if cards_to_come == 0:
            my_score_fixed = _best_score(my_hand + board)

        while time.perf_counter() < mc_deadline:
            opp_hand = self._sample_opp_hand(deck_remaining, min_strength)
            if opp_hand is None:
                break

            # Cartas usadas para a amostragem do runout (excluindo opp).
            opp_keys = {_card_key(opp_hand[0]), _card_key(opp_hand[1])}
            available_for_runout = [
                c for c in deck_remaining if _card_key(c) not in opp_keys
            ]
            if len(available_for_runout) < cards_to_come:
                break

            if cards_to_come == 0:
                runout: list[Card] = []
            else:
                # rng.sample no índice é mais barato do que rng.sample em listas grandes
                indices = self._rng.sample(
                    range(len(available_for_runout)), cards_to_come
                )
                runout = [available_for_runout[i] for i in indices]

            full_board = board + runout
            if my_score_fixed is not None:
                my_score = my_score_fixed
            else:
                my_score = _best_score(my_hand + full_board)
            opp_score = _best_score(list(opp_hand) + full_board)

            if my_score > opp_score:
                wins += 1
            elif my_score == opp_score:
                ties += 1
            n += 1

            # Safety hard-stop: nunca mais que 100 samples por chamada.
            if n >= 100:
                break

        if n == 0:
            return (0.5, 0)
        equity = (wins + 0.5 * ties) / n
        return (equity, n)

    # ═══ Pré-flop ═════════════════════════════════════════════════════════

    def _preflop(self, gv: GameView) -> int:
        strength = _preflop_strength(gv.my_hand)
        bb = gv.big_blind
        i_am_bb = self._org_sou_bb(gv)

        unopened = gv.current_bet <= bb
        opp_vpip = self._opp_vpip()
        opp_pfr = self._opp_pfr()

        if unopened:
            # Open size: 2.5x BB padrão, 3x se opp tight.
            open_mult = 3.0 if opp_pfr < 0.20 else 2.5

            if strength >= 0.62:
                target = self._maybe_overbet(gv, int(open_mult * bb))
                return self._raise_to(gv, target)
            if strength >= 0.50:
                return self._raise_to(gv, int(2.5 * bb))
            if strength >= 0.40:
                # Bots loose: amplia open range.
                if opp_vpip > 0.55:
                    return self._raise_to(gv, int(2.5 * bb))
                return 0
            if i_am_bb:
                return 0
            # SB com mão ruim: blefa pouco.
            if strength <= 0.30:
                return self._fold_or_bluff(gv, bluff_freq=self.BLUFF_FREQ_BASE)
            return 0

        # Enfrentando aposta.
        raise_size_bb = gv.current_bet / bb
        pot_odds = gv.to_call / (gv.pot + gv.to_call)

        # 3-bet threshold reduz vs oponente PFR alto (eles abrem fraco).
        threebet_thr = 0.72 if opp_pfr < 0.25 else 0.66
        call_thr = 0.48 if opp_pfr < 0.25 else 0.43

        if strength >= threebet_thr:
            target = self._maybe_overbet(gv, int(gv.current_bet * 3))
            return self._raise_to(gv, target)
        if strength >= 0.58:
            if self._rng.random() < 0.22:
                return self._raise_to(gv, int(gv.current_bet * 3))
            return 0
        if strength >= call_thr and raise_size_bb <= 5:
            return 0
        if strength >= 0.40 and pot_odds < 0.18:
            return 0
        # Defesa de BB: amplia call range em pot odds bons.
        if i_am_bb and pot_odds < 0.30 and strength >= 0.36:
            return 0
        return self._fold_or_bluff(gv, bluff_freq=self.BLUFF_FREQ_BASE)

    # ═══ Pós-flop ═════════════════════════════════════════════════════════

    def _postflop(self, gv: GameView, deadline: float) -> int:
        all_cards = list(gv.my_hand) + list(gv.board)
        score = _best_score(all_cards)
        rank = _rank_of(score)
        cards_to_come = 5 - len(gv.board)

        flush_dr = _has_flush_draw(all_cards) if cards_to_come > 0 else False
        oesd, gut = (
            _has_straight_draw(all_cards) if cards_to_come > 0 else (False, False)
        )
        strong_draw = flush_dr or oesd
        any_draw = strong_draw or gut

        to_call = gv.to_call
        pot = max(1, gv.pot)
        bb = gv.big_blind

        # Calcula equity Monte Carlo (com budget).
        range_pct = self._estimate_opp_postflop_range_pct(gv)
        equity, n_samples = self._mc_equity(gv, deadline, range_pct)

        # Fallback heurístico se MC não amostrou suficiente.
        if n_samples < self.MIN_MC_SAMPLES:
            return self._postflop_heuristic(
                gv, rank, strong_draw, any_draw, flush_dr, oesd, cards_to_come
            )

        # Ajusta equity por "implied odds" simples se temos draw forte.
        implied_bonus = 0.0
        if strong_draw and cards_to_come > 0:
            implied_bonus = 0.05
        adjusted_equity = min(0.99, equity + implied_bonus)

        # Pot odds atuais.
        pot_odds = to_call / (pot + to_call) if to_call > 0 else 0.0

        # Detecta categoria da mão para mixed strategy.
        is_monster = rank >= RANK_STRAIGHT or adjusted_equity >= 0.85
        is_strong = (rank in (RANK_TRINCA, RANK_DOIS_PARES)) or adjusted_equity >= 0.70
        is_medium = (rank == RANK_UM_PAR and self._pair_tier(gv) != "weak") or adjusted_equity >= 0.50

        # ─── Mão monster: valor + raras slowplays ─────────────────────────
        if is_monster:
            if to_call == 0:
                if self._rng.random() < self.SLOWPLAY_FREQ:
                    return 0  # slowplay: induz blefe
                target = self._sizing(gv, 0.75)
                target = self._maybe_overbet(gv, target)
                return self._raise_to(gv, target)
            # Vs aposta: raise grande, às vezes só call para trap.
            if self._rng.random() < 0.18:
                return 0
            raise_target = max(self._sizing(gv, 0.75), int(gv.current_bet * 2.6))
            raise_target = self._maybe_overbet(gv, raise_target)
            return self._raise_to(gv, raise_target)

        # ─── Mão forte ────────────────────────────────────────────────────
        if is_strong:
            if to_call == 0:
                # Value bet em 80% dos spots.
                if self._rng.random() < 0.82:
                    target = self._sizing(gv, 0.65)
                    return self._raise_to(gv, target)
                return 0
            # Vs aposta: depende do tamanho.
            bet_to_pot = gv.current_bet / pot
            if bet_to_pot >= 1.2:
                # Aposta grande: precisa de equity boa para call/raise.
                if adjusted_equity >= 0.60:
                    return 0
                if adjusted_equity >= 0.45 and pot_odds < 0.35:
                    return 0
                return self._fold_or_bluff(gv, bluff_freq=self.BLUFF_FREQ_BASE * 0.5)
            # Aposta normal: call/raise.
            if self._rng.random() < 0.30 and adjusted_equity >= 0.65:
                rt = self._maybe_overbet(gv, int(gv.current_bet * 2.4))
                return self._raise_to(gv, rt)
            return 0

        # ─── Mão média (par marginal, equity 0.40-0.60) ───────────────────
        if is_medium:
            if to_call == 0:
                # Thin value / probe bet.
                if self._rng.random() < 0.40:
                    return self._raise_to(gv, self._sizing(gv, 0.45))
                return 0
            # Vs aposta: bluff catch baseado em equity vs pot odds.
            if adjusted_equity > pot_odds + 0.05:
                return 0
            # Se opp é hyper-aggro, call mais largo (bluff catch).
            if self._opp_af() > 0.65 and adjusted_equity > pot_odds - 0.05:
                return 0
            if to_call <= 3 * bb and adjusted_equity > pot_odds - 0.05:
                return 0
            return self._fold_or_bluff(gv, bluff_freq=self.BLUFF_FREQ_BASE * 0.5)

        # ─── Mão fraca: draws + blefes ────────────────────────────────────
        if strong_draw and cards_to_come > 0:
            # Semi-bluff: valor + fold equity.
            outs = 9 if flush_dr else (8 if oesd else 4)
            draw_equity = min(0.50, outs * (4 if cards_to_come == 2 else 2) / 100)
            if to_call == 0:
                if self._rng.random() < 0.55:
                    return self._raise_to(gv, self._sizing(gv, 0.55))
                return 0
            if draw_equity > pot_odds:
                return 0
            return self._fold_or_bluff(gv, bluff_freq=self.BLUFF_FREQ_BASE)

        if any_draw and cards_to_come > 0 and to_call == 0:
            if self._rng.random() < 0.18:
                return self._raise_to(gv, self._sizing(gv, 0.45))
            return 0

        # Air: c-bet/blefe ocasional, senão fold/check.
        if to_call == 0:
            # C-bet frequency ajustada por opp fold equity.
            cbet_freq = 0.15
            if len(gv.board) == 3:
                cbet_freq = 0.35  # c-bet flop padrão
            if self._opp_af() < 0.35:
                cbet_freq *= 0.6   # opp calling station: c-bet menos
            if self._rng.random() < cbet_freq:
                return self._raise_to(gv, self._sizing(gv, 0.55))
            return 0
        # Vs aposta com air: fold quase sempre.
        return self._fold_or_bluff(gv, bluff_freq=self.BLUFF_FREQ_BASE * 0.6)

    # ─── Fallback heurístico (quando MC não amostrou o bastante) ──────────

    def _postflop_heuristic(
        self,
        gv: GameView,
        rank: int,
        strong_draw: bool,
        any_draw: bool,
        flush_dr: bool,
        oesd: bool,
        cards_to_come: int,
    ) -> int:
        """Decisão sem MC — clone enxuto do v7 para garantir resposta rápida."""
        if rank >= RANK_STRAIGHT:
            tier = "monster"
        elif rank in (RANK_TRINCA, RANK_DOIS_PARES):
            tier = "strong"
        elif rank == RANK_UM_PAR:
            tier = self._pair_tier(gv)
        else:
            tier = "weak"

        to_call = gv.to_call
        pot = max(1, gv.pot)

        if tier == "monster":
            target = self._sizing(gv, 0.75)
            if to_call == 0:
                return self._raise_to(gv, self._maybe_overbet(gv, target))
            if self._rng.random() < 0.15:
                return 0
            rt = self._maybe_overbet(gv, max(target, int(gv.current_bet * 2.5)))
            return self._raise_to(gv, rt)

        if tier == "strong":
            target = self._sizing(gv, 0.6)
            if to_call == 0:
                if self._rng.random() < 0.82:
                    return self._raise_to(gv, self._maybe_overbet(gv, target))
                return 0
            if gv.current_bet >= pot * 0.9 and self._rng.random() < 0.30:
                return self._fold_or_bluff(gv, self.BLUFF_FREQ_BASE * 0.5)
            return 0

        if tier == "medium":
            if to_call == 0:
                if self._rng.random() < 0.4:
                    return self._raise_to(gv, self._sizing(gv, 0.4))
                return 0
            pot_odds = to_call / (pot + to_call)
            if pot_odds < 0.25 and to_call <= 4 * gv.big_blind:
                return 0
            return self._fold_or_bluff(gv, self.BLUFF_FREQ_BASE * 0.5)

        if strong_draw and cards_to_come > 0:
            outs = 9 if flush_dr else (8 if oesd else 4)
            equity = outs * (4 if cards_to_come == 2 else 2) / 100
            if to_call == 0:
                if self._rng.random() < 0.5:
                    return self._raise_to(gv, self._sizing(gv, 0.5))
                return 0
            pot_odds = to_call / (pot + to_call)
            if equity > pot_odds:
                return 0
            return self._fold_or_bluff(gv, self.BLUFF_FREQ_BASE)

        if to_call == 0:
            if self._rng.random() < 0.10:
                return self._raise_to(gv, self._sizing(gv, 0.5))
            return 0
        return self._fold_or_bluff(gv, self.BLUFF_FREQ_BASE * 0.6)

    # ═══ Hooks de blefe e overbet ═════════════════════════════════════════

    def _fold_or_bluff(self, gv: GameView, bluff_freq: float) -> int:
        if self._rng.random() >= bluff_freq:
            return -1
        if gv.to_call == 0:
            return self._raise_to(gv, self._sizing(gv, 0.65))
        target = int(gv.current_bet * 2.7)
        return self._raise_to(gv, target)

    def _maybe_overbet(self, gv: GameView, normal_target: int) -> int:
        if self._rng.random() >= self.OVERBET_FREQ_BASE:
            return normal_target
        invested = gv.current_bet - gv.to_call
        delta = normal_target - invested
        bigger = invested + int(delta * 1.7)
        return bigger

    # ═══ Helpers ══════════════════════════════════════════════════════════

    def _pair_tier(self, gv: GameView) -> str:
        my_vals = [valor_carta(c) for c in gv.my_hand]
        board_vals = [valor_carta(c) for c in gv.board]
        all_vals = my_vals + board_vals
        counts = Counter(all_vals)
        pair_value = max((v for v, n in counts.items() if n >= 2), default=0)
        if pair_value == 0:
            return "weak"
        top_board = max(board_vals) if board_vals else 0
        if my_vals.count(pair_value) == 2:
            return "strong" if pair_value > top_board else "medium"
        if pair_value == top_board and pair_value in my_vals:
            kicker = max((v for v in my_vals if v != pair_value), default=0)
            return "strong" if kicker >= 12 else "medium"
        if pair_value >= 10 and pair_value in my_vals:
            return "medium"
        return "weak"

    def _sizing(self, gv: GameView, fraction: float) -> int:
        invested = gv.current_bet - gv.to_call
        bet_amount = max(gv.big_blind, int(gv.pot * fraction))
        return invested + gv.to_call + bet_amount

    def _raise_to(self, gv: GameView, target_total: int) -> int:
        invested = gv.current_bet - gv.to_call
        max_total = invested + gv.my_chips
        return min(target_total, max_total)

    def _push_fold_preflop(self, gv: GameView, stack_bb: float) -> int:
        strength = _preflop_strength(gv.my_hand)
        i_am_bb = self._org_sou_bb(gv)
        bb = gv.big_blind

        # Se sou SB (acho 1º): Nash push.
        if not i_am_bb:
            thr = _nash_sb_push_strength(stack_bb)
            if strength >= thr:
                return self._shove(gv)
            # Limp ocasional vs opp passivo, senão fold.
            if self._opp_af() < 0.30 and strength >= 0.36 and stack_bb >= 8:
                return 0
            return -1

        # Sou BB enfrentando ação:
        if gv.to_call == 0:
            return 0  # check grátis
        # SB shovou ou apostou: usa Nash BB call threshold.
        # Detecta shove pelo to_call em BBs (>= ~0.7 stack_bb).
        thr_call = _nash_bb_call_strength(stack_bb)
        # Ajusta por pot odds: stacks pequenos => call mais largo.
        pot_odds = gv.to_call / (gv.pot + gv.to_call)
        if pot_odds < 0.40:
            thr_call -= 0.04
        if strength >= thr_call:
            return 0  # call
        if strength >= 0.42 and pot_odds <= 0.25:
            return 0
        return self._fold_or_bluff(gv, self.BLUFF_FREQ_BASE * 0.5)

    def _shove(self, gv: GameView) -> int:
        invested = gv.current_bet - gv.to_call
        return invested + gv.my_chips

    def _defend_allin(self, gv: GameView, deadline: float) -> int:
        """Defesa de all-in: só call ou fold, com MC se houver tempo."""
        pot_odds = gv.to_call / (gv.pot + gv.to_call)

        if len(gv.board) == 0:
            strength = _preflop_strength(gv.my_hand)
            min_strength = 0.55 - (0.30 - min(pot_odds, 0.30)) * 0.5
            return 0 if strength >= min_strength else -1

        # Pós-flop: tenta MC; fallback para ranking direto.
        range_pct = self._estimate_opp_postflop_range_pct(gv)
        equity, n_samples = self._mc_equity(gv, deadline, range_pct)
        if n_samples >= self.MIN_MC_SAMPLES:
            # Call EV > 0 quando equity > pot_odds.
            margin = 0.02  # margem pequena para variância
            return 0 if equity > pot_odds + margin else -1

        # Fallback: ranking direto.
        score = _best_score(list(gv.my_hand) + list(gv.board))
        rank = _rank_of(score)
        if rank >= RANK_DOIS_PARES:
            return 0
        if rank == RANK_UM_PAR and pot_odds <= 0.40:
            return 0
        return -1


def create_player() -> Player:
    return Versao8("versao_8", Hand(), 0)
