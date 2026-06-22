from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from players.player import Player
from game.game_view import GameView
from cards.cards import Hand

# ── bibliotecas permitidas ──────────────────────────────────────────────────
import random
from collections import Counter
# ───────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

VALOR = {
    "2": 2,  "3": 3,  "4": 4,  "5": 5,  "6": 6,
    "7": 7,  "8": 8,  "9": 9,  "10": 10,
    "J": 11, "Q": 12, "K": 13, "A": 14,
}


# ---------------------------------------------------------------------------
# Avaliação pré-flop — Chen Score normalizado
# ---------------------------------------------------------------------------

def _chen(v1, v2, suited):
    high, low, gap = max(v1, v2), min(v1, v2), abs(v1 - v2)
    if high == 14:   pts = 10.0
    elif high == 13: pts = 8.0
    elif high == 12: pts = 7.0
    elif high == 11: pts = 6.0
    else:            pts = high / 2.0
    if v1 == v2:
        pts = max(pts * 2, 5)
    else:
        if suited:      pts += 2
        if gap == 1:    pts -= 1
        elif gap == 2:  pts -= 2
        elif gap == 3:  pts -= 4
        else:           pts -= 5
        if gap <= 1 and low >= 2:
            pts += 1
    return pts


def _equity_preflop(hand):
    v1, v2 = VALOR[hand[0].value], VALOR[hand[1].value]
    suited = hand[0].suit == hand[1].suit
    norm   = (_chen(v1, v2, suited) + 2) / 22.0
    return max(0.25, min(0.90, 0.30 + norm * 0.65))


# ---------------------------------------------------------------------------
# Avaliação de 7 cartas — SEM iterar combinações (rápido)
# ---------------------------------------------------------------------------

def _eval7(raw):
    """
    Avalia a melhor mão de 5 entre 7 cartas diretamente.
    raw: lista de (val_int: int, suit: str)
    Retorna tupla comparável (categoria, [desempate]).
    8=str-flush 7=quadra 6=full 5=flush 4=straight 3=trio 2=2par 1=par 0=alta
    """
    vals   = [v for v, s in raw]
    suits  = [s for v, s in raw]
    counts = Counter(vals)
    freq   = sorted(counts.values(), reverse=True)

    # ── straight flush / flush ───────────────────────────────────────────
    suit_cnt   = Counter(suits)
    flush_suit = next((s for s, c in suit_cnt.items() if c >= 5), None)
    if flush_suit:
        # Pega todas as cartas do naipe, sem truncar prematuramente
        all_fv = sorted([v for v, s in raw if s == flush_suit], reverse=True)
        uv_flush = sorted(set(all_fv), reverse=True)
        
        best_sf = None
        ext_f = uv_flush + ([1] if 14 in uv_flush else [])
        for i in range(len(ext_f) - 4):
            w = ext_f[i:i + 5]
            if w[0] - w[4] == 4 and len(set(w)) == 5:
                best_sf = w
                break
                
        if best_sf is None and {14, 2, 3, 4, 5}.issubset(set(all_fv)):
            best_sf = [5, 4, 3, 2, 1]
            
        if best_sf:
            return (8, best_sf)
        
        # Se não for straight flush, retorna o flush com as 5 maiores cartas
        return (5, all_fv[:5])

    # ── quadra ───────────────────────────────────────────────────────────
    if freq[0] == 4:
        q = max(v for v, c in counts.items() if c == 4)
        k = max(v for v, c in counts.items() if v != q)
        return (7, [q, k])

    # ── full house ───────────────────────────────────────────────────────
    trios = sorted([v for v, c in counts.items() if c >= 3], reverse=True)
    pairs = sorted([v for v, c in counts.items() if c >= 2 and v not in trios],
                   reverse=True)
    if trios:
        if len(trios) >= 2: return (6, [trios[0], trios[1]])
        if pairs:           return (6, [trios[0], pairs[0]])

    # ── straight ─────────────────────────────────────────────────────────
    uv         = sorted(set(vals), reverse=True)
    best_strt  = None
    ext        = uv + ([1] if 14 in uv else [])
    for i in range(len(ext) - 4):
        w = ext[i:i + 5]
        if w[0] - w[4] == 4 and len(set(w)) == 5:
            best_strt = w
            break
    if best_strt is None and {14, 2, 3, 4, 5}.issubset(set(vals)):
        best_strt = [5, 4, 3, 2, 1]

    # ── trio ─────────────────────────────────────────────────────────────
    if trios:
        kk = sorted([v for v in vals if v != trios[0]], reverse=True)[:2]
        trio_score = (3, [trios[0]] + kk)
        return max((4, best_strt), trio_score) if best_strt else trio_score

    if best_strt: return (4, best_strt)

    # ── dois pares ───────────────────────────────────────────────────────
    all_pairs = sorted([v for v, c in counts.items() if c >= 2], reverse=True)
    if len(all_pairs) >= 2:
        k = max(v for v in vals if v != all_pairs[0] and v != all_pairs[1])
        return (2, [all_pairs[0], all_pairs[1], k])

    # ── par ──────────────────────────────────────────────────────────────
    if all_pairs:
        kk = sorted([v for v in vals if v != all_pairs[0]], reverse=True)[:3]
        return (1, [all_pairs[0]] + kk)

    # ── carta alta ───────────────────────────────────────────────────────
    return (0, sorted(vals, reverse=True)[:5])


# ---------------------------------------------------------------------------
# Monte Carlo pós-flop  (usa _eval7 — sem combinations)
# ---------------------------------------------------------------------------

def _deck_livre(conhecidas):
    return [
        (VALOR[v], s)
        for v in VALOR for s in ("s", "h", "d", "c")
        if (VALOR[v], s) not in conhecidas
    ]


def _equity(hand, board):
    """Equidade estimada [0, 1]."""
    if not board:
        return _equity_preflop(hand)

    conhecidas = set()
    my_raw, board_raw = [], []
    for c in hand:
        t = (VALOR[c.value], c.suit); conhecidas.add(t); my_raw.append(t)
    for c in board:
        t = (VALOR[c.value], c.suit); conhecidas.add(t); board_raw.append(t)

    faltam = 5 - len(board)
    pool   = _deck_livre(conhecidas)

    # Margem de segurança máxima sem timer: 120 sims garantem que não bate 50ms
    nsims = 120

    wins = ties = total = 0
    for _ in range(nsims):
        s      = random.sample(pool, 2 + faltam)
        opp    = s[:2]
        runout = s[2:]
        full   = board_raw + runout

        me_score  = _eval7(my_raw + full)
        opp_score = _eval7(list(opp) + full)

        if me_score > opp_score:    wins += 1
        elif me_score == opp_score: ties += 1
        total += 1

    return (wins + ties * 0.5) / total if total else 0.5


# ---------------------------------------------------------------------------
# Detector de draws
# ---------------------------------------------------------------------------

def _tem_draw(hand, board):
    if len(board) < 3:
        return False
    cards = list(hand) + list(board)
    suits = [c.suit for c in cards]
    vals  = sorted({VALOR[c.value] for c in cards})

    if max(Counter(suits).values()) >= 4:  # flush draw
        return True
    for i in range(len(vals) - 3):         # straight draw
        if vals[i + 3] - vals[i] <= 4:
            return True
    return False


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class BotITAJr(Player):
    """
    Bot para o Torneio ITA Jr — Texas Hold'em heads-up.

    Estratégia:
    • Equidade por Chen Score (pré-flop) ou Monte Carlo com _eval7 (pós-flop)
      _eval7 avalia 7 cartas diretamente. 120 sims rodam muito abaixo do limite.
    • Call/fold baseado em equidade vs. pot odds (matematicamente correto)
    • Raise de valor proporcional ao pot e à força da mão
    • Push/fold com stack < 8 BBs (sobrevive aos blinds crescentes)
    • Bluffs e semi-bluffs calibrados com random (anti-exploração)
    • Memória do oponente: ajusta limiares conforme agressividade observada
    """

    def __init__(self, name, hand, chips):
        super().__init__(name, hand, chips)
        self.opp_raises   = 0
        self.maos         = 0
        self._opp_bet_ant = 0

    # ── ponto de entrada ──────────────────────────────────────────────────

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
        gv       = game_view
        bb       = gv.big_blind
        my_chips = gv.my_chips
        pot      = gv.pot
        to_call  = gv.to_call
        board    = gv.board
        opp      = gv.opponents[0]
        sou_bb   = self._org_sou_bb(gv)

        # atualiza modelo do oponente
        if opp.current_bet_in_round < self._opp_bet_ant:
            self.maos += 1
        if opp.current_bet_in_round > self._opp_bet_ant:
            self.opp_raises += 1
        self._opp_bet_ant = opp.current_bet_in_round

        # regra absoluta: nunca fold em check grátis
        if to_call == 0:
            return self._acao_gratis(gv, sou_bb, board, bb, pot, my_chips)

        # push/fold com stack curto
        if my_chips < bb * 8:
            return self._short_stack(gv, bb, my_chips, to_call)

        # equidade + ajuste pelo oponente
        eq   = _equity(gv.my_hand, board)
        aggr = self._aggr()
        if aggr > 1.5:   eq = max(0.0, eq - 0.04)
        elif aggr < 0.5: eq = min(1.0, eq + 0.03)

        pot_odds = to_call / max(pot + to_call, 1)
        return self._decisao(gv, eq, pot_odds, to_call, pot, bb, my_chips, sou_bb, board)

    # ── ação grátis (to_call == 0) ────────────────────────────────────────

    def _acao_gratis(self, gv, sou_bb, board, bb, pot, my_chips):
        eq = _equity(gv.my_hand, board)

        if eq > 0.75:
            return self._bet_valor(pot, bb, my_chips, eq)
        if eq > 0.58:
            if random.random() < 0.70:
                return self._bet_valor(pot, bb, my_chips, eq)
            return 0
        if eq < 0.40 and len(board) >= 3:
            freq = 0.22 if sou_bb else 0.13
            if self._aggr() < 0.5:
                freq = min(freq * 1.4, 0.35)
            if random.random() < freq:
                return self._bet_bluff(pot, bb, my_chips)
        return 0

    # ── decisão principal ─────────────────────────────────────────────────

    def _decisao(self, gv, eq, pot_odds, to_call, pot, bb, my_chips, sou_bb, board):

        if eq > 0.75:
            return self._raise_valor(gv, pot, bb, my_chips, eq)
        if eq > 0.60:
            if random.random() < 0.55:
                return self._raise_valor(gv, pot, bb, my_chips, eq)
            return 0
        if eq > 0.45:
            if eq >= pot_odds + 0.03: return 0
            if to_call <= bb * 2:     return 0
            return -1
        if eq >= pot_odds + 0.02:
            if _tem_draw(gv.my_hand, board) and random.random() < 0.35:
                return self._semi_bluff(gv, bb, my_chips)
            return 0

        # 3-bet bluff pré-flop ocasional
        if (not board and eq < 0.32
                and random.random() < 0.14
                and my_chips > bb * 18):
            alvo = gv.current_bet * 3
            if alvo < my_chips:
                return alvo

        return -1

    # ── push / fold ───────────────────────────────────────────────────────

    def _short_stack(self, gv, bb, my_chips, to_call):
        v1, v2 = VALOR[gv.my_hand[0].value], VALOR[gv.my_hand[1].value]
        suited = gv.my_hand[0].suit == gv.my_hand[1].suit
        score  = _chen(v1, v2, suited)
        high   = max(v1, v2)

        if my_chips < bb * 5:
            if score >= 4 or high >= 9: return my_chips
            if to_call == 0:            return 0
            if to_call <= bb:           return 0
            return -1

        if score >= 7 or (v1 == v2 and v1 >= 6) or high >= 11:
            return my_chips
        if to_call == 0:        return 0
        if to_call <= bb * 1.5: return 0
        return -1

    # ── sizing ────────────────────────────────────────────────────────────

    def _bet_valor(self, pot, bb, my_chips, eq):
        frac = min(0.50 + (eq - 0.75) * 1.2, 0.80)
        return min(max(int(pot * frac), bb), my_chips)

    def _raise_valor(self, gv, pot, bb, my_chips, eq):
        cb     = gv.current_bet
        by_bet = int(cb * (2.5 + eq * 1.5))
        by_pot = int((pot + gv.to_call) * 0.65)
        return min(max(by_bet, by_pot, cb + bb * 2), my_chips)

    def _bet_bluff(self, pot, bb, my_chips):
        return min(max(int(pot * 0.55), bb * 2), my_chips)

    def _semi_bluff(self, gv, bb, my_chips):
        return min(max(int(gv.current_bet * 2.5), gv.current_bet + bb * 2), my_chips)

    # ── modelo do oponente ────────────────────────────────────────────────

    def _aggr(self):
        if self.maos < 5:
            return 1.0
        return (self.opp_raises / max(self.maos, 1)) / 1.5


def create_player() -> Player:
    return BotITAJr("BotITAJr", Hand(), 0)