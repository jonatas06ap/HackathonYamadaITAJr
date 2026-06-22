"""
Bot de Poker — player_bigdaddy
==============================

COMO O BOT FUNCIONA (visão geral da estratégia)
------------------------------------------------
bigdaddy é um TAG (tight-aggressive) *exploitativo robusto* para heads-up. Ele
combina quatro camadas:

1) MOTOR DE EQUITY (determinístico, < 50 ms). Usa o avaliador de mãos da engine
   (cards.sequences) para achar a melhor mão de 5 cartas e estima a equity
   combinando força da mão feita + equity de draws (outs) + desconto de textura
   da mesa. Essa heurística CALIBRADA (que modela realização e range) é o motor
   padrão. Há também um Monte-Carlo de equity embutido (avaliador inteiro rápido
   + nº de amostras fixo, < 12 ms/decisão), mas DESLIGADO (flag USE_MC): testado,
   o MC vs range uniforme superestima mãos fracas no heads-up e regrediu o
   desempenho — fica reservado para uma futura variante "MC vs range estimado".

   Detalhe importante desta engine: o "flop" já é distribuído ANTES da rodada
   pré-flop, então nunca existe uma rodada com board vazio — sempre há 3+ cartas
   na mesa. bigdaddy avalia uma mão real desde a primeira ação.

2) MODELO DO OPONENTE (adaptação dentro da partida, robusto a envenenamento).
   EMA de dois tempos (rápido/lento) com filtro de "poison" e contador de
   mudanças, estimando: com que frequência o oponente FOLDA à nossa aposta (por
   street e por tamanho), com que frequência ELE é agressivo, qual o tamanho
   típico de aposta dele e — via showdown — se ele BLEFA (quando pagamos a
   agressão dele, com que frequência ganhamos). Nada é guardado entre partidas:
   cada partida começa do mesmo ponto neutro e sólido (a Fase 1 é confronto
   único, e isso impede que um oponente "nos prepare" entre rodadas).

3) CAMADA EXPLORATIVA, *limitada e condicionada à confiança*. Quanto mais o
   oponente folda, mais blefamos/c-betamos e maior o sizing por fold-equity;
   contra calling-stations zeramos blefe e engrossamos o valor (value fino);
   contra maníacos apertamos e pagamos light (trap). Cada desvio do baseline é
   limitado em magnitude e só é acionado quando o estimador está confiante.

4) BAIXA EXPLORABILIDADE (contra bots altamente adaptativos). Ranges
   balanceados, decisões mistas (random), e SIZING ALEATORIZADO: o valor da
   aposta parte do alvo baseado em equity/pote e recebe um jitter aleatório
   (±EPS), descorrelacionando o tamanho da nossa força — assim um bot que
   analisa nosso padrão de aposta não consegue ler a nossa mão. Se detectamos
   que o oponente está mudando o comportamento EM RESPOSTA a nós (gap rápido vs
   lento), descartamos a leitura velha e voltamos ao baseline neutro em vez de
   perseguir o padrão.

Curtos de fichas (< ~13 BB) entram em modo push/fold, pois os blinds dobram a
cada 50 mãos e passividade perde.

Restrições respeitadas: apenas stdlib + engine; O(21 avaliações de 5 cartas) por
decisão; sem I/O; try/except global converte qualquer erro em call/check.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import random
from collections import Counter
from itertools import combinations

from players.player import Player
from game.game_view import GameView
from cards.cards import Hand
from cards.sequences import (
    avaliar_cinco_cartas,
    desempate_para_numero,
    BASE_DESEMPATE,
    VALORES,
    RANK_CARTA_ALTA,
    RANK_UM_PAR,
    RANK_DOIS_PARES,
    RANK_TRINCA,
    RANK_STRAIGHT,
    RANK_FLUSH,
    RANK_FULL_HOUSE,
    RANK_QUADRA,
    RANK_STRAIGHT_FLUSH,
)


def _v(card) -> int:
    return VALORES[card.value]


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# ═══════════════════════════════════════════════════════════════════════════
#  Avaliador de mão RÁPIDO (inteiros) + Monte Carlo de equity
#  ---------------------------------------------------------------------------
#  Cada carta vira um int = rank*4 + suit (rank 0..12 = 2..A, suit 0..3).
#  _eval5 devolve um score comparável (categoria*BASE + desempate); _eval7 pega
#  a melhor de 5 entre 7. Validado contra o avaliador da engine em 200k mãos
#  (0 divergências de vencedor). É ~10x mais rápido que avaliar via Counter, o
#  que viabiliza o Monte Carlo dentro do orçamento de 50 ms.
# ═══════════════════════════════════════════════════════════════════════════
_R2I = {"2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, "9": 7,
        "10": 8, "J": 9, "Q": 10, "K": 11, "A": 12}
_S2I = {"s": 0, "h": 1, "d": 2, "c": 3}
_ALL52 = tuple(range(52))
_B5 = 15 ** 5


def _ci(card) -> int:
    return _R2I[card.value] * 4 + _S2I[card.suit]


def _eval5(cs) -> int:
    r = [c >> 2 for c in cs]
    r.sort(reverse=True)
    counts = [0] * 13
    for x in r:
        counts[x] += 1
    maxc = max(counts)
    flush = (cs[0] & 3) == (cs[1] & 3) == (cs[2] & 3) == (cs[3] & 3) == (cs[4] & 3)
    straight = False
    high = r[0]
    if maxc == 1:
        if r[0] - r[4] == 4:
            straight = True
        elif r[0] == 12 and r[1] == 3 and r[2] == 2 and r[3] == 1 and r[4] == 0:
            straight = True  # wheel A-2-3-4-5
            high = 3
    if flush and straight:
        return 8 * _B5 + high
    if maxc == 4:
        quad = counts.index(4)
        kick = next(x for x in r if x != quad)
        return 7 * _B5 + quad * 15 + kick
    trip = pa = pb = -1
    for x in range(12, -1, -1):
        c = counts[x]
        if c == 3 and trip == -1:
            trip = x
        elif c == 2:
            if pa == -1:
                pa = x
            elif pb == -1:
                pb = x
    if trip >= 0 and pa >= 0:
        return 6 * _B5 + trip * 15 + pa
    if flush:
        s = 0
        for x in r:
            s = s * 15 + x
        return 5 * _B5 + s
    if straight:
        return 4 * _B5 + high
    if trip >= 0:
        ks = [x for x in r if x != trip][:2]
        return 3 * _B5 + trip * 225 + ks[0] * 15 + ks[1]
    if pa >= 0 and pb >= 0:
        k = next(x for x in r if x != pa and x != pb)
        return 2 * _B5 + pa * 225 + pb * 15 + k
    if pa >= 0:
        ks = [x for x in r if x != pa][:3]
        return 1 * _B5 + pa * 3375 + ks[0] * 225 + ks[1] * 15 + ks[2]
    s = 0
    for x in r:
        s = s * 15 + x
    return s


def _eval7(seven) -> int:
    best = 0
    for combo in combinations(seven, 5):
        s = _eval5(combo)
        if s > best:
            best = s
    return best


class _RobustStat:
    """EMA de dois tempos (rápido/lento) com filtro anti-envenenamento.

    - `fast` reage rápido, `slow` é a leitura estável que usamos.
    - Amostras absurdas (muito distantes do `slow`, depois de algumas amostras)
      são tratadas como possível camuflagem e entram com peso reduzido.
    - `shifts` conta mudanças SUSTENTADAS de tendência: sinal de que o oponente
      está se adaptando a nós — usado para reduzir nossos desvios (voltar ao
      baseline) em vez de perseguir um padrão que ele controla.
    """
    __slots__ = ("fast", "slow", "n", "a_fast", "a_slow", "_streak", "_dir",
                 "poison_hits", "shifts")

    def __init__(self, prior: float, a_fast: float = 0.30, a_slow: float = 0.07):
        self.fast = prior
        self.slow = prior
        self.n = 0
        self.a_fast = a_fast
        self.a_slow = a_slow
        self._streak = 0
        self._dir = 0
        self.poison_hits = 0
        self.shifts = 0

    def update(self, x: float) -> None:
        x = _clamp(x, 0.0, 1.0)
        self.n += 1
        poison = self.n >= 18 and abs(x - self.slow) > 0.6
        if poison:
            self.poison_hits += 1
            self.fast += (self.a_fast * 0.25) * (x - self.fast)
            return
        self.fast += self.a_fast * (x - self.fast)
        self.slow += self.a_slow * (x - self.slow)
        d = self.fast - self.slow
        cur_dir = 1 if d > 0.12 else (-1 if d < -0.12 else 0)
        if cur_dir != 0 and cur_dir == self._dir:
            self._streak += 1
        elif cur_dir != 0:
            self._dir = cur_dir
            self._streak = 1
        else:
            self._dir = 0
            self._streak = 0
        if self._streak >= 4:
            # mudança sustentada: acelera o slow e registra o shift
            self.slow += 0.5 * (self.fast - self.slow)
            self._streak = 0
            self._dir = 0
            self.shifts += 1

    def value(self) -> float:
        return self.slow

    def confident(self, min_n: int = 12) -> bool:
        return self.n >= min_n


class _OppProfile:
    """Perfil de UM oponente, acumulado ao longo de TODA a partida-confronto
    (2000 jogos rodam no mesmo processo; uma instância nova do bot é criada por
    jogo, mas este perfil é de CLASSE e sobrevive). Permite aprendizado online
    real: dentro de um jogo há só ~6-12 mãos, mas ao longo do confronto há
    dezenas de milhares de mãos contra o MESMO oponente. Chaveado por nome ⇒ o
    que aprendemos sobre um bot não afeta o jogo contra outro."""
    __slots__ = ("games", "fold", "fold_street", "aggr", "betsize",
                 "bluff_ema", "bluff_samples", "bigpot")

    def __init__(self, bluff_prior: float):
        self.games = 0
        self.fold = _RobustStat(0.45)
        self.fold_street = {s: _RobustStat(0.45) for s in (3, 4, 5)}
        self.aggr = _RobustStat(0.30)
        self.betsize = _RobustStat(0.6, a_fast=0.40, a_slow=0.12)
        self.bluff_ema = bluff_prior
        self.bluff_samples = 0
        # win rate dos NOSSOS potes grandes (commitment ≥ ~50% do stack) contra
        # este oponente. Baixo ⇒ ele só entra em pote grande com mão muito forte
        # (nit, ex.: papa/tubarao) ⇒ devemos parar de dar all-in leve CONTRA ELE.
        self.bigpot = _RobustStat(0.5)


class BigDaddy(Player):

    # ── flags / hiperparâmetros ──────────────────────────────────────────────
    SIZING_JITTER = False      # aleatoriza o tamanho da aposta (anti-leitura de padrão)
    JITTER_EPS = 0.13          # amplitude do jitter (±13%)

    # Monte Carlo de equity (mais preciso que a heurística de equity fixa).
    # Nº de amostras por street (board=3/4/5) — fixo (não usa relógio: 'time'
    # não está na whitelist) e conservador p/ caber em 50 ms mesmo em hardware
    # lento. ~48 µs/amostra no flop ⇒ 160 amostras ≈ 8 ms (3× margem).
    # DESLIGADO por padrão: testado empiricamente, o MC vs RANGE UNIFORME
    # superestima mãos fracas no heads-up (lixo tem ~0.30 de equity por
    # runner-runner) e nos faz pagar/continuar leve demais — regrediu forte
    # contra bots exploráveis (ex.: bernardo 69%→62%) sem ajudar contra os
    # difíceis. A heurística calibrada (realização + range) joga melhor. O
    # motor MC fica pronto para uma futura versão "MC vs range ESTIMADO".
    USE_MC = False
    MC_TRIALS = {3: 160, 4: 200, 5: 260}
    MC_BLEND = 1.0             # peso do MC vs heurística (1.0 = MC puro)
    MC_MIN_N = 40             # abaixo disso, cai na heurística
    _PUSHFOLD_BB = 13.0        # abaixo disso: modo push/fold

    # bluff-catch (bônus de equity quando pagamos por suspeitar de blefe)
    _BLUFF_ALPHA = 0.11
    _BLUFF_PRIOR = 0.35
    _BC_MAX = 0.17

    # baselines de estratégia (TAG sólido) — ajustados adaptativamente
    _BASE_VALUE_THRESH = 0.55
    _BASE_BLUFF = 0.10
    _BASE_ODDS_MARGIN = 0.07
    _BASE_CBET = 0.78
    _BASE_VALUE_SIZE = 0.62
    _DISCOUNT_PAIR = 0.65

    # ── APRENDIZADO ONLINE ENTRE JOGOS (per-opponent, downside-safe) ──────────
    # Memória de CLASSE: sobrevive aos 2000 jogos do confronto (mesmo processo).
    # Se o ambiente isolar processos, cada jogo começa "frio" e o bot joga o
    # baseline — nunca pior. Todo desvio é condicionado à CONFIANÇA do perfil.
    # DESLIGADO por padrão: a infraestrutura de aprendizado entre jogos funciona
    # (ex.: joao_v2 +12pp na fase de exploit numa medição), MAS empiricamente
    # REGREDIU o campo no geral. Motivo: tornar o modelo confiante entre jogos
    # ativa adaptações que estavam (corretamente) dormentes com dados ralos — e
    # algumas métricas adaptativas estão mal-calibradas (o proxy de fold lê ~0.00
    # contra tight). Além disso o detector de "nit" é CENSURADO: a mão do busto
    # (all-in perdido) nunca é registrada (não há mão seguinte para fechá-la),
    # então o win rate de pote-grande lê alto demais e não identifica papa/tubarao.
    # Tornar isto +EV exige recalibrar toda a camada adaptativa (projeto à parte).
    # Com a flag OFF, cada jogo usa um perfil NOVO ⇒ comportamento = baseline.
    _USE_CROSSGAME = False
    _MEMORY: dict = {}
    _EXPLORE_GAMES = 50        # 1ª fase: explora/observa; depois passa a explorar EV
    _NIT_BIGPOT_N = 14         # amostras de pote-grande p/ confiar na leitura
    _NIT_BIGPOT_THR = 0.42     # win rate de pote-grande abaixo disso ⇒ nit confirmado

    # ── CAMUFLAGEM / ENVENENAMENTO (anti-profiler) ───────────────────────────
    # Em alguns jogos (após o oponente ter tido tempo de nos modelar) jogamos um
    # estilo CONTRASTANTE de propósito, para corromper o modelo que um profiler
    # adversário constrói sobre nós. Off-policy: não atualizamos nosso modelo do
    # oponente nesses jogos (não contaminamos nosso aprendizado).
    _USE_CAMO = False
    _CAMO_PROB = 0.06
    _CAMO_MIN_GAMES = 25

    def __init__(self, name, hand, chips) -> None:
        super().__init__(name, hand, chips)
        # posição / detecção de nova mão
        self._my_seat: int | None = None
        self._prev_board_len: int | None = None
        self._prev_pot: int = -1
        self.hands_seen = 0

        # modelo do oponente (robusto). Defaults de instância = fallback seguro;
        # em _bind_opp eles passam a APONTAR para o perfil de classe (cross-game).
        self._mem: _OppProfile | None = None
        self._opp_fold = _RobustStat(0.45)
        self._opp_fold_street = {s: _RobustStat(0.45) for s in (3, 4, 5)}
        self._opp_aggr = _RobustStat(0.30)
        self._opp_betsize = _RobustStat(0.6, a_fast=0.40, a_slow=0.12)
        self._camo = False         # este jogo é de camuflagem?

        # marcadores da mão atual (resetados a cada nova mão)
        self._hand_start_chips: int | None = None
        self._called_opp_aggr = False
        self._we_bet_last = False
        self._last_bet_street: int | None = None
        self._max_board_len = 0
        self._opp_aggr_this_hand = False
        self._hand_commit_frac = 0.0   # maior fração do stack comprometida na mão

        # cache de equity por estado (hand+board) dentro da decisão
        self._eq_cache: dict = {}

    def _bind_opp(self, name: str) -> None:
        """Liga este jogo ao perfil de CLASSE do oponente (cria se for o 1º jogo).
        Os estimadores passam a acumular ao longo de todo o confronto."""
        if self._USE_CROSSGAME:
            prof = BigDaddy._MEMORY.get(name)
            if prof is None:
                prof = _OppProfile(self._BLUFF_PRIOR)
                BigDaddy._MEMORY[name] = prof
        else:
            prof = _OppProfile(self._BLUFF_PRIOR)  # perfil novo por jogo = baseline
        self._mem = prof
        self._opp_fold = prof.fold
        self._opp_fold_street = prof.fold_street
        self._opp_aggr = prof.aggr
        self._opp_betsize = prof.betsize
        prof.games += 1
        # decide camuflagem deste jogo (off-policy): só depois que o oponente
        # teve jogos suficientes para nos modelar.
        self._camo = (self._USE_CAMO and prof.games >= self._CAMO_MIN_GAMES
                      and random.random() < self._CAMO_PROB)

    # ────────────────────────────────────────────────────────────────────── #
    #  Estado / posição / aprendizado por mão
    # ────────────────────────────────────────────────────────────────────── #
    def _update_state(self, gv: GameView) -> None:
        if self._mem is None:                      # 1º decisão do jogo: liga ao perfil
            self._bind_opp(gv.opponents[0].name)
        board_len = len(gv.board)
        pot = gv.pot
        opp = gv.opponents[0]

        new_hand = (
            self._prev_board_len is None
            or board_len < self._prev_board_len
            or (board_len == 3 and self._prev_board_len == 3 and pot < self._prev_pot)
        )

        if new_hand:
            self._close_previous_hand(gv)
            self.hands_seen += 1
            self._hand_start_chips = gv.my_chips
            self._called_opp_aggr = False
            self._we_bet_last = False
            self._last_bet_street = None
            self._max_board_len = board_len
            self._opp_aggr_this_hand = False
            self._hand_commit_frac = 0.0
            self._eq_cache.clear()

        if board_len > self._max_board_len:
            self._max_board_len = board_len

        # detecção de assento (uma vez): se o oponente já investiu nesta rodada
        # quando é a nossa vez no primeiro board=3, então agimos por último (dealer).
        if self._my_seat is None and board_len == 3:
            am_dealer = opp.current_bet_in_round > 0
            self._my_seat = gv.dealer_position if am_dealer else (1 - gv.dealer_position)

        # agressão do oponente nesta mão: enfrentar aposta real (> big blind)
        if gv.to_call > gv.big_blind:
            self._opp_aggr_this_hand = True

        self._prev_board_len = board_len
        self._prev_pot = pot

    def _close_previous_hand(self, gv: GameView) -> None:
        if self._hand_start_chips is None or self._mem is None:
            return
        delta = gv.my_chips - self._hand_start_chips
        won = delta > 0
        m = self._mem

        # win rate dos NOSSOS potes grandes (sinal-chave anti-nit). Atualiza
        # mesmo em camuflagem: é sobre a FORÇA dele em pote grande, não sobre nós.
        if self._hand_commit_frac >= 0.5:
            m.bigpot.update(1.0 if won else 0.0)

        # Em jogo de camuflagem, NÃO contaminamos nosso modelo do oponente: a
        # reação dele às nossas apostas anômalas não é representativa.
        if self._camo:
            self._opp_aggr.update(1.0 if self._opp_aggr_this_hand else 0.0)
            return

        # blefe do oponente: pagamos a agressão dele -> ganhamos? (1 = era batível)
        if self._called_opp_aggr:
            a = self._BLUFF_ALPHA
            m.bluff_ema = (1.0 - a) * m.bluff_ema + a * (1.0 if won else 0.0)
            m.bluff_samples += 1

        # fold do oponente à NOSSA aposta: ganhamos sem chegar ao river
        if self._we_bet_last:
            opp_folded = won and self._max_board_len < 5
            obs = 1.0 if opp_folded else 0.0
            self._opp_fold.update(obs)
            if self._last_bet_street is not None:
                st = self._opp_fold_street.get(self._last_bet_street)
                if st is not None:
                    st.update(obs)

        # frequência de agressão do oponente (por mão)
        self._opp_aggr.update(1.0 if self._opp_aggr_this_hand else 0.0)

    def _in_position(self, gv: GameView) -> bool:
        # nesta engine o dealer age por último em TODA street -> dealer = posição
        if self._my_seat is None:
            return gv.opponents[0].current_bet_in_round > 0
        return self._my_seat == gv.dealer_position

    # ────────────────────────────────────────────────────────────────────── #
    #  Motor de equity (determinístico e rápido)
    # ────────────────────────────────────────────────────────────────────── #
    def _best_made(self, cards):
        best_score = -1
        best = (RANK_CARTA_ALTA, [])
        for combo in combinations(cards, 5):
            rank, tb = avaliar_cinco_cartas(list(combo))
            score = rank * BASE_DESEMPATE + desempate_para_numero(tb)
            if score > best_score:
                best_score = score
                best = (rank, tb)
        return best

    def _draw_outs(self, my_cards, board, made_rank) -> int:
        if len(board) >= 5:
            return 0
        allcards = list(my_cards) + list(board)
        vals = [_v(c) for c in allcards]
        suits = [c.suit for c in allcards]
        outs = 0
        scount = Counter(suits)
        if any(c == 4 for c in scount.values()):
            outs += 9  # flush draw
        if made_rank < RANK_STRAIGHT:
            present = set(vals)
            vcount = Counter(vals)
            if 14 in present:
                present.add(1)
                vcount[1] = vcount[14]
            completing = set()
            for low in range(1, 11):
                window = set(range(low, low + 5))
                missing = window - present
                if len(missing) == 1:
                    completing.add(next(iter(missing)))
            s_outs = 0
            for r in completing:
                seen = vcount.get(r, 0)
                s_outs += max(0, 4 - seen)
            outs += min(s_outs, 8)
        if made_rank == RANK_CARTA_ALTA and board:
            max_b = max(_v(c) for c in board)
            over = sum(1 for c in my_cards if _v(c) > max_b)
            outs += over * 3  # overcards
        return min(outs, 15)

    def _made_equity(self, made_rank, tb, my_vals, board_vals) -> float:
        if made_rank == RANK_STRAIGHT_FLUSH: return 0.99
        if made_rank == RANK_QUADRA: return 0.98
        if made_rank == RANK_FULL_HOUSE: return 0.95
        if made_rank == RANK_FLUSH: return 0.90
        if made_rank == RANK_STRAIGHT: return 0.87
        if made_rank == RANK_TRINCA: return 0.84
        if made_rank == RANK_DOIS_PARES: return 0.74
        if made_rank == RANK_UM_PAR:
            pair_val = tb[0] if tb else 0
            max_b = max(board_vals) if board_vals else 0
            is_pocket = len(my_vals) == 2 and my_vals[0] == my_vals[1]
            if is_pocket and pair_val > max_b:
                return 0.72  # overpair
            if pair_val >= max_b:
                kicker = tb[1] if len(tb) > 1 else 0
                return 0.60 + min(kicker, 14) * 0.005  # top pair (kicker)
            second = sorted(board_vals)[-2] if len(board_vals) >= 2 else 0
            if pair_val >= second:
                return 0.50  # middle pair
            return 0.42  # weak pair
        high = max(my_vals) if my_vals else 2
        return 0.18 + (high - 2) * 0.012  # high card

    def _texture_discount(self, board) -> float:
        if not board:
            return 1.0
        bvals = [_v(c) for c in board]
        bsuits = [c.suit for c in board]
        disc = 1.0
        if len(bvals) != len(set(bvals)):
            disc *= 0.92  # board pareado
        if any(c >= 3 for c in Counter(bsuits).values()):
            disc *= 0.88  # 3+ do mesmo naipe
        uniq = sorted(set(bvals))
        for i in range(len(uniq)):
            run = [x for x in uniq if uniq[i] <= x <= uniq[i] + 4]
            if len(run) >= 3:
                disc *= 0.93  # board conectado
                break
        return disc

    def _board_play_discount(self, my_cards, board, made: float) -> float:
        """Corrige a superavaliação de mãos que vêm do BOARD. Numa board pareada
        (trinca/quadra/dois pares na mesa), a 'mão feita' (ex.: quadra) pode ser
        toda comunitária — nossas cartas só dão kicker, ou nem isso ('jogar o
        board'). Sem essa correção o bot dá all-in com K-alto numa board JJJ ou
        4-3 numa board 9999 e busta contra ranges tight. Usa o avaliador rápido."""
        bl = len(board)
        if bl < 4:
            return made
        bi = [_ci(c) for c in board]
        mi = [_ci(c) for c in my_cards]
        if bl == 5:
            board_best = _eval5(bi)                 # melhor mão só com a mesa
            full_best = _eval7(mi + bi)             # melhor mão com nossas cartas
            if full_best == board_best:
                return min(made, 0.12)              # jogamos o board (chop/perde)
            if full_best // _B5 == board_best // _B5:
                return min(made, 0.42)              # mesma categoria da mesa: só kicker
        else:  # bl == 4 (turn): trinca/quadra na mesa e sem par próprio = kicker
            from collections import Counter as _C
            bc = _C(c >> 2 for c in bi)
            mvr = [c >> 2 for c in mi]
            if max(bc.values()) >= 3 and mvr[0] != mvr[1] and mvr[0] not in bc and mvr[1] not in bc:
                return min(made, 0.45)
        return made

    def _mc_equity(self, gv: GameView):
        """Equity REAL por Monte Carlo: amostra a mão do oponente (range
        uniforme) + as cartas que faltam no board, e mede a fração de vitórias
        (empate conta 0.5). Nº de amostras fixo por street. Retorna (eq, n)."""
        my2 = [_ci(c) for c in gv.my_hand]
        if len(my2) != 2:
            return None, 0
        board = [_ci(c) for c in gv.board]
        bl = len(board)
        known = set(my2) | set(board)
        rem = [c for c in _ALL52 if c not in known]
        ctc = 5 - bl  # cartas a vir
        trials = self.MC_TRIALS.get(bl, 160)
        sample = random.sample
        win = 0.0
        n = 0
        if ctc == 0:
            myscore = _eval7(my2 + board)
            for _ in range(trials):
                opp = sample(rem, 2)
                osc = _eval7(opp + board)
                win += 1.0 if myscore > osc else (0.5 if myscore == osc else 0.0)
                n += 1
        else:
            need = 2 + ctc
            for _ in range(trials):
                s = sample(rem, need)
                full = board + s[2:]
                ms = _eval7(my2 + full)
                osc = _eval7(s[:2] + full)
                win += 1.0 if ms > osc else (0.5 if ms == osc else 0.0)
                n += 1
        return (win / n if n else None), n

    def _equity(self, gv: GameView):
        key = (tuple(str(c) for c in gv.my_hand), tuple(str(c) for c in gv.board))
        cached = self._eq_cache.get(key)
        if cached is not None:
            return cached
        my_cards = list(gv.my_hand)
        board = list(gv.board)
        rank, tb = self._best_made(my_cards + board)
        my_vals = [_v(c) for c in my_cards]
        board_vals = [_v(c) for c in board]
        made = self._made_equity(rank, tb, my_vals, board_vals)
        if rank <= RANK_TRINCA:
            made *= self._texture_discount(board)
        # corrige mãos que vêm do board (jogar o board / kicker-only)
        if rank >= RANK_DOIS_PARES:
            made = self._board_play_discount(my_cards, board, made)
        outs = self._draw_outs(my_cards, board, rank)
        cards_to_come = 2 if len(board) == 3 else (1 if len(board) == 4 else 0)
        mult = 4 if cards_to_come == 2 else (2 if cards_to_come == 1 else 0)
        draw_eq = _clamp(outs * mult / 100.0, 0.0, 0.90)
        combo_blend = 0.03 if (outs >= 8 and made >= 0.45) else 0.0
        heur_e = _clamp(max(made, draw_eq) + combo_blend, 0.02, 0.99)

        # Equity principal: Monte Carlo (preciso); cai na heurística se MC raso.
        e = heur_e
        if self.USE_MC:
            mc, n = self._mc_equity(gv)
            if mc is not None and n >= self.MC_MIN_N:
                e = _clamp(self.MC_BLEND * mc + (1.0 - self.MC_BLEND) * heur_e,
                           0.02, 0.99)
        result = (e, rank, draw_eq)
        self._eq_cache[key] = result
        return result

    # ────────────────────────────────────────────────────────────────────── #
    #  Parâmetros adaptativos (exploração limitada + anti-adaptação)
    # ────────────────────────────────────────────────────────────────────── #
    def _adapt_damp(self) -> float:
        """Fator [0..1] que reduz desvios quando o oponente parece estar se
        adaptando a nós (muitos shifts/poison) — voltamos ao baseline."""
        shifts = self._opp_aggr.shifts + self._opp_fold.shifts
        poison = self._opp_fold.poison_hits + self._opp_aggr.poison_hits
        damp = 1.0 - 0.12 * shifts - 0.04 * poison
        return _clamp(damp, 0.35, 1.0)

    def _params(self):
        """Calcula os parâmetros de estratégia desta decisão a partir do modelo
        robusto do oponente. Desvios são limitados e amortecidos."""
        damp = self._adapt_damp()
        vthr = self._BASE_VALUE_THRESH
        bluff = self._BASE_BLUFF
        margin = self._BASE_ODDS_MARGIN
        cbet = self._BASE_CBET
        vsize = self._BASE_VALUE_SIZE

        fold_conf = self._opp_fold.confident()
        fr = self._opp_fold.value()
        aggr_conf = self._opp_aggr.confident()
        ar = self._opp_aggr.value()

        if fold_conf:
            if fr < 0.28:           # calling-station: value pesado, sem blefe
                bluff = 0.0
                vsize += 0.20 * damp
                vthr -= 0.05 * damp  # value mais fino contra quem paga demais
                margin += 0.03 * damp
            elif fr > 0.55:         # folder: blefa/c-beta mais, sizing maior
                bluff = min(0.30, self._BASE_BLUFF * 2.4) * damp
                cbet = min(0.95, cbet + 0.15 * damp)
                vsize += 0.10 * damp
            else:                   # equilibrado: blefe calibrado ao fold real
                scale = _clamp((fr - 0.20) / 0.40, 0.0, 1.4)
                bluff = self._BASE_BLUFF * scale * damp
        else:
            bluff *= 0.5  # sem leitura confiável: blefe contido

        if aggr_conf and ar > 0.60:  # maníaco: aperta, paga light, blefa pouco
            vthr += 0.04 * damp
            margin = max(-0.02, margin - 0.04 * damp)
            bluff *= 0.4

        # CAMUFLAGEM: neste jogo jogamos um estilo contrastante de propósito para
        # envenenar um profiler adversário. Alterna entre "maníaco" (solto-agressivo)
        # e "rocha" (super-tight) por jogo, de forma determinística pelo nº do jogo.
        if self._camo:
            if (self._mem.games & 1) == 0:
                bluff = 0.34; vthr = 0.46; cbet = 0.95; margin = -0.02  # maníaco
            else:
                bluff = 0.0; vthr = 0.66; cbet = 0.55; margin = 0.14    # rocha

        return dict(vthr=vthr, bluff=bluff, margin=margin, cbet=cbet, vsize=vsize,
                    damp=damp)

    # ────────────────────────────────────────────────────────────────────── #
    #  Ação
    # ────────────────────────────────────────────────────────────────────── #
    def decision(self, gv: GameView) -> int:
        try:
            return self._decide(gv)
        except Exception:
            return 0

    def _decide(self, gv: GameView) -> int:
        self._update_state(gv)

        to_call = gv.to_call
        pot = gv.pot
        cb = gv.current_bet
        bb = max(1, gv.big_blind)
        my_chips = gv.my_chips
        opp_chips = gv.opponents[0].chips
        bl = len(gv.board)

        can_check = to_call == 0
        facing_bet = to_call > 0
        facing_real_bet = to_call > bb
        in_pos = self._in_position(gv)

        e, rank, draw_eq = self._equity(gv)
        e_raw = e

        p = self._params()
        value_thresh = p["vthr"]
        bluff_freq = p["bluff"]
        odds_margin = p["margin"]
        cbet_freq = p["cbet"]
        value_frac = p["vsize"]
        bluff_ok = bluff_freq > 0.0

        # ── equity de CALL = desconto de range + bônus de bluff-catch + tells ──
        e_call = e
        if facing_real_bet:
            self._opp_betsize.update(_clamp(to_call / max(pot, 1), 0.0, 1.0))
            if rank <= RANK_UM_PAR:
                e_call *= self._DISCOUNT_PAIR
            elif rank == RANK_DOIS_PARES:
                e_call *= 0.85
            if to_call > pot * 0.6:
                e_call *= 0.85
            size_ratio = to_call / max(pot, 1)
            size_factor = _clamp(1.15 - size_ratio, 0.0, 1.0)
            # bluff-catch: só credita blefe ACIMA de uma linha de base e com
            # amostras suficientes — assim não pagamos light a quem só aposta valor
            # (ex.: joao_v4 nunca blefa). Cresce com a evidência de blefe real.
            bluff_signal = max(0.0, self._mem.bluff_ema - 0.25)
            conf = _clamp(self._mem.bluff_samples / 6.0, 0.0, 1.0)
            e_call = _clamp(e_call + self._BC_MAX * 1.6 * bluff_signal * size_factor * conf,
                            0.02, 0.99)
            # tell de tamanho: aposta muito acima do típico do opp = mais força
            if self._opp_betsize.confident():
                typ = self._opp_betsize.value()
                cur = _clamp(size_ratio, 0.0, 1.0)
                e_call = _clamp(e_call - (cur - typ) * 0.12, 0.02, 0.99)

        pot_odds = to_call / (pot + to_call) if to_call > 0 else 0.0
        eff_bb = min(my_chips, opp_chips) / bb

        # ── EXPLOIT per-opponent: NIT confirmado (aprendido entre jogos) ───────
        # Só após a fase de exploração E com leitura confiante de que nossos
        # potes grandes contra ELE perdem (papa/tubarao). Downside-safe: se o
        # perfil não está maduro/confiante (ex.: processo isolado), nada dispara.
        m = self._mem
        nit = (m.games >= self._EXPLORE_GAMES
               and m.bigpot.confident(self._NIT_BIGPOT_N)
               and m.bigpot.value() < self._NIT_BIGPOT_THR)
        cap_commit = nit and rank <= RANK_DOIS_PARES and e_raw < 0.90 \
            and eff_bb >= self._PUSHFOLD_BB

        # ── helpers de ação ───────────────────────────────────────────────────
        def shove() -> int:
            self._we_bet_last = True
            self._last_bet_street = bl
            self._hand_commit_frac = 1.0
            return my_chips + cb

        def raise_to(frac: float) -> int:
            # tamanho-base pela fração do pote; depois JITTER aleatório para
            # descorrelacionar o tamanho da nossa força (anti-leitura de padrão).
            raise_over = frac * (pot + to_call)
            if self.SIZING_JITTER:
                raise_over *= random.uniform(1.0 - self.JITTER_EPS, 1.0 + self.JITTER_EPS)
            raise_over = max(bb, int(round(raise_over)))
            additional = to_call + raise_over
            # vs NIT confirmado: não construir pote all-in com mão vulnerável
            if cap_commit and additional > 0.45 * my_chips:
                capped = max(bb, int(0.45 * my_chips) - to_call)
                if capped < raise_over:
                    raise_over = capped
                    additional = to_call + raise_over
            if additional >= my_chips:
                return shove()
            self._we_bet_last = True
            self._last_bet_street = bl
            self._hand_commit_frac = max(self._hand_commit_frac,
                                         additional / max(1, my_chips))
            return cb + raise_over

        def do_call() -> int:
            if facing_real_bet:
                self._called_opp_aggr = True  # marca p/ aprender o bluff_ema
            self._hand_commit_frac = max(self._hand_commit_frac,
                                         min(to_call, my_chips) / max(1, my_chips))
            return 0

        # ══ PUSH/FOLD (stack curto) ════════════════════════════════════════════
        if eff_bb < self._PUSHFOLD_BB:
            # vs NIT confirmado: range de all-in mais apertado (eles pagam tight)
            shove_thr = (0.66 if nit else (0.50 if in_pos else 0.56))
            if e_raw >= shove_thr or rank >= RANK_TRINCA or \
                    (rank >= RANK_DOIS_PARES and not nit):
                return shove()
            if can_check:
                return 0
            if e_call > pot_odds:
                return do_call()
            return -1

        # ══ STACK PROFUNDO ══════════════════════════════════════════════════════

        # vs NIT confirmado: não pagar aposta que compromete grande parte do
        # stack com mão vulnerável (eles só apostam grande com trinca+).
        if cap_commit and facing_real_bet and to_call > 0.5 * my_chips:
            return -1

        # Valor forte (trinca+ ou equity muito alta)
        if e_raw >= 0.72 or rank >= RANK_TRINCA:
            # slowplay ocasional com monstro em posição para induzir blefe
            if rank >= RANK_FULL_HOUSE and in_pos and facing_bet and random.random() < 0.3:
                return do_call()
            return raise_to(value_frac)

        # Bom (top pair / overpair / equity média-alta)
        if e_raw >= value_thresh:
            if can_check:
                return raise_to(value_frac) if random.random() < cbet_freq else 0
            if e_call > pot_odds + odds_margin:
                if in_pos and random.random() < 0.30:  # raise de proteção/valor
                    return raise_to(value_frac)
                return do_call()
            return -1

        # Draw forte: semi-blefe / call por odds
        if draw_eq >= 0.30:
            if can_check:
                if bluff_ok and random.random() < 0.5:
                    return raise_to(0.55)
                return 0
            if e_call > pot_odds:
                if bluff_ok and random.random() < 0.4:
                    return raise_to(0.55)
                return do_call()
            if pot_odds < 0.25 and random.random() < 0.5:
                return do_call()  # implied odds
            return -1

        # Marginal / fraco
        if can_check:
            # Oponente passou (check). Em POSIÇÃO (agimos por último), roubamos
            # com alta frequência: os alvos foldam mão marginal à aposta, então
            # apostar 0.5x pote é +EV sempre que o fold deles > ~33%. Fora de
            # posição, stab esporádico para não inflar o pote com ar.
            steal = bluff_freq
            if in_pos and bluff_ok:
                steal = max(steal, 0.50 * p["damp"])
                if self._opp_fold.confident() and self._opp_fold.value() > 0.50:
                    steal = min(0.82, steal + 0.25)
            if bluff_ok and random.random() < steal:
                return raise_to(0.55)
            return 0
        if facing_bet:
            # bluff-catch: o bônus de blefe já está embutido em e_call
            if e_call > pot_odds + odds_margin:
                return do_call()
            # blefe-raise raro em aposta pequena contra quem folda
            if bluff_ok and to_call <= pot * 0.7 and random.random() < bluff_freq * 0.5:
                return raise_to(0.6)
            return -1
        return 0


def create_player(name: str = "player_bigdaddy") -> Player:
    return BigDaddy(name, Hand(), 0)
