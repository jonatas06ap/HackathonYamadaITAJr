from players.player import Player
from importlib import import_module
from inspect import signature
from pathlib import Path
import importlib.util
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from cards.cards import FullDeck, Board, Hand

_DECISION_TIMEOUT_S = 0.05  # 50 ms por decisão

# Timer de CPU disponível? (POSIX: setitimer/ITIMER_PROF/SIGPROF)
_HAS_CPU_TIMER = (
    hasattr(signal, "setitimer")
    and hasattr(signal, "ITIMER_PROF")
    and hasattr(signal, "SIGPROF")
)


class _DecisionTimeout(BaseException):
    """
    Sinaliza que uma decisão estourou seu orçamento de TEMPO DE CPU.

    Herda de BaseException (não Exception) de propósito: assim um
    `except Exception` dentro do bot não engole o estouro — a decisão é
    realmente interrompida aos 50 ms de CPU, de forma justa e determinística,
    independente da carga da máquina (ao contrário de um timeout de relógio).
    """

class Game:
    def __init__(self, players: list[Player]) -> None:
        self.players = players
        # Blinds (small/big) e progressão
        self.small_blind: int = 5
        self.big_blind: int = 10
        self.hands_played: int = 0
        self.blind_increase_every: int = 50  # a cada N mãos
        self.blind_increase_factor: int = 2  # dobra

        # Stack inicial padrão (se chips já veio setado, mantém)
        for p in self.players:
            if p.chips <= 0:
                p.chips = self.big_blind * 500
        self.full_deck = FullDeck()
        self.board = Board()
        self.pot: int = 0
        self.current_bet: int = 0
        self.current_raiser: int | None = None
        self.current_caller: int | None = None
        self.dealer: int = 0
        self.verbose: bool = True
        # Silencia os avisos por-decisão (timeout/exceção/ação inválida). Útil em
        # torneios em massa, onde bots lentos geram milhões dessas linhas.
        self.suppress_warnings: bool = False
        # Conta quantas decisões estouraram o timeout (forçadas a call). Permite
        # medir se a paralelização está inflando o relógio e distorcendo resultados.
        self.timeouts: int = 0

        # Estado exposto para `Player.decision(game)`
        self.acting_player_idx: int | None = None
        self.to_call: int = 0

        # Timeout por decisão (segundos). None desativa — útil em testes.
        self.decision_timeout_s: float | None = _DECISION_TIMEOUT_S
        self._executor: ThreadPoolExecutor | None = None


    def pre_flop(self) -> None:
        # início de mão: aumenta blinds se necessário e cobra SB/BB
        self.increase_blinds_if_needed()
        self.post_blinds()
        for player in self.players:
            player.hand.give_cards(self.full_deck)
        self.board.flop(self.full_deck)

    def flop(self) -> None:
        self.board.flop(self.full_deck)

    def turn(self) -> None:
        self.board.turn(self.full_deck)

    def river(self) -> None:
        self.board.river(self.full_deck)

    def showdown(self) -> None:
        from cards.sequences import FullHand

        ativos = [(i, p) for i, p in enumerate(self.players) if p.in_game]
        if not ativos:
            return
        if len(ativos) == 1:
            # vencedor por fold
            winner_idx, winner = ativos[0]
            winner.chips += self.pot
            self.pot = 0
            if self.verbose:
                print(f"Vencedor: {winner.name} (por fold)")
            return

        scores: list[tuple[int, int]] = []  # (idx, score)
        for i, p in ativos:
            score = FullHand(p.hand, self.board).score_hand()
            scores.append((i, score))

        best_score = max(s for _, s in scores)
        winners = [i for i, s in scores if s == best_score]

        if self.verbose:
            # Exibe mãos/scores dos ativos
            print(f"Board: {self.board}")
            for i, p in ativos:
                s = next(sc for idx, sc in scores if idx == i)
                print(f"{p.name}: {p.hand} | Score: {s}")

        # Distribui pote (split em empate)
        share = self.pot // len(winners)
        remainder = self.pot % len(winners)

        # desempate do resto: começa à esquerda do dealer entre os vencedores
        ordered = sorted(winners, key=lambda x: (x - (self.dealer + 1)) % len(self.players))
        for j, idx in enumerate(ordered):
            self.players[idx].chips += share + (1 if j < remainder else 0)

        nomes = ", ".join(self.players[i].name for i in ordered)
        if self.verbose:
            if len(ordered) == 1:
                print(f"Vencedor: {nomes}")
            else:
                print(f"Empate entre: {nomes}")

        self.pot = 0

    def _build_game_view(self, acting_idx: int, invested: list[int]) -> "GameView":
        """
        Constrói um snapshot somente-leitura do estado público do jogo.

        CONCEITO DE POO: ENCAPSULAMENTO
            Método privado (prefixo _): só o Game deve chamar isso.
            O bot recebe um GameView — nunca o Game diretamente.
        """
        from game.game_view import GameView, PublicPlayerInfo
        acting = self.players[acting_idx]
        opponents = tuple(
            PublicPlayerInfo(
                name=p.name,
                chips=p.chips,
                current_bet_in_round=invested[i],
                is_active=p.in_game,
            )
            for i, p in enumerate(self.players)
            if i != acting_idx
        )
        return GameView(
            board=tuple(self.board.cards),
            my_hand=tuple(acting.hand.cards),
            my_chips=acting.chips,
            my_name=acting.name,
            pot=self.pot,
            current_bet=self.current_bet,
            to_call=self.to_call,
            dealer_position=self.dealer,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            opponents=opponents,
        )

    # ─── Obtenção da decisão do bot, com orçamento de tempo ────────────────────

    def _ask_decision(self, p: Player, game_view: "GameView") -> int:
        """
        Pede a decisão do bot respeitando o orçamento de `decision_timeout_s`.

        Estratégia (preferindo a mais justa disponível):
        1. Timeout de TEMPO DE CPU via SIGPROF (POSIX, thread principal): o bot é
           interrompido de fato aos 50 ms de CPU, independente da carga da máquina.
           Não vaza threads e dá resultados determinísticos em série ou em paralelo.
        2. Fallback de relógio via ThreadPoolExecutor (não-POSIX, ou quando não
           estamos na thread principal e signals não podem ser usados).

        Levanta `_DecisionTimeout` (CPU) ou `FuturesTimeoutError` (relógio) no
        estouro — tratados em bet_round como conversão para call.
        """
        timeout = self.decision_timeout_s
        if timeout is None:
            return p.decision(game_view)

        # SIGPROF só pode ser armado/entregue na thread principal do processo.
        if _HAS_CPU_TIMER and threading.current_thread() is threading.main_thread():
            return self._decide_cpu_timed(p, game_view, timeout)

        return self._decide_wall_timed(p, game_view, timeout)

    def _decide_cpu_timed(self, p: Player, game_view: "GameView", timeout: float) -> int:
        """Roda decision() interrompendo-o aos `timeout` segundos de CPU (SIGPROF)."""
        def _on_prof(signum, frame):  # disparado dentro do código do bot
            raise _DecisionTimeout()

        old_handler = signal.signal(signal.SIGPROF, _on_prof)
        try:
            # Timer one-shot de tempo de CPU (user+sys) do processo.
            signal.setitimer(signal.ITIMER_PROF, timeout)
            try:
                return p.decision(game_view)
            finally:
                signal.setitimer(signal.ITIMER_PROF, 0)  # desarma
        finally:
            signal.signal(signal.SIGPROF, old_handler)

    def _decide_wall_timed(self, p: Player, game_view: "GameView", timeout: float) -> int:
        """Fallback: timeout de relógio via thread auxiliar (não interrompe o bot)."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)
        _fut = self._executor.submit(p.decision, game_view)
        return _fut.result(timeout=timeout)

    def bet_round(self) -> None:
        """
        Executa uma rodada de apostas até estabilizar (todos que continuam no jogo
        igualaram a aposta atual ou estão all-in) antes de virar a próxima carta.

        Convenção usada para `Player.decision(game)`:
        - retorna -1: fold
        - retorna 0: check/call
        - retorna N > 0: raise para um total N (valor total que o jogador quer ter investido nesta rodada)
        Se `decision` não estiver implementado, a decisão é pedida via input().
        """
        n = len(self.players)
        if n == 0:
            return

        # contribuição desta rodada (não side-pot; all-in é tratado como "não precisa igualar")
        invested: list[int] = [0] * n
        self.current_bet = max(self.current_bet, 0)

        def precisa_acao(i: int) -> bool:
            p = self.players[i]
            if not p.in_game:
                return False
            if p.chips == 0:
                return False  # all-in: não precisa reagir
            return invested[i] < self.current_bet

        def proximo_indice(i: int) -> int:
            return (i + 1) % n

        # Começa à esquerda do dealer
        idx = proximo_indice(self.dealer)

        # Se só 0/1 jogador ativo, não tem rodada de aposta
        if len([i for i in range(n) if self.players[i].in_game]) <= 1:
            return

        # Loop até ninguém precisar agir (sem raises pendentes)
        limite_iter = 0
        while True:
            limite_iter += 1
            if limite_iter > 10_000:
                raise RuntimeError("Loop infinito detectado em bet_round")

            # Se só sobrou 1 jogador na mão, encerra a rodada imediatamente
            if sum(1 for pl in self.players if pl.in_game) <= 1:
                break

            # Se terminou: ninguém precisa igualar (ou está all-in)
            if not any(precisa_acao(i) for i in range(n)):
                break

            p = self.players[idx]
            if not p.in_game:
                idx = proximo_indice(idx)
                continue

            to_call = max(0, self.current_bet - invested[idx])
            self.acting_player_idx = idx
            self.to_call = to_call

            # Se está all-in, pula
            if p.chips == 0:
                idx = proximo_indice(idx)
                continue

            # Obtém decisão — passa GameView (nunca o Game diretamente)
            action: int = 0
            try:
                game_view = self._build_game_view(idx, invested)
                action = self._ask_decision(p, game_view)
            except _DecisionTimeout:
                self.timeouts += 1
                if not self.suppress_warnings:
                    print(f"[AVISO] {p.name} excedeu {int(self.decision_timeout_s*1000)}ms de CPU em decision(), convertido para call.")
                action = 0
            except FuturesTimeoutError:
                self.timeouts += 1
                if not self.suppress_warnings:
                    print(f"[AVISO] {p.name} excedeu {int(self.decision_timeout_s*1000)}ms em decision(), convertido para call.")
                action = 0
            except Exception as e:
                if not self.suppress_warnings:
                    print(f"[AVISO] {p.name} lançou exceção em decision(): {e!r}, convertido para call.")
                action = 0

            if not isinstance(action, int) or isinstance(action, bool):
                if not self.suppress_warnings:
                    print(f"[AVISO] {p.name} retornou tipo inválido ({type(action).__name__}: {action!r}), convertido para call.")
                action = 0
            elif action < -1:
                if not self.suppress_warnings:
                    print(f"[AVISO] {p.name} retornou ação inválida ({action}), convertido para call.")
                action = 0

            # Aplica ação
            if action == -1:
                if self.verbose:
                    print(f"[AÇÃO] {p.name} fold")
                self.fold(idx)
                idx = proximo_indice(idx)
                continue

            if action == 0:
                if to_call == 0:
                    if self.verbose:
                        print(f"[AÇÃO] {p.name} check")
                    _ = self.check(idx)
                else:
                    if to_call >= p.chips:
                        if self.verbose:
                            print(f"[AÇÃO] {p.name} all-in {p.chips}")
                        paid = self.all_in(idx)
                    else:
                        if self.verbose:
                            print(f"[AÇÃO] {p.name} call {to_call}")
                        paid = self.call(idx, to_call)
                    invested[idx] += paid
                    self.pot += paid
                self.current_caller = idx
                idx = proximo_indice(idx)
                continue

            if action > 0:
                # action = novo TOTAL investido pelo jogador nesta rodada
                desired_total = action
                if desired_total < self.current_bet:
                    # Não permite "raise" menor que a aposta atual: trata como call/check
                    desired_total = self.current_bet

                delta = desired_total - invested[idx]
                if delta <= 0:
                    idx = proximo_indice(idx)
                    continue

                if delta >= p.chips:
                    if self.verbose:
                        print(f"[AÇÃO] {p.name} all-in {p.chips} (raise para {invested[idx] + p.chips})")
                    paid = self.all_in(idx)
                    invested[idx] += paid
                    self.pot += paid
                    # all-in pode (ou não) aumentar a aposta atual
                    self.current_bet = max(self.current_bet, invested[idx])
                else:
                    if self.verbose:
                        print(f"[AÇÃO] {p.name} raise +{delta} (total {desired_total})")
                    paid = self.raise_bet(idx, delta, to_call)
                    invested[idx] += paid
                    self.pot += paid
                    self.current_bet = max(self.current_bet, invested[idx])

                self.current_raiser = idx
                # Após raise, pode haver gente que precisa agir; o loop continua
                idx = proximo_indice(idx)
                continue

            idx = proximo_indice(idx)

    def fold(self, player_idx: int) -> int:
        self.players[player_idx].in_game = False
        return -1

    def check(self, player_idx: int) -> int:
        return 0

    def call(self, player_idx: int, amount: int) -> int:
        p = self.players[player_idx]
        if amount > p.chips:
            raise ValueError("Player does not have enough chips")
        p.chips -= amount
        return amount

    def raise_bet(self, player_idx: int, raise_amount: int, current_to_call: int) -> int:
        p = self.players[player_idx]
        if raise_amount > p.chips:
            raise ValueError("Player does not have enough chips")
        if raise_amount < current_to_call:
            raise ValueError("Raise amount must be greater than bet amount")
        p.chips -= raise_amount
        return raise_amount

    def all_in(self, player_idx: int) -> int:
        p = self.players[player_idx]
        coins = p.chips
        p.chips = 0
        return coins

    def _take_chips(self, player_idx: int, amount: int) -> int:
        """Tira até `amount` chips do jogador (pode ser all-in). Retorna quanto foi pago."""
        p = self.players[player_idx]
        if not p.in_game or amount <= 0 or p.chips <= 0:
            return 0
        pay = min(p.chips, amount)
        p.chips -= pay
        return pay

    def post_blinds(self) -> None:
        """
        Cobra small blind e big blind (à esquerda do dealer) e adiciona ao pote.
        Define `current_bet` como o valor efetivo do big blind (considerando all-in).
        """
        n = len(self.players)
        if n < 2:
            return
        if sum(1 for pl in self.players if pl.in_game) <= 1:
            return

        sb_idx = (self.dealer + 1) % n
        bb_idx = (self.dealer + 2) % n

        sb_paid = self._take_chips(sb_idx, self.small_blind)
        bb_paid = self._take_chips(bb_idx, self.big_blind)

        self.pot += sb_paid + bb_paid
        self.current_bet = max(sb_paid, bb_paid)
        self.current_raiser = bb_idx
        self.current_caller = sb_idx if sb_paid == self.current_bet else None

    def increase_blinds_if_needed(self) -> None:
        """
        Aumenta progressivamente os blinds a cada `blind_increase_every` mãos.
        Chame no início de cada mão.
        """
        self.hands_played += 1
        if self.blind_increase_every <= 0:
            return
        if self.hands_played % self.blind_increase_every != 0:
            return

        factor = max(2, int(self.blind_increase_factor))
        self.small_blind *= factor
        self.big_blind *= factor

    def play_game(self, max_hands: int = 1000) -> Player | None:
        """
        Roda uma partida completa (múltiplas mãos) até sobrar 1 jogador com chips.
        Retorna o vencedor (Player). Se max_hands for atingido (blinds normalmente
        forçam término antes de ~500 mãos), retorna o líder em chips.
        """
        while True:
            vivos = [p for p in self.players if p.chips > 0]
            if len(vivos) <= 1:
                return vivos[0] if vivos else None
            if self.hands_played >= max_hands:
                return max(self.players, key=lambda p: p.chips)

            # Reset de estado da mão
            self.pot = 0
            self.current_bet = 0
            self.current_raiser = None
            self.current_caller = None
            self.full_deck = FullDeck()
            self.board = Board()

            for p in self.players:
                p.in_game = p.chips > 0
                p.hand = Hand()

            # Pré-flop (cobra blinds, dá cartas, vira flop) + rodada de apostas
            self.pre_flop()
            self.bet_round()

            if sum(1 for pl in self.players if pl.in_game) <= 1:
                self.showdown()
                self.dealer = (self.dealer + 1) % len(self.players)
                continue

            # Flop -> apostas
            self.bet_round()
            if sum(1 for pl in self.players if pl.in_game) <= 1:
                self.showdown()
                self.dealer = (self.dealer + 1) % len(self.players)
                continue

            # Turn -> apostas
            self.turn()
            self.bet_round()
            if sum(1 for pl in self.players if pl.in_game) <= 1:
                self.showdown()
                self.dealer = (self.dealer + 1) % len(self.players)
                continue

            # River -> apostas
            self.river()
            self.bet_round()

            # Showdown e distribuição do pote
            self.showdown()

            # Próxima mão
            self.dealer = (self.dealer + 1) % len(self.players)
        


def find_players() -> list[Player]:
    """
    Descobre todos os bots na pasta `players/` (raiz do projeto) e retorna
    uma lista de instâncias prontas para jogar.

    Regras de descoberta:
    - Qualquer arquivo `player*.py` dentro de `players/`
    - Exceto `player.py` (classe base) e `player_template.py` (template dos alunos)
    - O arquivo precisa ter uma função `create_player()` sem argumentos
      (ou com 1 argumento: o nome do arquivo será passado como nome do bot)
    """
    # parents[2]: src/game/ -> src/ -> PokerEntregavel/ (raiz do projeto)
    project_root = Path(__file__).resolve().parents[2]
    base_dir = (project_root / "players").resolve()

    # Garante que src/ esteja no path para que os players importem a engine
    src_dir = str(Path(__file__).resolve().parents[1])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    players: list[Player] = []

    for py_file in sorted(base_dir.glob("player*.py")):
        if py_file.name in ("player.py", "player_template.py"):
            continue

        # Carrega o módulo diretamente pelo caminho do arquivo
        spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"  [aviso] Erro ao carregar {py_file.name}: {e}")
            continue

        create_fn = getattr(module, "create_player", None)
        if create_fn is None or not callable(create_fn):
            continue

        try:
            params = list(signature(create_fn).parameters.values())
        except Exception:
            params = []

        try:
            if len(params) == 0:
                p = create_fn()
            elif len(params) == 1:
                p = create_fn(py_file.stem)
            else:
                continue
            players.append(p)
        except Exception as e:
            print(f"  [aviso] Erro ao instanciar bot de {py_file.name}: {e}")
            continue

    return players

    
