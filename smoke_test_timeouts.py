"""
Smoke test de timeouts por bot.

Roda N partidas de cada bot contra um oponente simples (sempre chama)
e reporta quantas decisões estouraram o limite de 50 ms.

Uso:
    python3 smoke_test_timeouts.py [--games N] [--players-dir DIR]
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cards.cards import Hand
from game.game import Game
from players.player import Player
from tournament.tournament import Tournament


class _AlwaysCall(Player):
    """Bot de referência: sempre chama, nunca estoura o timeout."""

    def __init__(self):
        super().__init__("_ref_always_call", Hand(), 0)

    def decision(self, game_view) -> int:
        return 1  # call


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200,
                        help="Partidas por bot (padrão: 200)")
    parser.add_argument("--players-dir", type=str, default=None,
                        help="Pasta com os bots (padrão: players/)")
    args = parser.parse_args()

    players_dir = Path(args.players_dir) if args.players_dir else ROOT / "players"
    n_games = args.games

    t = Tournament(players_dir=players_dir, num_games=0)
    factories = t._load_player_factories()

    if not factories:
        print(f"Nenhum bot encontrado em {players_dir}")
        sys.exit(1)

    print(f"\nSmoke test de timeouts — {n_games} partidas por bot vs _ref_always_call")
    print(f"Pasta: {players_dir}")
    print(f"Limite de tempo: 50 ms de CPU por decisão\n")
    print(f"{'Bot':<40} {'Timeouts':>10} {'Partidas':>10} {'TO/partida':>12}  {'Status'}")
    print("-" * 82)

    results = []
    for factory in sorted(factories, key=lambda f: f().name):
        bot = factory()
        name = bot.name
        total_timeouts = 0
        games_done = 0

        for _ in range(n_games):
            ref = _AlwaysCall()
            game = Game([bot, ref])
            game.verbose = False
            game.suppress_warnings = True
            try:
                game.play_game(max_hands=150)
            except Exception:
                pass
            total_timeouts += game.timeouts
            games_done += 1
            bot = factory()  # instância fresca a cada partida

        per_game = total_timeouts / max(1, games_done)
        if per_game >= 1.0:
            status = "*** ALTO ***"
        elif per_game >= 0.1:
            status = "moderado"
        else:
            status = "ok"

        results.append((name, total_timeouts, games_done, per_game, status))
        print(f"{name:<40} {total_timeouts:>10,} {games_done:>10} {per_game:>12.2f}  {status}")

    print("-" * 82)
    total_all = sum(r[1] for r in results)
    print(f"\nTotal de timeouts: {total_all:,} em {len(results)} bots × {n_games} partidas")

    problematic = [r for r in results if r[3] >= 0.1]
    if problematic:
        print(f"\nBots com timeouts elevados (>= 0.1/partida):")
        for name, to, g, pp, st in sorted(problematic, key=lambda x: -x[3]):
            print(f"  {name}: {pp:.2f} TO/partida ({to:,} total)")
    else:
        print("\nNenhum bot com timeout elevado detectado.")


if __name__ == "__main__":
    main()
