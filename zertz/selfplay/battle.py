"""Battle mode: pit two Zertz models against each other (or a model vs the
heuristic alpha-beta bot).

Pass ``model2_path="alphabeta"`` (with ``bot_depth=N``) to run the model's NN
MCTS against the alphabeta bot in a sequential per-game loop. Two real model
paths route through the fast batched ``ZertzSelfPlaySession.play_battle``."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import colorama
import numpy as np
import torch
from tqdm import tqdm

colorama.init()
_RESET = colorama.Style.RESET_ALL
_BRIGHT = colorama.Style.BRIGHT
_cg = lambda v: f"{colorama.Fore.GREEN}{_BRIGHT}{v}{_RESET}"
_cy = lambda v: f"{colorama.Fore.YELLOW}{_BRIGHT}{v}{_RESET}"
_cr = lambda v: f"{colorama.Fore.RED}{_BRIGHT}{v}{_RESET}"
_cc = lambda v: f"{colorama.Fore.CYAN}{_BRIGHT}{v}{_RESET}"
_cm = lambda v: f"{colorama.Fore.MAGENTA}{_BRIGHT}{v}{_RESET}"

from ..nn.model import load_checkpoint


@dataclass
class _BattleStats:
    wins_model1: int = 0
    wins_model2: int = 0
    draws: int = 0
    wins_white: int = 0
    wins_grey: int = 0
    wins_black: int = 0
    wins_combo: int = 0
    game_lengths: list[int] = field(default_factory=list)


def _make_eval_fn(model, device):
    device_type = "cuda" if "cuda" in device else "cpu"
    def eval_fn(board_tensor_np, reserve_np):
        board = torch.from_numpy(np.array(board_tensor_np)).to(device, dtype=torch.float32)
        reserve = torch.from_numpy(np.array(reserve_np)).to(device, dtype=torch.float32)
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                place, cap_dir, value = model(board, reserve)
        return (
            place.float().cpu().numpy(),
            cap_dir.float().cpu().numpy(),
            value.float().cpu().numpy().squeeze(1),
        )
    return eval_fn


def run_battle(
    model1_path: str,
    model2_path: str,
    num_games: int = 100,
    simulations: int | None = None,
    device: str = "cuda",
    play_batch_size: int = 2,
    bot_depth: int = 3,
):
    if model2_path == "alphabeta":
        return _run_battle_vs_bot(
            model1_path=model1_path,
            num_games=num_games,
            simulations=simulations,
            device=device,
            bot_depth=bot_depth,
        )

    from engine_zero import ZertzSelfPlaySession

    model1, ckpt1 = load_checkpoint(model1_path)
    model2, ckpt2 = load_checkpoint(model2_path)

    meta1 = ckpt1.get("metadata", {})
    meta2 = ckpt2.get("metadata", {})
    iter1 = ckpt1.get("generation", "?")
    iter2 = ckpt2.get("generation", "?")

    sims1 = meta1.get("simulations", simulations or 800)
    sims2 = meta2.get("simulations", simulations or 800)
    sims = simulations if simulations is not None else max(sims1, sims2)

    name1 = f"{model1_path} (iter {iter1})"
    name2 = f"{model2_path} (iter {iter2})"

    print(f"\n{'='*60}")
    print(f"  Battle: {_cc(name1)}")
    print(f"      vs: {_cm(name2)}")
    print(f"  Games: {num_games}  Simulations: {sims}")
    print(f"{'='*60}")

    model1.to(device).eval()
    model2.to(device).eval()

    eval_fn1 = _make_eval_fn(model1, device)
    eval_fn2 = _make_eval_fn(model2, device)

    session = ZertzSelfPlaySession(
        num_games=num_games,
        simulations=sims,
        temp_threshold=0,
        playout_cap_p=0.0,
        fast_cap=sims,
        play_batch_size=play_batch_size,
    )

    pbar = tqdm(total=65, unit="turn", desc="  Battle", leave=False)
    turn = [0]

    def progress_fn(finished, total, active, total_moves):
        turn[0] += 1
        advance = turn[0] - pbar.n
        if advance > 0:
            pbar.update(advance)
        if turn[0] >= pbar.total:
            pbar.total = turn[0] + 10
            pbar.refresh()
        pbar.set_postfix(done=f"{finished}/{total}")

    start = time.time()
    result = session.play_battle(eval_fn1, eval_fn2, progress_fn)
    elapsed = time.time() - start
    pbar.update(pbar.total - pbar.n)
    pbar.close()

    stats = _BattleStats(
        wins_model1=result.wins_model1,
        wins_model2=result.wins_model2,
        draws=result.draws,
        wins_white=result.wins_white,
        wins_grey=result.wins_grey,
        wins_black=result.wins_black,
        wins_combo=result.wins_combo,
        game_lengths=list(result.game_lengths),
    )
    _print_battle_results(name1, name2, stats, elapsed)


def _print_battle_results(
    name1: str,
    name2: str,
    stats: _BattleStats,
    elapsed: float,
) -> None:
    w1, w2, d = stats.wins_model1, stats.wins_model2, stats.draws
    total = w1 + w2 + d
    score1 = (w1 + 0.5 * d) / total if total > 0 else 0.5

    lengths = stats.game_lengths
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    med_len = int(statistics.median(lengths)) if lengths else 0
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0

    decisive = w1 + w2
    _CW = "\x1b[38;2;255;160;50m"
    _CG = "\x1b[38;2;100;180;255m"
    _CB = "\x1b[38;2;255;60;180m"
    _CC = "\x1b[38;2;80;220;80m"

    print(f"\n{'='*60}")
    print(f"  Results ({total} games, {elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"  {_cc(name1)}")
    print(f"    Wins: {_cg(str(w1))}  Draws: {_cy(str(d))}  Losses: {_cr(str(w2))}")
    print(f"    Score: {_cg(f'{score1:.1%}')}")
    print()
    print(f"  {_cm(name2)}")
    print(f"    Wins: {_cg(str(w2))}  Draws: {_cy(str(d))}  Losses: {_cr(str(w1))}")
    print(f"    Score: {_cm(f'{1-score1:.1%}')}")
    print()
    if decisive > 0:
        print(f"  Win conditions (of {decisive} decisive games):")
        for label, color, count in [
            ("white", _CW, stats.wins_white),
            ("grey",  _CG, stats.wins_grey),
            ("black", _CB, stats.wins_black),
            ("combo", _CC, stats.wins_combo),
        ]:
            pct = count / decisive * 100
            print(f"    {color}{label}{_RESET}: {count} ({pct:.0f}%)")
    print()
    print(f"  Game length: avg={avg_len:.1f}  med={med_len}  min={min_len}  max={max_len}")

    if score1 > 0.55:
        print(f"\n  Winner: {_cg('Model 1')} ({_cc(name1)})")
    elif score1 < 0.45:
        print(f"\n  Winner: {_cg('Model 2')} ({_cm(name2)})")
    else:
        print(f"\n  Result: {_cy('Too close to call')}")
    print()


def _run_battle_vs_bot(
    model1_path: str,
    num_games: int,
    simulations: int | None,
    device: str,
    bot_depth: int,
):
    """Sequential model-vs-alphabeta loop. The model side runs MCTS via
    ``ZertzGame.best_move`` (one Python eval callback per inference batch);
    the bot side calls ``ZertzGame.best_move_alphabeta(depth)``. Games are
    paired: half have the model as P1, half have the bot as P1, so a single
    side's color preference can't bias the score.

    No cross-game batching here (unlike ``play_battle``) — for evaluation
    runs at a few dozen games the sequential overhead is the right tradeoff
    against not having to thread alphabeta through the Rust battle session.
    """
    import engine_zero

    model, ckpt = load_checkpoint(model1_path)
    meta = ckpt.get("metadata", {})
    iter_n = ckpt.get("generation", "?")
    sims = simulations if simulations is not None else meta.get("simulations", 800)

    model.to(device).eval()
    eval_fn = _make_eval_fn(model, device)

    name1 = f"{model1_path} (iter {iter_n})"
    name2 = f"alphabeta depth={bot_depth}"

    print(f"\n{'='*60}")
    print(f"  Battle: {_cc(name1)}")
    print(f"      vs: {_cm(name2)}")
    print(f"  Games: {num_games}  Simulations: {sims}  Bot depth: {bot_depth}")
    print(f"{'='*60}")

    # Half the games run with the model as P1, the other half with the bot
    # as P1. For odd counts the model gets the extra game as P1.
    model_is_p1 = [True] * ((num_games + 1) // 2) + [False] * (num_games // 2)

    stats = _BattleStats()
    pbar = tqdm(total=num_games, unit="game", desc="  Battle", leave=False)
    start = time.time()
    for game_idx, m_first in enumerate(model_is_p1):
        game = engine_zero.ZertzGame()
        moves = 0
        while game.outcome() == "ongoing":
            current = game.next_player()  # 0 == P1, 1 == P2
            model_to_move = (current == 0) == m_first
            if model_to_move:
                mv = game.best_move(eval_fn, simulations=sims)
            else:
                mv = game.best_move_alphabeta(bot_depth)
            game.play(mv)
            moves += 1

        outcome = game.outcome()
        stats.game_lengths.append(moves)
        if outcome == "draw":
            stats.draws += 1
        else:
            p1_won = outcome == "p1"
            model_won = p1_won == m_first
            if model_won:
                stats.wins_model1 += 1
            else:
                stats.wins_model2 += 1
            wt = game.win_type()
            if wt == "white":
                stats.wins_white += 1
            elif wt == "grey":
                stats.wins_grey += 1
            elif wt == "black":
                stats.wins_black += 1
            elif wt == "combo":
                stats.wins_combo += 1

        pbar.update(1)
        pbar.set_postfix(
            w=stats.wins_model1,
            l=stats.wins_model2,
            d=stats.draws,
        )
    pbar.close()
    elapsed = time.time() - start

    _print_battle_results(name1, name2, stats, elapsed)
