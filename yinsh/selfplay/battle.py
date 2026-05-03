"""Battle mode: pit two Yinsh models against each other (or a model vs the
heuristic alpha-beta bot).

Pass ``model2_path="alphabeta"`` (with ``bot_depth=N``) to run the model's NN
MCTS against the alphabeta bot. Two real model paths route through the
batched ``YinshSelfPlaySession.play_battle`` path.
"""

from __future__ import annotations

import statistics
import time

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


def _make_eval_fn(model, device):
    device_type = "cuda" if "cuda" in device else "cpu"

    def eval_fn(board_tensor_np, reserve_np):
        board = torch.from_numpy(np.array(board_tensor_np)).to(device, dtype=torch.float32)
        reserve = torch.from_numpy(np.array(reserve_np)).to(device, dtype=torch.float32)
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                policy, wdl_logits = model(board, reserve)
        wdl = torch.softmax(wdl_logits, dim=1)
        return (
            policy.float().cpu().numpy(),
            wdl.float().cpu().numpy(),
        )

    return eval_fn


def run_battle(
    model1_path: str,
    model2_path: str,
    num_games: int = 100,
    simulations: int | None = None,
    device: str = "cuda",
    play_batch_size: int = 8,
    bot_depth: int = 3,
):
    if model2_path == "alphabeta":
        return _run_battle_vs_bot(
            model1_path=model1_path,
            num_games=num_games,
            simulations=simulations,
            device=device,
            play_batch_size=play_batch_size,
            bot_depth=bot_depth,
        )

    from engine_zero import YinshSelfPlaySession

    model1, ckpt1 = load_checkpoint(model1_path)
    model2, ckpt2 = load_checkpoint(model2_path)

    meta1 = ckpt1.get("metadata", {})
    meta2 = ckpt2.get("metadata", {})
    iter1 = ckpt1.get("generation", "?")
    iter2 = ckpt2.get("generation", "?")

    sims1 = meta1.get("simulations", simulations or 400)
    sims2 = meta2.get("simulations", simulations or 400)
    sims = simulations if simulations is not None else max(sims1, sims2)

    name1 = f"{model1_path} (iter {iter1})"
    name2 = f"{model2_path} (iter {iter2})"

    print(f"\n{'=' * 60}")
    print(f"  Battle: {_cc(name1)}")
    print(f"      vs: {_cm(name2)}")
    print(f"  Games: {num_games}  Simulations: {sims}")
    print(f"{'=' * 60}")

    model1.to(device).eval()
    model2.to(device).eval()

    eval_fn1 = _make_eval_fn(model1, device)
    eval_fn2 = _make_eval_fn(model2, device)

    session = YinshSelfPlaySession(
        num_games=num_games,
        simulations=sims,
        temp_threshold=0,
        playout_cap_p=0.0,
        fast_cap=sims,
        play_batch_size=play_batch_size,
    )

    pbar, progress_fn = _make_progress(num_games)

    start = time.time()
    result = session.play_battle(eval_fn1, eval_fn2, progress_fn)
    elapsed = time.time() - start
    pbar.update(pbar.total - pbar.n)
    pbar.close()

    _print_battle_results(name1, name2, result, elapsed)


def _run_battle_vs_bot(
    model1_path: str,
    num_games: int,
    simulations: int | None,
    device: str,
    play_batch_size: int,
    bot_depth: int,
):
    """Run the model (NN+MCTS) against the alpha-beta bot in parallel via
    ``YinshSelfPlaySession.play_battle_vs_bot``. Each ply, the Rust core
    partitions active games by whose turn it is: model-turn games batch their
    MCTS leaves through one ``eval_fn`` call (controlled by
    ``play_batch_size``); bot-turn games are resolved synchronously via
    alphabeta. Pairing (model as P1 vs P2) is handled inside the session.
    """
    from engine_zero import YinshSelfPlaySession

    model, ckpt = load_checkpoint(model1_path)
    meta = ckpt.get("metadata", {})
    iter_n = ckpt.get("generation", "?")
    sims = simulations if simulations is not None else meta.get("simulations", 400)

    model.to(device).eval()
    eval_fn = _make_eval_fn(model, device)

    name1 = f"{model1_path} (iter {iter_n})"
    name2 = f"alphabeta depth={bot_depth}"

    print(f"\n{'=' * 60}")
    print(f"  Battle: {_cc(name1)}")
    print(f"      vs: {_cm(name2)}")
    print(f"  Games: {num_games}  Simulations: {sims}  Bot depth: {bot_depth}")
    print(f"{'=' * 60}")

    session = YinshSelfPlaySession(
        num_games=num_games,
        simulations=sims,
        temp_threshold=0,
        playout_cap_p=0.0,
        fast_cap=sims,
        play_batch_size=play_batch_size,
    )

    pbar, progress_fn = _make_progress(num_games)

    start = time.time()
    result = session.play_battle_vs_bot(eval_fn, bot_depth, progress_fn)
    elapsed = time.time() - start
    pbar.update(pbar.total - pbar.n)
    pbar.close()

    _print_battle_results(name1, name2, result, elapsed)


def _make_progress(num_games: int):
    """Create a tqdm bar plus a progress_fn that ticks per ply round and
    auto-grows past its initial total (Yinsh games have no hard move cap)."""
    pbar = tqdm(total=80, unit="turn", desc="  Battle", leave=False)
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

    return pbar, progress_fn


def _print_battle_results(name1: str, name2: str, result, elapsed: float) -> None:
    w1 = result.wins_model1
    w2 = result.wins_model2
    d = result.draws
    total = w1 + w2 + d
    score1 = (w1 + 0.5 * d) / total if total > 0 else 0.5

    lengths = list(result.game_lengths)
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    med_len = int(statistics.median(lengths)) if lengths else 0
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0

    print(f"\n{'=' * 60}")
    print(f"  Results ({total} games, {elapsed:.0f}s)")
    print(f"{'=' * 60}")
    print(f"  {_cc(name1)}")
    print(f"    Wins: {_cg(str(w1))}  Draws: {_cy(str(d))}  Losses: {_cr(str(w2))}")
    print(f"    Score: {_cg(f'{score1:.1%}')}")
    print()
    print(f"  {_cm(name2)}")
    print(f"    Wins: {_cg(str(w2))}  Draws: {_cy(str(d))}  Losses: {_cr(str(w1))}")
    print(f"    Score: {_cm(f'{1 - score1:.1%}')}")
    print()
    print(f"  Color split: white wins={result.wins_white}  black wins={result.wins_black}")
    print(f"  Game length: avg={avg_len:.1f}  med={med_len}  min={min_len}  max={max_len}")

    if score1 > 0.55:
        print(f"\n  Winner: {_cg('Model 1')} ({_cc(name1)})")
    elif score1 < 0.45:
        print(f"\n  Winner: {_cg('Model 2')} ({_cm(name2)})")
    else:
        print(f"\n  Result: {_cy('Too close to call')}")
    print()
