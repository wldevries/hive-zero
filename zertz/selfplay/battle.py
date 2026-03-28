"""Battle mode: pit two Zertz models against each other."""

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
                policy_logits, value = model(board, reserve)
        policy = torch.softmax(policy_logits, dim=1).float().cpu().numpy()
        value = value.float().cpu().numpy().squeeze(1)
        return policy, value
    return eval_fn


def run_battle(
    model1_path: str,
    model2_path: str,
    num_games: int = 100,
    simulations: int | None = None,
    device: str = "cuda",
    max_moves: int = 40,
    play_batch_size: int = 2,
):
    from hive_engine import ZertzSelfPlaySession

    model1, ckpt1 = load_checkpoint(model1_path)
    model2, ckpt2 = load_checkpoint(model2_path)

    meta1 = ckpt1.get("metadata", {})
    meta2 = ckpt2.get("metadata", {})
    iter1 = ckpt1.get("iteration", "?")
    iter2 = ckpt2.get("iteration", "?")

    sims1 = meta1.get("simulations", simulations or 800)
    sims2 = meta2.get("simulations", simulations or 800)
    sims = simulations if simulations is not None else max(sims1, sims2)

    name1 = f"{model1_path} (iter {iter1})"
    name2 = f"{model2_path} (iter {iter2})"

    print(f"\n{'='*60}")
    print(f"  Battle: {_cc(name1)}")
    print(f"      vs: {_cm(name2)}")
    print(f"  Games: {num_games}  Simulations: {sims}  Max moves: {max_moves}")
    print(f"{'='*60}")

    model1.to(device).eval()
    model2.to(device).eval()

    eval_fn1 = _make_eval_fn(model1, device)
    eval_fn2 = _make_eval_fn(model2, device)

    session = ZertzSelfPlaySession(
        num_games=num_games,
        simulations=sims,
        max_moves=max_moves,
        temp_threshold=0,
        playout_cap_p=0.0,
        fast_cap=sims,
        play_batch_size=play_batch_size,
    )

    pbar = tqdm(total=max_moves, unit="turn", desc="  Battle", leave=False)
    turn = [0]

    def progress_fn(finished, total, active, total_moves):
        turn[0] += 1
        advance = turn[0] - pbar.n
        if advance > 0:
            pbar.update(advance)
        pbar.set_postfix(done=f"{finished}/{total}")

    start = time.time()
    result = session.play_battle(eval_fn1, eval_fn2, progress_fn)
    elapsed = time.time() - start
    pbar.update(pbar.total - pbar.n)
    pbar.close()

    w1 = result.wins_model1
    w2 = result.wins_model2
    d  = result.draws
    total = w1 + w2 + d
    score1 = (w1 + 0.5 * d) / total if total > 0 else 0.5

    lengths = result.game_lengths
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
            ("white", _CW, result.wins_white),
            ("grey",  _CG, result.wins_grey),
            ("black", _CB, result.wins_black),
            ("combo", _CC, result.wins_combo),
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
