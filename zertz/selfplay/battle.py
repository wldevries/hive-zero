"""Battle mode: pit two Zertz models against each other (or a model vs the
heuristic alpha-beta bot).

Pass ``model2_path="alphabeta"`` (with ``bot_depth=N``) to run the model's NN
MCTS against the alphabeta bot in a sequential per-game loop. Two real model
paths route through the fast batched ``ZertzSelfPlaySession.play_battle``."""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm import tqdm

from shared.console import cg as _cg, cy as _cy, cr as _cr, cc as _cc, cm as _cm, styled

from ..nn.model import load_checkpoint


def _onnx_path_for(pt_path: str) -> str:
    """Sibling .onnx for a given .pt checkpoint, validated to exist."""
    onnx_path = os.path.splitext(pt_path)[0] + ".onnx"
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(
            f"--use-ort requires an .onnx file alongside the .pt checkpoint; "
            f"expected {onnx_path}"
        )
    return onnx_path


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
    bot_time_ms: int | None = None,
    use_ort: bool = False,
    temperature: float = 1.0,
    temp_threshold: int = 4,
):
    if model2_path == "alphabeta":
        return _run_battle_vs_bot(
            model1_path=model1_path,
            num_games=num_games,
            simulations=simulations,
            device=device,
            play_batch_size=play_batch_size,
            bot_depth=bot_depth,
            bot_time_ms=bot_time_ms,
            use_ort=use_ort,
            temperature=temperature,
            temp_threshold=temp_threshold,
        )

    from engine_zero import ZertzSelfPlaySession

    # Always load each .pt (cheap for small models) so we can print iter/sims
    # metadata in the header regardless of which inference backend we use.
    ckpt1_meta = torch.load(model1_path, weights_only=False, map_location="cpu")
    ckpt2_meta = torch.load(model2_path, weights_only=False, map_location="cpu")
    meta1 = ckpt1_meta.get("metadata", {})
    meta2 = ckpt2_meta.get("metadata", {})
    iter1 = ckpt1_meta.get("generation", "?")
    iter2 = ckpt2_meta.get("generation", "?")
    sims1 = meta1.get("simulations", simulations or 800)
    sims2 = meta2.get("simulations", simulations or 800)
    sims = simulations if simulations is not None else max(sims1, sims2)

    onnx_path1: str | None = None
    onnx_path2: str | None = None
    eval_fn1 = None
    eval_fn2 = None
    if use_ort:
        onnx_path1 = _onnx_path_for(model1_path)
        onnx_path2 = _onnx_path_for(model2_path)
    else:
        model1, _ = load_checkpoint(model1_path)
        model2, _ = load_checkpoint(model2_path)
        model1.to(device).eval()
        model2.to(device).eval()
        eval_fn1 = _make_eval_fn(model1, device)
        eval_fn2 = _make_eval_fn(model2, device)

    name1 = f"{model1_path} (iter {iter1})"
    name2 = f"{model2_path} (iter {iter2})"
    backend_str = "  Backend: ORT pipelined" if use_ort else ""

    print(f"\n{'='*60}")
    print(f"  Battle: {_cc(name1)}")
    print(f"      vs: {_cm(name2)}")
    print(f"  Games: {num_games}  Simulations: {sims}{backend_str}")
    print(f"{'='*60}")

    session = ZertzSelfPlaySession(
        num_games=num_games,
        simulations=sims,
        temperature=temperature,
        temp_threshold=temp_threshold,
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
    result = session.play_battle(
        eval_fn1=eval_fn1,
        eval_fn2=eval_fn2,
        progress_fn=progress_fn,
        onnx_path1=onnx_path1,
        onnx_path2=onnx_path2,
    )
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
            ("white", "#ffa032", stats.wins_white),
            ("grey",  "#64b4ff", stats.wins_grey),
            ("black", "#ff3cb4", stats.wins_black),
            ("combo", "#50dc50", stats.wins_combo),
        ]:
            pct = count / decisive * 100
            print(f"    {styled(label, color)}: {count} ({pct:.0f}%)")
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
    play_batch_size: int,
    bot_depth: int,
    bot_time_ms: int | None,
    use_ort: bool = False,
    temperature: float = 1.0,
    temp_threshold: int = 4,
):
    """Run the model (NN+MCTS) against the alpha-beta bot in parallel via
    ``ZertzSelfPlaySession.play_battle_vs_bot``. Each ply, the Rust core
    partitions active games by whose turn it is: model-turn games batch
    their MCTS leaves (controlled by ``play_batch_size`` like the
    model-vs-model path); bot-turn games are resolved via alphabeta in
    parallel across rayon workers. With ``bot_time_ms`` set, each alphabeta
    call caps wall-clock time and falls back to the deepest fully-completed
    iteration if it expires. Progress ticks per ply round. Pairing (model
    as P1 vs P2) is handled inside the session.

    When ``use_ort`` is set, NN inference runs through a Rust-native
    pipelined ORT worker (loaded from the .onnx alongside the .pt) so
    leaf-selection for batch N+1 overlaps the GPU forward for batch N.
    """
    from engine_zero import ZertzSelfPlaySession

    ckpt = torch.load(model1_path, weights_only=False, map_location="cpu")
    meta = ckpt.get("metadata", {})
    iter_n = ckpt.get("generation", "?")
    sims = simulations if simulations is not None else meta.get("simulations", 800)

    onnx_path: str | None = None
    eval_fn = None
    if use_ort:
        onnx_path = _onnx_path_for(model1_path)
    else:
        model, _ = load_checkpoint(model1_path)
        model.to(device).eval()
        eval_fn = _make_eval_fn(model, device)

    name1 = f"{model1_path} (iter {iter_n})"
    bot_label = f"alphabeta depth={bot_depth}"
    if bot_time_ms is not None and bot_time_ms > 0:
        bot_label += f" time<={bot_time_ms}ms"
    name2 = bot_label

    print(f"\n{'='*60}")
    print(f"  Battle: {_cc(name1)}")
    print(f"      vs: {_cm(name2)}")
    budget_str = f"  Bot time: {bot_time_ms}ms" if bot_time_ms else ""
    backend_str = "  Backend: ORT pipelined" if use_ort else ""
    print(f"  Games: {num_games}  Simulations: {sims}  Bot depth: {bot_depth}{budget_str}{backend_str}")
    print(f"{'='*60}")

    session = ZertzSelfPlaySession(
        num_games=num_games,
        simulations=sims,
        temperature=temperature,
        temp_threshold=temp_threshold,
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
    result = session.play_battle_vs_bot(
        eval_fn,
        bot_depth,
        progress_fn,
        bot_time_ms=bot_time_ms,
        onnx_path=onnx_path,
    )
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
