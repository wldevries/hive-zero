"""Battle mode: pit two Hive models against each other (or one model vs Mzinga).

Also exposes `ModelEngine` + `run_parallel_match` for in-process model-vs-model
evaluation matches used by checkpoint eval in selfplay.
"""

from __future__ import annotations

import statistics
import subprocess
import time

import numpy as np
import torch
from tqdm import tqdm

from shared.console import cg as _cg, cy as _cy, cr as _cr, cc as _cc, cm as _cm

from ..nn.model import load_checkpoint


def _make_eval_fn(model, device):
    device_type = "cuda" if "cuda" in device else "cpu"

    def eval_fn(board_batch, reserve_batch):
        board_4d = np.asarray(board_batch)
        reserves = np.asarray(reserve_batch)
        use_pinned = device_type == "cuda"
        bt = torch.from_numpy(board_4d).view(torch.bfloat16)
        rv = torch.from_numpy(reserves).view(torch.bfloat16)
        if use_pinned:
            bt = bt.pin_memory()
            rv = rv.pin_memory()
        bt = bt.to(device, non_blocking=use_pinned)
        rv = rv.to(device, non_blocking=use_pinned)
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                policy_logits, wdl_logits, _ = model(bt, rv)
                wdl = torch.softmax(wdl_logits, dim=1)
        policy = policy_logits.float().cpu().numpy().astype(np.float32)
        return policy, wdl.float().cpu().numpy().astype(np.float32)

    return eval_fn


class ModelEngine:
    """Lightweight container bundling a model with its eval params + a name.

    Used as the input pair to `run_parallel_match` for in-process evaluation
    matches (e.g. checkpoint eval: challenger vs best).
    """

    def __init__(self, model, device: str = "cpu", simulations: int = 800,
                 name: str = "HiveZero"):
        self.model = model
        self.device = device
        self.simulations = simulations
        self.name = name


def run_parallel_match(
    engine1: ModelEngine,
    engine2: ModelEngine,
    num_games: int = 10,
    max_moves: int = 200,
    verbose: bool = True,
    show_progress: bool = False,
) -> dict:
    """Play num_games of engine1 vs engine2 in parallel via RustSelfPlaySession.

    Even-indexed games: engine1=White; odd-indexed games: engine2=White. Returns
    a summary dict with engine1/engine2 names, wins, draws, total, score.
    """
    from engine_zero import RustSelfPlaySession

    if verbose:
        print(f"Match: {engine1.name} vs {engine2.name} ({num_games} games parallel)")

    grid_size = getattr(engine1.model, "grid_size", 23) if engine1.model else 23

    eval_fn1 = _make_eval_fn(engine1.model, engine1.device)
    eval_fn2 = _make_eval_fn(engine2.model, engine2.device)

    session = RustSelfPlaySession(
        num_games=num_games,
        simulations=engine1.simulations,
        max_moves=max_moves,
        grid_size=grid_size,
        c_puct=1.5,
        play_batch_size=16,
    )

    if show_progress:
        pbar = tqdm(total=num_games, unit="game", desc="  Eval", leave=False)
        last_done = [0]

        def progress(finished, total, active, moves):
            advance = finished - last_done[0]
            if advance > 0:
                pbar.update(advance)
                last_done[0] = finished

        result = session.play_battle(eval_fn1, eval_fn2, progress)
        pbar.update(pbar.total - pbar.n)
        pbar.close()
    else:
        result = session.play_battle(eval_fn1, eval_fn2)

    e1_wins = result.wins_model1
    e2_wins = result.wins_model2
    draws = result.draws
    total = e1_wins + e2_wins + draws

    summary = {
        "engine1": engine1.name,
        "engine2": engine2.name,
        "engine1_wins": e1_wins,
        "engine2_wins": e2_wins,
        "draws": draws,
        "total": total,
        "engine1_score": (e1_wins + 0.5 * draws) / total if total else 0,
    }

    if verbose:
        print(f"  Results: {engine1.name} {e1_wins}W / {draws}D / {e2_wins}L")
        print(f"  Score: {summary['engine1_score']:.0%}")

    return summary


def run_battle(
    model1_path: str,
    model2_path: str,
    num_games: int = 100,
    simulations: int | None = None,
    device: str = "cuda",
    max_moves: int = 200,
    c_puct: float = 1.5,
    play_batch_size: int = 1,
):
    from engine_zero import RustSelfPlaySession

    model1, ckpt1 = load_checkpoint(model1_path)
    model2, ckpt2 = load_checkpoint(model2_path)

    meta1 = ckpt1.get("metadata", {})
    meta2 = ckpt2.get("metadata", {})
    iter1 = ckpt1.get("generation", "?")
    iter2 = ckpt2.get("generation", "?")

    sims1 = meta1.get("simulations", simulations or 800)
    sims2 = meta2.get("simulations", simulations or 800)
    sims = simulations if simulations is not None else max(sims1, sims2)

    grid1 = getattr(model1, "grid_size", 23)
    grid2 = getattr(model2, "grid_size", 23)
    if grid1 != grid2:
        raise ValueError(
            f"Models have different grid sizes: {model1_path} uses {grid1}, "
            f"{model2_path} uses {grid2}. Both must match."
        )
    grid_size = grid1

    name1 = f"{model1_path} (gen {iter1})"
    name2 = f"{model2_path} (gen {iter2})"

    print(f"\n{'='*60}")
    print(f"  Battle: {_cc(name1)}")
    print(f"      vs: {_cm(name2)}")
    print(f"  Games: {num_games}  Simulations: {sims}  Max moves: {max_moves}")
    print(f"  Grid size: {grid_size}")
    print(f"{'='*60}")

    model1.to(device).eval()
    model2.to(device).eval()

    eval_fn1 = _make_eval_fn(model1, device)
    eval_fn2 = _make_eval_fn(model2, device)

    session = RustSelfPlaySession(
        num_games=num_games,
        simulations=sims,
        max_moves=max_moves,
        temperature=0.0,
        temp_threshold=0,
        playout_cap_p=0.0,
        fast_cap=sims,
        c_puct=c_puct,
        dir_alpha=0.0,
        dir_epsilon=0.0,
        play_batch_size=play_batch_size,
        grid_size=grid_size,
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
    d = result.draws
    total = w1 + w2 + d
    score1 = (w1 + 0.5 * d) / total if total > 0 else 0.5

    lengths = result.game_lengths
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    med_len = int(statistics.median(lengths)) if lengths else 0
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0

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
    print(f"  Game length: avg={avg_len:.1f}  med={med_len}  min={min_len}  max={max_len}")

    if score1 > 0.55:
        print(f"\n  Winner: {_cg('Model 1')} ({_cc(name1)})")
    elif score1 < 0.45:
        print(f"\n  Winner: {_cg('Model 2')} ({_cm(name2)})")
    else:
        print(f"\n  Result: {_cy('Too close to call')}")
    print()


class _MzingaProcess:
    """Minimal UHP subprocess wrapper for driving MzingaEngine."""

    def __init__(self, path: str, bestmove_args: str):
        self.path = path
        self.bestmove_args = bestmove_args
        self.process: subprocess.Popen | None = None
        self.name = "Mzinga"

    def start(self):
        self.process = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        for line in self._read():
            if line.startswith("id "):
                self.name = line[3:]
                break

    def _read(self) -> list[str]:
        lines = []
        while True:
            line = self.process.stdout.readline().strip()
            if line == "ok":
                break
            if line:
                lines.append(line)
        return lines

    def _send(self, command: str) -> list[str]:
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        return self._read()

    def newgame(self):
        self._send("newgame Base")

    def play(self, move: str):
        self._send(f"play {move}")

    def play_pass(self):
        self._send("pass")

    def bestmove(self) -> str:
        resp = self._send(f"bestmove {self.bestmove_args}")
        return resp[0] if resp else "pass"

    def stop(self):
        if self.process is None:
            return
        try:
            self.process.stdin.write("exit\n")
            self.process.stdin.flush()
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()
        self.process = None


def _make_eval_fn_f32(model, device):
    """Eval callback for HiveGame.best_move (single-game sequential MCTS).

    Unlike _make_eval_fn (which expects bf16-bit-encoded uint16 arrays from
    play_battle), best_move passes plain f32 arrays.
    """
    device_type = "cuda" if "cuda" in device else "cpu"

    def eval_fn(board_batch, reserve_batch):
        bt = torch.from_numpy(np.asarray(board_batch)).to(device, non_blocking=device_type == "cuda")
        rv = torch.from_numpy(np.asarray(reserve_batch)).to(device, non_blocking=device_type == "cuda")
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                policy_logits, wdl_logits, _ = model(bt, rv)
                wdl = torch.softmax(wdl_logits, dim=1)
        policy = policy_logits.float().cpu().numpy().astype(np.float32)
        return policy, wdl.float().cpu().numpy().astype(np.float32)

    return eval_fn


def run_battle_mzinga(
    model_path: str,
    mzinga_path: str,
    mzinga_time: int = 2,
    mzinga_depth: int | None = None,
    num_games: int = 20,
    simulations: int | None = None,
    device: str = "cuda",
    max_moves: int = 200,
    c_puct: float = 1.5,
):
    from engine_zero import HiveGame

    model, ckpt = load_checkpoint(model_path)
    meta = ckpt.get("metadata", {})
    iter_n = ckpt.get("generation", "?")
    sims = simulations if simulations is not None else meta.get("simulations", 800)
    grid_size = getattr(model, "grid_size", 23)

    if mzinga_depth is not None:
        bestmove_args = f"depth {mzinga_depth}"
    else:
        h, m, s = mzinga_time // 3600, (mzinga_time % 3600) // 60, mzinga_time % 60
        bestmove_args = f"time {h:02d}:{m:02d}:{s:02d}"

    model.to(device).eval()
    eval_fn = _make_eval_fn_f32(model, device)

    mzinga = _MzingaProcess(mzinga_path, bestmove_args=bestmove_args)
    mzinga.start()

    name1 = f"{model_path} (gen {iter_n})"
    name2 = f"{mzinga.name} ({bestmove_args})"

    print(f"\n{'='*60}")
    print(f"  Battle: {_cc(name1)}")
    print(f"      vs: {_cm(name2)}")
    print(f"  Games: {num_games}  Simulations: {sims}  Max moves: {max_moves}")
    print(f"  Grid size: {grid_size}")
    print(f"{'='*60}")

    w1 = w2 = d = 0
    lengths: list[int] = []
    start = time.time()

    try:
        pbar = tqdm(range(num_games), unit="game", desc="  Battle", leave=False)
        for g in pbar:
            game = HiveGame(grid_size=grid_size)
            mzinga.newgame()
            model_is_white = (g % 2 == 0)
            move_count = 0
            while move_count < max_moves and not game.is_game_over:
                turn_white = (game.turn_color == "w")
                model_to_move = (turn_white == model_is_white)
                if model_to_move:
                    # Inject Dirichlet noise at the root so each game's opening
                    # diverges; without this every game from the standard start
                    # plays out identically (Mzinga's alpha-beta is deterministic).
                    move = game.best_move(eval_fn, sims, c_puct,
                                          dir_alpha=0.3, dir_epsilon=0.25)
                else:
                    move = mzinga.bestmove()
                if not move or move.startswith("err"):
                    break
                if move.lower() == "pass":
                    game.play_pass()
                    mzinga.play_pass()
                else:
                    if not game.play_move_uhp(move):
                        break
                    mzinga.play(move)
                move_count += 1

            lengths.append(move_count)
            state = game.state
            if state == "WhiteWins":
                if model_is_white:
                    w1 += 1
                else:
                    w2 += 1
            elif state == "BlackWins":
                if model_is_white:
                    w2 += 1
                else:
                    w1 += 1
            else:
                d += 1
            pbar.set_postfix(W=w1, D=d, L=w2)
        pbar.close()
    finally:
        mzinga.stop()

    elapsed = time.time() - start

    total = w1 + w2 + d
    score1 = (w1 + 0.5 * d) / total if total > 0 else 0.5
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    med_len = int(statistics.median(lengths)) if lengths else 0
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0

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
    print(f"  Game length: avg={avg_len:.1f}  med={med_len}  min={min_len}  max={max_len}")

    if score1 > 0.55:
        print(f"\n  Winner: {_cg('Model')} ({_cc(name1)})")
    elif score1 < 0.45:
        print(f"\n  Winner: {_cm('Mzinga')} ({_cm(name2)})")
    else:
        print(f"\n  Result: {_cy('Too close to call')}")
    print()
