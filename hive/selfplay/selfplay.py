"""Self-play training loop for Hive AI."""

from __future__ import annotations
import torch
import numpy as np
import os
from typing import Optional

from ..training_log import LOG_HEADER

import colorama
colorama.init()
_RESET = colorama.Style.RESET_ALL
_BRIGHT = colorama.Style.BRIGHT


def _c(val, color: str) -> str:
    return f"{color}{_BRIGHT}{val}{_RESET}"


_cg = lambda v: _c(v, colorama.Fore.GREEN)    # wins
_cy = lambda v: _c(v, colorama.Fore.YELLOW)   # draws / secondary losses
_cr = lambda v: _c(v, colorama.Fore.RED)      # total loss / losses in eval
_cc = lambda v: _c(v, colorama.Fore.CYAN)     # scores / percentages

from ..encoding.move_encoder import POLICY_SIZE


from ..nn.model import HiveNet, create_model, save_checkpoint, load_checkpoint
from ..nn.training import HiveDataset, Trainer


class SelfPlayTrainer:
    """Full self-play training pipeline."""

    def __init__(self, model_path: str = "model.pt", device: str = "cpu",
                 num_blocks: int = 6, channels: int = 64,
                 checkpoint_dir: str = "checkpoints", lr: float = 0.02):
        self.model_path = model_path
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.num_blocks = num_blocks
        self.channels = channels
        self.start_iteration = 0

        if os.path.exists(model_path):
            self.model, ckpt = load_checkpoint(model_path)
            self.start_iteration = ckpt.get("iteration", 0)
            blocks = len(self.model.res_blocks)
            ch = self.model.input_conv.out_channels
            params = sum(p.numel() for p in self.model.parameters())
            print(f"Resumed from {model_path} (iteration {self.start_iteration}, "
                  f"{blocks} blocks, {ch} channels, {params/1e6:.2f}M params)")
            if blocks != num_blocks or ch != channels:
                print(f"  WARNING: --blocks {num_blocks} --channels {channels} ignored "
                      f"(checkpoint shape: {blocks} blocks, {ch} channels)")
        else:
            self.model = create_model(num_blocks, channels)
            print(f"Created new model ({num_blocks} blocks, {channels} channels)")

        self.model.to(device)
        self.trainer = Trainer(self.model, device=device, lr=lr)

    def run(self, num_iterations: int | None = None, games_per_iter: int = 10,
            simulations: int = 100, epochs_per_iter: int = 1,
            batch_size: int = 64, max_moves: int = 200,
            time_limit_minutes: float | None = None,
            eval_config: dict | None = None,
            checkpoint_eval_games: int | None = None,
            checkpoint_eval_simulations: int | None = None,
            checkpoint_every: int = 10,
            checkpoint_eval: bool = False,
            resign_threshold: float = -0.97,
            resign_moves: int = 5,
            resign_min_moves: int = 20,
            calibration_frac: float = 0.1,
            playout_cap_p: float = 0.0,
            fast_cap: int = 20,
            replay_window: int = 8,
            leaf_batch_size: int = 512,
            comment: str = ""):
        """Run the full training loop.

        Args:
            eval_config: If set, run evaluation vs Mzinga periodically.
                Keys: every, games, simulations, mzinga_path, mzinga_time.
            checkpoint_eval_games: Games per checkpoint self-eval (default 2x games_per_iter).
            checkpoint_eval_simulations: Simulations for checkpoint eval.
                Defaults to same as `simulations`.
            playout_cap_p: Probability of full search per turn (0=disabled,
                all turns are full). KataGo-style playout cap randomization.
            fast_cap: Number of simulations for fast-search turns.
        """
        import time
        start_time = time.time()
        self._comment = comment

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Training log (CSV, truncated on fresh start)
        log_path = f"{self.model_name}_log.csv"
        header = LOG_HEADER
        if self.start_iteration == 0:
            self._log = open(log_path, "w")
            self._log.write(header)
        else:
            needs_header = True
            if os.path.exists(log_path):
                with open(log_path) as f:
                    first = f.readline()
                needs_header = not first.startswith("iter,")
            self._log = open(log_path, "a")
            if needs_header:
                self._log.write(header)
        self._log.flush()

        # Bootstrap eval: if best_model.pt doesn't exist, run it immediately
        # rather than waiting until the next checkpoint iteration.
        if checkpoint_eval:
            best_model_path = os.path.join(os.path.dirname(self.model_path) or ".", "best_model.pt")
            if not os.path.exists(best_model_path):
                eval_sims = checkpoint_eval_simulations if checkpoint_eval_simulations is not None else simulations
                eval_games = checkpoint_eval_games if checkpoint_eval_games is not None else 2 * games_per_iter
                self._run_checkpoint_eval(self.start_iteration, eval_sims, eval_games)

        # Replay buffer: keep last `replay_window` iterations of data (worst case: all games hit max_moves)
        replay_buffer = HiveDataset(max_size=replay_window * games_per_iter * max_moves)

        cap_label = ""
        if playout_cap_p > 0:
            cap_label = f" [sims={simulations}, fast={fast_cap}, p={playout_cap_p}]"
        else:
            cap_label = f" [sims={simulations}]"

        import itertools
        for i in (range(num_iterations) if num_iterations is not None else itertools.count()):
            iteration = self.start_iteration + i + 1

            if time_limit_minutes is not None:
                elapsed = (time.time() - start_time) / 60.0
                if elapsed >= time_limit_minutes:
                    print(f"\nTime limit reached ({elapsed:.1f}m / {time_limit_minutes}m)")
                    break

            elapsed_str = f" [{(time.time() - start_time) / 60:.1f}m]" if time_limit_minutes else ""

            from .rust_selfplay import RustParallelSelfPlay

            mode_label = "MCTS"
            print(f"\n=== Iteration {iteration} [{mode_label}]{cap_label}{elapsed_str} ===")

            # Generate self-play games
            iter_start = time.time()
            torch.cuda.empty_cache()
            _print_vram("pre-play")

            sp = RustParallelSelfPlay(
                model=self.model, device=self.device,
                simulations=simulations, max_moves=max_moves,
                resign_threshold=resign_threshold,
                resign_moves=resign_moves,
                resign_min_moves=resign_min_moves,
                calibration_frac=calibration_frac,
                playout_cap_p=playout_cap_p,
                fast_cap=fast_cap,
                leaf_batch_size=leaf_batch_size,
            )

            result = sp.play_games(games_per_iter)
            play_time = time.time() - iter_start
            _print_vram("post-play")

            # Insert training data into replay buffer
            buf_start = time.time()
            boards, reserves, policies, values, weights, value_only_flags, policy_only_flags = result.training_data()
            replay_buffer.add_batch(boards, reserves, policies, values, weights, value_only_flags, policy_only_flags)
            buf_time = time.time() - buf_start

            total_positions = result.num_samples
            fast_positions = sum(value_only_flags)
            game_time = time.time() - iter_start
            fast_str = ""
            if fast_positions > 0:
                fast_str = f" ({fast_positions} value-only)"

            wins_w = result.wins_w
            wins_b = result.wins_b
            draws = result.draws
            resignations = result.resignations
            resign_suffix = f" (resigned={resignations})" if resignations else ""
            print(f"  Results: W={_cg(wins_w)} B={_cg(wins_b)} D/unfinished={_cy(draws)}{resign_suffix}")
            if result.use_playout_cap:
                print(f"  Playout cap: {result.full_search_turns}/{result.total_turns} full-search turns "
                      f"({100*result.full_search_turns/max(result.total_turns,1):.0f}%)")
            if result.calibration_would_resign > 0:
                print(f"  Calibration: {result.calibration_would_resign}/{result.calibration_total} "
                      f"would resign, {result.calibration_false_positives} false positives")
            print(f"  {games_per_iter} games: {total_positions} new positions{fast_str} "
                  f"(play={play_time:.1f}s, buf={buf_time:.1f}s, "
                  f"{total_positions / max(game_time, 0.1):.0f} pos/s), "
                  f"buffer: {len(replay_buffer)}")

            # Show boards of decisive games
            finished_games = result.final_games()
            decisive_games = []
            from ..core.render import render_board
            for g in finished_games:
                state = g.state if isinstance(g.state, str) else g.state.value
                if state in ("WhiteWins", "BlackWins") and len(decisive_games) < 3:
                    decisive_games.append(g)

            if decisive_games:
                labels, board_strs = [], []
                for g in decisive_games:
                    state = g.state if isinstance(g.state, str) else g.state.value
                    winner = "White" if state == "WhiteWins" else "Black"
                    move_count = g.move_count if hasattr(g, 'move_count') else len(g.move_history)
                    labels.append(f"{winner} wins ({move_count} moves)")
                    board_strs.append(render_board(g))
                rendered = _render_boards_horizontally(board_strs, labels=labels)
                print('\n'.join('    ' + line for line in rendered.split('\n')))

            # Train on replay buffer
            train_start = time.time()
            for epoch in range(epochs_per_iter):
                losses = self.trainer.train_epoch(replay_buffer, batch_size=batch_size)
                lr = self.trainer._current_lr
                total_s = f"{losses['total_loss']:.4f}"
                policy_s = f"{losses['policy_loss']:.4f}"
                value_s = f"{losses['value_loss']:.4f}"
                print(f"  Epoch {epoch + 1}: loss={_cr(total_s)} "
                      f"(policy={_cy(policy_s)}, value={_cy(value_s)}, lr={lr})")
            train_time = time.time() - train_start
            _print_vram("post-train")

            # Log to CSV
            self._log.write(f"{iteration},MCTS,{simulations},"
                            f"{wins_w},{wins_b},{draws},{resignations},{total_positions},"
                            f"{len(replay_buffer)},{losses['total_loss']:.6f},"
                            f"{losses['policy_loss']:.6f},{losses['value_loss']:.6f},"
                            f"{lr:.8f},{play_time + train_time:.1f},{self._comment}\n")
            self._comment = ""
            self._log.flush()

            # Save latest model + periodic checkpoint
            metadata = {
                "total_loss": losses["total_loss"],
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"],
                "samples": len(replay_buffer),
            }
            save_checkpoint(self.model, self.model_path, iteration, metadata)

            if iteration % checkpoint_every == 0:
                ckpt_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_iter{iteration}.pt")
                save_checkpoint(self.model, ckpt_path, iteration, metadata)
                print(f"  Checkpoint saved to {ckpt_path}")
                if checkpoint_eval:
                    eval_sims = checkpoint_eval_simulations if checkpoint_eval_simulations is not None else simulations
                    eval_games = checkpoint_eval_games if checkpoint_eval_games is not None else 2 * games_per_iter
                    self._run_checkpoint_eval(iteration, eval_sims, eval_games)

            print(f"  Model saved to {self.model_path} (iteration {iteration})")

            # Periodic evaluation vs Mzinga
            if eval_config and iteration % eval_config["every"] == 0:
                self._run_eval(eval_config, iteration, replay_buffer)


    def _find_prev_checkpoint(self, current_iteration: int) -> str | None:
        """Return path of the latest checkpoint strictly before current_iteration, or None."""
        import glob as _glob
        prefix = f"{self.model_name}_iter"
        pattern = os.path.join(self.checkpoint_dir, f"{prefix}*.pt")
        candidates = []
        for path in _glob.glob(pattern):
            name = os.path.basename(path)
            try:
                iter_num = int(name[len(prefix):-len(".pt")])
                if iter_num < current_iteration:
                    candidates.append((iter_num, path))
            except ValueError:
                pass
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[0])[1]

    def _run_checkpoint_eval(self, iteration: int, simulations: int, num_games: int):
        """Pit current model against best_model.pt. Winner becomes new best and model.pt."""
        import shutil
        from ..eval.engine_match import ModelEngine, run_parallel_match

        WIN_THRESHOLD = 0.5
        best_model_path = os.path.join(os.path.dirname(self.model_path) or ".", "best_model.pt")

        if not os.path.exists(best_model_path):
            # Bootstrap: pit model.pt against the previous checkpoint to find the best.
            prev_ckpt = self._find_prev_checkpoint(iteration)
            if prev_ckpt is None:
                print(f"  No best_model.pt and no prior checkpoint — current model is now best")
                save_checkpoint(self.model, best_model_path, iteration)
                shutil.copy2(best_model_path, self.model_path)
                return
            print(f"\n--- Bootstrap eval: model.pt vs {os.path.basename(prev_ckpt)} ({num_games} games) ---")
            self.model.eval()
            engine1 = ModelEngine(model=self.model, device=self.device,
                                  simulations=simulations, name=f"current-i{iteration}")
            prev_model, prev_ckpt_data = load_checkpoint(prev_ckpt)
            prev_iter = prev_ckpt_data.get("iteration", 0)
            prev_model.to(self.device)
            prev_model.eval()
            engine2 = ModelEngine(model=prev_model, device=self.device,
                                  simulations=simulations, name=f"prev-i{prev_iter}")
            try:
                summary = run_parallel_match(engine1, engine2, num_games=num_games, max_moves=200, verbose=False, show_progress=True)
            finally:
                self.model.train()
            score = summary["engine1_score"]
            w, d, l = summary["engine1_wins"], summary["draws"], summary["engine2_wins"]
            if score >= 0.5:
                winner_label = engine1.name
                save_checkpoint(self.model, best_model_path, iteration)
            else:
                winner_label = engine2.name
                shutil.copy2(prev_ckpt, best_model_path)
            shutil.copy2(best_model_path, self.model_path)
            print(f"  {_cg(w)}W/{_cy(d)}D/{_cr(l)}L → best model: {winner_label}")
            self._log.write(f"{iteration},pit-bootstrap,{simulations},{w},{l},{d},0,0,0,"
                            f"{score:.6f},0,0,0,0,{self._comment}\n")
            self._log.flush()
            return

        print(f"\n--- Checkpoint eval: challenger (i{iteration}) vs best ({num_games} games) ---")
        self.model.eval()

        challenger = ModelEngine(
            model=self.model, device=self.device,
            simulations=simulations, name=f"challenger-i{iteration}",
        )
        best_model, _ = load_checkpoint(best_model_path)
        best_model.to(self.device)
        best_model.eval()
        defender = ModelEngine(
            model=best_model, device=self.device,
            simulations=simulations, name="best",
        )

        try:
            summary = run_parallel_match(challenger, defender, num_games=num_games, max_moves=200, verbose=False, show_progress=True)
        finally:
            self.model.train()

        score = summary["engine1_score"]
        w, d, l = summary["engine1_wins"], summary["draws"], summary["engine2_wins"]
        print(f"  {_cg(w)}W/{_cy(d)}D/{_cr(l)}L (score: {_cc(f'{score:.0%}')})", end="")

        if score >= WIN_THRESHOLD:
            save_checkpoint(self.model, best_model_path, iteration)
            print(f" → NEW BEST (iter {iteration})")
        else:
            print(f" → no improvement (defender holds)")

        # model.pt always mirrors best_model.pt so fresh restarts use the best known weights
        shutil.copy2(best_model_path, self.model_path)

        self._log.write(f"{iteration},pit,{simulations},{w},{l},{d},0,0,0,"
                        f"{score:.6f},0,0,0,0,{self._comment}\n")
        self._log.flush()

    def _run_eval(self, eval_config: dict, iteration: int, replay_buffer=None):
        """Run evaluation games against Mzinga and feed samples back."""
        from ..eval.engine_match import EngineConfig, UHPProcess, ModelEngine, run_match

        print(f"\n--- Eval vs Mzinga (iter {iteration}) ---")
        self.model.eval()

        our_engine = ModelEngine(
            model=self.model, device=self.device,
            simulations=eval_config["simulations"],
            name=f"HiveZero-i{iteration}",
        )

        t = eval_config["mzinga_time"]
        h, m, s = t // 3600, (t % 3600) // 60, t % 60
        mzinga_config = EngineConfig(
            path=eval_config["mzinga_path"],
            bestmove_args=f"time {h:02d}:{m:02d}:{s:02d}",
        )
        mzinga = UHPProcess(mzinga_config)

        try:
            summary = run_match(
                our_engine, mzinga,
                num_games=eval_config["games"],
                max_moves=200,
                verbose=False,
            )
            score = summary["engine1_score"]
            w = summary["engine1_wins"]
            d = summary["draws"]
            l = summary["engine2_wins"]

            # Feed samples back into replay buffer
            samples = summary.get("training_samples", [])
            if replay_buffer is not None and samples:
                for bt, rv, pv, vt, wt in samples:
                    replay_buffer.add_sample(bt, rv, pv, vt, wt)
                print(f"  vs {summary['engine2']}: {_cg(w)}W/{_cy(d)}D/{_cr(l)}L "
                      f"(score: {_cc(f'{score:.0%}')}, {len(samples)} samples -> buffer)")
            else:
                print(f"  vs {summary['engine2']}: {_cg(w)}W/{_cy(d)}D/{_cr(l)}L "
                      f"(score: {_cc(f'{score:.0%}')})")

            # Log to CSV
            self._log.write(f"{iteration},eval,{eval_config['simulations']},{w},{l},{d},0,{len(samples)},"
                            f"{len(replay_buffer) if replay_buffer else 0},"
                            f"{score:.6f},0,0,0,0,{self._comment}\n")
            self._log.flush()
        except Exception as e:
            print(f"  Eval failed: {e}")
        finally:
            self.model.train()


def _print_vram(label: str, enabled: bool = False) -> None:
    """Print VRAM usage and warn if spilling into shared memory."""
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    reserv = torch.cuda.memory_reserved()
    spill = max(0, reserv - total)
    if spill:
        print(f"  *** WARNING [{label}]: {spill/1e9:.2f}GB spilled into shared memory "
              f"({reserv/1e9:.2f}GB reserved > {total/1e9:.2f}GB VRAM) ***")
    if enabled:
        print(f"  VRAM [{label}]: {alloc/1e9:.2f}GB live, {reserv/1e9:.2f}GB reserved, "
              f"{free/1e9:.2f}GB free / {total/1e9:.2f}GB")


def _render_boards_horizontally(board_strings: list[str], labels: list[str] | None = None, sep: str = "   ") -> str:
    """Render multiple board strings side-by-side, handling ANSI escape codes."""
    import re
    _ANSI = re.compile(r'\033\[[0-9;]*m')

    def visual_len(s: str) -> int:
        return len(_ANSI.sub('', s))

    boards_lines = [b.split('\n') for b in board_strings]
    board_widths = [max((visual_len(line) for line in lines), default=0) for lines in boards_lines]

    if labels:
        board_widths = [max(w, len(labels[i])) for i, w in enumerate(board_widths)]

    max_height = max(len(lines) for lines in boards_lines)
    all_lines = []

    if labels:
        all_lines.append(sep.join(lbl.ljust(board_widths[i]) for i, lbl in enumerate(labels)))

    for row in range(max_height):
        parts = []
        for bi, lines in enumerate(boards_lines):
            if row < len(lines):
                line = lines[row]
                padding = board_widths[bi] - visual_len(line)
                parts.append(line + ' ' * padding)
            else:
                parts.append(' ' * board_widths[bi])
        all_lines.append(sep.join(parts))

    return '\n'.join(all_lines)


def main():
    """Entry point for self-play training."""
    import argparse
    parser = argparse.ArgumentParser(description="Hive self-play training")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", type=str, default="model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--parallel", type=int, default=8)
    args = parser.parse_args()

    trainer = SelfPlayTrainer(
        model_path=args.model, device=args.device,
        num_blocks=args.blocks, channels=args.channels
    )
    trainer.run(
        num_iterations=args.iterations, games_per_iter=args.games,
        simulations=args.simulations, epochs_per_iter=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
