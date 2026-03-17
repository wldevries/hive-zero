"""Self-play training loop for Hive AI."""

from __future__ import annotations
import numpy as np
import os
from typing import Optional

from ..encoding.move_encoder import POLICY_SIZE


from ..nn.model import HiveNet, create_model, save_checkpoint, load_checkpoint
from ..nn.training import HiveDataset, Trainer


class SelfPlayTrainer:
    """Full self-play training pipeline."""

    def __init__(self, model_path: str = "model.pt", device: str = "cpu",
                 num_blocks: int = 6, channels: int = 64,
                 checkpoint_dir: str = "checkpoints"):
        self.model_path = model_path
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.num_blocks = num_blocks
        self.channels = channels
        self.start_iteration = 0

        if os.path.exists(model_path):
            self.model, ckpt = load_checkpoint(model_path)
            self.start_iteration = ckpt.get("iteration", 0)
            print(f"Resumed from {model_path} (iteration {self.start_iteration})")
        else:
            self.model = create_model(num_blocks, channels)
            print(f"Created new model ({num_blocks} blocks, {channels} channels)")

        self.model.to(device)
        self.trainer = Trainer(self.model, device=device)
        if self.start_iteration > 0:
            # Start a fresh cosine cycle from the resume point
            self.trainer._last_restart = self.start_iteration
            self.trainer.update_lr(self.start_iteration)

    def run(self, num_iterations: int = 100, games_per_iter: int = 10,
            simulations: int = 100, epochs_per_iter: int = 1,
            batch_size: int = 64, max_moves: int = 200,
            time_limit_minutes: float | None = None,
            eval_config: dict | None = None,
            checkpoint_eval_games: int | None = None,
            checkpoint_eval_simulations: int | None = None,
            checkpoint_every: int = 10,
            resign_threshold: float = -0.97,
            resign_moves: int = 5,
            calibration_frac: float = 0.1,
            playout_cap_p: float = 0.0,
            fast_cap: int = 20):
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

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Training log (TSV, truncated on fresh start)
        log_path = "training_log.tsv"
        if self.start_iteration == 0:
            self._log = open(log_path, "w")
            self._log.write("iter\tmode\tsimulations\twins_w\twins_b\tdraws\tpositions\tbuffer\t"
                            "loss\tpolicy_loss\tvalue_loss\tlr\tplay_s\n")
        else:
            self._log = open(log_path, "a")
        self._log.flush()

        # Bootstrap eval: if best_model.pt doesn't exist, run it immediately
        # rather than waiting until the next checkpoint iteration.
        best_model_path = os.path.join(os.path.dirname(self.model_path) or ".", "best_model.pt")
        if not os.path.exists(best_model_path):
            eval_sims = checkpoint_eval_simulations if checkpoint_eval_simulations is not None else simulations
            eval_games = checkpoint_eval_games if checkpoint_eval_games is not None else 2 * games_per_iter
            self._run_checkpoint_eval(self.start_iteration, eval_sims, eval_games)

        # Replay buffer: keep last ~50k positions across iterations
        replay_buffer = HiveDataset(max_size=50_000)

        cap_label = ""
        if playout_cap_p > 0:
            cap_label = f" [cap p={playout_cap_p}, fast={fast_cap}]"

        for i in range(num_iterations):
            iteration = self.start_iteration + i + 1

            if time_limit_minutes is not None:
                elapsed = (time.time() - start_time) / 60.0
                if elapsed >= time_limit_minutes:
                    print(f"\nTime limit reached ({elapsed:.1f}m / {time_limit_minutes}m)")
                    break

            elapsed_str = f" [{(time.time() - start_time) / 60:.1f}m]" if time_limit_minutes else ""

            from .rust_selfplay import RustParallelSelfPlay

            mode_label = "MCTS"
            print(f"\n=== Iteration {iteration} [{mode_label}]{cap_label} [Rust]{elapsed_str} ===")

            # Generate self-play games
            iter_start = time.time()

            sp = RustParallelSelfPlay(
                model=self.model, device=self.device,
                simulations=simulations, max_moves=max_moves,
                resign_threshold=resign_threshold,
                resign_moves=resign_moves,
                calibration_frac=calibration_frac,
                playout_cap_p=playout_cap_p,
                fast_cap=fast_cap,
            )

            all_game_samples, finished_games = sp.play_games(games_per_iter)
            play_time = time.time() - iter_start

            total_positions = 0
            skipped_fast = 0
            buf_start = time.time()
            for gi, samples in enumerate(all_game_samples):
                for sample in samples:
                    bt, rv, pv, vt, wt = sample[0], sample[1], sample[2], sample[3], sample[4]
                    is_value_only = sample[5] if len(sample) > 5 else False
                    if is_value_only:
                        skipped_fast += 1
                        continue
                    replay_buffer.add_sample(bt, rv, pv, vt, wt)
                    total_positions += 1
            buf_time = time.time() - buf_start

            game_time = time.time() - iter_start
            fast_str = ""
            if skipped_fast > 0:
                fast_str = f" ({skipped_fast} fast-only skipped)"
            print(f"  {games_per_iter} games: {total_positions} new positions{fast_str} "
                  f"(play={play_time:.1f}s, buf={buf_time:.1f}s, "
                  f"{total_positions / max(game_time, 0.1):.0f} pos/s), "
                  f"buffer: {len(replay_buffer)}")

            # Count results and show boards of up to 3 decisive games side-by-side
            wins_w, wins_b, draws = 0, 0, 0
            decisive_games = []
            from ..core.render import render_board
            for g in finished_games:
                state = g.state if isinstance(g.state, str) else g.state.value
                if state == "WhiteWins":
                    wins_w += 1
                elif state == "BlackWins":
                    wins_b += 1
                else:
                    draws += 1
                if state in ("WhiteWins", "BlackWins") and len(decisive_games) < 3:
                    decisive_games.append(g)

            if decisive_games:
                labels, boards = [], []
                for g in decisive_games:
                    state = g.state if isinstance(g.state, str) else g.state.value
                    winner = "White" if state == "WhiteWins" else "Black"
                    move_count = g.move_count if hasattr(g, 'move_count') else len(g.move_history)
                    labels.append(f"{winner} wins ({move_count} moves)")
                    boards.append(render_board(g))
                print(_render_boards_horizontally(boards, labels=labels))

            # Update learning rate based on schedule
            self.trainer.update_lr(iteration)

            # Train on replay buffer
            for epoch in range(epochs_per_iter):
                losses = self.trainer.train_epoch(replay_buffer, batch_size=batch_size)
                lr = self.trainer._current_lr
                print(f"  Epoch {epoch + 1}: loss={losses['total_loss']:.4f} "
                      f"(policy={losses['policy_loss']:.4f}, value={losses['value_loss']:.4f}, lr={lr})")

            # Log to TSV
            self._log.write(f"{iteration}\tMCTS\t{simulations}\t"
                            f"{wins_w}\t{wins_b}\t{draws}\t{total_positions}\t"
                            f"{len(replay_buffer)}\t{losses['total_loss']:.6f}\t"
                            f"{losses['policy_loss']:.6f}\t{losses['value_loss']:.6f}\t"
                            f"{lr:.8f}\t{play_time:.1f}\n")
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
                ckpt_path = os.path.join(self.checkpoint_dir, f"model_iter{iteration:04d}.pt")
                save_checkpoint(self.model, ckpt_path, iteration, metadata)
                print(f"  Checkpoint saved to {ckpt_path}")
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
        pattern = os.path.join(self.checkpoint_dir, "model_iter*.pt")
        candidates = []
        for path in _glob.glob(pattern):
            name = os.path.basename(path)
            try:
                iter_num = int(name[len("model_iter"):-len(".pt")])
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
        from ..eval.engine_match import ModelEngine, run_match

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
                summary = run_match(engine1, engine2, num_games=num_games, max_moves=200, verbose=False, show_progress=True)
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
            print(f"  {w}W/{d}D/{l}L → best model: {winner_label}")
            self._log.write(f"{iteration}\tpit-bootstrap\t{simulations}\t{w}\t{l}\t{d}\t0\t0\t"
                            f"{score:.6f}\t0\t0\t0\t0\n")
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
            summary = run_match(challenger, defender, num_games=num_games, max_moves=200, verbose=False, show_progress=True)
        finally:
            self.model.train()

        score = summary["engine1_score"]
        w, d, l = summary["engine1_wins"], summary["draws"], summary["engine2_wins"]
        print(f"  {w}W/{d}D/{l}L (score: {score:.0%})", end="")

        if score >= WIN_THRESHOLD:
            save_checkpoint(self.model, best_model_path, iteration)
            print(f" → NEW BEST (iter {iteration})")
        else:
            print(f" → no improvement (defender holds)")

        # model.pt always mirrors best_model.pt so fresh restarts use the best known weights
        shutil.copy2(best_model_path, self.model_path)

        self._log.write(f"{iteration}\tpit\t{simulations}\t{w}\t{l}\t{d}\t0\t0\t"
                        f"{score:.6f}\t0\t0\t0\t0\n")
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
                print(f"  vs {summary['engine2']}: {w}W/{d}D/{l}L "
                      f"(score: {score:.0%}, {len(samples)} samples -> buffer)")
            else:
                print(f"  vs {summary['engine2']}: {w}W/{d}D/{l}L "
                      f"(score: {score:.0%})")

            # Log to TSV
            self._log.write(f"{iteration}\teval\t{eval_config['simulations']}\t{w}\t{l}\t{d}\t{len(samples)}\t"
                            f"{len(replay_buffer) if replay_buffer else 0}\t"
                            f"{score:.6f}\t0\t0\t0\t0\n")
            self._log.flush()
        except Exception as e:
            print(f"  Eval failed: {e}")
        finally:
            self.model.train()


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
