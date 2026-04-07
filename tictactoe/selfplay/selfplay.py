"""Self-play training loop for Tic-Tac-Toe AI."""

from __future__ import annotations

import os
import statistics
import time
from typing import Optional

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

from shared.lr_scheduler import LRScheduler
from shared.training_log import csv_comment

from ..nn.model import TicTacToeNet, create_model, load_checkpoint, save_checkpoint, export_onnx
from ..nn.training import Trainer, TicTacToeDataset

LOG_HEADER = (
    "gen,epoch,"
    "simulations,games,positions,buffer,"
    "wins_p1,wins_p2,draws,"
    "avg_game_len,med_game_len,min_game_len,max_game_len,"
    "loss,policy_loss,value_loss,"
    "lr,duration_s,comment\n"
)


def _median(lst):
    return int(statistics.median(lst)) if lst else 0


class SelfPlayTrainer:
    """Full self-play training pipeline for Tic-Tac-Toe."""

    def __init__(
        self,
        model_path: str = "tictactoe.pt",
        device: str = "cuda",
        num_blocks: int = 2,
        channels: int = 32,
        lr: float = 0.02,
        lr_scheduler: Optional[LRScheduler] = None,
        checkpoint_dir: str = "checkpoints/tictactoe",
    ):
        self.model_path = model_path
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.start_generation = 0

        if os.path.exists(model_path):
            self.model, ckpt = load_checkpoint(model_path)
            self.start_generation = ckpt.get("generation", 0)
            blocks = len(self.model.res_blocks)
            ch = self.model.input_conv.out_channels
            params = sum(p.numel() for p in self.model.parameters())
            print(
                f"Resumed from {model_path} (gen {self.start_generation}, "
                f"{blocks} blocks, {ch} channels, {params / 1e6:.2f}M params)"
            )
        else:
            self.model = create_model(num_blocks=num_blocks, channels=channels)
            params = sum(p.numel() for p in self.model.parameters())
            print(
                f"New model: {num_blocks} blocks, {channels} channels, {params / 1e6:.2f}M params"
            )

        self.model.to(device)
        self.lr_scheduler = lr_scheduler
        self.trainer = Trainer(model=self.model, device=device, lr=lr)

    def _eval_fn(self, board_tensor_np):
        """NN inference callback for Rust self-play.

        Returns (policy_logits, value) as numpy arrays.
        """
        board = torch.from_numpy(np.array(board_tensor_np)).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            policy, value = self.model(board)
        return (
            policy.float().cpu().numpy(),
            value.float().cpu().numpy().squeeze(1),
        )

    def run(
        self,
        num_generations: Optional[int] = None,
        games_per_gen: int = 100,
        simulations: int = 50,
        epochs_per_gen: int = 1,
        batch_size: int = 256,
        max_moves: int = 9,
        replay_window: int = 8,
        checkpoint_every: int = 10,
        playout_cap_p: float = 0.0,
        fast_cap: int = 10,
        temperature: float = 1.0,
        temp_threshold: int = 5,
        c_puct: float = 1.5,
        dir_alpha: float = 0.5,
        dir_epsilon: float = 0.25,
        time_limit_minutes: Optional[float] = None,
        comment: str = "",
        value_loss_scale: float = 1.0,
    ):
        from engine_zero import TTTSelfPlaySession

        log_path = self.model_name + "_log.csv"
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write(LOG_HEADER)

        max_buffer = games_per_gen * max_moves * replay_window * 8  # 8 symmetries
        dataset = TicTacToeDataset(max_size=max_buffer)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        train_params = {
            "simulations": simulations,
            "games_per_gen": games_per_gen,
            "epochs_per_gen": epochs_per_gen,
            "batch_size": batch_size,
            "max_moves": max_moves,
            "replay_window": replay_window,
            "playout_cap_p": playout_cap_p,
            "fast_cap": fast_cap,
            "temperature": temperature,
            "temp_threshold": temp_threshold,
            "c_puct": c_puct,
            "dir_alpha": dir_alpha,
            "dir_epsilon": dir_epsilon,
        }

        start_time = time.time()
        generation = self.start_generation

        while True:
            if (
                num_generations is not None
                and (generation - self.start_generation) >= num_generations
            ):
                break
            if time_limit_minutes is not None:
                if (time.time() - start_time) / 60 >= time_limit_minutes:
                    break

            generation += 1
            gen_start = time.time()

            # Apply LR schedule
            if self.lr_scheduler is not None:
                scheduled_lr = self.lr_scheduler.get_scheduled_lr(generation)
                if scheduled_lr is not None:
                    for pg in self.trainer.optimizer.param_groups:
                        pg['lr'] = scheduled_lr

            cap_str = (
                f", fast={fast_cap}, cap={int(playout_cap_p * 100)}%"
                if playout_cap_p > 0
                else ""
            )
            print(
                f"\n=== {_cc(self.model_name)}  Gen {generation}  [sims={simulations}{cap_str}] ==="
            )

            self.model.eval()

            # --- Self-play ---
            session = TTTSelfPlaySession(
                num_games=games_per_gen,
                simulations=simulations,
                max_moves=max_moves,
                temperature=temperature,
                temp_threshold=temp_threshold,
                c_puct=c_puct,
                dir_alpha=dir_alpha,
                dir_epsilon=dir_epsilon,
                playout_cap_p=playout_cap_p,
                fast_cap=fast_cap,
            )

            pbar = tqdm(total=games_per_gen, unit="game", desc="  Self-play", leave=False)
            last_finished = [0]

            def progress_fn(finished, total, active, total_moves):
                advance = finished - last_finished[0]
                if advance > 0:
                    pbar.update(advance)
                    last_finished[0] = finished

            play_start = time.time()
            result = session.play_games(self._eval_fn, progress_fn=progress_fn)
            play_time = time.time() - play_start
            pbar.update(pbar.total - pbar.n)
            pbar.close()

            # --- Results ---
            p1 = result.wins_p1
            p2 = result.wins_p2
            d = result.draws
            lengths = list(result.game_lengths)

            print(
                f"  Results: P1(X)={_cg(f'{p1}')}  P2(O)={_cr(f'{p2}')}  Draw={_cy(f'{d}')}"
            )

            if lengths:
                avg = sum(lengths) / len(lengths)
                med = _median(lengths)
                mn, mx = min(lengths), max(lengths)
                print(f"  Game length:  min={mn}  avg={avg:.1f}  med={med}  max={mx}")

            # Add to replay buffer
            boards, policies, values, weights, value_only = result.training_data()
            dataset.add_batch(
                board_tensors=np.array(boards),
                policy_targets=np.array(policies),
                value_targets=np.array(values),
                weights=np.array(weights),
                value_only=list(value_only),
            )

            pos_per_s = result.num_samples / play_time if play_time > 0 else 0
            print(
                f"  {games_per_gen} games: {result.num_samples} positions "
                f"(play={play_time:.1f}s, {pos_per_s:.0f} pos/s), "
                f"buffer: {len(dataset)}"
            )

            # Show sample games side-by-side (up to 9)
            final_boards = result.final_boards()
            if final_boards:
                import random
                picks = random.sample(final_boards, min(9, len(final_boards)))
                _print_boards_grid(picks)

            # --- Training ---
            avg_gl = f"{sum(lengths) / len(lengths):.1f}" if lengths else ""
            med_gl = str(_median(lengths)) if lengths else ""
            min_gl = str(min(lengths)) if lengths else ""
            max_gl = str(max(lengths)) if lengths else ""

            losses = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}
            for epoch in range(epochs_per_gen):
                losses = self.trainer.train_epoch(dataset, batch_size=batch_size,
                                                   value_loss_scale=value_loss_scale)
                lr = self.trainer._current_lr
                total_s = f"{losses['total_loss']:.4f}"
                policy_s = f"{losses['policy_loss']:.4f}"
                value_s = f"{losses['value_loss']:.4f}"
                print(
                    f"  Epoch {epoch + 1}: loss={_cr(total_s)} "
                    f"(policy={_cy(policy_s)}, value={_cy(value_s)}, lr={lr:.4f})"
                )

                duration = time.time() - gen_start

                with open(log_path, "a") as f:
                    f.write(
                        f"{generation},{epoch + 1},"
                        f"{simulations},{games_per_gen},{result.num_samples},{len(dataset)},"
                        f"{result.wins_p1},{result.wins_p2},{result.draws},"
                        f"{avg_gl},{med_gl},{min_gl},{max_gl},"
                        f"{losses['total_loss']:.6f},{losses['policy_loss']:.6f},"
                        f"{losses['value_loss']:.6f},"
                        f"{lr:.6f},{duration:.1f},"
                        f"{csv_comment(comment)}\n"
                    )
                comment = ""

            # --- Save model ---
            metadata = {**train_params, "lr": lr}
            save_checkpoint(self.model, self.model_path, generation=generation, metadata=metadata)
            print(f"  Model saved: {self.model_path} (gen {generation})")

            # --- Checkpoint ---
            if generation % checkpoint_every == 0:
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"{self.model_name}_gen{generation:05d}.pt"
                )
                save_checkpoint(self.model, ckpt_path, generation=generation, metadata=metadata)
                print(f"  Checkpoint saved to {ckpt_path}")

        print(f"\nTraining complete after gen {generation}. Final model: {self.model_path}")


def _print_boards_grid(boards_with_outcome, cols=9):
    """Print final boards in a grid, up to `cols` per row.

    boards_with_outcome: list of (board[9], outcome) where outcome is 0=draw, 1=X, 2=O.
    """
    _x = f"{colorama.Fore.MAGENTA}{_BRIGHT}X{_RESET}"
    _o = f"{colorama.Fore.CYAN}{_BRIGHT}O{_RESET}"
    CELL = {0: '\u00b7', 1: _x, 2: _o}
    # Pre-padded to 5 visible chars to avoid ANSI breaking format specs
    OUTCOME_LABEL = {0: '  =  ', 1: f'  {_x}  ', 2: f'  {_o}  '}
    SEP = "  "

    for start in range(0, len(boards_with_outcome), cols):
        chunk = boards_with_outcome[start:start + cols]
        labels = [OUTCOME_LABEL.get(oc, '  ?  ') for _, oc in chunk]
        print("    " + SEP.join(labels))
        # Board rows
        for row in range(3):
            parts = []
            for board, _ in chunk:
                cells = " ".join(CELL[board[row * 3 + col]] for col in range(3))
                parts.append(cells)
            print("    " + SEP.join(parts))
        print()
