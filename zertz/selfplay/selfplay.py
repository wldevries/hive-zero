"""Self-play training loop for Zertz AI."""

from __future__ import annotations
import os
import time
import statistics
import torch
import numpy as np
from typing import Optional
from tqdm import tqdm

import colorama
colorama.init()
_RESET = colorama.Style.RESET_ALL
_BRIGHT = colorama.Style.BRIGHT
_cg = lambda v: f"{colorama.Fore.GREEN}{_BRIGHT}{v}{_RESET}"
_cy = lambda v: f"{colorama.Fore.YELLOW}{_BRIGHT}{v}{_RESET}"
_cr = lambda v: f"{colorama.Fore.RED}{_BRIGHT}{v}{_RESET}"
_cc = lambda v: f"{colorama.Fore.CYAN}{_BRIGHT}{v}{_RESET}"

from shared.training_log import csv_comment
from ..nn.model import ZertzNet, create_model, save_checkpoint, load_checkpoint
from ..nn.training import ZertzDataset, Trainer

LOG_HEADER = (
    "iter,simulations,wins_p1,wins_p2,draws,positions,buffer,"
    "loss,policy_loss,value_loss,lr,duration_s,comment\n"
)


def _median(lst):
    return int(statistics.median(lst)) if lst else 0


class SelfPlayTrainer:
    """Full self-play training pipeline for Zertz."""

    def __init__(self, model_path: str = "zertz.pt", device: str = "cuda",
                 num_blocks: int = 6, channels: int = 64, lr: float = 0.02,
                 checkpoint_dir: str = "checkpoints/zertz"):
        self.model_path = model_path
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.start_iteration = 0

        if os.path.exists(model_path):
            self.model, ckpt = load_checkpoint(model_path)
            self.start_iteration = ckpt.get("iteration", 0)
            blocks = len(self.model.res_blocks)
            ch = self.model.input_conv.out_channels
            params = sum(p.numel() for p in self.model.parameters())
            print(f"Resumed from {model_path} (iteration {self.start_iteration}, "
                  f"{blocks} blocks, {ch} channels, {params/1e6:.2f}M params)")
        else:
            self.model = create_model(num_blocks=num_blocks, channels=channels)
            params = sum(p.numel() for p in self.model.parameters())
            print(f"New model: {num_blocks} blocks, {channels} channels, {params/1e6:.2f}M params")

        self.model.to(device)
        self.trainer = Trainer(model=self.model, device=device, lr=lr)

    def _eval_fn(self, board_tensor_np):
        """NN inference callback for Rust self-play."""
        board = torch.from_numpy(np.array(board_tensor_np)).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda" if self.device != "cpu" else "cpu",
                dtype=torch.bfloat16,
            ):
                policy_logits, value = self.model(board)
        policy = torch.softmax(policy_logits, dim=1).float().cpu().numpy()
        value = value.float().cpu().numpy().squeeze(1)
        return policy, value

    def run(
        self,
        num_iterations: Optional[int] = None,
        games_per_iter: int = 20,
        simulations: int = 100,
        epochs_per_iter: int = 1,
        batch_size: int = 256,
        max_moves: int = 200,
        replay_window: int = 8,
        checkpoint_every: int = 10,
        playout_cap_p: float = 0.0,
        fast_cap: int = 20,
        play_batch_size: int = 2,
        time_limit_minutes: Optional[float] = None,
        comment: str = "",
    ):
        from hive_engine import ZertzSelfPlaySession

        log_path = self.model_name + "_log.csv"
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write(LOG_HEADER)

        max_buffer = games_per_iter * max_moves * replay_window
        dataset = ZertzDataset(max_size=max_buffer)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        start_time = time.time()
        iteration = self.start_iteration

        while True:
            if num_iterations is not None and (iteration - self.start_iteration) >= num_iterations:
                break
            if time_limit_minutes is not None:
                if (time.time() - start_time) / 60 >= time_limit_minutes:
                    break

            iteration += 1
            iter_start = time.time()

            # Header
            cap_str = f", fast={fast_cap}, cap={int(playout_cap_p*100)}%" if playout_cap_p > 0 else ""
            print(f"\n=== {_cc(self.model_name)}  Iteration {iteration}  [sims={simulations}{cap_str}] ===")

            # --- Self-play ---
            session = ZertzSelfPlaySession(
                num_games=games_per_iter,
                simulations=simulations,
                max_moves=max_moves,
                playout_cap_p=playout_cap_p,
                fast_cap=fast_cap,
                play_batch_size=play_batch_size,
            )
            self.model.eval()

            pbar = tqdm(total=max_moves, unit="turn", desc="  Self-play", leave=False)
            turn = [0]
            def progress_fn(finished, total, active, total_moves):
                turn[0] += 1
                advance = turn[0] - pbar.n
                if advance > 0:
                    pbar.update(advance)
                pbar.set_postfix(active=f"{active}/{total}")

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
            decisive = list(result.decisive_lengths)

            print(f"  Results: {_cg(f'P1={p1}')}  {_cr(f'P2={p2}')}  {_cy(f'D/timeout={d}')}")

            if lengths:
                avg = sum(lengths) / len(lengths)
                med = _median(lengths)
                mn, mx = min(lengths), max(lengths)
                dec_str = ""
                if decisive:
                    dec_str = f"  decisive: avg={sum(decisive)/len(decisive):.0f} med={_median(decisive)}"
                print(f"  Game length: avg={avg:.0f} med={med} min={mn} max={mx}{dec_str}")

            if playout_cap_p > 0:
                fs = result.full_search_turns
                tt = result.total_turns
                pct = 100 * fs / tt if tt > 0 else 0
                print(f"  Playout cap: {fs}/{tt} full-search turns ({pct:.0f}%)")

            buf_start = time.time()
            boards, policies, values, weights, value_only = result.training_data()
            dataset.add_batch(
                board_tensors=np.array(boards),
                policy_targets=np.array(policies),
                value_targets=np.array(values),
                weights=np.array(weights),
                value_only=list(value_only),
            )
            buf_time = time.time() - buf_start

            n_vo = sum(1 for v in value_only if v)
            pos_per_s = result.num_samples / play_time if play_time > 0 else 0
            print(
                f"  {games_per_iter} games: {result.num_samples} new positions "
                f"({n_vo} value-only) "
                f"(play={play_time:.1f}s, buf={buf_time:.2f}s, {pos_per_s:.0f} pos/s), "
                f"buffer: {len(dataset)}"
            )

            # --- Training ---
            losses = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}
            for epoch in range(epochs_per_iter):
                losses = self.trainer.train_epoch(dataset, batch_size=batch_size)
                lr = self.trainer._current_lr
                total_s = f"{losses['total_loss']:.4f}"
                policy_s = f"{losses['policy_loss']:.4f}"
                value_s = f"{losses['value_loss']:.4f}"
                print(
                    f"  Epoch {epoch + 1}: loss={_cy(total_s)} "
                    f"(policy={_cc(policy_s)}, "
                    f"value={_cc(value_s)}, "
                    f"lr={lr:.4f})"
                )

            duration = time.time() - iter_start

            # --- Save model ---
            save_checkpoint(self.model, self.model_path, iteration=iteration)
            print(f"  Model saved: {self.model_path} (iteration {iteration})")

            # --- Checkpoint ---
            if iteration % checkpoint_every == 0:
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"{self.model_name}_iter{iteration:05d}.pt"
                )
                save_checkpoint(self.model, ckpt_path, iteration=iteration)
                print(f"  Checkpoint saved to {ckpt_path}")

            # --- Log ---
            with open(log_path, "a") as f:
                f.write(
                    f"{iteration},{simulations},"
                    f"{result.wins_p1},{result.wins_p2},{result.draws},"
                    f"{result.num_samples},{len(dataset)},"
                    f"{losses['total_loss']:.6f},{losses['policy_loss']:.6f},"
                    f"{losses['value_loss']:.6f},{lr:.6f},{duration:.1f},"
                    f"{csv_comment(comment)}\n"
                )

        print(f"\nTraining complete. Final model: {self.model_path}")
