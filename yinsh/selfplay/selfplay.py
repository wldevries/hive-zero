"""Self-play training pipeline for Yinsh."""

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

from ..nn.model import (
    create_model,
    export_onnx,
    load_checkpoint,
    save_checkpoint,
)
from ..nn.training import Trainer, YinshDataset

LOG_HEADER = (
    "gen,epoch,"
    "simulations,games,positions,buffer,"
    "wins_p1,wins_p2,draws,timeouts,"
    "avg_game_len,med_game_len,min_game_len,max_game_len,"
    "loss,policy_loss,value_loss,"
    "lr,duration_s,comment\n"
)


def _median(lst):
    return int(statistics.median(lst)) if lst else 0


class SelfPlayTrainer:
    """Self-play trainer orchestrating Rust self-play + Python NN training."""

    def __init__(
        self,
        name: str = "yinsh",
        device: str = "cuda",
        num_blocks: int = 8,
        channels: int = 96,
        lr: float = 0.02,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        self.name = name
        self.model_dir = os.path.join("models", name)
        self.model_path = os.path.join(self.model_dir, f"{name}.pt")
        self.onnx_path = os.path.join(self.model_dir, f"{name}.onnx")
        self.log_path = os.path.join(self.model_dir, f"{name}_log.csv")
        self.checkpoint_dir = os.path.join(self.model_dir, "checkpoints")
        self.device = device
        self.start_generation = 0

        if os.path.exists(self.model_path):
            self.model, ckpt = load_checkpoint(self.model_path)
            self.start_generation = ckpt.get("generation", 0)
            blocks = len(self.model.res_blocks)
            ch = self.model.input_conv.out_channels
            params = sum(p.numel() for p in self.model.parameters())
            print(
                f"Resumed from {self.model_path} (gen {self.start_generation}, "
                f"{blocks} blocks, {ch} channels, {params / 1e6:.2f}M params)"
            )
        else:
            os.makedirs(self.model_dir, exist_ok=True)
            self.model = create_model(num_blocks=num_blocks, channels=channels)
            params = sum(p.numel() for p in self.model.parameters())
            print(
                f"New model '{name}': {num_blocks} blocks, {channels} channels, "
                f"{params / 1e6:.2f}M params"
            )

        self.model.to(device)
        self.lr_scheduler = lr_scheduler
        self.trainer = Trainer(model=self.model, device=device, lr=lr)

    def _eval_fn(self, board_tensor_np, reserve_np):
        """Python NN eval callback for the Rust self-play loop.

        Rust passes `(board[N,9,11,11], reserve[N,6])`; we return
        `(policy[N,7139], value[N])` as float32 numpy arrays.
        """
        board = torch.from_numpy(np.array(board_tensor_np)).to(
            self.device, dtype=torch.float32
        )
        reserve = torch.from_numpy(np.array(reserve_np)).to(
            self.device, dtype=torch.float32
        )
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda" if self.device != "cpu" else "cpu",
                dtype=torch.bfloat16,
            ):
                policy, value = self.model(board, reserve)
        return (
            policy.float().cpu().numpy(),
            value.float().cpu().numpy().squeeze(1),
        )

    def run(
        self,
        num_generations: Optional[int] = None,
        games_per_gen: int = 16,
        simulations: int = 200,
        epochs_per_gen: int = 1,
        batch_size: int = 256,
        max_moves: int = 400,
        replay_window: int = 8,
        checkpoint_every: int = 10,
        playout_cap_p: float = 0.0,
        fast_cap: int = 30,
        play_batch_size: int = 8,
        temperature: float = 1.0,
        temp_threshold: int = 20,
        c_puct: float = 1.5,
        dir_alpha: float = 0.3,
        dir_epsilon: float = 0.25,
        time_limit_minutes: Optional[float] = None,
        comment: str = "",
        augment_symmetry: bool = False,
        use_ort: bool = False,
        value_loss_scale: float = 1.0,
        buf_dir: Optional[str] = None,
    ):
        from engine_zero import YinshSelfPlaySession

        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write(LOG_HEADER)

        resolved_buf_dir = buf_dir if buf_dir is not None else self.model_dir
        max_buffer = games_per_gen * max_moves * replay_window
        dataset = YinshDataset(max_size=max_buffer, buf_dir=resolved_buf_dir)
        dataset.augment_symmetry = augment_symmetry
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
            "play_batch_size": play_batch_size,
            "augment_symmetry": augment_symmetry,
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

            if self.lr_scheduler is not None:
                scheduled_lr = self.lr_scheduler.get_scheduled_lr(generation)
                if scheduled_lr is not None:
                    for pg in self.trainer.optimizer.param_groups:
                        pg["lr"] = scheduled_lr

            cap_str = (
                f", fast={fast_cap}, cap={int(playout_cap_p * 100)}%"
                if playout_cap_p > 0
                else ""
            )
            print(
                f"\n=== {_cc(self.name)}  Gen {generation}  "
                f"[sims={simulations}{cap_str}] ==="
            )

            onnx_path = None
            if use_ort:
                onnx_path = self.onnx_path
                if not os.path.exists(onnx_path):
                    print(f"  ORT requested but {onnx_path} not found, exporting...")
                    export_onnx(self.model, onnx_path)

            if not use_ort:
                self.model.eval()

            session = YinshSelfPlaySession(
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
                play_batch_size=play_batch_size,
            )

            pbar = tqdm(total=max_moves, unit="turn", desc="  Self-play", leave=False)
            turn = [0]

            def progress_fn(finished, total, active, total_moves):
                turn[0] += 1
                advance = turn[0] - pbar.n
                if advance > 0:
                    pbar.update(advance)
                pbar.set_postfix(active=f"{active}/{total}")

            play_start = time.time()
            if onnx_path:
                result = session.play_games(progress_fn=progress_fn, onnx_path=onnx_path)
            else:
                result = session.play_games(self._eval_fn, progress_fn=progress_fn)
            play_time = time.time() - play_start
            pbar.update(pbar.total - pbar.n)
            pbar.close()

            p1 = result.wins_p1
            p2 = result.wins_p2
            d = result.draws
            to = result.timeouts
            lengths = list(result.game_lengths)

            print(
                f"  Results: white={_cg(f'{p1}')}  black={_cr(f'{p2}')}  "
                f"draw={_cy(f'{d}')}  timeout={_cy(f'{to}')}"
            )

            if lengths:
                avg = sum(lengths) / len(lengths)
                med = _median(lengths)
                mn, mx = min(lengths), max(lengths)
                print(
                    f"  Game length:  min=\033[1;37m{mn}\033[0m  "
                    f"avg=\033[1;37m{avg:.1f}\033[0m  "
                    f"med=\033[1;37m{med}\033[0m  max=\033[1;37m{mx}\033[0m"
                )

            if playout_cap_p > 0:
                fs = result.full_search_turns
                tt = result.total_turns
                pct = 100 * fs / tt if tt > 0 else 0
                print(f"  Playout cap: {fs}/{tt} full-search turns ({pct:.0f}%)")

            buf_start = time.time()
            boards, reserves, policies, values, value_only, phase_flags = (
                result.training_data()
            )
            dataset.add_batch(
                board_tensors=np.array(boards),
                reserve_vectors=np.array(reserves),
                policy_targets=np.array(policies),
                value_targets=np.array(values),
                value_only=list(value_only),
                phase_flags=list(phase_flags),
                generation=generation,
            )
            buf_time = time.time() - buf_start

            n_vo = sum(1 for v in value_only if v)
            pos_per_s = result.num_samples / play_time if play_time > 0 else 0
            print(
                f"  {games_per_gen} games: {result.num_samples} new positions "
                f"({n_vo} value-only) "
                f"(play={play_time:.1f}s, buf={buf_time:.2f}s, {pos_per_s:.0f} pos/s), "
                f"buffer: {len(dataset)}"
            )

            samples = result.sample_boards()
            if samples:
                board_strs = [b for _, b in samples]
                labels = [lbl for lbl, _ in samples]
                rendered = _render_boards_horizontally(board_strs, labels=labels)
                print("\n".join("    " + line for line in rendered.split("\n")))

            avg_gl = f"{sum(lengths) / len(lengths):.1f}" if lengths else ""
            med_gl = str(_median(lengths)) if lengths else ""
            min_gl = str(min(lengths)) if lengths else ""
            max_gl = str(max(lengths)) if lengths else ""

            losses = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}
            lr = self.trainer._current_lr
            for epoch in range(epochs_per_gen):
                losses = self.trainer.train_epoch(
                    dataset, batch_size=batch_size, value_loss_scale=value_loss_scale
                )
                lr = self.trainer._current_lr
                total_s = f"{losses['total_loss']:.4f}"
                policy_s = f"{losses['policy_loss']:.4f}"
                value_s = f"{losses['value_loss']:.4f}"
                print(
                    f"  Epoch {epoch + 1}: loss={_cr(total_s)} "
                    f"(policy={_cy(policy_s)}, value={_cy(value_s)}, lr={lr:.4f})"
                )

                duration = time.time() - gen_start

                with open(self.log_path, "a") as f:
                    f.write(
                        f"{generation},{epoch + 1},"
                        f"{simulations},{games_per_gen},{result.num_samples},{len(dataset)},"
                        f"{result.wins_p1},{result.wins_p2},{result.draws},{result.timeouts},"
                        f"{avg_gl},{med_gl},{min_gl},{max_gl},"
                        f"{losses['total_loss']:.6f},"
                        f"{losses['policy_loss']:.6f},"
                        f"{losses['value_loss']:.6f},"
                        f"{lr:.6f},{duration:.1f},"
                        f"{csv_comment(comment)}\n"
                    )
                comment = ""

            metadata = {**train_params, "lr": lr}
            save_checkpoint(self.model, self.model_path, generation=generation, metadata=metadata)
            try:
                export_onnx(self.model, self.onnx_path)
            except Exception as e:
                print(f"  ONNX export failed (non-fatal): {e}")
            print(f"  Model saved: {self.model_path} (gen {generation})")

            if generation % checkpoint_every == 0:
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"{self.name}_gen{generation:05d}.pt"
                )
                save_checkpoint(self.model, ckpt_path, generation=generation, metadata=metadata)
                print(f"  Checkpoint saved to {ckpt_path}")

        print(f"\nTraining complete after gen {generation}. Final model: {self.model_path}")


def _render_boards_horizontally(
    board_strings: list, labels: list | None = None, sep: str = "   "
) -> str:
    """Render multiple board strings side-by-side, handling ANSI escape codes."""
    import re

    _ANSI = re.compile(r"\033\[[0-9;]*m")

    def visual_len(s: str) -> int:
        return len(_ANSI.sub("", s))

    boards_lines = [b.split("\n") for b in board_strings]
    board_widths = [
        max((visual_len(line) for line in lines), default=0) for lines in boards_lines
    ]

    if labels:
        board_widths = [max(w, visual_len(labels[i])) for i, w in enumerate(board_widths)]

    max_height = max(len(lines) for lines in boards_lines)
    all_lines = []

    if labels:
        all_lines.append(
            sep.join(
                lbl + " " * (board_widths[i] - visual_len(lbl)) for i, lbl in enumerate(labels)
            ).rstrip()
        )

    for row in range(max_height):
        parts = []
        for bi, lines in enumerate(boards_lines):
            if row < len(lines):
                line = lines[row]
                padding = board_widths[bi] - visual_len(line)
                parts.append(line + " " * padding)
            else:
                parts.append(" " * board_widths[bi])
        all_lines.append(sep.join(parts).rstrip())

    return "\n".join(all_lines)
