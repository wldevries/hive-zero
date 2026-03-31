"""Self-play training loop for Zertz AI."""

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

from shared.training_log import csv_comment

from ..nn.model import ZertzNet, create_model, load_checkpoint, save_checkpoint
from ..nn.training import Trainer, ZertzDataset

LOG_HEADER = (
    "gen,epoch,"
    "simulations,games,positions,buffer,"
    "wins_p1,wins_p2,draws,"
    "wins_white,wins_grey,wins_black,wins_combo,"
    "avg_game_len,med_game_len,min_game_len,max_game_len,"
    "isolation_captures,jump_captures,"
    "loss,policy_loss,value_loss,"
    "place_policy_loss,capture_policy_loss,"
    "place_value_loss,capture_value_loss,"
    "lr,duration_s,comment\n"
)


def _median(lst):
    return int(statistics.median(lst)) if lst else 0


class SelfPlayTrainer:
    """Full self-play training pipeline for Zertz."""

    def __init__(
        self,
        model_path: str = "zertz.pt",
        device: str = "cuda",
        num_blocks: int = 6,
        channels: int = 64,
        lr: float = 0.02,
        lr_schedule: Optional[list[tuple[int, float]]] = None,
        checkpoint_dir: str = "checkpoints/zertz",
    ):
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
            print(
                f"Resumed from {model_path} (gen {self.start_iteration}, "
                f"{blocks} blocks, {ch} channels, {params / 1e6:.2f}M params)"
            )
        else:
            self.model = create_model(num_blocks=num_blocks, channels=channels)
            params = sum(p.numel() for p in self.model.parameters())
            print(
                f"New model: {num_blocks} blocks, {channels} channels, {params / 1e6:.2f}M params"
            )

        self.model.to(device)
        self.lr_schedule = lr_schedule
        self.trainer = Trainer(model=self.model, device=device, lr=lr)

    def _get_scheduled_lr(self, iteration: int) -> Optional[float]:
        """Return the LR for this iteration by linearly interpolating between waypoints."""
        if not self.lr_schedule:
            return None
        # Before first waypoint or at/after last: clamp
        if iteration <= self.lr_schedule[0][0]:
            return self.lr_schedule[0][1]
        if iteration >= self.lr_schedule[-1][0]:
            return self.lr_schedule[-1][1]
        # Find surrounding waypoints and interpolate
        for i in range(len(self.lr_schedule) - 1):
            it0, lr0 = self.lr_schedule[i]
            it1, lr1 = self.lr_schedule[i + 1]
            if it0 <= iteration <= it1:
                t = (iteration - it0) / (it1 - it0)
                return lr0 + t * (lr1 - lr0)
        return self.lr_schedule[-1][1]

    def _eval_fn(self, board_tensor_np, reserve_np):
        """NN inference callback for Rust self-play.

        Returns 4-tuple: (place_logits, cap_source_logits, cap_dest_logits, value)
        as numpy arrays. Rust MCTS computes softmax over legal moves internally.
        """
        board = torch.from_numpy(np.array(board_tensor_np)).to(self.device, dtype=torch.float32)
        reserve = torch.from_numpy(np.array(reserve_np)).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda" if self.device != "cpu" else "cpu",
                dtype=torch.bfloat16,
            ):
                place, source, dest, value = self.model(board, reserve)
        return (
            place.float().cpu().numpy(),
            source.float().cpu().numpy(),
            dest.float().cpu().numpy(),
            value.float().cpu().numpy().squeeze(1),
        )

    def run(
        self,
        num_generations: Optional[int] = None,
        games_per_gen: int = 20,
        simulations: int = 100,
        epochs_per_gen: int = 1,
        batch_size: int = 256,
        max_moves: int = 200,
        replay_window: int = 8,
        checkpoint_every: int = 10,
        playout_cap_p: float = 0.0,
        fast_cap: int = 20,
        play_batch_size: int = 2,
        temperature: float = 1.0,
        temp_threshold: int = 10,
        c_puct: float = 1.5,
        dir_alpha: float = 0.3,
        dir_epsilon: float = 0.25,
        time_limit_minutes: Optional[float] = None,
        comment: str = "",
        augment_symmetry: bool = False,
    ):
        from hive_engine import ZertzSelfPlaySession

        log_path = self.model_name + "_log.csv"
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write(LOG_HEADER)

        max_buffer = games_per_gen * max_moves * replay_window
        dataset = ZertzDataset(max_size=max_buffer)
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
        generation = self.start_iteration

        while True:
            if (
                num_generations is not None
                and (generation - self.start_iteration) >= num_generations
            ):
                break
            if time_limit_minutes is not None:
                if (time.time() - start_time) / 60 >= time_limit_minutes:
                    break

            generation += 1
            iter_start = time.time()

            # Apply LR schedule
            scheduled_lr = self._get_scheduled_lr(generation)
            if scheduled_lr is not None:
                for pg in self.trainer.optimizer.param_groups:
                    pg['lr'] = scheduled_lr

            # Header
            cap_str = (
                f", fast={fast_cap}, cap={int(playout_cap_p * 100)}%"
                if playout_cap_p > 0
                else ""
            )
            print(
                f"\n=== {_cc(self.model_name)}  Gen {generation}  [sims={simulations}{cap_str}] ==="
            )

            # --- Self-play ---
            session = ZertzSelfPlaySession(
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

            print(
                f"  Results: P1={_cg(f'{p1}')}  P2={_cr(f'{p2}')}  D/timeout={_cy(f'{d}')}"
            )

            if lengths:
                avg = sum(lengths) / len(lengths)
                med = _median(lengths)
                mn, mx = min(lengths), max(lengths)
                print(f"  Game length:  min=\033[1;37m{mn}\033[0m  avg=\033[1;37m{avg:.1f}\033[0m  med=\033[1;37m{med}\033[0m  max=\033[1;37m{mx}\033[0m")

            decisive_total = result.wins_white + result.wins_grey + result.wins_black + result.wins_combo
            if decisive_total > 0:
                _CW = "\x1b[38;2;255;160;50m"   # orange     — white balls
                _CG = "\x1b[38;2;100;180;255m"  # steel blue — grey balls
                _CB = "\x1b[38;2;255;60;180m"   # hot pink   — black balls
                _CC = "\x1b[38;2;80;220;80m"    # green      — combo
                def _wc(c, n): return f"{c}{n}{_RESET}"
                print(
                    f"  Win cons: white={_wc(_CW, result.wins_white)}"
                    f"  grey={_wc(_CG, result.wins_grey)}"
                    f"  black={_wc(_CB, result.wins_black)}"
                    f"  combo={_wc(_CC, result.wins_combo)}"
                )

            if playout_cap_p > 0:
                fs = result.full_search_turns
                tt = result.total_turns
                pct = 100 * fs / tt if tt > 0 else 0
                print(f"  Playout cap: {fs}/{tt} full-search turns ({pct:.0f}%)")

            buf_start = time.time()
            boards, reserves, policies, values, weights, value_only, capture_turn, mid_capture_turn = result.training_data()
            dataset.add_batch(
                board_tensors=np.array(boards),
                reserve_vectors=np.array(reserves),
                policy_targets=np.array(policies),
                value_targets=np.array(values),
                weights=np.array(weights),
                value_only=list(value_only),
                capture_turn=list(capture_turn),
                mid_capture_turn=list(mid_capture_turn),
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

            # --- Sample boards ---
            samples = result.sample_boards()
            if samples:
                import random
                picks = random.sample(samples, min(4, len(samples)))
                board_strs = [b for _, b in picks]
                labels = [lbl for lbl, _ in picks]
                rendered = _render_boards_horizontally(board_strs, labels=labels)
                print("\n".join("    " + line for line in rendered.split("\n")))

            # --- Training ---
            avg_gl = f"{sum(lengths) / len(lengths):.1f}" if lengths else ""
            med_gl = str(_median(lengths)) if lengths else ""
            min_gl = str(min(lengths)) if lengths else ""
            max_gl = str(max(lengths)) if lengths else ""

            losses = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}
            for epoch in range(epochs_per_gen):
                losses = self.trainer.train_epoch(dataset, batch_size=batch_size)
                lr = self.trainer._current_lr
                total_s = f"{losses['total_loss']:.4f}"
                policy_s = f"{losses['policy_loss']:.4f}"
                value_s = f"{losses['value_loss']:.4f}"
                place_pol_s = f"{losses['place_policy_loss']:.4f}"
                cap_pol_s = f"{losses['capture_policy_loss']:.4f}"
                print(
                    f"  Epoch {epoch + 1}: loss={_cr(total_s)} "
                    f"(policy={_cy(policy_s)} place_pol={_cy(place_pol_s)} cap_pol={_cy(cap_pol_s)}, "
                    f"value={_cy(value_s)}, "
                    f"lr={lr:.4f})"
                )

                duration = time.time() - iter_start

                # --- Log each epoch ---
                with open(log_path, "a") as f:
                    f.write(
                        f"{generation},{epoch + 1},"
                        f"{simulations},{games_per_gen},{result.num_samples},{len(dataset)},"
                        f"{result.wins_p1},{result.wins_p2},{result.draws},"
                        f"{result.wins_white},{result.wins_grey},{result.wins_black},{result.wins_combo},"
                        f"{avg_gl},{med_gl},{min_gl},{max_gl},"
                        f"{result.isolation_captures},{result.jump_captures},"
                        f"{losses['total_loss']:.6f},{losses['policy_loss']:.6f},"
                        f"{losses['value_loss']:.6f},"
                        f"{losses['place_policy_loss']:.6f},{losses['capture_policy_loss']:.6f},"
                        f"{losses['place_value_loss']:.6f},{losses['capture_value_loss']:.6f},"
                        f"{lr:.6f},{duration:.1f},"
                        f"{csv_comment(comment)}\n"
                    )
                comment = ""

            # --- Save model ---
            metadata = {**train_params, "lr": lr}
            save_checkpoint(self.model, self.model_path, iteration=generation, metadata=metadata)
            print(f"  Model saved: {self.model_path} (gen {generation})")

            # --- Checkpoint ---
            if generation % checkpoint_every == 0:
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"{self.model_name}_gen{generation:05d}.pt"
                )
                save_checkpoint(self.model, ckpt_path, iteration=generation, metadata=metadata)
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
            sep.join(lbl + " " * (board_widths[i] - visual_len(lbl)) for i, lbl in enumerate(labels))
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
        all_lines.append(sep.join(parts))

    return "\n".join(all_lines)
