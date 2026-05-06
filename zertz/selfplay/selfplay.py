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

from shared.lr_scheduler import LRScheduler
from shared.optimizer_defaults import resolve_resumed_lr
from shared.selfplay_config import MctsConfig, PlayoutCapConfig
from shared.training_log import csv_comment

from ..nn.model import ZertzNet, _describe_trunk, create_model, load_checkpoint, save_checkpoint, export_onnx
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
    "lr,duration_s,"
    "mcts_top1_mean,mcts_top1_std,mcts_depth_mean,mcts_depth_std,mcts_moves_mean,mcts_moves_std,"
    "comment\n"
)


def _median(lst):
    return int(statistics.median(lst)) if lst else 0


class SelfPlayTrainer:
    """Full self-play training pipeline for Zertz."""

    def __init__(
        self,
        name: str = "zertz",
        device: str = "cuda",
        model_config: Optional[dict] = None,
        lr: Optional[float] = None,
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

        ckpt: dict = {}
        if os.path.exists(self.model_path):
            self.model, ckpt = load_checkpoint(self.model_path)
            self.start_generation = ckpt.get("generation", 0)
            ch = self.model.input_conv.out_channels
            trunk_desc = _describe_trunk(self.model.trunk_spec)
            params = sum(p.numel() for p in self.model.parameters())
            print(
                f"Resumed from {self.model_path} (gen {self.start_generation}, "
                f"{ch}ch, {trunk_desc}, {params / 1e6:.2f}M params)"
            )
        else:
            os.makedirs(self.model_dir, exist_ok=True)
            self.model = create_model(model_config)
            ch = self.model.input_conv.out_channels
            trunk_desc = _describe_trunk(self.model.trunk_spec)
            params = sum(p.numel() for p in self.model.parameters())
            print(
                f"New model '{name}': {ch}ch, {trunk_desc}, "
                f"{params / 1e6:.2f}M params"
            )

        self.model.to(device)
        self.lr_scheduler = lr_scheduler
        opt_state = ckpt.get("optimizer_state_dict")
        effective_lr, lr_source = resolve_resumed_lr(lr, opt_state)
        self.trainer = Trainer(model=self.model, device=device, lr=effective_lr)
        if opt_state is not None:
            try:
                self.trainer.optimizer.load_state_dict(opt_state)
                for pg in self.trainer.optimizer.param_groups:
                    pg["lr"] = effective_lr
                print(f"  Restored optimizer state (momentum buffers); lr={effective_lr:g} from {lr_source}")
            except (ValueError, KeyError) as e:
                print(f"  Skipped optimizer state (incompatible: {e})")

    def _eval_fn(self, board_tensor_np, reserve_np):
        """NN inference callback for Rust self-play.

        Returns 3-tuple: (place_logits, cap_dir_logits, value)
        as numpy arrays. Rust MCTS computes softmax over legal moves internally.
        """
        board = torch.from_numpy(np.array(board_tensor_np)).to(self.device, dtype=torch.float32)
        reserve = torch.from_numpy(np.array(reserve_np)).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda" if self.device != "cpu" else "cpu",
                dtype=torch.bfloat16,
            ):
                place, cap_dir, value = self.model(board, reserve)
        return (
            place.float().cpu().numpy(),
            cap_dir.float().cpu().numpy(),
            value.float().cpu().numpy().squeeze(1),
        )

    def run(
        self,
        mcts: MctsConfig,
        playout_cap: PlayoutCapConfig = PlayoutCapConfig(),
        num_generations: Optional[int] = None,
        games_per_gen: int = 20,
        epochs_per_gen: int = 1,
        batch_size: int = 256,
        replay_window: int = 8,
        checkpoint_every: int = 10,
        time_limit_minutes: Optional[float] = None,
        comment: str = "",
        augment_symmetry: bool = True,
        use_ort: bool = False,
        value_loss_scale: float = 1.0,
        buf_dir: Optional[str] = None,
    ):
        from engine_zero import ZertzSelfPlaySession

        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write(LOG_HEADER)

        resolved_buf_dir = buf_dir if buf_dir is not None else self.model_dir
        max_buffer = games_per_gen * 65 * replay_window
        dataset = ZertzDataset(max_size=max_buffer, buf_dir=resolved_buf_dir)
        dataset.augment_symmetry = augment_symmetry
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Flat metadata stored in the checkpoint — keep keys stable for any
        # downstream tooling that reads them.
        train_params = {
            "simulations": mcts.simulations,
            "games_per_gen": games_per_gen,
            "epochs_per_gen": epochs_per_gen,
            "batch_size": batch_size,
            "replay_window": replay_window,
            "playout_cap_p": playout_cap.p,
            "fast_cap": playout_cap.fast_cap,
            "temperature": mcts.temperature,
            "temp_threshold": mcts.temp_threshold,
            "c_puct": mcts.c_puct,
            "dir_alpha": mcts.dir_alpha,
            "dir_epsilon": mcts.dir_epsilon,
            "play_batch_size": mcts.play_batch_size,
            "augment_symmetry": augment_symmetry,
            "forced_playouts": mcts.forced_playouts,
            "draw_contempt": mcts.draw_contempt,
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

            # Header
            cap_str = (
                f", fast={playout_cap.fast_cap}, cap={int(playout_cap.p * 100)}%"
                if playout_cap.p > 0
                else ""
            )
            print(
                f"\n=== {_cc(self.name)}  Gen {generation}  [sims={mcts.simulations}{cap_str}] ==="
            )

            onnx_path = None
            if use_ort:
                onnx_path = self.onnx_path
                if not os.path.exists(onnx_path):
                    print(f"  ORT requested but {onnx_path} not found, exporting...")
                    export_onnx(self.model, onnx_path)

            if not use_ort:
                self.model.eval()

            # --- Self-play ---
            session = ZertzSelfPlaySession(
                num_games=games_per_gen,
                **mcts.session_kwargs(),
                **playout_cap.session_kwargs(),
            )

            pbar = tqdm(total=65, unit="turn", desc="  Self-play", leave=False)
            turn = [0]

            def progress_fn(finished, total, active, total_moves):
                turn[0] += 1
                advance = turn[0] - pbar.n
                if advance > 0:
                    pbar.update(advance)
                if turn[0] >= pbar.total:
                    pbar.total = turn[0] + 10
                    pbar.refresh()
                pbar.set_postfix(active=f"{active}/{total}")

            play_start = time.time()
            if onnx_path:
                result = session.play_games(progress_fn=progress_fn, onnx_path=onnx_path)
            else:
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

            ss = result.search_stats
            print(
                f"  MCTS: top-1 {ss.top1_mean:.2f}±{ss.top1_std:.2f}  "
                f"depth {ss.depth_mean:.1f}±{ss.depth_std:.1f}  "
                f"moves {ss.valid_moves_mean:.1f}±{ss.valid_moves_std:.1f}"
            )

            if playout_cap.p > 0:
                fs = result.full_search_turns
                tt = result.total_turns
                pct = 100 * fs / tt if tt > 0 else 0
                print(f"  Playout cap: {fs}/{tt} full-search turns ({pct:.0f}%)")

            buf_start = time.time()
            boards, reserves, policies, values, value_only, capture_turn, mid_capture_turn = result.training_data()
            dataset.add_batch(
                board_tensors=np.array(boards),
                reserve_vectors=np.array(reserves),
                policy_targets=np.array(policies),
                value_targets=np.array(values),
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
                losses = self.trainer.train_epoch(dataset, batch_size=batch_size, value_loss_scale=value_loss_scale)
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

                duration = time.time() - gen_start

                # --- Log each epoch ---
                with open(self.log_path, "a") as f:
                    f.write(
                        f"{generation},{epoch + 1},"
                        f"{mcts.simulations},{games_per_gen},{result.num_samples},{len(dataset)},"
                        f"{result.wins_p1},{result.wins_p2},{result.draws},"
                        f"{result.wins_white},{result.wins_grey},{result.wins_black},{result.wins_combo},"
                        f"{avg_gl},{med_gl},{min_gl},{max_gl},"
                        f"{result.isolation_captures},{result.jump_captures},"
                        f"{losses['total_loss']:.6f},{losses['policy_loss']:.6f},"
                        f"{losses['value_loss']:.6f},"
                        f"{losses['place_policy_loss']:.6f},{losses['capture_policy_loss']:.6f},"
                        f"{losses['place_value_loss']:.6f},{losses['capture_value_loss']:.6f},"
                        f"{lr:.6f},{duration:.1f},"
                        f"{ss.top1_mean:.4f},{ss.top1_std:.4f},"
                        f"{ss.depth_mean:.4f},{ss.depth_std:.4f},"
                        f"{ss.valid_moves_mean:.4f},{ss.valid_moves_std:.4f},"
                        f"{csv_comment(comment)}\n"
                    )
                comment = ""

            # --- Save model ---
            metadata = {**train_params, "lr": lr}
            save_checkpoint(self.model, self.model_path, generation=generation,
                            metadata=metadata, optimizer=self.trainer.optimizer)
            export_onnx(self.model, self.onnx_path)
            print(f"  Model saved: {self.model_path} (gen {generation})")

            # --- Checkpoint ---
            if generation % checkpoint_every == 0:
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"{self.name}_gen{generation:05d}.pt"
                )
                save_checkpoint(self.model, ckpt_path, generation=generation,
                                metadata=metadata, optimizer=self.trainer.optimizer)
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
