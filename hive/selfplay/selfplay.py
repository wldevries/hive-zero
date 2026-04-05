"""Self-play training loop for Hive AI."""

from __future__ import annotations
import torch
import numpy as np
import os
import time
from typing import Optional

from shared.lr_scheduler import LRScheduler
from shared.training_log import csv_comment

LOG_HEADER = (
    "iter,mode,simulations,wins_w,wins_b,draws,resignations,positions,buffer,"
    "loss,policy_loss,value_loss,qd_loss,lr,duration_s,comment,qe_loss,mob_loss,"
    "avg_game_len,med_game_len,avg_decisive_len,med_decisive_len\n"
)

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

from ..encoding.move_encoder import policy_size as compute_policy_size


from ..nn.model import HiveNet, create_model, save_checkpoint, load_checkpoint, export_onnx
from ..nn.training import HiveDataset, Trainer


class SelfPlayTrainer:
    """Full self-play training pipeline."""

    def __init__(self,
                 model_path: str = "model.pt",
                 device: str = "cpu",
                 num_blocks: int = 6,
                 channels: int = 64,
                 num_attention_layers: int = 0,
                 checkpoint_dir: str = "checkpoints",
                 lr: float = 0.02,
                 lr_scheduler: Optional[LRScheduler] = None,
                 grid_size: int = 23):
        self.model_path = model_path
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.num_blocks = num_blocks
        self.channels = channels
        self.grid_size = grid_size
        self.start_generation = 0

        if os.path.exists(model_path):
            self.model, ckpt = load_checkpoint(model_path)
            self.start_generation = ckpt.get("generation", 0)
            blocks = len(self.model.res_blocks)
            ch = self.model.input_conv.out_channels
            attn = len(self.model.attention_layers)
            gs = self.model.grid_size
            params = sum(p.numel() for p in self.model.parameters())
            print(f"Resumed from {model_path} (generation {self.start_generation}, "
                  f"{blocks} blocks, {ch} channels, {attn} attn layers, grid {gs}x{gs}, {params/1e6:.2f}M params)")
            if blocks != num_blocks or ch != channels:
                print(f"  WARNING: --blocks {num_blocks} --channels {channels} ignored "
                      f"(checkpoint shape: {blocks} blocks, {ch} channels)")
            self.grid_size = gs
        else:
            self.model = create_model(num_blocks, channels, grid_size=grid_size,
                                      num_attention_layers=num_attention_layers)
            print(f"Created new model ({num_blocks} blocks, {channels} channels, "
                  f"{num_attention_layers} attn layers, grid {grid_size}x{grid_size})")

        self.model.to(device)
        self.lr_scheduler = lr_scheduler
        self.trainer = Trainer(self.model, device=device, lr=lr)

    def run(self, num_generations: int | None = None, games_per_gen: int = 10,
            simulations: int = 100, epochs_per_gen: int = 1,
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
            leaf_batch_size: int = 1,
            fixed_batch_size: int | None = None,
            temperature: float = 1.0,
            temp_threshold: int = 30,
            c_puct: float = 1.5,
            dir_alpha: float = 0.3,
            dir_epsilon: float = 0.25,
            random_opening_moves: int | tuple[int, int] = 0,
            opening_games_csv: str | None = None,
            opening_boardspace_dir: str | None = None,
            boardspace_frac: float = 1.0,
            opening_min_elo: float = 1600.0,
            skip_timeout_games: bool = False,
            augment_symmetry: bool = False,
            comment: str = "",
            use_ort: bool = False,
            value_loss_scale: float = 1.0):
        """Run the full training loop.

        Args:
            eval_config: If set, run evaluation vs Mzinga periodically.
                Keys: every, games, simulations, mzinga_path, mzinga_time.
            checkpoint_eval_games: Games per checkpoint self-eval (default 2x games_per_gen).
            checkpoint_eval_simulations: Simulations for checkpoint eval.
                Defaults to same as `simulations`.
            playout_cap_p: Probability of full search per turn (0=disabled,
                all turns are full). KataGo-style playout cap randomization.
            fast_cap: Number of simulations for fast-search turns.
        """
        from .rust_selfplay import RustParallelSelfPlay
        start_time = time.time()
        self._comment = comment

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
            "leaf_batch_size": leaf_batch_size,
            "augment_symmetry": augment_symmetry,
        }

        # Training log (CSV, truncated on fresh start)
        self._log_path = f"{self.model_name}_log.csv"
        log_path = self._log_path
        if self.start_generation == 0:
            with open(log_path, "w") as f:
                f.write(LOG_HEADER)

        # Bootstrap eval: if best_model.pt doesn't exist, run it immediately
        # rather than waiting until the next checkpoint generation.
        if checkpoint_eval:
            best_model_path = os.path.join(os.path.dirname(self.model_path) or ".", "best_model.pt")
            if not os.path.exists(best_model_path):
                eval_sims = checkpoint_eval_simulations if checkpoint_eval_simulations is not None else simulations
                eval_games = checkpoint_eval_games if checkpoint_eval_games is not None else 2 * games_per_gen
                self._run_checkpoint_eval(self.start_generation, eval_sims, eval_games)

        # Replay buffer: keep last `replay_window` generations of data (worst case: all games hit max_moves)
        replay_buffer = HiveDataset(max_size=replay_window * games_per_gen * max_moves,
                                     grid_size=self.grid_size)
        replay_buffer.augment_symmetry = augment_symmetry

        # Opening book: load boardspace game sequences if configured
        opening_book: list[list[str]] = []
        if opening_games_csv and opening_boardspace_dir:
            opening_book = self._load_opening_book(
                opening_games_csv, opening_boardspace_dir, opening_min_elo,
            )

        sim_label = ""
        if playout_cap_p > 0:
            sim_label = f" [sims={simulations}, fast={fast_cap}, p={playout_cap_p}]"
        else:
            sim_label = f" [sims={simulations}]"

        generation = self.start_generation

        interrupted = False
        while True:
            if interrupted:
                break

            if (num_generations is not None) and generation >= self.start_generation + num_generations:
                print(f"\nReached target of {num_generations} generations, stopping.")
                break

            if time_limit_minutes is not None:
                elapsed = (time.time() - start_time) / 60.0
                if elapsed >= time_limit_minutes:
                    print(f"\nTime limit reached ({elapsed:.1f}m / {time_limit_minutes}m)")
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
            elapsed_str = f" [{(time.time() - start_time) / 60:.1f}m]" if time_limit_minutes else ""

            mode_label = "MCTS"
            sim_label = (
                f" [sims={simulations}, fast={fast_cap}, p={playout_cap_p}]"
                if playout_cap_p > 0
                else f" [sims={simulations}]"
            )
            opening_label = ""
            if opening_book:
                rand_str = f"{random_opening_moves[0]}-{random_opening_moves[1]}" if isinstance(random_opening_moves, tuple) else random_opening_moves
                opening_label = f" [book={boardspace_frac:.0%} rand={rand_str}]" if random_opening_moves else f" [book={boardspace_frac:.0%}]"
            elif random_opening_moves:
                rand_str = f"{random_opening_moves[0]}-{random_opening_moves[1]}" if isinstance(random_opening_moves, tuple) else random_opening_moves
                opening_label = f" [rand={rand_str}]"
            print(
                f"\n=== {_cc(self.model_name)}  Gen {generation} [{mode_label}]{sim_label}{opening_label}{elapsed_str} ==="
            )

            # Generate self-play games
            torch.cuda.empty_cache()
            _print_vram("pre-play")

            _book_opening_moves = random_opening_moves[1] if isinstance(random_opening_moves, tuple) else random_opening_moves
            opening_sequences = _make_opening_sequences(
                games_per_gen, opening_book, boardspace_frac, _book_opening_moves,
            ) if opening_book else None

            sp = RustParallelSelfPlay(
                model=self.model, device=self.device,
                simulations=simulations, max_moves=max_moves,
                temperature=temperature, temp_threshold=temp_threshold,
                c_puct=c_puct, dir_alpha=dir_alpha, dir_epsilon=dir_epsilon,
                resign_threshold=resign_threshold,
                resign_moves=resign_moves,
                resign_min_moves=resign_min_moves,
                calibration_frac=calibration_frac,
                playout_cap_p=playout_cap_p,
                fast_cap=fast_cap,
                leaf_batch_size=leaf_batch_size,
                fixed_batch_size=fixed_batch_size,
                random_opening_moves=random_opening_moves,
                skip_timeout_games=skip_timeout_games,
            )

            ort_path = None
            if use_ort:
                ort_path = self.model_path.rsplit(".", 1)[0] + ".onnx"
                # Re-export if missing or if batch size is fixed (static ONNX must match).
                needs_export = not os.path.exists(ort_path)
                if not needs_export and fixed_batch_size:
                    import onnx
                    try:
                        m = onnx.load(ort_path)
                        dim = m.graph.input[0].type.tensor_type.shape.dim[0]
                        baked = dim.dim_value  # 0 means dynamic
                        needs_export = (baked != fixed_batch_size)
                    except Exception:
                        needs_export = True
                if needs_export:
                    print(f"  ORT requested but {ort_path} not found or stale, exporting...")
                    export_onnx(self.model, ort_path, batch_size=fixed_batch_size)

            if not use_ort:
                self.model.eval()
                self.model.bfloat16()
            try:
                result = sp.play_games(games_per_gen, opening_sequences=opening_sequences,
                                       onnx_path=ort_path)
            except BaseException as e:
                if not isinstance(e, KeyboardInterrupt) and "KeyboardInterrupt" not in str(e):
                    raise
                print("\n  Ctrl-C received, exiting cleanly.")
                if not use_ort:
                    self.model.float()
                break
            if not use_ort:
                self.model.float()
            play_time = time.time() - gen_start
            _print_vram("post-play")

            # Insert training data into replay buffer
            buf_start = time.time()
            (boards, reserves, policies, values, weights, value_only_flags, policy_only_flags,
             aux_targets) = result.training_data()
            # aux_targets is [N, 6]: [my_qd, opp_qd, my_qe, opp_qe, my_mob, opp_mob]
            replay_buffer.add_batch(boards, reserves, policies, values, weights,
                                    value_only_flags, policy_only_flags,
                                    my_queen_danger=aux_targets[:, 0],
                                    opp_queen_danger=aux_targets[:, 1],
                                    my_queen_escape=aux_targets[:, 2],
                                    opp_queen_escape=aux_targets[:, 3],
                                    my_mobility=aux_targets[:, 4],
                                    opp_mobility=aux_targets[:, 5])
            buf_time = time.time() - buf_start

            total_positions = result.num_samples
            fast_positions = sum(value_only_flags)
            game_time = time.time() - gen_start
            fast_str = ""
            if fast_positions > 0:
                fast_str = f" ({fast_positions} value-only)"

            wins_w = result.wins_w
            wins_b = result.wins_b
            draws = result.draws
            resignations = result.resignations
            resign_suffix = f" (resigned={resignations})" if resignations else ""
            print(f"  Results: W={_cg(wins_w)} B={_cg(wins_b)} D/unfinished={_cy(draws)}{resign_suffix}")

            # Game length stats
            finished_games_all = result.final_games()
            game_lengths = sorted([g.move_count for g in finished_games_all])
            decisive_lengths = sorted([g.move_count for g in finished_games_all
                                       if (g.state if isinstance(g.state, str) else g.state.value) in ("WhiteWins", "BlackWins")])
            if game_lengths:
                median = game_lengths[len(game_lengths) // 2]
                avg = sum(game_lengths) / len(game_lengths)
                print(f"  Game length: avg={avg:.0f} med={median} min={game_lengths[0]} max={game_lengths[-1]}", end="")
                if decisive_lengths:
                    d_med = decisive_lengths[len(decisive_lengths) // 2]
                    d_avg = sum(decisive_lengths) / len(decisive_lengths)
                    print(f"  decisive: avg={d_avg:.0f} med={d_med}")
                else:
                    print()
            if result.use_playout_cap:
                print(f"  Playout cap: {result.full_search_turns}/{result.total_turns} full-search turns "
                      f"({100*result.full_search_turns/max(result.total_turns,1):.0f}%)")
            if result.calibration_would_resign > 0:
                print(f"  Calibration: {result.calibration_would_resign}/{result.calibration_total} "
                      f"would resign, {result.calibration_false_positives} false positives")
            print(f"  {games_per_gen} games: {total_positions} new positions{fast_str} "
                  f"(play={play_time:.1f}s, buf={buf_time:.1f}s, "
                  f"{total_positions / max(game_time, 0.1):.0f} pos/s), "
                  f"buffer: {len(replay_buffer)}")

            # Show boards of decisive games
            decisive_games = []
            for g in finished_games_all:
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
                    board_strs.append(g.render_board())
                rendered = _render_boards_horizontally(board_strs, labels=labels)
                print('\n'.join('    ' + line for line in rendered.split('\n')))

            # Train on replay buffer
            train_start = time.time()
            try:
                for epoch in range(epochs_per_gen):
                    losses = self.trainer.train_epoch(replay_buffer, batch_size=batch_size,
                                                       value_loss_scale=value_loss_scale)
                    lr = self.trainer._current_lr
                    total_s = f"{losses['total_loss']:.4f}"
                    policy_s = f"{losses['policy_loss']:.4f}"
                    value_s = f"{losses['value_loss']:.4f}"
                    qd_s = f"{losses.get('qd_loss', 0):.4f}"
                    qe_s = f"{losses.get('qe_loss', 0):.4f}"
                    mob_s = f"{losses.get('mob_loss', 0):.4f}"
                    print(f"  Epoch {epoch + 1}: loss={_cr(total_s)} "
                          f"(policy={_cy(policy_s)}, value={_cy(value_s)}, "
                          f"qd={_cy(qd_s)}, qe={_cy(qe_s)}, mob={_cy(mob_s)}, lr={lr})")
            except KeyboardInterrupt:
                print("\n  Ctrl-C during training, saving model...")
                interrupted = True
            train_time = time.time() - train_start
            _print_vram("post-train")

            # Log to CSV
            avg_gl = sum(game_lengths) / len(game_lengths) if game_lengths else 0
            med_gl = game_lengths[len(game_lengths) // 2] if game_lengths else 0
            avg_dl = sum(decisive_lengths) / len(decisive_lengths) if decisive_lengths else 0
            med_dl = decisive_lengths[len(decisive_lengths) // 2] if decisive_lengths else 0
            with open(log_path, "a") as f:
                f.write(f"{generation},MCTS,{simulations},"
                        f"{wins_w},{wins_b},{draws},{resignations},{total_positions},"
                        f"{len(replay_buffer)},{losses['total_loss']:.6f},"
                        f"{losses['policy_loss']:.6f},{losses['value_loss']:.6f},"
                        f"{losses.get('qd_loss', 0):.6f},"
                        f"{lr:.8f},{play_time + train_time:.1f},{csv_comment(self._comment)},"
                        f"{losses.get('qe_loss', 0):.6f},{losses.get('mob_loss', 0):.6f},"
                        f"{avg_gl:.1f},{med_gl},{avg_dl:.1f},{med_dl}\n")
            self._comment = ""

            # --- Save model ---
            metadata = {**train_params, "lr": self.trainer._current_lr}
            save_checkpoint(self.model, self.model_path, generation, metadata)
            onnx_path = self.model_path.rsplit(".", 1)[0] + ".onnx"
            export_onnx(self.model, onnx_path, batch_size=fixed_batch_size)

            # --- Checkpoint ---            
            if generation % checkpoint_every == 0:
                ckpt_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_gen{generation:05d}.pt")
                save_checkpoint(self.model, ckpt_path, generation, metadata)
                print(f"  Checkpoint saved to {ckpt_path}")
                if checkpoint_eval:
                    eval_sims = checkpoint_eval_simulations if checkpoint_eval_simulations is not None else simulations
                    eval_games = checkpoint_eval_games if checkpoint_eval_games is not None else 2 * games_per_gen
                    self._run_checkpoint_eval(generation, eval_sims, eval_games)

            print(f"  Model saved to {self.model_path} (generation {generation})")

            # Periodic evaluation vs Mzinga
            if eval_config and generation % eval_config["every"] == 0:
                self._run_eval(eval_config, generation, replay_buffer)


    def _load_opening_book(
        self, games_csv: str, boardspace_dir: str, min_elo: float = 1600.0,
    ) -> list[list[str]]:
        """Load boardspace game move sequences for use as opening positions.

        Returns a list of move-string lists (one per game), normalized to Rust
        canonical UHP format. Short games (< 12 moves) are skipped.
        """
        import zipfile
        from tqdm import tqdm
        from ..supervised.pretrain import load_filtered_games, build_zip_index
        from engine_zero import parse_sgf_moves as parse_moves

        elo_csv = os.path.join(os.path.dirname(games_csv) or ".", "player_elo.csv")
        games = load_filtered_games(games_csv, elo_csv, min_elo=min_elo, min_games=20)
        zip_index = build_zip_index(boardspace_dir)

        sequences: list[list[str]] = []
        errors = 0
        for zip_file, sgf_name, _result in tqdm(games, desc="Loading opening book", unit="game"):
            zip_path = zip_index.get(zip_file)
            if not zip_path:
                continue
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    content = zf.read(sgf_name).decode('iso-8859-1')
                moves = [_normalize_uhp_move(m) for m in parse_moves(content)
                         if m != 'pass']
                if len(moves) >= 12:
                    sequences.append(moves)
            except Exception:
                errors += 1

        print(f"  Opening book: {len(sequences)} games loaded ({errors} errors)")
        return sequences

    def _find_prev_checkpoint(self, current_generation: int) -> str | None:
        """Return path of the latest checkpoint strictly before current_generation, or None."""
        import glob as _glob
        import re
        pattern = os.path.join(self.checkpoint_dir, f"{self.model_name}_*.pt")
        candidates = []
        # Match both old _iter{N} and new _gen{N} formats
        pat = re.compile(rf"^{re.escape(self.model_name)}_(?:iter|gen)(\d+)\.pt$")
        for path in _glob.glob(pattern):
            m = pat.match(os.path.basename(path))
            if m:
                num = int(m.group(1))
                if num < current_generation:
                    candidates.append((num, path))
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[0])[1]

    def _run_checkpoint_eval(self, generation: int, simulations: int, num_games: int):
        """Pit current model against best_model.pt. Winner becomes new best and model.pt."""
        import shutil
        from ..eval.engine_match import ModelEngine, run_parallel_match

        WIN_THRESHOLD = 0.5
        best_model_path = os.path.join(os.path.dirname(self.model_path) or ".", "best_model.pt")

        if not os.path.exists(best_model_path):
            # Bootstrap: pit model.pt against the previous checkpoint to find the best.
            prev_ckpt = self._find_prev_checkpoint(generation)
            if prev_ckpt is None:
                print(f"  No best_model.pt and no prior checkpoint — current model is now best")
                save_checkpoint(self.model, best_model_path, generation)
                shutil.copy2(best_model_path, self.model_path)
                return
            print(f"\n--- Bootstrap eval: model.pt vs {os.path.basename(prev_ckpt)} ({num_games} games) ---")
            self.model.eval()
            engine1 = ModelEngine(model=self.model, device=self.device,
                                  simulations=simulations, name=f"current-g{generation}")
            prev_model, prev_ckpt_data = load_checkpoint(prev_ckpt)
            prev_gen = prev_ckpt_data.get("generation", 0)
            prev_model.to(self.device)
            prev_model.eval()
            engine2 = ModelEngine(model=prev_model, device=self.device,
                                  simulations=simulations, name=f"prev-g{prev_gen}")
            try:
                summary = run_parallel_match(engine1, engine2, num_games=num_games, max_moves=200, verbose=False, show_progress=True)
            finally:
                self.model.train()
            score = summary["engine1_score"]
            w, d, l = summary["engine1_wins"], summary["draws"], summary["engine2_wins"]
            if score >= 0.5:
                winner_label = engine1.name
                save_checkpoint(self.model, best_model_path, generation)
            else:
                winner_label = engine2.name
                shutil.copy2(prev_ckpt, best_model_path)
            shutil.copy2(best_model_path, self.model_path)
            print(f"  {_cg(w)}W/{_cy(d)}D/{_cr(l)}L → best model: {winner_label}")
            with open(self._log_path, "a") as f:
                f.write(f"{generation},pit-bootstrap,{simulations},{w},{l},{d},0,0,0,"
                        f"{score:.6f},0,0,0,0,0,{csv_comment(self._comment)},0,0\n")
            return

        print(f"\n--- Checkpoint eval: challenger (g{generation}) vs best ({num_games} games) ---")
        self.model.eval()

        challenger = ModelEngine(
            model=self.model, device=self.device,
            simulations=simulations, name=f"challenger-g{generation}",
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
            save_checkpoint(self.model, best_model_path, generation)
            print(f" → NEW BEST (gen {generation})")
        else:
            print(f" → no improvement (defender holds)")

        # model.pt always mirrors best_model.pt so fresh restarts use the best known weights
        shutil.copy2(best_model_path, self.model_path)

        with open(self._log_path, "a") as f:
            f.write(f"{generation},pit,{simulations},{w},{l},{d},0,0,0,"
                    f"{score:.6f},0,0,0,0,0,{csv_comment(self._comment)},0,0\n")

    def _run_eval(self, eval_config: dict, generation: int, replay_buffer=None):
        """Run evaluation games against Mzinga and feed samples back."""
        from ..eval.engine_match import EngineConfig, UHPProcess, ModelEngine, run_match

        print(f"\n--- Eval vs Mzinga (gen {generation}) ---")
        self.model.eval()

        our_engine = ModelEngine(
            model=self.model, device=self.device,
            simulations=eval_config["simulations"],
            name=f"HiveZero-g{generation}",
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
            with open(self._log_path, "a") as f:
                f.write(f"{generation},eval,{eval_config['simulations']},{w},{l},{d},0,{len(samples)},"
                        f"{len(replay_buffer) if replay_buffer else 0},"
                        f"{score:.6f},0,0,0,0,0,{csv_comment(self._comment)},0,0\n")
        except Exception as e:
            print(f"  Eval failed: {e}")
        finally:
            self.model.train()


from ..uhp import normalize_move as _normalize_uhp_move


def _make_opening_sequences(
    num_games: int,
    book: list[list[str]],
    boardspace_frac: float,
    opening_moves: int,
) -> list[list[str]]:
    """Generate per-game opening sequences from the book.

    boardspace_frac fraction of games get a boardspace prefix of exactly
    opening_moves moves (sampled from a random human game); the rest get an
    empty list (Rust will fall back to random_opening_moves for those games).
    Games shorter than opening_moves moves are skipped.
    """
    import random as _random
    eligible = [g for g in book if len(g) >= opening_moves]
    sequences = []
    for _ in range(num_games):
        if eligible and _random.random() < boardspace_frac:
            game_moves = _random.choice(eligible)
            sequences.append(game_moves[:opening_moves])
        else:
            sequences.append([])
    return sequences


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
    parser.add_argument("--generations", type=int, default=100)
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
        num_generations=args.generations, games_per_gen=args.games,
        simulations=args.simulations, epochs_per_gen=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
