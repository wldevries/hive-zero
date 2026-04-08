"""Hyperparameter sweep for Tic-Tac-Toe self-play training.

Runs multiple experiments with different hyperparameter combinations,
each for a fixed number of generations. Results are logged to separate
CSV files and a summary is printed at the end.

Usage:
    uv run python -m tictactoe.sweep [--device cuda] [--gens 50] [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Experiment:
    """A single hyperparameter configuration to test."""
    name: str
    simulations: int = 100
    games: int = 200
    lr: float = 0.001
    optimizer: str = "sgd"
    temperature: float = 1.0
    temp_threshold: int = 9
    c_puct: float = 1.5
    dir_alpha: float = 0.5
    dir_epsilon: float = 0.25
    replay_window: int = 3
    value_loss_scale: float = 1.0
    blocks: int = 2
    channels: int = 32
    augment_symmetry: bool = False


def build_sweep() -> list[Experiment]:
    """Define the hyperparameter sweep.

    Full grid over the axes most likely to matter.
    No fixed baseline — every combination is tested so the best
    combo becomes the new default.

    2 × 2 × 2 × 2 = 16 experiments.
    """
    experiments: list[Experiment] = []

    for sims in [200, 800]:
        for blocks, channels in [(2, 32), (4, 64)]:
            for replay in [1, 3]:
                for dir_eps in [0.0, 0.25]:
                    name = (
                        f"s{sims}_b{blocks}c{channels}"
                        f"_r{replay}_d{dir_eps}"
                    )
                    experiments.append(Experiment(
                        name=name,
                        simulations=sims,
                        blocks=blocks,
                        channels=channels,
                        replay_window=replay,
                        dir_epsilon=dir_eps,
                    ))

    return experiments


def run_experiment(exp: Experiment, device: str, num_generations: int, sweep_dir: str):
    """Run a single experiment."""
    from tictactoe.selfplay.selfplay import SelfPlayTrainer

    model_path = os.path.join(sweep_dir, f"{exp.name}.pt")

    trainer = SelfPlayTrainer(
        model_path=model_path,
        device=device,
        num_blocks=exp.blocks,
        channels=exp.channels,
        lr=exp.lr,
        checkpoint_dir=os.path.join(sweep_dir, "checkpoints", exp.name),
        history_length=1,
        optimizer=exp.optimizer,
    )
    trainer.run(
        num_generations=num_generations,
        games_per_gen=exp.games,
        simulations=exp.simulations,
        epochs_per_gen=1,
        batch_size=64,
        max_moves=9,
        replay_window=exp.replay_window,
        checkpoint_every=999,  # don't checkpoint during sweep
        temperature=exp.temperature,
        temp_threshold=exp.temp_threshold,
        c_puct=exp.c_puct,
        dir_alpha=exp.dir_alpha,
        dir_epsilon=exp.dir_epsilon,
        value_loss_scale=exp.value_loss_scale,
        augment_symmetry=exp.augment_symmetry,
        comment=f"sweep:{exp.name}",
    )


def summarize_sweep(sweep_dir: str):
    """Parse all experiment logs and print a comparison table."""
    import csv

    print("\n" + "=" * 100)
    print("SWEEP SUMMARY")
    print("=" * 100)

    results = []
    log_files = sorted(f for f in os.listdir(sweep_dir) if f.endswith("_log.csv"))

    for log_file in log_files:
        path = os.path.join(sweep_dir, log_file)
        name = log_file.replace("_log.csv", "")

        gens = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gens.append(row)

        if not gens:
            continue

        # Aggregate stats over last 10 generations
        tail = gens[-10:]
        total_games = sum(int(g["games"]) for g in tail)
        total_p1 = sum(int(g["wins_p1"]) for g in tail)
        total_p2 = sum(int(g["wins_p2"]) for g in tail)
        total_draws = sum(int(g["draws"]) for g in tail)

        draw_pct = total_draws / total_games * 100 if total_games > 0 else 0
        p1_pct = total_p1 / total_games * 100 if total_games > 0 else 0
        p2_pct = total_p2 / total_games * 100 if total_games > 0 else 0

        avg_vloss = np.mean([float(g["value_loss"]) for g in tail])
        avg_ploss = np.mean([float(g["policy_loss"]) for g in tail])

        # Check stability: draw rate over windows of 5 gens
        draw_rates_over_time = []
        for i in range(0, len(gens), 5):
            chunk = gens[i:i+5]
            if chunk:
                cg = sum(int(g["games"]) for g in chunk)
                cd = sum(int(g["draws"]) for g in chunk)
                draw_rates_over_time.append(cd / cg * 100 if cg > 0 else 0)

        # Stability = std of draw rates over 5-gen windows
        stability = np.std(draw_rates_over_time) if len(draw_rates_over_time) > 1 else 0

        results.append({
            "name": name,
            "draw_pct": draw_pct,
            "p1_pct": p1_pct,
            "p2_pct": p2_pct,
            "vloss": avg_vloss,
            "ploss": avg_ploss,
            "stability": stability,
            "num_gens": len(gens),
        })

    # Sort by draw rate (higher is better for TTT)
    results.sort(key=lambda r: r["draw_pct"], reverse=True)

    # Print table
    header = f"{'Experiment':<25} {'Draws%':>7} {'P1%':>6} {'P2%':>6} {'VLoss':>8} {'PLoss':>8} {'Stab':>6} {'Gens':>5}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<25} {r['draw_pct']:>6.1f}% {r['p1_pct']:>5.1f}% {r['p2_pct']:>5.1f}% "
            f"{r['vloss']:>8.4f} {r['ploss']:>8.4f} {r['stability']:>5.1f}% {r['num_gens']:>5}"
        )

    print(f"\n(Draws%/P1%/P2% = last 10 gens. Stab = std of draw% over 5-gen windows, lower=more stable)")


def main():
    parser = argparse.ArgumentParser(description="Tic-Tac-Toe hyperparameter sweep")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gens", type=int, default=50, help="Generations per experiment")
    parser.add_argument("--sweep-dir", type=str, default="sweep_ttt",
                        help="Directory for sweep outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print experiments without running")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only experiments matching this substring")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only print summary of existing results")
    args = parser.parse_args()

    os.makedirs(args.sweep_dir, exist_ok=True)

    if args.summary_only:
        summarize_sweep(args.sweep_dir)
        return

    experiments = build_sweep()

    if args.only:
        experiments = [e for e in experiments if args.only in e.name]

    print(f"Sweep: {len(experiments)} experiments, {args.gens} generations each\n")
    for i, exp in enumerate(experiments):
        print(f"  [{i+1:2d}] {exp.name:<25} sims={exp.simulations} lr={exp.lr} "
              f"cpuct={exp.c_puct} dir_eps={exp.dir_epsilon} "
              f"replay={exp.replay_window} vls={exp.value_loss_scale} "
              f"sym={exp.augment_symmetry}")

    if args.dry_run:
        print("\n(dry run — not executing)")
        return

    print()
    for i, exp in enumerate(experiments):
        # Skip if already completed
        log_path = os.path.join(args.sweep_dir, f"{exp.name}_log.csv")
        if os.path.exists(log_path):
            import csv
            with open(log_path) as f:
                existing_gens = sum(1 for _ in csv.DictReader(f))
            if existing_gens >= args.gens:
                print(f"\n>>> [{i+1}/{len(experiments)}] {exp.name} — already done ({existing_gens} gens), skipping")
                continue

        print(f"\n>>> [{i+1}/{len(experiments)}] Running: {exp.name}")
        t0 = time.time()

        # Override model log path so it writes to sweep dir
        orig_cwd = os.getcwd()
        os.chdir(args.sweep_dir)
        try:
            run_experiment(exp, args.device, args.gens, args.sweep_dir)
        except Exception as e:
            print(f"  !!! Experiment {exp.name} failed: {e}")
        finally:
            os.chdir(orig_cwd)

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.0f}s")

    summarize_sweep(args.sweep_dir)


if __name__ == "__main__":
    main()
