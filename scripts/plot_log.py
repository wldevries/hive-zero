#!/usr/bin/env python3
"""Plot training performance log (perf_log.csv)."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_perf_log(csv_path: Path, output: Path | None = None) -> None:
    df = pd.read_csv(csv_path, skipinitialspace=True)

    total = df["wins_w"] + df["wins_b"] + df["draws"]
    df["win_w_pct"] = df["wins_w"] / total * 100
    df["win_b_pct"] = df["wins_b"] / total * 100
    df["draw_pct"] = df["draws"] / total * 100

    iters = df["iter"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(csv_path.name, fontsize=13)

    # --- Win percentages ---
    ax1.plot(iters, df["win_w_pct"], label="White wins %", color="steelblue")
    ax1.plot(iters, df["win_b_pct"], label="Black wins %", color="tomato")
    ax1.plot(iters, df["draw_pct"], label="Draws %", color="gray", linestyle="--")
    ax1.set_ylabel("Percentage")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Game outcomes per iteration")

    # Mark any row with a comment
    commented = df[df["comment"].notna() & df["comment"].astype(str).str.strip().ne("")]
    for _, row in commented.iterrows():
        label = str(row["comment"]).strip()
        for ax in (ax1, ax2):
            ax.axvline(row["iter"], color="green", linestyle=":", alpha=0.6, linewidth=1)
        ax1.text(row["iter"], 98, label, fontsize=7, color="green", va="top", ha="left", rotation=90)

    # --- Loss values ---
    ax2.plot(iters, df["loss"], label="Total loss", color="black", linewidth=1.5)
    ax2.plot(iters, df["policy_loss"], label="Policy loss", color="royalblue", linewidth=1)
    ax2.plot(iters, df["value_loss"], label="Value loss", color="darkorange", linewidth=1)
    ax2.plot(iters, df["qd_loss"], label="Queen danger loss", color="mediumpurple", linewidth=1)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Training losses")

    ax1.set_xlim(left=iters.min())

    plt.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        print(f"Saved to {output}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot perf_log.csv")
    parser.add_argument("csv", nargs="?", default="perf_log.csv", help="Path to CSV file")
    parser.add_argument("-o", "--output", help="Save plot to file instead of showing it")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    plot_perf_log(csv_path, Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
