#!/usr/bin/env python3
"""Plot training performance log (perf_log.csv)."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_GEOM_FILE = Path(__file__).parent / ".plot_log_geom.json"


def _load_geometry() -> dict | None:
    try:
        return json.loads(_GEOM_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _save_geometry(fig) -> None:
    try:
        win = fig.canvas.manager.window
        # TkAgg: win is a Tk window with geometry() returning "WxH+X+Y"
        geom = win.geometry()
        _GEOM_FILE.write_text(json.dumps({"geom": geom}))
    except Exception:
        pass


def _restore_geometry(fig) -> None:
    geom = _load_geometry()
    if not geom or "geom" not in geom:
        return
    try:
        fig.canvas.manager.window.geometry(geom["geom"])
    except Exception:
        pass


def plot_perf_log(csv_path: Path, output: Path | None = None, rolling: int = 1) -> None:
    df = pd.read_csv(csv_path, skipinitialspace=True, on_bad_lines="warn")

    def smooth(s: pd.Series) -> pd.Series:
        if rolling > 1:
            return s.rolling(rolling, min_periods=1).mean()
        return s

    # Detect format: new (gen/epoch) vs legacy (iter)
    has_gen = "gen" in df.columns
    if has_gen:
        # New format: multiple rows per generation (one per epoch).
        # For self-play stats, use first epoch row per gen (all identical).
        # For losses, use last epoch row per gen (final training state).
        x_col = "gen"
        gen_first = df.groupby("gen").first().reset_index()
        gen_last = df.groupby("gen").last().reset_index()
        x_label = "Generation"
    else:
        x_col = "iter"
        gen_first = df
        gen_last = df
        x_label = "Iteration"

    col_w = "wins_w" if "wins_w" in gen_first.columns else "wins_p1"
    col_b = "wins_b" if "wins_b" in gen_first.columns else "wins_p2"
    total = gen_first[col_w] + gen_first[col_b] + gen_first["draws"]
    gen_first = gen_first.copy()
    gen_first["win_w_pct"] = gen_first[col_w] / total * 100
    gen_first["win_b_pct"] = gen_first[col_b] / total * 100
    gen_first["draw_pct"] = gen_first["draws"] / total * 100

    iters = gen_first[x_col]
    loss_iters = gen_last[x_col]

    # For per-epoch loss plotting, use all rows
    if has_gen:
        # Create a fractional x-axis: gen + (epoch-1)/max_epoch
        max_epoch = df["epoch"].max()
        if max_epoch > 1:
            df = df.copy()
            df["x_loss"] = df["gen"] + (df["epoch"] - 1) / max_epoch
        else:
            df = df.copy()
            df["x_loss"] = df["gen"].astype(float)
        loss_x = df["x_loss"]
        loss_df = df
    else:
        loss_x = gen_last[x_col]
        loss_df = gen_last

    has_game_len = "avg_game_len" in gen_first.columns and pd.to_numeric(gen_first["avg_game_len"], errors="coerce").notna().any()
    has_wincon = "wins_white" in gen_first.columns and pd.to_numeric(gen_first["wins_white"], errors="coerce").notna().any()
    has_capture_ratio = "isolation_captures" in gen_first.columns and pd.to_numeric(gen_first["isolation_captures"], errors="coerce").notna().any()
    has_lr = "lr" in loss_df.columns
    nrows = 2 + has_game_len + has_wincon + has_capture_ratio + has_lr
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 4 * nrows), sharex=True)
    ax_idx = 0
    ax1 = axes[ax_idx]; ax_idx += 1
    ax2 = axes[ax_idx]; ax_idx += 1
    ax3 = axes[ax_idx] if has_game_len else None
    if has_game_len: ax_idx += 1
    ax4 = axes[ax_idx] if has_wincon else None
    if has_wincon: ax_idx += 1
    ax5 = axes[ax_idx] if has_capture_ratio else None
    if has_capture_ratio: ax_idx += 1
    ax6 = axes[ax_idx] if has_lr else None
    fig.suptitle(csv_path.name, fontsize=13)

    # --- Win percentages ---
    label_w = "White wins %" if col_w == "wins_w" else "P1 wins %"
    label_b = "Black wins %" if col_b == "wins_b" else "P2 wins %"
    ax1.plot(iters, smooth(gen_first["win_w_pct"]), label=label_w, color="steelblue")
    ax1.plot(iters, smooth(gen_first["win_b_pct"]), label=label_b, color="tomato")
    ax1.plot(iters, smooth(gen_first["draw_pct"]), label="Draws %", color="gray", linestyle="--")
    trend = np.poly1d(np.polyfit(iters, gen_first["draw_pct"], 1))
    ax1.plot(iters, trend(iters), color="gray", linestyle="-", linewidth=1, alpha=0.5, label="Draws trend")
    ax1.set_ylabel("Percentage")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.1), ncol=4, mode="expand", borderaxespad=0, fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Game outcomes per generation")

    # Mark any row with a comment
    comment_col = gen_first["comment"] if "comment" in gen_first.columns else pd.Series(dtype=str)
    commented = gen_first[comment_col.notna() & comment_col.astype(str).str.strip().ne("")]
    for _, row in commented.iterrows():
        label = str(row["comment"]).strip()
        for ax in [a for a in [ax1, ax2, ax3, ax4, ax5, ax6] if a is not None]:
            ax.axvline(row[x_col], color="green", linestyle=":", alpha=0.6, linewidth=1)
        ax1.text(row[x_col], 98, label, fontsize=7, color="green", va="top", ha="left", rotation=90)

    # --- Loss values ---
    ax2.plot(loss_x, loss_df["loss"], label="Total loss", color="black", linewidth=1.5)
    ax2.plot(loss_x, loss_df["policy_loss"], label="Policy loss", color="royalblue", linewidth=1)
    ax2.plot(loss_x, loss_df["value_loss"], label="Value loss", color="darkorange", linewidth=1)
    if "qd_loss" in loss_df.columns:
        ax2.plot(loss_x, loss_df["qd_loss"], label="QD loss", color="mediumpurple", linewidth=1)
    if "qe_loss" in loss_df.columns:
        qe = pd.to_numeric(loss_df["qe_loss"], errors="coerce")
        if qe.notna().any() and (qe > 0).any():
            ax2.plot(loss_x, qe, label="QE loss", color="mediumseagreen", linewidth=1)
    if "mob_loss" in loss_df.columns:
        mob = pd.to_numeric(loss_df["mob_loss"], errors="coerce")
        if mob.notna().any() and (mob > 0).any():
            ax2.plot(loss_x, mob, label="Mob loss", color="indianred", linewidth=1)
    if "place_value_loss" in loss_df.columns:
        pvl = pd.to_numeric(loss_df["place_value_loss"], errors="coerce")
        if pvl.notna().any() and (pvl > 0).any():
            ax2.plot(loss_x, pvl, label="Place val loss", color="darkorange", linewidth=1, linestyle="--")
    if "capture_value_loss" in loss_df.columns:
        cvl = pd.to_numeric(loss_df["capture_value_loss"], errors="coerce")
        if cvl.notna().any() and (cvl > 0).any():
            ax2.plot(loss_x, cvl, label="Capture val loss", color="darkorange", linewidth=1, linestyle=":")
    if "place_policy_loss" in loss_df.columns:
        ppl = pd.to_numeric(loss_df["place_policy_loss"], errors="coerce")
        if ppl.notna().any() and (ppl > 0).any():
            ax2.plot(loss_x, ppl, label="Place pol loss", color="royalblue", linewidth=1, linestyle="--")
    if "capture_policy_loss" in loss_df.columns:
        cpl = pd.to_numeric(loss_df["capture_policy_loss"], errors="coerce")
        if cpl.notna().any() and (cpl > 0).any():
            ax2.plot(loss_x, cpl, label="Capture pol loss", color="royalblue", linewidth=1, linestyle=":")
    ax2.set_yscale("log")
    ax2.set_ylabel("Loss (log scale)")
    ax2.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.1), ncol=6, mode="expand", borderaxespad=0, fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Training losses")

    # --- Game length ---
    if ax3 is not None:
        avg_gl = pd.to_numeric(gen_first["avg_game_len"], errors="coerce")
        med_gl = pd.to_numeric(gen_first.get("med_game_len", pd.Series(dtype=float)), errors="coerce") if "med_game_len" in gen_first.columns else pd.Series(dtype=float)
        min_gl = pd.to_numeric(gen_first.get("min_game_len", pd.Series(dtype=float)), errors="coerce") if "min_game_len" in gen_first.columns else pd.Series(dtype=float)
        max_gl = pd.to_numeric(gen_first.get("max_game_len", pd.Series(dtype=float)), errors="coerce") if "max_game_len" in gen_first.columns else pd.Series(dtype=float)
        avg_dl = pd.to_numeric(gen_first.get("avg_decisive_len", pd.Series(dtype=float)), errors="coerce") if "avg_decisive_len" in gen_first.columns else pd.Series(dtype=float)
        med_dl = pd.to_numeric(gen_first.get("med_decisive_len", pd.Series(dtype=float)), errors="coerce") if "med_decisive_len" in gen_first.columns else pd.Series(dtype=float)
        if avg_gl.notna().any():
            ax3.plot(iters, smooth(avg_gl), label="Avg all", color="steelblue", linewidth=1.5)
        if med_gl.notna().any():
            ax3.plot(iters, smooth(med_gl), label="Med all", color="steelblue", linewidth=1, linestyle="--")
        if min_gl.notna().any():
            ax3.plot(iters, smooth(min_gl), label="Min all", color="steelblue", linewidth=1, linestyle=":")
        if max_gl.notna().any():
            ax3.plot(iters, smooth(max_gl), label="Max all", color="steelblue", linewidth=1, linestyle=(0, (3, 1, 1, 1)))
        if avg_dl.notna().any():
            ax3.plot(iters, smooth(avg_dl), label="Avg decisive", color="tomato", linewidth=1.5)
        if med_dl.notna().any():
            ax3.plot(iters, smooth(med_dl), label="Med decisive", color="tomato", linewidth=1, linestyle="--")
        ax3.set_ylabel("Moves")
        ax3.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.1), ncol=4, mode="expand", borderaxespad=0, fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_title("Game length")

    # --- Win conditions ---
    if ax4 is not None:
        decisive = pd.to_numeric(gen_first["wins_white"] + gen_first["wins_grey"] + gen_first["wins_black"] + gen_first["wins_combo"], errors="coerce").replace(0, np.nan)
        for col, label, color in [
            ("wins_white", "white", "#FFA032"),
            ("wins_grey",  "grey",  "#64B4FF"),
            ("wins_black", "black", "#FF3CB4"),
            ("wins_combo", "combo", "#50DC50"),
        ]:
            pct = pd.to_numeric(gen_first[col], errors="coerce") / decisive * 100
            if pct.notna().any():
                ax4.plot(iters, smooth(pct), label=label, color=color, linewidth=1.5)
        ax4.set_ylabel("% of decisive games")
        ax4.set_ylim(0, 100)
        ax4.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.1), ncol=4, mode="expand", borderaxespad=0, fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.set_title("Win condition (% of decisive games)")

    # --- Capture ratio ---
    if ax5 is not None:
        iso = pd.to_numeric(gen_first["isolation_captures"], errors="coerce")
        jmp = pd.to_numeric(gen_first["jump_captures"], errors="coerce")
        total_cap = (iso + jmp).replace(0, np.nan)
        ratio = iso / total_cap
        ax5.plot(iters, smooth(ratio), color="mediumpurple", linewidth=1.5, label="Isolation ratio")
        ax5.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax5.set_ylabel("Fraction")
        ax5.set_ylim(0, 1)
        ax5.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.1), ncol=2, mode="expand", borderaxespad=0, fontsize=8)
        ax5.grid(True, alpha=0.3)
        ax5.set_title("Isolation captures / total captures")

    # --- Learning rate ---
    if ax6 is not None:
        lr = pd.to_numeric(loss_df["lr"], errors="coerce")
        ax6.plot(loss_x, lr, color="teal", linewidth=1.5)
        ax6.set_ylabel("Learning rate")
        ax6.set_yscale("log")
        ax6.grid(True, alpha=0.3)
        ax6.set_title("Learning rate schedule")

    axes[-1].set_xlabel(x_label)
    ax1.set_xlim(left=iters.min())

    plt.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        print(f"Saved to {output}")
    else:
        _restore_geometry(fig)
        fig.canvas.mpl_connect("close_event", lambda _: _save_geometry(fig))
        try:
            import mplcursors
            _cursor = mplcursors.cursor(hover=True)  # noqa: F841 — must stay in scope
        except ImportError:
            pass
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot perf_log.csv")
    parser.add_argument("csv", nargs="?", help="Path to CSV file (default: perf_log.csv)")
    parser.add_argument("--game", help="Game type prefix, e.g. 'zertz' — picks most recent <game>_*_log.csv")
    parser.add_argument("-o", "--output", help="Save plot to file instead of showing it")
    parser.add_argument("--rolling", type=int, default=1, metavar="N", help="Rolling average window (default: 1 = no smoothing). Applies to all series except losses.")
    args = parser.parse_args()

    if args.game:
        matches = sorted(Path(".").glob(f"{args.game}*_log.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            print(f"No log files found matching '{args.game}*_log.csv'", file=sys.stderr)
            sys.exit(1)
        csv_path = matches[0]
    else:
        csv_path = Path(args.csv) if args.csv else Path("model_log.csv")

    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    plot_perf_log(csv_path, Path(args.output) if args.output else None, rolling=args.rolling)


if __name__ == "__main__":
    main()
