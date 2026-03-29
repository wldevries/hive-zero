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

    col_w = "wins_w" if "wins_w" in df.columns else "wins_p1"
    col_b = "wins_b" if "wins_b" in df.columns else "wins_p2"
    total = df[col_w] + df[col_b] + df["draws"]
    df["win_w_pct"] = df[col_w] / total * 100
    df["win_b_pct"] = df[col_b] / total * 100
    df["draw_pct"] = df["draws"] / total * 100

    iters = df["iter"]

    if "avg_game_len" not in df.columns and "positions" in df.columns:
        total_games = df[col_w] + df[col_b] + df["draws"]
        df["avg_game_len"] = df["positions"] / total_games.replace(0, np.nan)
    has_game_len = "avg_game_len" in df.columns and pd.to_numeric(df["avg_game_len"], errors="coerce").notna().any()
    has_wincon = "wins_white" in df.columns and pd.to_numeric(df["wins_white"], errors="coerce").notna().any()
    has_capture_ratio = "isolation_captures" in df.columns and pd.to_numeric(df["isolation_captures"], errors="coerce").notna().any()
    nrows = 2 + has_game_len + has_wincon + has_capture_ratio
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 4 * nrows), sharex=True)
    ax1, ax2 = axes[0], axes[1]
    ax3 = axes[2] if has_game_len else None
    ax4 = axes[2 + has_game_len] if has_wincon else None
    ax5 = axes[2 + has_game_len + has_wincon] if has_capture_ratio else None
    fig.suptitle(csv_path.name, fontsize=13)

    # --- Win percentages ---
    label_w = "White wins %" if col_w == "wins_w" else "P1 wins %"
    label_b = "Black wins %" if col_b == "wins_b" else "P2 wins %"
    ax1.plot(iters, smooth(df["win_w_pct"]), label=label_w, color="steelblue")
    ax1.plot(iters, smooth(df["win_b_pct"]), label=label_b, color="tomato")
    ax1.plot(iters, smooth(df["draw_pct"]), label="Draws %", color="gray", linestyle="--")
    trend = np.poly1d(np.polyfit(iters, df["draw_pct"], 1))
    ax1.plot(iters, trend(iters), color="gray", linestyle="-", linewidth=1, alpha=0.5, label="Draws trend")
    ax1.set_ylabel("Percentage")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.1), ncol=4, mode="expand", borderaxespad=0, fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Game outcomes per iteration")

    # Mark any row with a comment
    commented = df[df["comment"].notna() & df["comment"].astype(str).str.strip().ne("")]
    for _, row in commented.iterrows():
        label = str(row["comment"]).strip()
        for ax in [a for a in [ax1, ax2, ax3, ax4, ax5] if a is not None]:
            ax.axvline(row["iter"], color="green", linestyle=":", alpha=0.6, linewidth=1)
        ax1.text(row["iter"], 98, label, fontsize=7, color="green", va="top", ha="left", rotation=90)

    # --- Loss values ---
    ax2.plot(iters, df["loss"], label="Total loss", color="black", linewidth=1.5)
    ax2.plot(iters, df["policy_loss"], label="Policy loss", color="royalblue", linewidth=1)
    ax2.plot(iters, df["value_loss"], label="Value loss", color="darkorange", linewidth=1)
    if "qd_loss" in df.columns:
        ax2.plot(iters, df["qd_loss"], label="QD loss", color="mediumpurple", linewidth=1)
    if "qe_loss" in df.columns:
        qe = pd.to_numeric(df["qe_loss"], errors="coerce")
        if qe.notna().any() and (qe > 0).any():
            ax2.plot(iters, qe, label="QE loss", color="mediumseagreen", linewidth=1)
    if "mob_loss" in df.columns:
        mob = pd.to_numeric(df["mob_loss"], errors="coerce")
        if mob.notna().any() and (mob > 0).any():
            ax2.plot(iters, mob, label="Mob loss", color="indianred", linewidth=1)
    if "place_value_loss" in df.columns:
        pvl = pd.to_numeric(df["place_value_loss"], errors="coerce")
        if pvl.notna().any() and (pvl > 0).any():
            ax2.plot(iters, pvl, label="Place val loss", color="darkorange", linewidth=1, linestyle="--")
    if "capture_value_loss" in df.columns:
        cvl = pd.to_numeric(df["capture_value_loss"], errors="coerce")
        if cvl.notna().any() and (cvl > 0).any():
            ax2.plot(iters, cvl, label="Capture val loss", color="darkorange", linewidth=1, linestyle=":")
    if "place_policy_loss" in df.columns:
        ppl = pd.to_numeric(df["place_policy_loss"], errors="coerce")
        if ppl.notna().any() and (ppl > 0).any():
            ax2.plot(iters, ppl, label="Place pol loss", color="royalblue", linewidth=1, linestyle="--")
    if "capture_policy_loss" in df.columns:
        cpl = pd.to_numeric(df["capture_policy_loss"], errors="coerce")
        if cpl.notna().any() and (cpl > 0).any():
            ax2.plot(iters, cpl, label="Capture pol loss", color="royalblue", linewidth=1, linestyle=":")
    if not has_game_len:
        ax2.set_xlabel("Iteration")
    ax2.set_yscale("log")
    ax2.set_ylabel("Loss (log scale)")
    ax2.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.1), ncol=6, mode="expand", borderaxespad=0, fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Training losses")

    # --- Game length ---
    if ax3 is not None:
        avg_gl = pd.to_numeric(df["avg_game_len"], errors="coerce")
        med_gl = pd.to_numeric(df.get("med_game_len", pd.Series(dtype=float)), errors="coerce") if "med_game_len" in df.columns else pd.Series(dtype=float)
        min_gl = pd.to_numeric(df.get("min_game_len", pd.Series(dtype=float)), errors="coerce") if "min_game_len" in df.columns else pd.Series(dtype=float)
        max_gl = pd.to_numeric(df.get("max_game_len", pd.Series(dtype=float)), errors="coerce") if "max_game_len" in df.columns else pd.Series(dtype=float)
        avg_dl = pd.to_numeric(df.get("avg_decisive_len", pd.Series(dtype=float)), errors="coerce") if "avg_decisive_len" in df.columns else pd.Series(dtype=float)
        med_dl = pd.to_numeric(df.get("med_decisive_len", pd.Series(dtype=float)), errors="coerce") if "med_decisive_len" in df.columns else pd.Series(dtype=float)
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
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Moves")
        ax3.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.1), ncol=4, mode="expand", borderaxespad=0, fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_title("Game length")

    # --- Win conditions ---
    if ax4 is not None:
        decisive = pd.to_numeric(df["wins_white"] + df["wins_grey"] + df["wins_black"] + df["wins_combo"], errors="coerce").replace(0, np.nan)
        for col, label, color in [
            ("wins_white", "white", "#FFA032"),
            ("wins_grey",  "grey",  "#64B4FF"),
            ("wins_black", "black", "#FF3CB4"),
            ("wins_combo", "combo", "#50DC50"),
        ]:
            pct = pd.to_numeric(df[col], errors="coerce") / decisive * 100
            if pct.notna().any():
                ax4.plot(iters, smooth(pct), label=label, color=color, linewidth=1.5)
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("% of decisive games")
        ax4.set_ylim(0, 100)
        ax4.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.1), ncol=4, mode="expand", borderaxespad=0, fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.set_title("Win condition (% of decisive games)")

    # --- Capture ratio ---
    if ax5 is not None:
        iso = pd.to_numeric(df["isolation_captures"], errors="coerce")
        jmp = pd.to_numeric(df["jump_captures"], errors="coerce")
        total_cap = (iso + jmp).replace(0, np.nan)
        ratio = iso / total_cap
        ax5.plot(iters, smooth(ratio), color="mediumpurple", linewidth=1.5, label="Isolation ratio")
        ax5.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax5.set_ylabel("Fraction")
        ax5.set_ylim(0, 1)
        ax5.set_xlabel("Iteration")
        ax5.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.1), ncol=2, mode="expand", borderaxespad=0, fontsize=8)
        ax5.grid(True, alpha=0.3)
        ax5.set_title("Isolation captures / total captures")

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
