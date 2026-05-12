"""Quick diagnostic: entropy of stored policy targets in a replay.h5.

Reports the distribution of CE-floor (target entropy) across stored samples,
split by `value_only` (whether the position is masked from policy loss) and
by an "is one-hot" check (max target prob > 0.99).

Usage:
    uv run python scripts/inspect_policy_targets.py models/hive-small-train-fix/replay.h5
"""
import argparse
import math
import sys

import h5py
import numpy as np


def entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5path")
    ap.add_argument("--sample", type=int, default=20000,
                    help="random sample size for speed (default: 20000; 0 = all)")
    args = ap.parse_args()

    with h5py.File(args.h5path, "r") as f:
        size = int(f.attrs["size"])
        print(f"replay buffer: size={size}, max_size={int(f.attrs['max_size'])}")

        n = size if args.sample == 0 else min(size, args.sample)
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(size, n, replace=False)) if n < size else np.arange(size)

        place_probs = f["place_probs"][idx]
        movement_probs = f["movement_probs"][idx]
        n_place = f["num_placements"][idx]
        n_move = f["num_movements"][idx]
        value_only = f["value_only"][idx]
        policy_only = f["policy_only"][idx]
        value_targets = f["value_targets"][idx]

    entropies = np.empty(n, dtype=np.float32)
    max_probs = np.empty(n, dtype=np.float32)
    n_legal = np.empty(n, dtype=np.int32)
    total_mass = np.empty(n, dtype=np.float32)

    for i in range(n):
        np_ = int(n_place[i])
        nm = int(n_move[i])
        pp = place_probs[i, :np_]
        mm = movement_probs[i, :nm]
        combined = np.concatenate([pp, mm])
        entropies[i] = entropy(combined)
        max_probs[i] = float(combined.max()) if combined.size else 0.0
        n_legal[i] = combined.size
        total_mass[i] = float(combined.sum())

    def report(mask, label):
        if not mask.any():
            print(f"  {label}: (none)")
            return
        h = entropies[mask]
        mp = max_probs[mask]
        nl = n_legal[mask]
        tm = total_mass[mask]
        one_hot = (mp > 0.99).mean() * 100
        empty = (nl == 0).mean() * 100
        sumto1 = (np.abs(tm - 1.0) < 1e-3).mean() * 100
        print(f"  {label} (n={mask.sum()}):")
        print(f"    entropy mean={h.mean():.3f}, p10={np.quantile(h,0.1):.3f}, "
              f"p50={np.quantile(h,0.5):.3f}, p90={np.quantile(h,0.9):.3f}")
        print(f"    max-prob   mean={mp.mean():.3f}, p10={np.quantile(mp,0.1):.3f}, "
              f"p50={np.quantile(mp,0.5):.3f}, p90={np.quantile(mp,0.9):.3f}")
        print(f"    n_legal    mean={nl.mean():.1f}, p10={np.quantile(nl,0.1):.0f}, "
              f"p50={np.quantile(nl,0.5):.0f}, p90={np.quantile(nl,0.9):.0f}")
        print(f"    one-hot (maxp>0.99): {one_hot:.1f}%  empty: {empty:.1f}%  "
              f"sum~1: {sumto1:.1f}%")
        # Bucket by n_legal so we can see if entropy scales with branching factor
        buckets = [(1, 5), (6, 20), (21, 40), (41, 60), (61, 100), (101, 9999)]
        print(f"    entropy by n_legal bucket:")
        for lo, hi in buckets:
            sel = mask & (n_legal >= lo) & (n_legal <= hi)
            if sel.any():
                print(f"      [{lo:3d}-{hi:>4}]  n={sel.sum():>5}  "
                      f"entropy mean={entropies[sel].mean():.3f}  "
                      f"log(n_legal_mean)={math.log(n_legal[sel].mean()):.3f}  "
                      f"max-prob mean={max_probs[sel].mean():.3f}")

    print()
    print("=== ALL samples ===")
    report(np.ones(n, dtype=bool), "all")
    print()
    print("=== value_only=True (masked from policy loss; fast-cap turns) ===")
    report(value_only, "vo=True")
    print()
    print("=== value_only=False (policy-trained; full-search turns) ===")
    report(~value_only, "vo=False")
    print()
    print("=== policy_only=True (masked-timeout games) ===")
    report(policy_only, "po=True")
    print()
    print("=== value_only=False & policy_only=False (full both-trained) ===")
    report((~value_only) & (~policy_only), "vo=False,po=False")

    print()
    print(f"value_targets: mean={value_targets.mean():.3f}, "
          f"abs-mean={np.abs(value_targets).mean():.3f}, "
          f"=0: {(value_targets == 0).mean()*100:.1f}%, "
          f"=+1: {(value_targets == 1).mean()*100:.1f}%, "
          f"=-1: {(value_targets == -1).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
