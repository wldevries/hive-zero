"""Inspect WDL head distribution on positions sampled from random self-play."""
import random
import sys
import numpy as np
import torch
import torch.nn.functional as F
from engine_zero import HiveGame
from hive.nn.model import load_checkpoint


def sample_positions(num_games: int, max_moves: int, grid_size: int, seed: int = 0):
    rng = random.Random(seed)
    positions = []
    for _ in range(num_games):
        g = HiveGame(grid_size=grid_size)
        depth = rng.randint(1, max_moves)
        for _ in range(depth):
            if g.is_game_over:
                break
            moves = g.valid_moves()
            if not moves:
                g.play_pass()
                continue
            piece, frm, to = rng.choice(moves)
            g.play_move(piece, frm, to)
        if g.is_game_over:
            continue
        board, reserve = g.encode_board()
        positions.append((np.asarray(board), np.asarray(reserve), g.move_count))
    return positions


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "models/hive_medium2/hive_medium2.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(ckpt_path)
    model.to(device).eval()
    grid_size = ckpt.get("grid_size", 23)
    gen = ckpt.get("generation", "?")
    print(f"loaded {ckpt_path} | gen={gen} | grid={grid_size} | device={device}")

    # Sample positions — short, mid, long mixes
    positions = []
    positions += sample_positions(60, max_moves=10, grid_size=grid_size, seed=1)   # opening
    positions += sample_positions(60, max_moves=30, grid_size=grid_size, seed=2)   # mid
    positions += sample_positions(60, max_moves=55, grid_size=grid_size, seed=3)   # late
    print(f"sampled {len(positions)} positions")

    boards = torch.from_numpy(np.stack([p[0] for p in positions])).to(device)
    reserves = torch.from_numpy(np.stack([p[1] for p in positions])).to(device)
    move_counts = np.array([p[2] for p in positions])

    with torch.no_grad():
        _, wdl_logits, _ = model(boards, reserves)
        wdl = F.softmax(wdl_logits, dim=1).cpu().numpy()  # (N, 3) = [W, D, L]

    W, D, L = wdl[:, 0], wdl[:, 1], wdl[:, 2]
    print()
    print(f"{'segment':<10} {'n':>4} {'W mean':>8} {'D mean':>8} {'L mean':>8} {'D>0.7':>7} {'D max':>7} {'D min':>7}")
    print("-" * 60)
    for label, mask in [
        ("all",    np.ones_like(move_counts, dtype=bool)),
        ("opening", move_counts <= 12),
        ("mid",     (move_counts > 12) & (move_counts <= 30)),
        ("late",    move_counts > 30),
    ]:
        if not mask.any():
            continue
        Wm, Dm, Lm = W[mask], D[mask], L[mask]
        print(f"{label:<10} {mask.sum():>4d} {Wm.mean():>8.3f} {Dm.mean():>8.3f} {Lm.mean():>8.3f} "
              f"{(Dm > 0.7).mean():>7.2%} {Dm.max():>7.3f} {Dm.min():>7.3f}")

    # Distribution of D over all positions
    print()
    pcts = [10, 25, 50, 75, 90]
    qs = np.percentile(D, pcts)
    print("Draw probability percentiles:", {f"p{p}": round(float(q), 3) for p, q in zip(pcts, qs)})
    print(f"Fraction with D > 0.5: {(D > 0.5).mean():.2%}")
    print(f"Fraction with D > 0.8: {(D > 0.8).mean():.2%}")
    print(f"Fraction with W > L:   {(W > L).mean():.2%}  (W-L symmetry sanity check)")
    print(f"Mean |W - L|:          {np.abs(W - L).mean():.3f}")

    # Effect of contempt 0.2 on the W-L-c*D scalar
    contempt = 0.2
    score = W - L - contempt * D
    print()
    print(f"With contempt={contempt}: W-L-c*D scalar — mean={score.mean():+.4f}, std={score.std():.4f}")
    print(f"Same with contempt=0:    W-L scalar      — mean={(W-L).mean():+.4f}, std={(W-L).std():.4f}")
    print(f"Mean shift from contempt: {(-contempt * D).mean():+.4f}  (should pull score toward L)")


if __name__ == "__main__":
    main()
