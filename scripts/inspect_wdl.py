"""Inspect WDL head distribution on in-distribution positions (boardspace SGFs).

Auto-detects game (hive | yinsh) from the checkpoint metadata. Walks
`--boardspace-dir`, picks --num-games random SGFs, samples one random
position per game, then prints WDL summary stats segmented by:
  - hive:  move_count buckets (opening / mid / late)
  - yinsh: phase (setup / normal / claim_row), with normal split by stage
"""
import argparse
import os
import random
import sys
import zipfile

import numpy as np
import torch
import torch.nn.functional as F


def detect_game(ckpt_path: str) -> str:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    return ckpt.get("game", "hive")


def _walk_zips(boardspace_dir: str) -> list[str]:
    paths = []
    for root, _, files in os.walk(boardspace_dir):
        for f in files:
            if f.endswith(".zip"):
                paths.append(os.path.join(root, f))
    return paths


def sample_hive_positions(boardspace_dir: str, num_games: int, grid_size: int, seed: int):
    import engine_zero
    from engine_zero import parse_sgf_moves
    from hive.supervised.pretrain import parse_uhp_move

    rng = random.Random(seed)
    zip_paths = _walk_zips(boardspace_dir)
    if not zip_paths:
        raise FileNotFoundError(f"no .zip files under {boardspace_dir}")
    rng.shuffle(zip_paths)

    boards, reserves, move_counts = [], [], []
    games_used = 0
    for zp in zip_paths:
        if games_used >= num_games:
            break
        try:
            zf = zipfile.ZipFile(zp)
        except zipfile.BadZipFile:
            continue
        with zf:
            names = [n for n in zf.namelist() if n.endswith(".sgf")]
            rng.shuffle(names)
            for name in names:
                if games_used >= num_games:
                    break
                try:
                    content = zf.read(name).decode("iso-8859-1")
                    moves = list(parse_sgf_moves(content))
                except Exception:
                    continue
                if len(moves) < 8:
                    continue
                target_step = rng.randrange(len(moves))

                g = engine_zero.HiveGame(grid_size=grid_size)
                ok = True
                for step, mv in enumerate(moves):
                    if g.is_game_over:
                        ok = False
                        break
                    if step == target_step:
                        b, r = g.encode_board()
                        boards.append(np.asarray(b, dtype=np.float32))
                        reserves.append(np.asarray(r, dtype=np.float32))
                        move_counts.append(g.move_count)
                        ok = True
                        break
                    if mv == "pass":
                        try:
                            g.play_pass()
                        except Exception:
                            ok = False
                            break
                        continue
                    try:
                        piece, frm, to = parse_uhp_move(g, mv)
                        g.play_move(piece, frm, to)
                    except Exception:
                        ok = False
                        break
                if ok:
                    games_used += 1

    if not boards:
        raise RuntimeError("no positions sampled — check boardspace dir")
    move_counts = np.array(move_counts)
    segments = {
        "all":     np.ones_like(move_counts, dtype=bool),
        "opening": move_counts <= 12,
        "mid":     (move_counts > 12) & (move_counts <= 30),
        "late":    move_counts > 30,
    }
    return np.stack(boards), np.stack(reserves), segments


def sample_yinsh_positions(boardspace_dir: str, num_games: int, seed: int):
    import engine_zero
    from yinsh.nn.model import NUM_CHANNELS, GRID_SIZE

    rng = random.Random(seed)
    zip_paths = _walk_zips(boardspace_dir)
    if not zip_paths:
        raise FileNotFoundError(f"no .zip files under {boardspace_dir}")
    rng.shuffle(zip_paths)

    boards, reserves, phases, step_idxs = [], [], [], []
    PHASE = {"setup": 0, "normal": 1, "claim_row": 2}
    games_used = 0
    for zp in zip_paths:
        if games_used >= num_games:
            break
        try:
            zf = zipfile.ZipFile(zp)
        except zipfile.BadZipFile:
            continue
        with zf:
            names = [n for n in zf.namelist() if n.endswith(".sgf")]
            rng.shuffle(names)
            for name in names:
                if games_used >= num_games:
                    break
                try:
                    content = zf.read(name).decode("iso-8859-1")
                    moves = engine_zero.parse_yinsh_sgf_moves(content)
                except Exception:
                    continue
                if len(moves) < 8:
                    continue
                target_step = rng.randrange(len(moves))

                g = engine_zero.YinshGame()
                ok = True
                for step, mv_str in enumerate(moves):
                    if g.outcome() != "ongoing":
                        ok = False
                        break
                    if step == target_step:
                        b, r = g.encode()
                        board = np.asarray(b, dtype=np.float32).reshape(
                            NUM_CHANNELS, GRID_SIZE, GRID_SIZE
                        )
                        boards.append(board)
                        reserves.append(np.asarray(r, dtype=np.float32))
                        phases.append(PHASE.get(g.phase(), 1))
                        step_idxs.append(step)
                        ok = True
                        break
                    try:
                        g.play(mv_str)
                    except Exception:
                        ok = False
                        break
                if ok:
                    games_used += 1

    if not boards:
        raise RuntimeError("no positions sampled — check boardspace dir")
    phases = np.array(phases)
    step_idxs = np.array(step_idxs)
    segments = {
        "all":         np.ones_like(phases, dtype=bool),
        "setup":       phases == 0,
        "normal":      phases == 1,
        "claim_row":   phases == 2,
        "norm.early":  (phases == 1) & (step_idxs < 30),
        "norm.late":   (phases == 1) & (step_idxs >= 30),
    }
    return np.stack(boards), np.stack(reserves), segments


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ckpt", help="Path to model .pt checkpoint")
    p.add_argument("--boardspace-dir", default=None,
                   help="Boardspace SGF dir (default: games/{game}/boardspace)")
    p.add_argument("--num-games", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--contempt", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    game = detect_game(args.ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if game == "hive":
        from hive.nn.model import load_checkpoint
        model, ckpt = load_checkpoint(args.ckpt)
        grid_size = ckpt.get("grid_size", 23)
        bdir = args.boardspace_dir or "games/hive/boardspace"
        print(f"loaded {args.ckpt} | game=hive | gen={ckpt.get('generation','?')} "
              f"| grid={grid_size} | device={device}")
        boards, reserves, segments = sample_hive_positions(bdir, args.num_games, grid_size, args.seed)
    elif game == "yinsh":
        from yinsh.nn.model import load_checkpoint
        model, ckpt = load_checkpoint(args.ckpt)
        bdir = args.boardspace_dir or "games/yinsh/boardspace"
        print(f"loaded {args.ckpt} | game=yinsh | gen={ckpt.get('generation','?')} | device={device}")
        boards, reserves, segments = sample_yinsh_positions(bdir, args.num_games, args.seed)
    else:
        sys.exit(f"unsupported game in checkpoint: {game!r}")

    model.to(device).eval()
    print(f"sampled {boards.shape[0]} positions from {bdir}")

    # Batched inference. Both models return (policy, wdl, ...) — wdl is index 1.
    boards_t = torch.from_numpy(boards).to(device)
    reserves_t = torch.from_numpy(reserves).to(device)
    wdl_chunks = []
    with torch.no_grad():
        for i in range(0, boards_t.size(0), args.batch_size):
            outs = model(boards_t[i:i + args.batch_size], reserves_t[i:i + args.batch_size])
            wdl_chunks.append(F.softmax(outs[1], dim=1).cpu())
    wdl = torch.cat(wdl_chunks).numpy()  # (N, 3) = [W, D, L]
    W, D, L = wdl[:, 0], wdl[:, 1], wdl[:, 2]

    print()
    print(f"{'segment':<11} {'n':>4} {'W mean':>8} {'D mean':>8} {'L mean':>8} {'D>0.7':>7} {'D max':>7} {'D min':>7}")
    print("-" * 64)
    for label, mask in segments.items():
        if not mask.any():
            continue
        Wm, Dm, Lm = W[mask], D[mask], L[mask]
        print(f"{label:<11} {mask.sum():>4d} {Wm.mean():>8.3f} {Dm.mean():>8.3f} {Lm.mean():>8.3f} "
              f"{(Dm > 0.7).mean():>7.2%} {Dm.max():>7.3f} {Dm.min():>7.3f}")

    print()
    pcts = [10, 25, 50, 75, 90]
    qs = np.percentile(D, pcts)
    print("Draw probability percentiles:", {f"p{p}": round(float(q), 3) for p, q in zip(pcts, qs)})
    print(f"Fraction with D > 0.5: {(D > 0.5).mean():.2%}")
    print(f"Fraction with D > 0.8: {(D > 0.8).mean():.2%}")
    print(f"Fraction with W > L:   {(W > L).mean():.2%}  (W-L symmetry sanity check)")
    print(f"Mean |W - L|:          {np.abs(W - L).mean():.3f}")

    score = W - L - args.contempt * D
    print()
    print(f"With contempt={args.contempt}: W-L-c*D scalar — mean={score.mean():+.4f}, std={score.std():.4f}")
    print(f"Same with contempt=0:    W-L scalar      — mean={(W-L).mean():+.4f}, std={(W-L).std():.4f}")
    print(f"Mean shift from contempt: {(-args.contempt * D).mean():+.4f}  (should pull score toward L)")


if __name__ == "__main__":
    main()
