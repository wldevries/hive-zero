# Overview of runs

Logs are in `{modelname}_log.csv`

Human boardspace stats for reference: 67% white wins, 5% grey, 12% black, 16% combo. Avg game length 26.2 turns. Second player wins 54.4%.

## Summary table (last 20 gens of each run)

| Run | Arch | Sims | Gens | Best Loss | Tail Loss | Val Loss | White | Grey | Black | Combo | Game Len |
|-----|------|------|------|-----------|-----------|----------|-------|------|-------|-------|----------|
| b4_c128_cycl_lr | 4b/128ch | 800 | 100 | 3.70 | 3.77 | 0.667 | 3 | 2 | 91 | 4 | 16.3 |
| b8_c96_1200sims_V2 | 8b/96ch | 1200 | 177 | 3.72 | 4.26 | 0.725 | 3 | 3 | 89 | 4 | 17.8 |
| b6_c96_cycl_lr_v2 | 6b/96ch | 800 | 294 | 4.08 | 5.38 | 0.826 | 3 | 78 | 6 | 13 | 21.8 |
| b8_c96_1200sims | 8b/96ch | 1200 | 105 | 4.23 | 4.87 | 0.996 | 4 | 0 | 95 | 1 | 16.9 |
| b8_c96_1200sims_V3 | 8b/96ch | 1200 | 50* | 5.58 | 7.08 | 0.859 | 10 | 16 | 62 | 12 | 19.8 |
| b4_c128_cycl_lr_v2 | 4b/128ch | 800 | 260 | 7.03 | 7.11 | 0.913 | 44 | 3 | 24 | 29 | 26.5 |

*V3 still running

## Key findings

### Win-condition collapse
The dominant issue across runs: stronger models (b6, b8) converge to a single win condition that differs from human play. b8 runs lock into black wins (89-95%), b6 into grey (78%). Only the weaker b4_v2 run maintains diverse win types and is the only run where white wins appear (~44%), closer to human play (67% white).

### Isolation capture trap
When black wins dominate, isolation captures make up 75-82% of all captures. The models learn a passive strategy: remove rings to strand marbles, passively accumulate the most abundant color (black). Human play is the opposite: aggressive jump captures targeting white (only 4 needed). The b4_v2 run maintains a balanced ~40-50% isolation ratio.

### Capacity vs diversity tradeoff
The b4_v2 model lacks capacity to discover and exploit the isolation-heavy degenerate strategy, accidentally preserving win-type diversity. But it also can't improve — policy loss plateaus at ~6.1 and value loss is stuck at 0.9. The b8 models reach much lower loss (3.7) but collapse into single-color wins.

### Cyclic LR is a double-edged sword
The b8 v1 (2-epoch) run found white wins early (gen 2-30: 72-84% white) but the 0.2 LR peak at gen 50 destroyed this strategy. It recovered into black wins and never escaped. Conversely, the V3 run (0.1 peak) was locked into black wins by gen 38 but the LR ramp toward gen 50 is currently disrupting it — win types are shuffling at gen 47-50.

### Dirichlet noise matters
b4_v2 (diverse wins) uses alpha=0.01, epsilon=0.4. All b8 runs (collapsed wins) use alpha=0.15, epsilon=0.25. Lower alpha + higher epsilon means fewer moves get boosted but more strongly, creating more disruptive exploration.

### Value head struggles everywhere
Value loss remains high across all runs (0.67-0.99), while policy loss drops from ~22 to ~3-4. The models learn what to play much better than who's winning.

## Run details

### zertz_b4_128_cycl_lr

1 cycle LR curve, 4 blocks, 128 channels, standard alphazero dirichlet. Reached best loss 3.70 but collapsed into 91% black wins.

simulations 800, games_per_gen 100, epochs_per_gen 2, batch_size 256, max_moves 40, replay_window 8, playout_cap_p 0.25, fast_cap 20, temp_threshold 10, play_batch_size 2, augment_symmetry true, lr 0.01

### zertz_b4_128_cycl_lr_v2

Different dirichlet noise (alpha=0.01, epsilon=0.4). Best run for win-type diversity: maintained 30-60% white, 10-25% black, 10-39% combo throughout 260 gens. Survived the 0.2 LR peak without strategy collapse. However policy loss plateaued at ~6.1 — the model is too weak to improve further.

```powershell
uv run zertz train --blocks 4 --channels 128 --games 100 --epochs-per-gen 2 --simulations 800 --device cuda --model "zertz_b4_c128_cycl_lr_v2.pt" --playout-cap-p 0.25 --comment "start" --play-batch-size 2 --augment-symmetry --temp-threshold 10 --lr-schedule "1:0.02,50:0.2,100:0.02,200:0.002" --dir-alpha 0.01 --dir-epsilon 0.4 --replay-window 3
```

### zertz_b6_cycl-lr_v2

More blocks, fewer channels. Longest run (294 gens). Converged to ~78% grey wins. P1/P2 balance stays ~50/50. Loss oscillates with cyclic LR (peaks near gen 50 at LR=0.2). Value loss stuck at ~0.82.

```powershell
uv run zertz train --blocks 6 --channels 96 --games 100 --epochs-per-gen 2 --simulations 800 --device cuda --model "zertz_b6_c96_cycl_lr_v2.pt" --playout-cap-p 0.25 --comment "start" --play-batch-size 2 --augment-symmetry --temp-threshold 10 --lr-schedule "1:0.02,50:0.2,100:0.02,200:0.002" --dir-alpha 0.15 --dir-epsilon 0.25 --replay-window 3
```

### zertz_b8_c96_cycl_1200sims

2 epochs per gen. Found white-win strategy early (gen 2-30: 72-84% white) but LR peak at gen 50 (0.2) destroyed it. Flipped to 95-100% black wins by gen 60 and never recovered. Value loss ~0.99 — value head essentially never learned.

```powershell
uv run zertz train --blocks 8 --channels 96 --games 100 --epochs-per-gen 2 --simulations 1200 --device cuda --model "zertz_b8_c96_cycl_1200sims.pt" --playout-cap-p 0.25 --comment "start" --play-batch-size 2 --augment-symmetry --temp-threshold 10 --lr-schedule "1:0.02,50:0.2,100:0.02,200:0.002" --dir-alpha 0.15 --dir-epsilon 0.25 --replay-window 3
```

### zertz_b8_c96_cycl_1200sims_V2

1 epoch per gen. Never found white wins — went straight to black by gen 8. Better value loss than v1 (0.63-0.72 vs ~0.99), confirming 1 epoch is better. Ended with P2 skew (33/67 at gen 173).

```powershell
uv run zertz train --blocks 8 --channels 96 --games 100 --epochs-per-gen 1 --simulations 1200 --device cuda --model "zertz_b8_c96_cycl_1200sims.pt" --playout-cap-p 0.25 --comment "start" --play-batch-size 2 --augment-symmetry --temp-threshold 10 --lr-schedule "1:0.02,50:0.2,100:0.02,200:0.002" --dir-alpha 0.15 --dir-epsilon 0.25 --replay-window 3
```

### zertz_b8_c96_cycl_1200sims_V3 (running)

Lower peak LR (0.1 instead of 0.2), 1 epoch. Locked into black wins (75-97%) by gen 38-46 but win types are diversifying at gen 47-50 as LR approaches the 0.1 peak. Currently at gen 50 with loss 7.19. Watching whether diversity survives the LR downslope.

```powershell
uv run zertz train --blocks 8 --channels 96 --games 100 --epochs-per-gen 1 --simulations 1200 --device cuda --model "zertz_b8_c96_cycl_1200sims_V3.pt" --playout-cap-p 0.25 --comment "start" --play-batch-size 2 --augment-symmetry --temp-threshold 10 --lr-schedule "1:0.02,50:0.1,100:0.02,200:0.002" --dir-alpha 0.15 --dir-epsilon 0.25 --replay-window 3
```

## Next experiments to try

- **b8 with b4_v2's dirichlet params** (alpha=0.01, epsilon=0.4): isolate whether noise or capacity drives win diversity
- **More games per generation** (200-400): reduce small-sample drift toward degenerate strategies
- **Higher dirichlet epsilon** (0.4-0.5) on b8: force more root exploration to prevent single-color lock-in
