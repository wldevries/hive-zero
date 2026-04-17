# Policy Head Design: Factorization and Its Limits

## The Problem

The sum-factorized policy head can only represent product distributions over (source, destination), so a specific pairing that is tactically decisive but individually unremarkable in either part gets underweighted by the prior.

## Current Design

Movement priors are computed as:

```
prior(src=A, type=T, dst=B) = src_logit[A] + dst_logit[T][B]
```

Training stores visit counts as **marginals**: visits for move (A→B) are added to both `src_logit[A]` and `dst_logit[T][B]` independently. The joint count is discarded.

This enforces an independence assumption: `P(A,B) ∝ exp(f(A)) × exp(g(B))`. The model cannot represent "A→B is the only winning move" if A is an unremarkable source on average and B is an unremarkable destination on average — the marginals pull both logits toward neutral.

The gradient issue is not about gradient flow (both factorized and bilinear couple gradients across legal moves through the softmax). The issue is **representational**: the factorized form constrains the distribution to a rank-1 exponential family.

Placement moves (`place_logit[T][B]`) do not have this problem — the (type, position) pair is a single joint decision and every combination is freely parameterized.

## Observed Effect

In Zertz, placement plays better than captures. The only architectural difference between the two is that placement uses a direct joint head while captures use `cap_source + cap_dest`. This is consistent with the factorization being the bottleneck.

## Proposed Solutions

### 1. Direction-based encoding for fixed-geometry moves

For moves where source + direction fully determines destination (Zertz captures, Hive Queen, Hive Beetle), replace the two-part head with a single `(6, H, W)` head:

```
prior(src=A, direction=d) = cap_dir[d][A]
```

- No independence assumption
- Compact: 6 channels × board cells, smaller than the two-head approach
- Mid-capture falls out naturally: source is fixed in the game state input, valid logits are the 6 direction channels at the fixed source position
- Training: visits for (A→B) go to `cap_dir[d][A]` where d = direction from A to B. No marginalization.

Applies to:
- **Zertz**: all captures (always a 1-hop jump in one of 6 directions)
- **Hive Queen**: moves exactly 1 step in one of 6 directions
- **Hive Beetle**: moves exactly 1 step in one of 6 directions (or stacks)
- **Hive Grasshopper**: jumps in a fixed direction, variable distance — direction still determines destination given the board, so same encoding works

Does not cleanly apply to Hive Spider (3-step slide) or Hive Ant (arbitrary reachable position).

### 2. Bilinear head (Q·K)

Replace `src_logit[A] + dst_logit[T][B]` with a dot product of learned per-cell embeddings:

```
prior(src=A, dst=B) = Q[A] · K[B] / sqrt(D)

Q = Conv1x1(trunk, D)   # (B, D, G, G)
K = Conv1x1(trunk, D)   # (B, D, G, G)
```

Q and K are outputs of the model. In Rust MCTS, the prior for each legal movement is computed as a dot product of the D-dimensional embeddings at src and dst. The full G²×G² logit matrix is never materialized at inference time.

During training, the full matrix `Q @ K.T` is computed for the cross-entropy loss. Training targets are **joint** (src, dst) visit counts — marginalized storage is incorrect here because the gradient of Q[A] depends on K[B] and vice versa.

This is more expressive than direction encoding but applies uniformly to all movement types including Ant and Spider.

### 3. Multi-step MCTS

Split each movement into two sequential MCTS decisions:

- **Step 1** (mode=Normal): choose a piece to pick up (board hex) or a piece type to place (reserve). Transitions to `Moving(src, type)` or `Placing(type)`.
- **Step 2** (mode=Moving or Placing): choose destination hex.

The step-2 policy is fully conditioned on the chosen piece because it is part of the game state seen by the network (encoded as a mode channel, similar to Zertz `mid_capture_source`). No independence assumption.

Cost: halves effective search depth per simulation budget since each real move takes 2 tree levels.

## Recommendation

### Zertz

Direction-based encoding is the cleanest fix: physically motivated, compact, and completely eliminates the factorization issue for all capture moves with no architectural overhead.

### Hive

Direction encoding covers Queen, Beetle, and Grasshopper. Spider and Ant do not fit — Spider takes an exact 3-step slide along the hive boundary and Ant can reach any connected position, neither of which is determined by direction alone.

A split approach (direction for some pieces, bilinear or multi-step for others) is not worth the complexity. The bilinear Q·K head applies uniformly to all piece types and is the recommended solution.

Additionally, the expansion pieces make any geometry-based encoding impractical: Mosquito copies the movement rules of any adjacent piece, so its valid moves are dynamic and unknowable from piece identity alone. Pillbug moves other pieces rather than itself. Ladybug moves 2 steps on top of the hive then 1 step down — 3 hexes total but not in a straight line. None of these can be represented by any fixed structural encoding. The bilinear head handles them correctly since it operates on board features rather than piece geometry.
