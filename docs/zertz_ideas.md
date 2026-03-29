# Zertz Ideas & Known Issues

## Policy Head Size (Critical)

The current policy head is a fully-connected layer of shape `[5587, trunk_flat]`,
which accounts for ~97% of all model parameters (35M of 36M for b4_c128).
This is because the trunk is flattened before the policy head instead of using conv1x1.

### Current encoding
- Place moves: 3 colors × 37 positions × 37 rings to remove = 4107 outputs
- Place-only moves: 3 × 37 = 111 outputs
- Capture moves: 37 × 37 from/to pairs = 1369 outputs
- **Total: 5587 outputs from a dense FC layer**

### Proposed fix: factorized conv1x1 heads

**Place policy** (4 channels × grid):
- Channels 0-2: "place white/grey/black ball here" — destination per color
- Channel 3: "remove this ring" — ring removal destination
- Placement and ring removal treated as independent (factorized)
- Total: 4 × 37 = 148 outputs

**Capture policy** (1-2 channels × grid):
- Option A: destination only — "capture chain ends here" (37 outputs)
- Option B: source + destination — 2 × 37 = 74 outputs
- Multi-hop chains: destination encoding is sufficient since chains
  are deterministic given source+destination in most cases

**Total policy size: ~185-222 outputs** vs current 5587.
Policy head parameters drop from ~35M to ~200-400K.

### Impact
- Requires changes to Rust move encoding AND Python model architecture
- Existing checkpoints are incompatible (clean start required)
- Worth doing before investing significant compute in the current architecture

---

## Value Loss Instability

Value loss degrades reliably when LR is reduced:
- At iter 100: value_loss = 0.877 (best observed)
- LR drop 0.02 → 0.01 at iter 144: value_loss spiked to 0.936
- Value loss never recovered after that point
- Best battle checkpoint found at iter 110 (beats iter 120, 100)

When restarting from iter 110/120 checkpoint, value loss rises again even
without changing LR — suggesting value degradation may be caused by
replay buffer content (old positions from better-calibrated policy) rather
than LR alone.

### Possible causes
- Replay buffer fills with positions from the current (weaker-policy) network,
  replacing the better-quality data from iter ~110
- Value targets become noisier as the network explores different strategies
- The value and policy heads may be competing for gradient signal

### Ideas to investigate
- Separate LR for value and policy heads
- Higher weight on value loss in the combined loss
- Freeze policy head briefly when value loss rises
