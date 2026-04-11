# MCTS Value Convention

## The choice

There are two standard ways to store backed-up values in an MCTS tree.

**Current-player perspective** (simple): each node stores the mean value
from the perspective of the player whose turn it is *at that node*. UCB
selection negates the child's value when evaluating it from the parent:

```
UCB(child) = -child.value() + c_puct * prior * sqrt(N_parent) / (1 + N_child)
```

**Parent-player perspective** (what this codebase uses): each node stores
the mean value from the perspective of the player who *chose the move that
led to this node* (the parent's player). UCB selection reads the child's
value directly without negation:

```
UCB(child) = child.value() + c_puct * prior * sqrt(N_parent) / (1 + N_child)
```

We use parent-player perspective everywhere: `core-game` MCTS (used by
Hive and TicTacToe) and `zertz-game` MCTS.

## Why parent-player perspective

- UCB is called in the innermost loop (once per child per simulation). No
  negation saves one operation per call.
- The stored value directly encodes "was this move good for the player who
  picked it?" which is a natural reading.

## How backpropagation works

The NN returns a value in `[-1, 1]` from the **current player's
perspective** at the leaf node. Backpropagation must transform this into
the parent-player perspective for each node along the path to the root.

### Rule

After storing `value` at node N (which represents "from N's parent's
player's perspective"), prepare the value to store at N's parent P by
flipping sign iff **P's player differs from P's parent's player**:

```
should_flip = (P.player != P.parent.player)   // P.parent == grandparent of N
```

If P is the root (no grandparent), always flip. This ensures `root_value()`
= `-root.value_sum / root.visit_count` correctly recovers the root player's
own expected return.

### Why this rule (not the naive "flip on every player change")

The naive approach flips sign when crossing the N→P boundary (N.player !=
P.player). This works for strictly alternating games because consecutive
nodes always have different players, making the N→P and P→grandparent
boundaries equivalent. It fails when the same player acts consecutively
(e.g. Zertz mid-capture continuations): an extra same-player step shifts
parity and inverts the sign at the mid-capture node, causing MCTS to
undervalue capture chains.

The correct rule looks one level higher: does the value need to flip to go
from P's frame into P.parent's frame? That only depends on whether P and
P.parent have the same or different players.

### Trace: normal alternating game

Tree: Root(P1) → B(P2) → C(P1), leaf C with NN value +0.8 for P1.

```
start:  value = -0.8   (negate: C stores from B's/P2's frame)
at C:   store -0.8     B→Root boundary: P2 ≠ P1 → flip → value = +0.8
at B:   store +0.8     Root is root boundary → flip → value = -0.8
at Root: store -0.8    root_value() = +0.8  ✓
```

`C.value() = -0.8` — from P2's frame, C is bad for P2 (P1 wins). P2 avoids C. ✓  
`B.value() = +0.8` — from P1's frame, B is good for P1. P1 chooses B. ✓

### Trace: Zertz mid-capture

Tree: Root(P1) → B(P1, mid-capture) → C(P2) → D(P1), leaf D with NN value +0.8 for P1.

```
start:  value = -0.8   (negate: D stores from C's/P2's frame)
at D:   store -0.8     C→B boundary: P2 ≠ P1 → flip → value = +0.8
at C:   store +0.8     B→Root boundary: P1 = P1 → no flip → value = +0.8
at B:   store +0.8     Root is root boundary → flip → value = -0.8
at Root: store -0.8    root_value() = +0.8  ✓
```

`D.value() = -0.8` — from P2's frame, D is bad for P2. ✓  
`C.value() = +0.8` — from P1's frame (B's player), P1 likes C. ✓  
`B.value() = +0.8` — from P1's frame (Root's player), P1 likes entering the capture chain. ✓

## Root value

The root has no parent, so its value_sum accumulates in the "opposite"
frame. `MctsSearch::root_value()` corrects for this:

```rust
pub fn root_value(&self) -> f32 {
    -self.arena.get(self.root).value()
}
```

This gives the root player's own expected return in `[-1, 1]`.

## Virtual loss

`apply_virtual_loss` and `correct_virtual_loss` use the same flip rule as
`backpropagate`. Virtual loss pessimistically marks a path as `-1` during
batch selection to deter other threads/batch entries from picking the same
path. `correct_virtual_loss` replaces the placeholder with the real NN
value once inference returns.
