# GNN Architecture for Hive

Exploration of replacing the CNN-on-fixed-grid approach with a Graph Neural Network.

## Motivation

The current model encodes the board as a `39 × grid_size × grid_size` tensor. This requires a fixed grid size baked into the checkpoint, wastes compute on empty cells (most of the grid), and forces a spatial inductive bias that doesn't match the actual structure of Hive (a sparse connected blob that drifts in space).

A GNN operates directly on the graph structure of the hive — pieces as nodes, adjacency as edges — and is naturally variable-size.

## Input Graph

**Nodes**: one per piece (placed or in reserve), plus one per candidate destination hex (empty hexes adjacent to the hive).

Node features:
- Piece type (one-hot: Queen, Spider, Beetle, Grasshopper, Ant)
- Color (current-player-relative)
- Stack depth (for beetles)
- In-reserve flag

**Edges**: directed, one per adjacency relationship.

Edge features:
- Direction (one-hot over 6 hex directions + up/down for stack)
- Is-stack-connection flag

No explicit coordinates are used. Position is encoded entirely through graph topology — after message passing, a node's embedding reflects "I am adjacent to piece X to my west and piece Y to my northeast," which fully determines its spatial context without needing (q, r) values (which are translation-variant and meaningless across games).

**Candidate destination nodes** (empty hexes adjacent to the hive) connect to their neighboring piece nodes via directed edges with direction features. Their position is thus encoded through connectivity, not coordinates.

## Message Passing Trunk

Standard message passing has a locality problem: after k layers, a node only knows about its k-hop neighborhood. Hive boards can have diameter ~15, so naive deep stacking is needed for global context.

Two mitigations:

**Global readout node**: one special node connected to all piece nodes. Aggregates global board state in one hop, broadcasts back in the next. Effectively gives every node global context after 2 extra layers.

**Graph Transformer (attention)**: every node attends to every other node regardless of graph distance. O(n²) attention, but n ≈ 30 nodes so this is cheap. Eliminates depth-for-propagation entirely.

Recommended: 6–8 message passing layers with a global readout node, or 4–6 Graph Transformer layers.

## Compute vs CNN

```
CNN:  289 positions × 128 channels × 8 blocks  → expensive per layer
GNN:  ~30 nodes    × 128 dim       × 8 layers  → ~10× cheaper per layer
```

The GNN trunk is much smaller in activations. Late game with all 22 pieces placed the gap narrows, but the board is still sparser than a 17×17 grid.

## Policy Head (Pointer Network)

AlphaZero requires a joint distribution over all moves in a single forward pass. For GNN this is done as a pointer network — the output "points back" to one of the input candidates.

For each legal move (piece node i → destination node j), compute a score:

```
score(i, j) = MLP(embedding_i || embedding_j)
```

Mask illegal (piece, destination) pairs, softmax over the rest → policy distribution.

If there are 10 placed pieces and 15 candidate destinations, you score up to 150 pairs. No fixed grid needed — the output space is exactly the set of legal moves for this position.

## Value Head

Global mean pooling over all node embeddings → MLP → scalar. Same as graph classification in standard GNN literature.

## Batching

Variable-size graphs don't batch as cleanly as fixed tensors. PyTorch Geometric handles this by stacking graphs into one large block-diagonal graph per batch — each game becomes a disconnected subgraph with offset node indices.

This works but has costs:
- Uneven graph sizes (early game: ~10 nodes, late game: ~35 nodes) → uneven GPU utilization
- More bookkeeping overhead than CNN batching
- The Rust MCTS batching assumes uniform structure — needs adjustment

In practice this is manageable. PYG's `DataLoader` with `Batch.from_data_list()` handles it automatically.

## Open Questions

- Does the relational inductive bias of GNN actually help over CNN for Hive? The CNN already implicitly learns connectivity patterns. Needs empirical testing.
- Graph Transformer vs message passing + global node: attention is cleaner but less standard in game-playing literature.
- Reserve pieces are disconnected nodes until placed — does the GNN learn useful representations for them without structural context?
