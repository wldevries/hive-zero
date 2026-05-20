/// Token-based Hive encoding for the transformer policy/value network.
///
/// A Hive position is encoded as a sequence of L = SEQ_LEN tokens. Each token
/// is a fixed-width record of categorical fields, axial position, and a small
/// flag bundle. Token slots are assigned in this order:
///
///   [0]                : `[GAME]` token (CLS-style, reads value + aux heads)
///   [1 .. 1+M]         : `PIECE` tokens — one per piece on the board.
///                        Stacked pieces share `(q, r)` and are distinguished
///                        by the per-token `z` (position-in-stack: 0=bottom,
///                        height-1=top). FLAG_TOP_OF_STACK and FLAG_BURIED
///                        encode the top/buried split.
///   [1+M .. 1+M+R]     : `RESERVE` tokens — one per (color, base_type) with
///                        remaining count > 0 (mine first, then opponent)
///   [1+M+R .. used]    : `CELL` tokens — one per unique destination hex
///                        appearing in any legal move from this state
///   [used .. SEQ_LEN]  : `[PAD]` tokens — masked out by attention
///
/// Movers are RESERVE tokens (for placements) and top-of-stack PIECE tokens
/// (for movements). Destinations are CELL tokens. Each legal move maps to a
/// unique `(mover_slot, dest_slot)` pair scored by the bilinear Q·K policy
/// head:
///
///   logit(move) = (Q[mover_slot] · K[dest_slot]) / sqrt(BILINEAR_DIM)
///
/// The MCTS policy layout in the flat `policy[B, 2*L*D]` tensor returned by
/// the inference callback is `[Q (L*D) || K (L*D)]`, with **each block in
/// D-major order** — i.e. `policy[d * L + token_slot]`. This matches the
/// existing `PolicyIndex::DotProduct` math (`policy[q_offset + d * g2 + src]`
/// with `g2 = L`). Python emits `Q` and `K` as `(B, L, D)` and permutes to
/// `(B, D, L)` before flattening.
///
/// All token features are current-player-relative: `color = 1` means "mine"
/// regardless of which physical color the player controls. The model never
/// sees a side-to-move bit; the encoding *is* always from the mover's POV.

use std::collections::HashMap;

use core_game::game::PolicyIndex;
use core_game::hex::{Hex, hex_distance};

use crate::game::{Game, Move};
use crate::piece::{Piece, PieceColor, PieceType, ALL_PIECE_TYPES, PIECE_COUNTS};

// -- vocabulary sizes -------------------------------------------------------

/// Padded token sequence length. Mid-game positions typically use 40–80
/// slots; the cap accommodates the (extremely rare) case of every ant being
/// unpinned with a wide frontier.
pub const SEQ_LEN: usize = 128;

/// Bilinear embedding dimension for the Q·K policy head. See plan: chosen to
/// match transformer head-dim convention rather than the CNN-era D=32.
pub const BILINEAR_DIM: usize = 64;

/// Number of per-token flag floats. See `FLAG_*` indices below.
pub const F_FLAGS: usize = 8;

// -- token kinds ------------------------------------------------------------

pub const TOKEN_KIND_PAD: u8 = 0;
pub const TOKEN_KIND_GAME: u8 = 1;
pub const TOKEN_KIND_PIECE: u8 = 2;
pub const TOKEN_KIND_RESERVE: u8 = 3;
pub const TOKEN_KIND_CELL: u8 = 4;
pub const NUM_TOKEN_KINDS: usize = 5;

// -- colors -----------------------------------------------------------------

pub const COLOR_NONE: u8 = 0;
pub const COLOR_MINE: u8 = 1;
pub const COLOR_OPP: u8 = 2;
pub const NUM_COLORS: usize = 3;

// -- piece types (current-player-relative: 1..=5 for base types, 0 = none) --

pub const PIECE_TYPE_NONE: u8 = 0;
pub const PIECE_TYPE_QUEEN: u8 = 1;
pub const PIECE_TYPE_SPIDER: u8 = 2;
pub const PIECE_TYPE_BEETLE: u8 = 3;
pub const PIECE_TYPE_GRASSHOPPER: u8 = 4;
pub const PIECE_TYPE_ANT: u8 = 5;
pub const NUM_PIECE_TYPE_TOKENS: usize = 6;

// -- per-token flag indices (into TokenBatch::flags) ------------------------

pub const FLAG_PINNED: usize = 0;
/// Set on every PIECE token where `z == height - 1` (top of its stack — true
/// for solo pieces too). Zero on buried PIECE tokens and on every non-PIECE
/// token. Anti-correlated with FLAG_BURIED for PIECE tokens.
pub const FLAG_TOP_OF_STACK: usize = 1;
pub const FLAG_ADJ_MY_QUEEN: usize = 2;
pub const FLAG_ADJ_OPP_QUEEN: usize = 3;
/// Set on PIECE tokens with at least one piece on top of them (z < height-1).
/// Reuses the slot index that previously held `FLAG_ON_FRONTIER` (which was a
/// perfect duplicate of `kind == CELL` and carried no information).
pub const FLAG_BURIED: usize = 4;
pub const FLAG_DIST_MY_QUEEN: usize = 5;
pub const FLAG_DIST_OPP_QUEEN: usize = 6;
pub const FLAG_RESERVED: usize = 7;

// -- stack depth max (clamp; deepest realistic stack is 7) -----------------

pub const MAX_STACK_DEPTH: u8 = 7;

// -- policy layout helpers --------------------------------------------------

/// Total size of the flat policy tensor returned by the NN: `2 * L * D`.
pub const POLICY_SIZE: usize = 2 * SEQ_LEN * BILINEAR_DIM;
pub const Q_SECTION_OFFSET: usize = 0;
pub const K_SECTION_OFFSET: usize = SEQ_LEN * BILINEAR_DIM;

#[inline]
fn base_piece_type_byte(t: PieceType) -> Option<u8> {
    match t {
        PieceType::Queen => Some(PIECE_TYPE_QUEEN),
        PieceType::Spider => Some(PIECE_TYPE_SPIDER),
        PieceType::Beetle => Some(PIECE_TYPE_BEETLE),
        PieceType::Grasshopper => Some(PIECE_TYPE_GRASSHOPPER),
        PieceType::Ant => Some(PIECE_TYPE_ANT),
        PieceType::Mosquito | PieceType::Ladybug | PieceType::Pillbug => None,
    }
}

#[inline]
fn relative_color(color: PieceColor, my_color: PieceColor) -> u8 {
    if color == my_color { COLOR_MINE } else { COLOR_OPP }
}

/// A batch of token features for a single Hive position. Buffers are
/// pre-padded to `SEQ_LEN` and `mask[i]` marks real (non-pad) slots.
#[derive(Clone, Debug)]
pub struct TokenBatch {
    /// Token kind per slot — one of `TOKEN_KIND_*`.
    pub kind: Vec<u8>,
    /// Piece type per slot (current-player-relative encoding for piece-type IDs).
    /// `PIECE_TYPE_NONE` for non-piece tokens.
    pub piece_type: Vec<u8>,
    /// Color per slot — `COLOR_MINE` / `COLOR_OPP` for piece/reserve tokens,
    /// `COLOR_NONE` otherwise.
    pub color: Vec<u8>,
    /// Position-in-stack (0..=MAX_STACK_DEPTH-1, 0=bottom) for PIECE tokens,
    /// 0 elsewhere. Pieces in a stack share `(q, r)`; `z` is the per-piece
    /// vertical coordinate that distinguishes them.
    pub z: Vec<u8>,
    /// Remaining count (1..=N) for RESERVE tokens, 0 elsewhere.
    pub count: Vec<u8>,
    /// Axial q coordinate for positional tokens (PIECE, CELL); 0 elsewhere.
    pub q: Vec<i8>,
    /// Axial r coordinate for positional tokens; 0 elsewhere.
    pub r: Vec<i8>,
    /// Flag/scalar bundle laid out as `[SEQ_LEN, F_FLAGS]` row-major.
    pub flags: Vec<f32>,
    /// Attention mask — `true` for real tokens, `false` for pad slots.
    pub mask: Vec<bool>,
    /// Number of real (non-pad) tokens in slots `0..used`.
    pub used: usize,
}

impl TokenBatch {
    pub fn new() -> Self {
        Self {
            kind: vec![TOKEN_KIND_PAD; SEQ_LEN],
            piece_type: vec![PIECE_TYPE_NONE; SEQ_LEN],
            color: vec![COLOR_NONE; SEQ_LEN],
            z: vec![0u8; SEQ_LEN],
            count: vec![0u8; SEQ_LEN],
            q: vec![0i8; SEQ_LEN],
            r: vec![0i8; SEQ_LEN],
            flags: vec![0.0f32; SEQ_LEN * F_FLAGS],
            mask: vec![false; SEQ_LEN],
            used: 0,
        }
    }

    #[inline]
    fn set_flag(&mut self, slot: usize, flag: usize, v: f32) {
        self.flags[slot * F_FLAGS + flag] = v;
    }
}

/// Tokenize the current game state into a `TokenBatch` and return the legal
/// moves paired with `PolicyIndex::DotProduct` lookups over (mover, dest)
/// token slots. Empty `indexed_moves` means a pass is required (terminal
/// positions are handled by the MCTS caller before this is invoked).
pub fn tokenize_and_priors(game: &mut Game) -> (TokenBatch, Vec<(PolicyIndex, Move)>) {
    let mut tokens = TokenBatch::new();
    let mut slot: usize = 0;

    let my_color = game.turn_color;
    let opp_color = my_color.opposite();
    let my_queen = Piece::new(my_color, PieceType::Queen, 1);
    let opp_queen = Piece::new(opp_color, PieceType::Queen, 1);
    let my_queen_pos = game.board.piece_position(my_queen);
    let opp_queen_pos = game.board.piece_position(opp_queen);
    let articulation_points = game.board.articulation_points();

    // ---- [GAME] token ----
    tokens.kind[slot] = TOKEN_KIND_GAME;
    if my_queen_pos.is_some() {
        tokens.set_flag(slot, FLAG_RESERVED, 1.0);
    }
    if opp_queen_pos.is_some() {
        // We piggyback FLAG_ADJ_OPP_QUEEN on the GAME token as "opp queen placed";
        // its standard per-cell meaning doesn't apply here.
        tokens.set_flag(slot, FLAG_ADJ_OPP_QUEEN, 1.0);
    }
    tokens.mask[slot] = true;
    slot += 1;

    // ---- PIECE tokens ----
    // One token per piece on the board (not per cell). A stack of N pieces
    // produces N tokens sharing `(q, r)` but with distinct `z` (0=bottom,
    // height-1=top). Only the top piece of each stack is a legal movement
    // mover, so we record only top-of-stack slots in `hex_to_top_piece_slot`.
    let mut hex_to_top_piece_slot: HashMap<Hex, usize> = HashMap::new();
    for (pos, stack) in game.board.iter_occupied() {
        let stack_height = stack.height();
        if stack_height == 0 { continue; }
        let is_articulation = articulation_points.contains(&pos);
        // Queen distance / adjacency are stack-level facts; compute once.
        let (my_q_dist_flag, my_q_adj) = match my_queen_pos {
            Some(qp) => {
                let d = hex_distance(pos, qp);
                (normalize_dist(d as i32), d == 1)
            }
            None => (1.0, false),
        };
        let (opp_q_dist_flag, opp_q_adj) = match opp_queen_pos {
            Some(qp) => {
                let d = hex_distance(pos, qp);
                (normalize_dist(d as i32), d == 1)
            }
            None => (1.0, false),
        };

        for (z, piece) in stack.iter().enumerate() {
            if slot >= SEQ_LEN { break; }
            let is_top = z as u8 == stack_height - 1;
            let pt_byte = match base_piece_type_byte(piece.piece_type()) {
                Some(b) => b,
                None => PIECE_TYPE_NONE, // expansion pieces show up as "unknown type"
            };

            tokens.kind[slot] = TOKEN_KIND_PIECE;
            tokens.piece_type[slot] = pt_byte;
            tokens.color[slot] = relative_color(piece.color(), my_color);
            tokens.z[slot] = (z as u8).min(MAX_STACK_DEPTH - 1);
            tokens.q[slot] = pos.0;
            tokens.r[slot] = pos.1;

            if is_top {
                tokens.set_flag(slot, FLAG_TOP_OF_STACK, 1.0);
            } else {
                tokens.set_flag(slot, FLAG_BURIED, 1.0);
            }
            // PINNED only applies when the top piece sits on a height-1 stack
            // that's an articulation point. Buried pieces can't move by
            // definition, and beetles on a stack of >= 2 can always move.
            if is_top && is_articulation && stack_height == 1 {
                tokens.set_flag(slot, FLAG_PINNED, 1.0);
            }
            // Queen flags apply only to the top piece — the model reads
            // "what's at this hex" through the top token's neighborhood
            // features; buried pieces don't add queen-distance signal.
            if is_top {
                if my_q_adj { tokens.set_flag(slot, FLAG_ADJ_MY_QUEEN, 1.0); }
                tokens.set_flag(slot, FLAG_DIST_MY_QUEEN, my_q_dist_flag);
                if opp_q_adj { tokens.set_flag(slot, FLAG_ADJ_OPP_QUEEN, 1.0); }
                tokens.set_flag(slot, FLAG_DIST_OPP_QUEEN, opp_q_dist_flag);
            }
            tokens.mask[slot] = true;

            if is_top {
                hex_to_top_piece_slot.insert(pos, slot);
            }
            slot += 1;
        }
    }

    // ---- RESERVE tokens ----
    // One per (color, base_type) with count > 0. Mine first so move-mover
    // lookup is a simple HashMap; opponent reserves are info-only.
    let mut my_reserve_slot: HashMap<PieceType, usize> = HashMap::new();
    for (i, &pt) in ALL_PIECE_TYPES.iter().enumerate() {
        let pt_byte = match base_piece_type_byte(pt) {
            Some(b) => b,
            None => continue, // skip expansion piece types
        };
        let count = game.reserve_count(my_color, pt);
        if count == 0 { continue; }
        if slot >= SEQ_LEN { break; }
        tokens.kind[slot] = TOKEN_KIND_RESERVE;
        tokens.piece_type[slot] = pt_byte;
        tokens.color[slot] = COLOR_MINE;
        tokens.count[slot] = count.min(PIECE_COUNTS[i]);
        tokens.mask[slot] = true;
        my_reserve_slot.insert(pt, slot);
        slot += 1;
    }
    for (i, &pt) in ALL_PIECE_TYPES.iter().enumerate() {
        let pt_byte = match base_piece_type_byte(pt) {
            Some(b) => b,
            None => continue,
        };
        let count = game.reserve_count(opp_color, pt);
        if count == 0 { continue; }
        if slot >= SEQ_LEN { break; }
        tokens.kind[slot] = TOKEN_KIND_RESERVE;
        tokens.piece_type[slot] = pt_byte;
        tokens.color[slot] = COLOR_OPP;
        tokens.count[slot] = count.min(PIECE_COUNTS[i]);
        tokens.mask[slot] = true;
        slot += 1;
    }

    // ---- CELL tokens ----
    // Enumerate legal moves (using the same `valid_moves()` path that
    // dedups placements to one piece per type) and collect the unique
    // destination hexes. Each becomes one CELL token.
    let valid_moves = game.valid_moves();
    let mut hex_to_cell_slot: HashMap<Hex, usize> = HashMap::new();
    let mut dest_order: Vec<Hex> = Vec::with_capacity(valid_moves.len());
    for mv in &valid_moves {
        if let Some(dest) = mv.to {
            if !hex_to_cell_slot.contains_key(&dest) {
                hex_to_cell_slot.insert(dest, usize::MAX); // placeholder, slot assigned below
                dest_order.push(dest);
            }
        }
    }
    // Stable order for reproducibility (and so D6 symmetry tests can compare
    // before/after deterministically).
    dest_order.sort();
    for dest in &dest_order {
        if slot >= SEQ_LEN {
            debug_assert!(false, "token sequence overflow — SEQ_LEN={} too small for legal move set", SEQ_LEN);
            break;
        }
        tokens.kind[slot] = TOKEN_KIND_CELL;
        tokens.q[slot] = dest.0;
        tokens.r[slot] = dest.1;
        if let Some(qp) = my_queen_pos {
            let d = hex_distance(*dest, qp);
            if d == 1 { tokens.set_flag(slot, FLAG_ADJ_MY_QUEEN, 1.0); }
            tokens.set_flag(slot, FLAG_DIST_MY_QUEEN, normalize_dist(d as i32));
        } else {
            tokens.set_flag(slot, FLAG_DIST_MY_QUEEN, 1.0);
        }
        if let Some(qp) = opp_queen_pos {
            let d = hex_distance(*dest, qp);
            if d == 1 { tokens.set_flag(slot, FLAG_ADJ_OPP_QUEEN, 1.0); }
            tokens.set_flag(slot, FLAG_DIST_OPP_QUEEN, normalize_dist(d as i32));
        } else {
            tokens.set_flag(slot, FLAG_DIST_OPP_QUEEN, 1.0);
        }
        tokens.mask[slot] = true;

        hex_to_cell_slot.insert(*dest, slot);
        slot += 1;
    }

    // Also flag hive-edge cells that don't appear as legal destinations —
    // these never become CELL tokens, so the model implicitly learns
    // "non-token = illegal." Empty-cell features on PIECE tokens (no-op for
    // them) are left zero. This is intentional: we don't pre-create CELL
    // tokens for non-destination cells to keep the sequence short.

    tokens.used = slot;

    // ---- Build legal-move index ----
    let mut indexed: Vec<(PolicyIndex, Move)> = Vec::with_capacity(valid_moves.len());
    for mv in valid_moves.into_iter() {
        let dest = match mv.to { Some(d) => d, None => continue };
        let dest_slot = match hex_to_cell_slot.get(&dest) {
            Some(&s) if s != usize::MAX => s,
            _ => continue, // dest overflowed the cap — drop the move
        };
        let mover_slot_opt: Option<usize> = if let Some(src) = mv.from {
            // Only the top of each stack is a valid movement mover.
            hex_to_top_piece_slot.get(&src).copied()
        } else if let Some(piece) = mv.piece {
            my_reserve_slot.get(&piece.piece_type()).copied()
        } else {
            None
        };
        let mover_slot = match mover_slot_opt {
            Some(s) => s,
            None => continue,
        };

        indexed.push((
            PolicyIndex::DotProduct {
                q_offset: Q_SECTION_OFFSET,
                k_offset: K_SECTION_OFFSET,
                src_cell: mover_slot,
                dst_cell: dest_slot,
                embed_dim: BILINEAR_DIM,
                g2: SEQ_LEN,
            },
            mv,
        ));
    }

    (tokens, indexed)
}

/// Normalize a hex distance into [0, 1] for the per-token distance features.
/// Beyond 6 the queen is functionally "far" and the gradient is uninformative,
/// so we saturate.
#[inline]
fn normalize_dist(d: i32) -> f32 {
    (d as f32 / 6.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;
    use crate::piece::{Piece, PieceColor, PieceType};

    #[test]
    fn empty_game_has_game_and_my_reserve_tokens_only() {
        let mut g = Game::new();
        let (tb, indexed) = tokenize_and_priors(&mut g);

        // 1 GAME + 5 my reserve types + 5 opp reserve types = 11 tokens, but
        // on an empty board there are no pieces and no CELL tokens until we
        // enumerate moves... we DO enumerate moves: first placement.
        // first ply: 5 placement options at (0,0). So dest cells = 1.
        // total = 1 + 0 piece + 5 my res + 5 opp res + 1 cell = 12.
        assert_eq!(tb.used, 12);
        assert_eq!(tb.kind[0], TOKEN_KIND_GAME);
        // first piece tokens... none. Reserve tokens follow.
        assert_eq!(tb.kind[1], TOKEN_KIND_RESERVE);
        assert_eq!(tb.kind[6], TOKEN_KIND_RESERVE); // opp reserves
        assert_eq!(tb.kind[11], TOKEN_KIND_CELL);
        // 5 placements, each (queen/spider/beetle/grasshopper/ant) at (0,0)
        // Tournament rule is off by default, so queen is among them.
        assert_eq!(indexed.len(), 5);
        // All point at the same dest token (the (0,0) CELL slot=11) but
        // different mover slots (5 distinct reserve tokens).
        let dest_slots: std::collections::HashSet<usize> = indexed.iter()
            .filter_map(|(p, _)| if let PolicyIndex::DotProduct { dst_cell, .. } = p { Some(*dst_cell) } else { None })
            .collect();
        assert_eq!(dest_slots.len(), 1);
        let mover_slots: std::collections::HashSet<usize> = indexed.iter()
            .filter_map(|(p, _)| if let PolicyIndex::DotProduct { src_cell, .. } = p { Some(*src_cell) } else { None })
            .collect();
        assert_eq!(mover_slots.len(), 5);
    }

    #[test]
    fn piece_token_carries_top_of_stack_and_pinned_flags() {
        // Build a 3-piece chain wQ at (0,0) — bS1 at (1,0) — bQ at (2,0)
        // directly via test_position to avoid recentering. bS1 is the
        // middle of the chain and is the articulation point.
        // It's Black's turn so my_color = Black → bS1 = mine.
        let wq  = Piece::new(PieceColor::White, PieceType::Queen, 1);
        let bs1 = Piece::new(PieceColor::Black, PieceType::Spider, 1);
        let bq  = Piece::new(PieceColor::Black, PieceType::Queen, 1);
        let mut g = Game::test_position(
            &[(wq, (0, 0)), (bs1, (1, 0)), (bq, (2, 0))],
            PieceColor::Black,
            3,
            23,
        );

        let (tb, _) = tokenize_and_priors(&mut g);
        let mid_slot = (0..tb.used).find(|&i|
            tb.kind[i] == TOKEN_KIND_PIECE
            && tb.color[i] == COLOR_MINE
            && tb.piece_type[i] == PIECE_TYPE_SPIDER
        ).expect("bS1 piece token should exist");
        assert_eq!(tb.flags[mid_slot * F_FLAGS + FLAG_TOP_OF_STACK], 1.0);
        assert_eq!(tb.flags[mid_slot * F_FLAGS + FLAG_PINNED], 1.0,
            "bS1 should be pinned — it's the middle of a 3-piece chain");

        // The end pieces (wQ, bQ) are NOT articulation points.
        let wq_slot = (0..tb.used).find(|&i|
            tb.kind[i] == TOKEN_KIND_PIECE
            && tb.color[i] == COLOR_OPP
            && tb.piece_type[i] == PIECE_TYPE_QUEEN
        ).expect("wQ piece token should exist");
        assert_eq!(tb.flags[wq_slot * F_FLAGS + FLAG_PINNED], 0.0,
            "wQ at chain end should not be pinned");
    }

    #[test]
    fn stacked_pieces_get_one_token_each_with_z_and_burial_flags() {
        // Place wQ at (0,0), then bB1 on top of it. It's Black's turn so
        // bB1 (top) is MINE and wQ (buried) is OPP. The stack should
        // produce two PIECE tokens sharing (q,r)=(0,0).
        let wq = Piece::new(PieceColor::White, PieceType::Queen, 1);
        let bb1 = Piece::new(PieceColor::Black, PieceType::Beetle, 1);
        let mut g = Game::test_position(
            &[(wq, (0, 0)), (bb1, (0, 0))],
            PieceColor::Black,
            2,
            23,
        );

        let (tb, indexed) = tokenize_and_priors(&mut g);

        // Find the two PIECE tokens at (0,0).
        let piece_slots: Vec<usize> = (0..tb.used)
            .filter(|&i| tb.kind[i] == TOKEN_KIND_PIECE && tb.q[i] == 0 && tb.r[i] == 0)
            .collect();
        assert_eq!(piece_slots.len(), 2, "stack of 2 must produce 2 PIECE tokens");

        // Identify bottom (z=0) and top (z=1) by their z field.
        let bottom = *piece_slots.iter().find(|&&s| tb.z[s] == 0)
            .expect("bottom z=0 token must exist");
        let top = *piece_slots.iter().find(|&&s| tb.z[s] == 1)
            .expect("top z=1 token must exist");

        // Bottom: wQ → opp queen, BURIED set, TOP_OF_STACK clear.
        assert_eq!(tb.piece_type[bottom], PIECE_TYPE_QUEEN);
        assert_eq!(tb.color[bottom], COLOR_OPP);
        assert_eq!(tb.flags[bottom * F_FLAGS + FLAG_BURIED], 1.0);
        assert_eq!(tb.flags[bottom * F_FLAGS + FLAG_TOP_OF_STACK], 0.0);

        // Top: bB1 → my beetle, TOP_OF_STACK set, BURIED clear.
        assert_eq!(tb.piece_type[top], PIECE_TYPE_BEETLE);
        assert_eq!(tb.color[top], COLOR_MINE);
        assert_eq!(tb.flags[top * F_FLAGS + FLAG_TOP_OF_STACK], 1.0);
        assert_eq!(tb.flags[top * F_FLAGS + FLAG_BURIED], 0.0);

        // Movement moves should only originate from the TOP piece's slot, never
        // from the buried wQ's slot. Walk indexed_moves looking for the bB1
        // beetle moving (its `from` is (0,0)) and confirm mover_slot == top.
        for (pi, mv) in &indexed {
            if let Some(src) = mv.from {
                if src == (0, 0) {
                    let mover = match pi {
                        PolicyIndex::DotProduct { src_cell, .. } => *src_cell,
                        _ => panic!("expected DotProduct policy index"),
                    };
                    assert_eq!(mover, top,
                        "movement mover slot must be the TOP piece, never the buried wQ");
                }
            }
        }
    }

    #[test]
    fn legal_moves_pair_unique_mover_dest_slots() {
        let mut g = Game::new();
        let (_tb, indexed) = tokenize_and_priors(&mut g);
        // Each (mover_slot, dest_slot) pair should be unique in the legal-move
        // list — valid_moves() already dedups placements by piece-type, so we
        // never get two RESERVE tokens that share a (mover, dest) pair.
        let mut pairs: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        for (p, _) in &indexed {
            if let PolicyIndex::DotProduct { src_cell, dst_cell, .. } = p {
                assert!(pairs.insert((*src_cell, *dst_cell)),
                    "duplicate (mover, dest) pair: ({}, {})", src_cell, dst_cell);
            }
        }
    }

    #[test]
    fn policy_index_uses_token_seq_len_as_g2() {
        let mut g = Game::new();
        let (_tb, indexed) = tokenize_and_priors(&mut g);
        for (p, _) in &indexed {
            match p {
                PolicyIndex::DotProduct { embed_dim, g2, q_offset, k_offset, .. } => {
                    assert_eq!(*embed_dim, BILINEAR_DIM);
                    assert_eq!(*g2, SEQ_LEN);
                    assert_eq!(*q_offset, Q_SECTION_OFFSET);
                    assert_eq!(*k_offset, K_SECTION_OFFSET);
                }
                other => panic!("expected DotProduct, got {:?}", other),
            }
        }
    }
}
