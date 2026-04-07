/// Movement rules for each Hive piece type.

use crate::hex::{Hex, DIRECTIONS, hex_add, hex_neighbors};
use crate::board::Board;
use crate::piece::{Piece, PieceColor, PieceType};

/// Get all legal destination hexes for a piece already on the board.
/// Temporarily removes the piece from the board to compute moves, then restores it.
pub fn get_moves(piece: Piece, board: &mut Board, articulation_points: &[Hex]) -> Vec<Hex> {
    let pos = match board.piece_position(piece) {
        Some(p) => p,
        None => return Vec::new(),
    };

    // Piece must be on top of its stack
    if board.top_piece(pos) != Some(piece) {
        return Vec::new();
    }

    // One Hive Rule: pieces on stacks can always move
    if board.stack_height(pos) == 1 {
        if articulation_points.contains(&pos) {
            return Vec::new();
        }
    }

    // Temporarily remove piece from board
    board.remove_piece(piece).unwrap();

    let moves = match piece.piece_type() {
        PieceType::Queen => queen_moves(pos, board),
        PieceType::Spider => spider_moves(pos, board),
        PieceType::Beetle => beetle_moves(pos, board),
        PieceType::Grasshopper => grasshopper_moves(pos, board),
        PieceType::Ant => ant_moves(pos, board),
    };

    // Restore piece
    board.place_piece(piece, pos).unwrap();

    moves
}

/// Get all legal placement positions for a player.
pub fn get_placements(color: PieceColor, board: &Board) -> Vec<Hex> {
    let my_pieces = board.pieces_on_board(color);
    let opp_color = color.opposite();
    let opp_pieces = board.pieces_on_board(opp_color);

    // First piece: place at origin
    if my_pieces.is_empty() && opp_pieces.is_empty() {
        return vec![(0, 0)];
    }

    // Second piece: adjacent to existing pieces
    if my_pieces.is_empty() && !opp_pieces.is_empty() {
        let mut result = Vec::new();
        for p in &opp_pieces {
            let pos = board.piece_position(*p).unwrap();
            for n in board.empty_neighbors(pos) {
                if !result.contains(&n) {
                    result.push(n);
                }
            }
        }
        return result;
    }

    // Normal: adjacent to friendly, not adjacent to enemy
    let friendly_positions: Vec<Hex> = my_pieces
        .iter()
        .filter_map(|p| board.piece_position(*p))
        .collect();

    // Collect all empty hexes adjacent to friendly pieces
    let mut candidates = Vec::new();
    for &fpos in &friendly_positions {
        for n in board.empty_neighbors(fpos) {
            if !candidates.contains(&n) {
                candidates.push(n);
            }
        }
    }

    // Filter out those adjacent to enemy pieces
    let mut valid = Vec::new();
    for cand in candidates {
        let mut adjacent_to_enemy = false;
        for &n in hex_neighbors(cand).iter() {
            if let Some(top) = board.top_piece(n) {
                if top.color() == opp_color {
                    adjacent_to_enemy = true;
                    break;
                }
            }
        }
        if !adjacent_to_enemy {
            valid.push(cand);
        }
    }

    valid
}

// ---- Piece-specific movement ----
// All functions below assume the piece has already been removed from `pos`.

fn queen_moves(pos: Hex, board: &Board) -> Vec<Hex> {
    let mut moves = Vec::new();
    for &n in hex_neighbors(pos).iter() {
        if !board.is_occupied(n) && board.can_slide(pos, n) {
            if hex_neighbors(n).iter().any(|&adj| board.is_occupied(adj)) {
                moves.push(n);
            }
        }
    }
    moves
}

fn beetle_moves(pos: Hex, board: &Board) -> Vec<Hex> {
    let mut moves = Vec::new();
    let src_height = board.stack_height(pos);

    for &n in hex_neighbors(pos).iter() {
        let dst_height = board.stack_height(n);

        if src_height == 0 && dst_height == 0 {
            // Ground level: normal slide rules
            if board.can_slide(pos, n) {
                if hex_neighbors(n).iter().any(|&adj| board.is_occupied(adj)) {
                    moves.push(n);
                }
            }
        } else {
            // Elevated movement: no gate blocking
            if dst_height > 0 || hex_neighbors(n).iter().any(|&adj| board.is_occupied(adj)) {
                moves.push(n);
            }
        }
    }

    moves
}

fn grasshopper_moves(pos: Hex, board: &Board) -> Vec<Hex> {
    let mut moves = Vec::new();
    for &d in DIRECTIONS.iter() {
        let mut current = hex_add(pos, d);
        if !board.is_occupied(current) {
            continue; // must jump over at least one piece
        }
        while board.is_occupied(current) {
            current = hex_add(current, d);
        }
        moves.push(current);
    }
    moves
}

fn spider_moves(pos: Hex, board: &Board) -> Vec<Hex> {
    walk_exactly_n(pos, 3, board)
}

fn ant_moves(pos: Hex, board: &Board) -> Vec<Hex> {
    walk_any(pos, board)
}

/// Find all hexes reachable by sliding exactly n steps.
fn walk_exactly_n(start: Hex, n: usize, board: &Board) -> Vec<Hex> {
    let mut results = Vec::new();

    struct State {
        pos: Hex,
        steps: usize,
        visited: Vec<Hex>,
    }

    let mut frontier = vec![State {
        pos: start,
        steps: 0,
        visited: vec![start],
    }];

    while let Some(state) = frontier.pop() {
        if state.steps == n {
            if state.pos != start && !results.contains(&state.pos) {
                results.push(state.pos);
            }
            continue;
        }

        for &neighbor in hex_neighbors(state.pos).iter() {
            if board.is_occupied(neighbor) {
                continue;
            }
            if state.visited.contains(&neighbor) {
                continue;
            }
            if !board.can_slide(state.pos, neighbor) {
                continue;
            }
            if !hex_neighbors(neighbor).iter().any(|&adj| board.is_occupied(adj)) {
                continue;
            }
            let mut new_visited = state.visited.clone();
            new_visited.push(neighbor);
            frontier.push(State {
                pos: neighbor,
                steps: state.steps + 1,
                visited: new_visited,
            });
        }
    }

    results
}

/// Find all hexes reachable by sliding any number of steps (ant movement).
fn walk_any(start: Hex, board: &Board) -> Vec<Hex> {
    let mut visited = vec![start];
    let mut results = Vec::new();
    let mut queue = vec![start];

    while let Some(pos) = queue.pop() {
        for &neighbor in hex_neighbors(pos).iter() {
            if board.is_occupied(neighbor) {
                continue;
            }
            if visited.contains(&neighbor) {
                continue;
            }
            if !board.can_slide(pos, neighbor) {
                continue;
            }
            if !hex_neighbors(neighbor).iter().any(|&adj| board.is_occupied(adj)) {
                continue;
            }
            visited.push(neighbor);
            results.push(neighbor);
            queue.push(neighbor);
        }
    }

    results
}

#[cfg(test)]
#[allow(unused_must_use)]
mod tests {
    use super::*;

    fn setup_basic_board() -> Board {
        let mut board = Board::new();
        // Place white queen at origin and black queen adjacent
        let wq = Piece::new(PieceColor::White, PieceType::Queen, 1);
        let bq = Piece::new(PieceColor::Black, PieceType::Queen, 1);
        board.place_piece(wq, (0, 0));
        board.place_piece(bq, (1, 0));
        board
    }

    #[test]
    fn test_first_placement() {
        let board = Board::new();
        let placements = get_placements(PieceColor::White, &board);
        assert_eq!(placements, vec![(0, 0)]);
    }

    #[test]
    fn test_second_placement() {
        let mut board = Board::new();
        let wq = Piece::new(PieceColor::White, PieceType::Queen, 1);
        board.place_piece(wq, (0, 0));

        let placements = get_placements(PieceColor::Black, &board);
        assert_eq!(placements.len(), 6);
    }

    #[test]
    fn test_queen_moves() {
        let mut board = setup_basic_board();
        let wq = Piece::new(PieceColor::White, PieceType::Queen, 1);
        let moves = get_moves(wq, &mut board, &[]);
        // Queen should be able to slide to 2 positions adjacent to both pieces
        assert!(!moves.is_empty());
    }

    #[test]
    fn test_grasshopper_jump() {
        let mut board = Board::new();
        let wq = Piece::new(PieceColor::White, PieceType::Queen, 1);
        let bq = Piece::new(PieceColor::Black, PieceType::Queen, 1);
        let wg = Piece::new(PieceColor::White, PieceType::Grasshopper, 1);

        board.place_piece(wq, (0, 0));
        board.place_piece(bq, (1, 0));
        board.place_piece(wg, (-1, 0));

        let moves = get_moves(wg, &mut board, &[]);
        // Should jump over wq and bq to land at (2, 0)
        assert!(moves.contains(&(2, 0)));
    }

    #[test]
    fn test_beetle_stacking() {
        let mut board = Board::new();
        let wq = Piece::new(PieceColor::White, PieceType::Queen, 1);
        let bq = Piece::new(PieceColor::Black, PieceType::Queen, 1);
        let wb = Piece::new(PieceColor::White, PieceType::Beetle, 1);

        board.place_piece(wq, (0, 0));
        board.place_piece(bq, (1, 0));
        board.place_piece(wb, (0, -1)); // adjacent to wq

        // Add more pieces so beetle isn't an articulation point
        let ws = Piece::new(PieceColor::White, PieceType::Spider, 1);
        board.place_piece(ws, (1, -1));

        let moves = get_moves(wb, &mut board, &[]);
        // Beetle should be able to move onto occupied hexes
        assert!(moves.contains(&(0, 0)) || moves.contains(&(1, 0)));
    }

    #[test]
    fn test_articulation_point_blocks_move() {
        let mut board = Board::new();
        let wq = Piece::new(PieceColor::White, PieceType::Queen, 1);
        let bq = Piece::new(PieceColor::Black, PieceType::Queen, 1);
        let wa = Piece::new(PieceColor::White, PieceType::Ant, 1);

        board.place_piece(wq, (0, 0));
        board.place_piece(bq, (1, 0));
        board.place_piece(wa, (2, 0));

        let aps = board.articulation_points();
        // bq at (1,0) is an articulation point
        let moves = get_moves(bq, &mut board, &aps);
        assert!(moves.is_empty());
    }
}
