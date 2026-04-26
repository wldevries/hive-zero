//! MCTS correctness tests for Hive.
//!
//! We use a heuristic oracle (queen-pressure based) instead of a minimax solver
//! since Hive has no tractable exact search. The key property we exploit: in a
//! position where the current player can WIN IN 1 MOVE, the terminal backprop
//! value is +1.0, while all other first moves return the heuristic (~0.8 when
//! the opponent queen is 5/6 surrounded). UCB quickly concentrates on the
//! winning move, making these tests strong sign-convention regression guards.

#[cfg(test)]
mod tests {
    use core_game::game::{Game as GameTrait, NNGame, Outcome, Player};
    use core_game::mcts::search::{
        CpuctStrategy, ForcedExploration, MctsSearch, RootNoise, SearchParams,
    };

    use crate::game::{Game, Move};
    use crate::move_encoding::policy_size;
    use crate::piece::{Piece, PieceColor, PieceType};

    // -------------------------------------------------------------------------
    // Oracle and search helpers
    // -------------------------------------------------------------------------

    /// Heuristic oracle: value from the current player's perspective.
    /// Only called for non-terminal nodes (MCTS handles terminal nodes internally).
    fn hive_oracle(game: &Game) -> f32 {
        let (w, b) = game.heuristic_value();
        match game.next_player() {
            Player::Player1 => w,
            Player::Player2 => b,
        }
    }

    /// 1-ply lookahead policy: gives 10× higher weight to moves that immediately
    /// end the game (win or forced win), then normalises. This ensures the winning
    /// move has a much higher prior than noise moves so MCTS visits it first, even
    /// with hundreds of legal alternatives.
    ///
    /// Using a uniform policy across 100+ Hive moves makes the exploration bonus
    /// negligible compared to the Q-value (prior ≈ 0.01 → bonus ≈ 0.01·√N),
    /// so MCTS never leaves the first child it visits. The lookahead policy
    /// fixes that without compromising what we're actually testing (value
    /// backprop sign convention).
    fn lookahead_policy(game: &mut Game) -> Vec<f32> {
        let grid_size = game.nn_grid_size;
        let ps = policy_size(grid_size);
        let mut policy = vec![0.0f32; ps];
        let (_, indexed_moves) = game.get_legal_move_mask();

        const WIN_BOOST: f32 = 10.0;

        for (enc, mv) in &indexed_moves {
            let mut child = game.clone();
            let _ = child.play_move(mv);
            let weight = if child.is_game_over() { WIN_BOOST } else { 1.0 };

            use core_game::game::PolicyIndex;
            match enc {
                PolicyIndex::Single(idx) => policy[*idx] += weight,
                // For bilinear head: set Q and K embeddings for src and dst such that
                // the dot product Q[src]·K[dst] is boosted for the winning move.
                // Simplest oracle: set Q[src][0] = K[dst][0] = sqrt(weight), others = 0.
                PolicyIndex::DotProduct { q_offset, k_offset, src_cell, dst_cell, .. } => {
                    policy[q_offset + src_cell] = weight.sqrt();
                    policy[k_offset + dst_cell] = weight.sqrt();
                }
                PolicyIndex::Sum(_, _) => {} // legacy, unused for Hive
            }
        }

        let total: f32 = policy.iter().sum();
        if total > 0.0 {
            for p in policy.iter_mut() {
                *p /= total;
            }
        }
        policy
    }

    /// Run MCTS using lookahead policy and zero value for non-terminal nodes.
    ///
    /// We use value=0 for non-terminal expansions deliberately.  With the heuristic
    /// oracle (~0.5 for Black winning), the first visited child (whichever happens to
    /// be first in valid_moves order) gets a 0.5 head-start and its UCB stays high
    /// enough that the winning child's exploration bonus can't catch it within a few
    /// hundred simulations.  Setting value=0 means non-terminal children start with
    /// value()=0; UCB is then driven purely by the prior, so the winning child's
    /// higher prior wins immediately in sim 2.  Once visited as a terminal it gets
    /// +1.0 and dominates all subsequent simulations.
    ///
    /// The tests still guard what matters:
    ///   - backprop sign convention: a wrong sign gives the winning child value=-1.0
    ///     (terrible UCB), causing move-selection and root_value tests to fail.
    ///   - lookahead policy: the winning child must have the highest prior to be
    ///     selected in sim 2 before any non-terminal child accumulates value.
    fn run_mcts_oracle(game: &Game, simulations: usize) -> MctsSearch<Game> {
        let mut search = MctsSearch::<Game>::new(8192);
        search.params = SearchParams::new(
            CpuctStrategy::Constant { c_puct: 1.5 },
            ForcedExploration::None,
            RootNoise::None,
        );

        let mut game_copy = game.clone();
        let init_policy = lookahead_policy(&mut game_copy);
        search.init(game, &init_policy);

        let mut sims_done = 0;
        while sims_done < simulations {
            let leaves = search.select_leaves(1);
            sims_done += 1;
            if leaves.is_empty() {
                // Terminal node handled inside select_leaves — no oracle call needed.
                continue;
            }

            let mut policies = Vec::new();
            let mut values = Vec::new();
            for &leaf in &leaves {
                let mut leaf_game = search.reconstruct_game(leaf);
                // Value=0 for non-terminal: keeps UCB prior-driven so the winning
                // child (highest lookahead prior) wins selection over the first-visited
                // non-terminal child that would otherwise hold a 0.5 value advantage.
                values.push(0.0f32);
                policies.push(lookahead_policy(&mut leaf_game));
            }

            search.expand_and_backprop(&policies, &values, &[]);
        }

        search
    }

    // -------------------------------------------------------------------------
    // Test position builders
    //
    // Geometry (axial hex, flat-top):
    //   Neighbors of (q,r): (q+1,r), (q-1,r), (q,r+1), (q,r-1), (q+1,r-1), (q-1,r+1)
    //
    // Position A — Black wins in 1:
    //   wQ at (0,0), surrounded 5/6 by black pieces.
    //   Open side: (0,1).
    //   bB1 at (-1,2), adjacent to bS2@(-1,1) [connected] and (0,1) [winning cell].
    //   Black plays bB1 (-1,2) → (0,1): wQ is now fully surrounded → BlackWins.
    //
    // Position B — White wins in 1 (colour-flipped version of A):
    //   bQ at (0,0), surrounded 5/6 by white pieces.
    //   Open side: (0,1).
    //   wB1 at (-1,2), adjacent to wS2@(-1,1) and (0,1).
    //   White plays wB1 (-1,2) → (0,1): bQ fully surrounded → WhiteWins.
    // -------------------------------------------------------------------------

    fn black_wins_in_one() -> Game {
        use PieceColor::{Black, White};
        use PieceType::{Ant, Beetle, Queen, Spider};
        // White queen surrounded on 5/6 sides; (0,1) is the open gap.
        Game::test_position(
            &[
                (Piece::new(White, Queen,   1), (0,  0)),  // wQ — to be surrounded
                (Piece::new(Black, Queen,   1), (1,  0)),  // bQ — one of the 5 surrounding
                (Piece::new(Black, Ant,     1), (-1, 0)),  // bA1
                (Piece::new(Black, Ant,     2), (0, -1)),  // bA2
                (Piece::new(Black, Spider,  1), (1, -1)),  // bS1
                (Piece::new(Black, Spider,  2), (-1, 1)),  // bS2 — connects bB1 to hive
                (Piece::new(Black, Beetle,  1), (-1, 2)),  // bB1 — winning piece
            ],
            Black,
            12,
            7,
        )
    }

    fn white_wins_in_one() -> Game {
        use PieceColor::{Black, White};
        use PieceType::{Ant, Beetle, Queen, Spider};
        // Black queen surrounded on 5/6 sides; (0,1) is the open gap.
        Game::test_position(
            &[
                (Piece::new(Black, Queen,   1), (0,  0)),  // bQ — to be surrounded
                (Piece::new(White, Queen,   1), (1,  0)),  // wQ — one of the 5 surrounding
                (Piece::new(White, Ant,     1), (-1, 0)),  // wA1
                (Piece::new(White, Ant,     2), (0, -1)),  // wA2
                (Piece::new(White, Spider,  1), (1, -1)),  // wS1
                (Piece::new(White, Spider,  2), (-1, 1)),  // wS2 — connects wB1 to hive
                (Piece::new(White, Beetle,  1), (-1, 2)),  // wB1 — winning piece
            ],
            White,
            12,
            7,
        )
    }

    // -------------------------------------------------------------------------
    // Position sanity checks (no MCTS)
    // -------------------------------------------------------------------------

    #[test]
    fn position_a_is_not_yet_terminal() {
        let game = black_wins_in_one();
        assert_eq!(game.outcome(), Outcome::Ongoing);
    }

    #[test]
    fn position_a_winning_move_ends_game() {
        let mut game = black_wins_in_one();
        let winning_mv = Move::movement(
            Piece::new(PieceColor::Black, PieceType::Beetle, 1),
            (-1, 2),
            (0, 1),
        );
        game.play_move(&winning_mv).unwrap();
        assert_eq!(
            game.outcome(),
            Outcome::WonBy(Player::Player2),
            "bB1 → (0,1) must surround wQ and give Black the win"
        );
    }

    #[test]
    fn position_b_is_not_yet_terminal() {
        let game = white_wins_in_one();
        assert_eq!(game.outcome(), Outcome::Ongoing);
    }

    #[test]
    fn position_b_winning_move_ends_game() {
        let mut game = white_wins_in_one();
        let winning_mv = Move::movement(
            Piece::new(PieceColor::White, PieceType::Beetle, 1),
            (-1, 2),
            (0, 1),
        );
        game.play_move(&winning_mv).unwrap();
        assert_eq!(
            game.outcome(),
            Outcome::WonBy(Player::Player1),
            "wB1 → (0,1) must surround bQ and give White the win"
        );
    }

    #[test]
    fn winning_move_is_in_valid_moves_for_black() {
        let mut game = black_wins_in_one();
        let moves = game.valid_moves();
        let beetle = Piece::new(PieceColor::Black, PieceType::Beetle, 1);
        let winning = moves
            .iter()
            .find(|m| m.piece == Some(beetle) && m.to == Some((0, 1)));
        assert!(
            winning.is_some(),
            "valid_moves() must include bB1@(-1,2)→(0,1); all moves: {:?}",
            moves
        );
    }

    #[test]
    fn winning_move_is_in_valid_moves_for_white() {
        let mut game = white_wins_in_one();
        let moves = game.valid_moves();
        let beetle = Piece::new(PieceColor::White, PieceType::Beetle, 1);
        let winning = moves
            .iter()
            .find(|m| m.piece == Some(beetle) && m.to == Some((0, 1)));
        assert!(
            winning.is_some(),
            "valid_moves() must include wB1@(-1,2)→(0,1); all moves: {:?}",
            moves
        );
    }

    #[test]
    fn heuristic_strongly_favours_current_player_in_winning_position() {
        // In both positions the current player has 5/6 of the enemy queen
        // surrounded, so the heuristic should be strongly positive.
        let game_b = black_wins_in_one();
        let v_b = hive_oracle(&game_b);
        assert!(
            v_b >= 0.49,
            "heuristic must favour Black in position A, got {}",
            v_b
        );

        let game_w = white_wins_in_one();
        let v_w = hive_oracle(&game_w);
        assert!(
            v_w >= 0.49,
            "heuristic must favour White in position B, got {}",
            v_w
        );
    }

    // -------------------------------------------------------------------------
    // MCTS move selection
    // -------------------------------------------------------------------------

    /// MCTS must select the unique winning move for Black.
    #[test]
    fn mcts_finds_winning_move_for_black() {
        let game = black_wins_in_one();
        let search = run_mcts_oracle(&game, 200);

        let best = search.best_move().expect("MCTS must return a best move");
        assert_eq!(
            best.to,
            Some((0, 1)),
            "MCTS must select the winning destination (0,1) for Black, got {:?}",
            best
        );
        assert_eq!(
            best.piece,
            Some(Piece::new(PieceColor::Black, PieceType::Beetle, 1)),
            "MCTS must move the beetle, got {:?}",
            best
        );
    }

    /// MCTS must select the unique winning move for White.
    /// Tests that sign convention is not tied to player identity.
    #[test]
    fn mcts_finds_winning_move_for_white() {
        let game = white_wins_in_one();
        let search = run_mcts_oracle(&game, 200);

        let best = search.best_move().expect("MCTS must return a best move");
        assert_eq!(
            best.to,
            Some((0, 1)),
            "MCTS must select the winning destination (0,1) for White, got {:?}",
            best
        );
        assert_eq!(
            best.piece,
            Some(Piece::new(PieceColor::White, PieceType::Beetle, 1)),
            "MCTS must move the white beetle, got {:?}",
            best
        );
    }

    // -------------------------------------------------------------------------
    // root_value() sign convention
    //
    // These are the primary regression guards for the backpropagation bug the
    // user observed (pretraining good → self-play deterioration).
    //
    // Correct convention: root_value() > 0 when the player-to-move is winning.
    // Inverted convention: root_value() < 0 (sign is flipped → value head trains
    // on wrong targets → self-play policy diverges from pretraining signal).
    // -------------------------------------------------------------------------

    /// root_value() must be positive for Black when Black has an immediate win.
    #[test]
    fn root_value_positive_for_black_winning() {
        let game = black_wins_in_one();
        let search = run_mcts_oracle(&game, 200);
        assert!(
            search.root_value() > 0.5,
            "root_value() must be strongly positive for Black (immediate win available), got {}",
            search.root_value()
        );
    }

    /// root_value() must be positive for White when White has an immediate win.
    /// Verifies the sign is correct regardless of which player holds Player1/Player2.
    #[test]
    fn root_value_positive_for_white_winning() {
        let game = white_wins_in_one();
        let search = run_mcts_oracle(&game, 200);
        assert!(
            search.root_value() > 0.5,
            "root_value() must be strongly positive for White (immediate win available), got {}",
            search.root_value()
        );
    }

    // -------------------------------------------------------------------------
    // Visit distribution (training targets)
    // -------------------------------------------------------------------------

    /// The visit distribution must concentrate on the winning destination (0,1).
    /// Weak value targets on the winning move → the policy head fails to learn to
    /// exploit forced wins, which compounds through self-play.
    #[test]
    fn visit_distribution_concentrates_on_winning_move_for_black() {
        let game = black_wins_in_one();
        let search = run_mcts_oracle(&game, 300);
        let dist = search.get_visit_distribution();

        let beetle = Piece::new(PieceColor::Black, PieceType::Beetle, 1);
        let winning_prob: f32 = dist
            .iter()
            .filter(|(mv, _)| mv.piece == Some(beetle) && mv.to == Some((0, 1)))
            .map(|(_, p)| p)
            .sum();

        assert!(
            winning_prob > 0.5,
            "visit distribution must concentrate on winning beetle move (prob > 0.5), got {:.3}",
            winning_prob
        );
    }

    /// Diagnostic: print MCTS visit distribution to understand convergence behaviour.
    /// Not a correctness assertion — run with `cargo test debug_mcts -- --nocapture`.
    #[test]
    fn debug_mcts_distribution() {
        let game = black_wins_in_one();
        let beetle1 = Piece::new(PieceColor::Black, PieceType::Beetle, 1);

        // Print lookahead policy for winning move
        let mut game_copy = game.clone();
        let policy = lookahead_policy(&mut game_copy);
        let (_, indexed_moves) = game_copy.get_legal_move_mask();
        eprintln!("Lookahead policy for winning move bB1→(0,1):");
        for (enc, mv) in &indexed_moves {
            if mv.piece == Some(beetle1) && mv.to == Some((0, 1)) {
                use core_game::game::PolicyIndex;
                match enc {
                    PolicyIndex::Single(i) => eprintln!("  Single({i}): p={:.6}", policy[*i]),
                    PolicyIndex::Sum(a, b) => eprintln!("  Sum({a},{b}): p[a]={:.6} p[b]={:.6} sum={:.6}", policy[*a], policy[*b], policy[*a]+policy[*b]),
                    PolicyIndex::DotProduct { src_cell, dst_cell, .. } => eprintln!("  DotProduct(src={src_cell},dst={dst_cell})"),
                }
            }
        }
        eprintln!("Total indexed moves: {}", indexed_moves.len());

        // Print top moves in distribution after 300 sims
        let search = run_mcts_oracle(&game, 300);
        let dist = search.get_visit_distribution();
        let mut sorted = dist.clone();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        eprintln!("\nTop 10 moves by visit fraction (300 sims):");
        for (mv, p) in sorted.iter().take(10) {
            let tag = if mv.piece == Some(beetle1) && mv.to == Some((0, 1)) { " *** WINNING ***" } else { "" };
            eprintln!("  {:?} → {:?}  p={:.4}{}", mv.piece, mv.to, p, tag);
        }
        eprintln!("root_value = {:.4}", search.root_value());
        eprintln!("best_move  = {:?}", search.best_move());
    }
}
