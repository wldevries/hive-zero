use rand::seq::IndexedRandom;
use yinsh_game::board::YinshBoard;
use yinsh_game::notation::move_to_str;
use core_game::game::Outcome;

pub fn run_random_game(max_moves: usize) {
    let mut board = YinshBoard::new();
    let mut rng = rand::rng();

    println!("--- Initial ---");
    println!("{board}");

    let mut n = 0usize;
    loop {
        if board.outcome != Outcome::Ongoing {
            break;
        }
        let legal = board.legal_moves();
        if legal.is_empty() || n >= max_moves {
            break;
        }

        let mv = *legal.choose(&mut rng).unwrap();
        let mv_str = move_to_str(&mv);
        let before = board.clone();
        board.apply_move(mv).unwrap();
        n += 1;

        let header_left = format!("--- Move {n}: {mv_str} (before) ---");
        let header_right = format!("--- Move {n}: {mv_str} (after) ---");
        print!("{}", side_by_side(&format!("{header_left}\n{before}"), &format!("{header_right}\n{board}"), 4));
    }

    match board.outcome {
        Outcome::WonBy(core_game::game::Player::Player1) => {
            println!("White wins after {n} moves")
        }
        Outcome::WonBy(core_game::game::Player::Player2) => {
            println!("Black wins after {n} moves")
        }
        Outcome::Draw => println!("Draw after {n} moves"),
        Outcome::Ongoing => println!("Stopped after {n} moves (ongoing)"),
    }
}

/// Visible width of a string, ignoring ANSI escape sequences.
fn visible_width(s: &str) -> usize {
    let mut w = 0;
    let mut in_esc = false;
    for c in s.chars() {
        if c == '\x1b' {
            in_esc = true;
        } else if in_esc {
            if c == 'm' {
                in_esc = false;
            }
        } else {
            w += 1;
        }
    }
    w
}

/// Join two multi-line strings side by side with `gap` spaces between columns.
fn side_by_side(left: &str, right: &str, gap: usize) -> String {
    let l_lines: Vec<&str> = left.lines().collect();
    let r_lines: Vec<&str> = right.lines().collect();
    let max_l = l_lines.iter().map(|s| visible_width(s)).max().unwrap_or(0);
    let rows = l_lines.len().max(r_lines.len());
    let mut out = String::new();
    for i in 0..rows {
        let ll = l_lines.get(i).copied().unwrap_or("");
        let rl = r_lines.get(i).copied().unwrap_or("");
        out.push_str(ll);
        for _ in visible_width(ll)..(max_l + gap) {
            out.push(' ');
        }
        out.push_str(rl);
        out.push('\n');
    }
    out
}
