/// YINSH move notation: "A2", "E5 G5", "Claim E5 E6 E7 E8 E9 D3".
///
/// `Claim` is a joint row+ring removal: 5 row cells (consecutive in one of the
/// 3 axial directions) followed by the ring cell to take off.

use crate::board::YinshMove;
use crate::hex::{ROW_DIRS, cell_index, index_to_cell, is_valid};

pub fn cell_to_str(col: u8, row: u8) -> String {
    let col_char = (b'A' + col) as char;
    let row_num = row + 1;
    format!("{}{}", col_char, row_num)
}

pub fn str_to_cell(s: &str) -> Option<(u8, u8)> {
    let s = s.trim();
    let bytes = s.as_bytes();
    if bytes.is_empty() { return None; }
    let col_char = bytes[0];
    if !(b'A'..=b'K').contains(&col_char) { return None; }
    let col = col_char - b'A';
    let row_num: u8 = s[1..].parse().ok()?;
    if row_num == 0 { return None; }
    let row = row_num - 1;
    if !is_valid(col, row) { return None; }
    Some((col, row))
}

pub fn move_to_str(mv: &YinshMove) -> String {
    match *mv {
        YinshMove::PlaceRing(idx) => {
            let (c, r) = index_to_cell(idx);
            cell_to_str(c, r)
        }
        YinshMove::MoveRing { from, to } => {
            let (fc, fr) = index_to_cell(from);
            let (tc, tr) = index_to_cell(to);
            format!("{} {}", cell_to_str(fc, fr), cell_to_str(tc, tr))
        }
        YinshMove::ClaimRow { start, dir, ring } => {
            let (sc, sr) = index_to_cell(start);
            let (dc, dr) = ROW_DIRS[dir];
            let row_cells: Vec<String> = (0i8..5).map(|k| {
                let c = (sc as i8 + dc * k) as u8;
                let r = (sr as i8 + dr * k) as u8;
                cell_to_str(c, r)
            }).collect();
            let (rc, rr) = index_to_cell(ring);
            format!("Claim {} {}", row_cells.join(" "), cell_to_str(rc, rr))
        }
        YinshMove::Pass => "Pass".to_string(),
    }
}

pub fn str_to_move(s: &str) -> Result<YinshMove, String> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("Pass") {
        return Ok(YinshMove::Pass);
    }
    if let Some(rest) = s.strip_prefix("Claim ")
        .or_else(|| s.strip_prefix("CLAIM "))
        .or_else(|| s.strip_prefix("claim "))
    {
        let parts: Vec<&str> = rest.split_whitespace().collect();
        if parts.len() != 6 {
            return Err(format!("Claim expects 5 row cells + 1 ring cell, got {}", parts.len()));
        }
        let cells: Vec<(u8, u8)> = parts.iter()
            .map(|p| str_to_cell(p).ok_or_else(|| format!("bad cell: {}", p)))
            .collect::<Result<_, _>>()?;
        let start_idx = cell_index(cells[0].0, cells[0].1);
        let (c0, r0) = cells[0];
        let (c1, r1) = cells[1];
        let dc = (c1 as i8 - c0 as i8).signum();
        let dr = (r1 as i8 - r0 as i8).signum();
        let dir = ROW_DIRS.iter().position(|&d| d == (dc, dr))
            .ok_or_else(|| format!("bad row direction ({}, {})", dc, dr))?;
        for k in 0i8..5 {
            let ec = c0 as i8 + dc * k;
            let er = r0 as i8 + dr * k;
            if cells[k as usize] != (ec as u8, er as u8) {
                return Err("row cells not consecutive in direction".to_string());
            }
        }
        let (rc, rr) = cells[5];
        return Ok(YinshMove::ClaimRow {
            start: start_idx,
            dir,
            ring: cell_index(rc, rr),
        });
    }
    // Otherwise: one or two cells
    let parts: Vec<&str> = s.split_whitespace().collect();
    match parts.len() {
        1 => {
            let (c, r) = str_to_cell(parts[0]).ok_or_else(|| format!("bad cell: {}", parts[0]))?;
            Ok(YinshMove::PlaceRing(cell_index(c, r)))
        }
        2 => {
            let (fc, fr) = str_to_cell(parts[0]).ok_or_else(|| format!("bad cell: {}", parts[0]))?;
            let (tc, tr) = str_to_cell(parts[1]).ok_or_else(|| format!("bad cell: {}", parts[1]))?;
            Ok(YinshMove::MoveRing {
                from: cell_index(fc, fr),
                to: cell_index(tc, tr),
            })
        }
        _ => Err(format!("unrecognized move: {}", s)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_roundtrip() {
        for i in 0..crate::hex::BOARD_SIZE {
            let (c, r) = index_to_cell(i);
            let s = cell_to_str(c, r);
            let (c2, r2) = str_to_cell(&s).unwrap();
            assert_eq!((c, r), (c2, r2));
        }
    }

    #[test]
    fn test_known_cells() {
        assert_eq!(cell_to_str(0, 1), "A2");
        assert_eq!(cell_to_str(10, 9), "K10");
        assert_eq!(str_to_cell("A2"), Some((0, 1)));
        assert_eq!(str_to_cell("K10"), Some((10, 9)));
        assert_eq!(str_to_cell("A1"), None); // invalid
    }

    #[test]
    fn test_move_roundtrip_place() {
        let mv = YinshMove::PlaceRing(cell_index(4, 4));
        let s = move_to_str(&mv);
        assert_eq!(s, "E5");
        assert_eq!(str_to_move(&s).unwrap(), mv);
    }

    #[test]
    fn test_move_roundtrip_move_ring() {
        let mv = YinshMove::MoveRing {
            from: cell_index(4, 4),
            to: cell_index(4, 7),
        };
        let s = move_to_str(&mv);
        assert_eq!(s, "E5 E8");
        assert_eq!(str_to_move(&s).unwrap(), mv);
    }

    #[test]
    fn test_move_roundtrip_claim_row() {
        let mv = YinshMove::ClaimRow {
            start: cell_index(4, 4),
            dir: 0, // vertical (0, 1)
            ring: cell_index(3, 2),
        };
        let s = move_to_str(&mv);
        assert_eq!(s, "Claim E5 E6 E7 E8 E9 D3");
        assert_eq!(str_to_move(&s).unwrap(), mv);
    }
}
