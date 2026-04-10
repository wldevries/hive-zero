/// Human-readable move and coordinate notation for Zertz.
///
/// Coordinate format: letter A-G (column, left-to-right) + digit 1-7 (row, top-to-bottom).
/// Examples: "D4" (center), "A1" (top-left corner of col A).
///
/// Move format:
///   Place:     "W D4 C3"  (color, place-at, remove)
///   PlaceOnly: "W D4"     (no ring removed — board edge or isolated)
///   Capture:   "CAP D4 D6 D8"  (from, then each landing after each hop)
///   Pass:      "pass"

use crate::hex::{is_valid, Hex, RADIUS};
use crate::zertz::{Marble, ZertzMove, MAX_CAPTURE_JUMPS};

pub fn hex_to_coord(h: Hex) -> String {
    let (q, r) = h;
    let col = (b'A' + (q + RADIUS) as u8) as char;
    let row = RADIUS + 1 - r; // r=-3→7, r=0→4, r=3→1
    format!("{}{}", col, row)
}

pub fn coord_to_hex(s: &str) -> Result<Hex, String> {
    let s = s.trim();
    let bytes = s.as_bytes();
    if bytes.len() != 2 {
        return Err(format!("Expected cell like D4, got '{}'", s));
    }
    let col_b = bytes[0].to_ascii_uppercase();
    let row_b = bytes[1];
    if !(b'A'..=b'G').contains(&col_b) {
        return Err(format!("Column must be A-G, got '{}'", col_b as char));
    }
    if !(b'1'..=b'7').contains(&row_b) {
        return Err(format!("Row must be 1-7, got '{}'", row_b as char));
    }
    let q = col_b as i8 - b'A' as i8 - RADIUS;
    let r = RADIUS + 1 - (row_b - b'0') as i8;
    let h = (q, r);
    if !is_valid(h) {
        return Err(format!("'{}' is not on the Zertz board", s));
    }
    Ok(h)
}

pub fn move_to_str(mv: ZertzMove) -> String {
    match mv {
        ZertzMove::Place { color, place_at, remove } =>
            format!("{} {} {}", color, hex_to_coord(place_at), hex_to_coord(remove)),
        ZertzMove::PlaceOnly { color, place_at } =>
            format!("{} {}", color, hex_to_coord(place_at)),
        ZertzMove::Capture { jumps, len } => {
            let mut s = format!("CAP {}", hex_to_coord(jumps[0].0));
            for i in 0..len as usize {
                s.push(' ');
                s.push_str(&hex_to_coord(jumps[i].2));
            }
            s
        }
        ZertzMove::Pass => "pass".to_string(),
    }
}

pub fn str_to_move(s: &str) -> Result<ZertzMove, String> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.is_empty() {
        return Err("Empty move string".to_string());
    }
    let first = parts[0].to_ascii_uppercase();
    match first.as_str() {
        "CAP" => {
            if parts.len() < 3 {
                return Err("CAP needs at least from + one landing: CAP D4 D6".to_string());
            }
            let positions: Result<Vec<Hex>, _> = parts[1..].iter().map(|s| coord_to_hex(s)).collect();
            let positions = positions?;
            let n_hops = positions.len() - 1;
            if n_hops > MAX_CAPTURE_JUMPS {
                return Err(format!("Too many hops (max {})", MAX_CAPTURE_JUMPS));
            }
            let mut jumps = [((0i8, 0i8), (0i8, 0i8), (0i8, 0i8)); MAX_CAPTURE_JUMPS];
            for i in 0..n_hops {
                let (fq, fr) = positions[i];
                let (tq, tr) = positions[i + 1];
                let dq = tq - fq;
                let dr = tr - fr;
                if dq % 2 != 0 || dr % 2 != 0 {
                    return Err(format!(
                        "Hop from {} to {} is not a valid 2-step jump",
                        hex_to_coord(positions[i]), hex_to_coord(positions[i + 1])
                    ));
                }
                jumps[i] = (positions[i], (fq + dq / 2, fr + dr / 2), positions[i + 1]);
            }
            Ok(ZertzMove::Capture { jumps, len: n_hops as u8 })
        }
        "PASS" => Ok(ZertzMove::Pass),
        "W" | "G" | "B" => {
            let color = match first.as_str() {
                "W" => Marble::White,
                "G" => Marble::Grey,
                _   => Marble::Black,
            };
            match parts.len() {
                2 => Ok(ZertzMove::PlaceOnly { color, place_at: coord_to_hex(parts[1])? }),
                3 => Ok(ZertzMove::Place { color, place_at: coord_to_hex(parts[1])?, remove: coord_to_hex(parts[2])? }),
                _ => Err(format!("Expected 'W/G/B place [remove]', got '{}'", s)),
            }
        }
        _ => Err(format!("Unknown move format '{}': expected CAP/W/G/B/pass", s)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coord_roundtrip() {
        use crate::hex::all_hexes;
        for h in all_hexes() {
            let s = hex_to_coord(h);
            let h2 = coord_to_hex(&s).expect(&s);
            assert_eq!(h, h2, "roundtrip failed for {:?} → {}", h, s);
        }
    }

    #[test]
    fn test_center_is_d4() {
        assert_eq!(hex_to_coord((0, 0)), "D4");
        assert_eq!(coord_to_hex("D4").unwrap(), (0, 0));
    }

    #[test]
    fn test_place_move_roundtrip() {
        let mv = ZertzMove::Place { color: Marble::White, place_at: (0, 0), remove: (1, 0) };
        let s = move_to_str(mv);
        let mv2 = str_to_move(&s).unwrap();
        assert_eq!(move_to_str(mv2), s);
    }
}
