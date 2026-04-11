/// Piece definitions for Hive.
/// Each piece is packed into a single u8:
///   bit 7: color (0=White, 1=Black)
///   bits 4-6: piece type (0=Queen, 1=Spider, 2=Beetle, 3=Grasshopper, 4=Ant)
///   bits 0-3: number (1-indexed)

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PieceColor {
    White = 0,
    Black = 1,
}

impl PieceColor {
    #[inline]
    pub fn opposite(self) -> PieceColor {
        match self {
            PieceColor::White => PieceColor::Black,
            PieceColor::Black => PieceColor::White,
        }
    }

    pub fn as_char(self) -> char {
        match self {
            PieceColor::White => 'w',
            PieceColor::Black => 'b',
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PieceType {
    Queen = 0,
    Spider = 1,
    Beetle = 2,
    Grasshopper = 3,
    Ant = 4,
    Mosquito = 5,
    Ladybug = 6,
    Pillbug = 7,
}

impl PieceType {
    pub fn as_char(self) -> char {
        match self {
            PieceType::Queen => 'Q',
            PieceType::Spider => 'S',
            PieceType::Beetle => 'B',
            PieceType::Grasshopper => 'G',
            PieceType::Ant => 'A',
            PieceType::Mosquito => 'M',
            PieceType::Ladybug => 'L',
            PieceType::Pillbug => 'P',
        }
    }

    pub fn from_char(c: char) -> Option<PieceType> {
        match c {
            'Q' => Some(PieceType::Queen),
            'S' => Some(PieceType::Spider),
            'B' => Some(PieceType::Beetle),
            'G' => Some(PieceType::Grasshopper),
            'A' => Some(PieceType::Ant),
            'M' => Some(PieceType::Mosquito),
            'L' => Some(PieceType::Ladybug),
            'P' => Some(PieceType::Pillbug),
            _ => None,
        }
    }

    /// Index for encoding channels (matches Python PIECE_TYPE_INDEX).
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// How many of each piece type per player (base game).
pub const PIECE_COUNTS: [u8; 5] = [1, 2, 2, 3, 3]; // Q, S, B, G, A

/// All piece types in order.
pub const ALL_PIECE_TYPES: [PieceType; 8] = [
    PieceType::Queen,
    PieceType::Spider,
    PieceType::Beetle,
    PieceType::Grasshopper,
    PieceType::Ant,
    PieceType::Mosquito,
    PieceType::Ladybug,
    PieceType::Pillbug,
];

/// Total pieces per player: 1+2+2+3+3 = 11
pub const PIECES_PER_PLAYER: usize = 11;

/// Packed piece representation.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece(pub u8);

impl Piece {
    #[inline]
    pub fn new(color: PieceColor, piece_type: PieceType, number: u8) -> Self {
        debug_assert!(number >= 1 && number <= 15);
        Piece(((color as u8) << 7) | ((piece_type as u8) << 4) | number)
    }

    #[inline]
    pub fn color(self) -> PieceColor {
        if self.0 >> 7 == 0 {
            PieceColor::White
        } else {
            PieceColor::Black
        }
    }

    #[inline]
    pub fn piece_type(self) -> PieceType {
        match (self.0 >> 4) & 0x07 {
            0 => PieceType::Queen,
            1 => PieceType::Spider,
            2 => PieceType::Beetle,
            3 => PieceType::Grasshopper,
            4 => PieceType::Ant,
            5 => PieceType::Mosquito,
            6 => PieceType::Ladybug,
            7 => PieceType::Pillbug,
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn number(self) -> u8 {
        self.0 & 0x0F
    }

    #[inline]
    pub fn raw(self) -> u8 {
        self.0
    }

    pub fn from_str(s: &str) -> Option<Piece> {
        let bytes = s.as_bytes();
        if bytes.len() < 2 {
            return None;
        }
        let color = match bytes[0] {
            b'w' => PieceColor::White,
            b'b' => PieceColor::Black,
            _ => return None,
        };
        let ptype = PieceType::from_char(bytes[1] as char)?;
        let num = if bytes.len() > 2 {
            (bytes[2] - b'0') as u8
        } else {
            1
        };
        Some(Piece::new(color, ptype, num))
    }

    pub fn to_string(self) -> String {
        format!(
            "{}{}{}",
            self.color().as_char(),
            self.piece_type().as_char(),
            self.number()
        )
    }

    /// UHP-compliant string: omit number for unique pieces (Queen).
    pub fn to_uhp_string(self) -> String {
        if PIECE_COUNTS[self.piece_type() as usize] == 1 {
            format!("{}{}", self.color().as_char(), self.piece_type().as_char())
        } else {
            self.to_string()
        }
    }

    /// Unique index for this piece across all pieces (0..21).
    /// White pieces: 0..10, Black pieces: 11..21.
    #[inline]
    pub fn linear_index(self) -> usize {
        let color_offset = if self.color() == PieceColor::White { 0 } else { PIECES_PER_PLAYER };
        let mut type_offset = 0;
        for i in 0..(self.piece_type() as usize) {
            type_offset += PIECE_COUNTS[i] as usize;
        }
        color_offset + type_offset + (self.number() as usize - 1)
    }
}

impl std::fmt::Debug for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}{}", self.color().as_char(), self.piece_type().as_char(), self.number())
    }
}

impl std::fmt::Display for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}{}", self.color().as_char(), self.piece_type().as_char(), self.number())
    }
}

/// Generate all pieces for one player.
pub fn player_pieces(color: PieceColor) -> Vec<Piece> {
    let mut pieces = Vec::with_capacity(PIECES_PER_PLAYER);
    for (i, &pt) in ALL_PIECE_TYPES.iter().enumerate() {
        for n in 1..=PIECE_COUNTS[i] {
            pieces.push(Piece::new(color, pt, n));
        }
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piece_pack_unpack() {
        let p = Piece::new(PieceColor::White, PieceType::Queen, 1);
        assert_eq!(p.color(), PieceColor::White);
        assert_eq!(p.piece_type(), PieceType::Queen);
        assert_eq!(p.number(), 1);

        let p2 = Piece::new(PieceColor::Black, PieceType::Ant, 3);
        assert_eq!(p2.color(), PieceColor::Black);
        assert_eq!(p2.piece_type(), PieceType::Ant);
        assert_eq!(p2.number(), 3);
    }

    #[test]
    fn test_piece_string() {
        let p = Piece::new(PieceColor::White, PieceType::Queen, 1);
        assert_eq!(p.to_string(), "wQ1");

        let p2 = Piece::from_str("bS2").unwrap();
        assert_eq!(p2.color(), PieceColor::Black);
        assert_eq!(p2.piece_type(), PieceType::Spider);
        assert_eq!(p2.number(), 2);
    }

    #[test]
    fn test_player_pieces() {
        let whites = player_pieces(PieceColor::White);
        assert_eq!(whites.len(), PIECES_PER_PLAYER);
        assert!(whites.iter().all(|p| p.color() == PieceColor::White));

        let blacks = player_pieces(PieceColor::Black);
        assert_eq!(blacks.len(), PIECES_PER_PLAYER);
    }

    #[test]
    fn test_linear_index() {
        let whites = player_pieces(PieceColor::White);
        let blacks = player_pieces(PieceColor::Black);
        for (i, p) in whites.iter().enumerate() {
            assert_eq!(p.linear_index(), i);
        }
        for (i, p) in blacks.iter().enumerate() {
            assert_eq!(p.linear_index(), PIECES_PER_PLAYER + i);
        }
    }
}
