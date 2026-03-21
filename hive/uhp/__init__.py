"""UHP (Universal Hive Protocol) utilities."""


def normalize_piece(piece_str: str) -> str:
    """Normalize a piece name to canonical UHP form (as returned by the Rust engine).

    Handles old-format quirks:
    - lowercase piece type: 'wg1' -> 'wG1'
    - queen with number: 'wQ1' -> 'wQ' (queen has no number in UHP)
    """
    if len(piece_str) < 2 or piece_str[0] not in "wb":
        return piece_str
    normalized = piece_str[0] + piece_str[1].upper() + piece_str[2:]
    if normalized[1] == 'Q':
        return normalized[:2]
    return normalized


def normalize_move(move_str: str) -> str:
    """Normalize a UHP move string to canonical form.

    Applies normalize_piece to both the moving piece and the reference piece,
    handling old-format quirks (wQ1->wQ, lowercase piece type).
    """
    parts = move_str.strip().split(None, 1)
    if not parts:
        return move_str
    piece = normalize_piece(parts[0])
    if len(parts) == 1:
        return piece
    ref_part = parts[1]
    dir_prefix = ''
    if ref_part and ref_part[0] in '-/\\':
        dir_prefix = ref_part[0]
        ref_part = ref_part[1:]
    dir_suffix = ''
    if ref_part and ref_part[-1] in '-/\\':
        dir_suffix = ref_part[-1]
        ref_part = ref_part[:-1]
    return f"{piece} {dir_prefix}{normalize_piece(ref_part)}{dir_suffix}"
