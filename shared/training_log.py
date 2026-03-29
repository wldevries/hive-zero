def csv_comment(comment: str) -> str:
    """Return comment as a properly quoted CSV field (RFC 4180)."""
    escaped = comment.replace('"', '""')
    if "," in escaped or '"' in comment:
        return f'"{escaped}"'
    return escaped
