LOG_HEADER = (
    "iter,mode,simulations,wins_w,wins_b,draws,resignations,positions,buffer,"
    "loss,policy_loss,value_loss,qd_loss,lr,duration_s,comment\n"
)


def csv_comment(comment: str) -> str:
    """Return comment as a properly quoted CSV field (RFC 4180)."""
    escaped = comment.replace('"', '""')
    if "," in escaped or '"' in comment:
        return f'"{escaped}"'
    return escaped
