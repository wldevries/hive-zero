"""Terminal styling helpers built on rich.

Drop-in replacements for the per-file colorama lambdas (`_cg`, `_cy`, `_cr`,
`_cc`, `_cm`) plus a generic `styled()` for ad-hoc styles. Each helper returns
a plain ANSI-coded string suitable for f-string interpolation, so existing
`print(f"... {cg(v)} ...")` callsites keep working unchanged.

Color semantics across the codebase:
    cg — green   (wins, positives)
    cy — yellow  (draws, secondary losses, intermediate)
    cr — red     (losses, total loss)
    cc — cyan    (scores, percentages, labels)
    cm — magenta (accent, X piece)
"""

from __future__ import annotations

from rich.console import Console

_console = Console()


def styled(value, style: str) -> str:
    with _console.capture() as cap:
        _console.print(str(value), style=style, end="", markup=False, highlight=False)
    return cap.get()


def cg(v) -> str:
    return styled(v, "green bold")


def cy(v) -> str:
    return styled(v, "yellow bold")


def cr(v) -> str:
    return styled(v, "red bold")


def cc(v) -> str:
    return styled(v, "cyan bold")


def cm(v) -> str:
    return styled(v, "magenta bold")


def bold_white(v) -> str:
    return styled(v, "bold white")
