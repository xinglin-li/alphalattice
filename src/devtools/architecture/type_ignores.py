"""Strict typing's escapes: each ``type: ignore`` comment, counted per error code.

The commit holds these counts per file to a baseline that only falls (scripts/check_counts.py),
so a suppression is added only by fixing the type instead. Comments are read as Python tokens, so
a ``type: ignore`` inside a string is data and never counted. Each code an ignore names counts on
its own: replacing ``[arg-type]`` with ``[arg-type, attr-defined]`` adds an ``attr-defined`` and
is refused. An ignore with no code hides every error on its line and counts as ``type-ignore``.
"""

from __future__ import annotations

import io
import re
import tokenize
from dataclasses import dataclass

IGNORE = re.compile(r"#\s*type:\s*ignore(?:\[([^\]]*)\])?")
UNCODED = "type-ignore"


@dataclass(frozen=True)
class Finding:
    """One ignore of one error code: where, and its rule ``type-ignore[code]``."""

    path: str
    line: int
    rule: str

    @property
    def message(self) -> str:
        """The repair, as the refusal says it."""
        if self.rule == UNCODED:
            return "a type: ignore with no error code hides every error on its line: fix the type"
        return f"a new {self.rule.replace('type-ignore', 'type: ignore', 1)}: fix the type instead"


def findings(path: str, text: str) -> list[Finding]:
    """Every error code each ``type: ignore`` comment in one Python source names."""
    found = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type != tokenize.COMMENT:
            continue
        for match in IGNORE.finditer(token.string):
            codes = {code.strip() for code in (match.group(1) or "").split(",")} - {""}
            rules = sorted(f"{UNCODED}[{code}]" for code in codes) or [UNCODED]
            found += [Finding(path, token.start[0], rule) for rule in rules]
    return found
