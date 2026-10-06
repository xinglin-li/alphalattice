"""Count the Local Web UI's stylistic sprawl against its laws (workbench-source/LAWS.md).

Static and fast (no browser, no build): it reads the workbench source sheets and app
scripts and prints one table. Each UI round quotes the numbers before and after. It
asserts nothing and keeps no baseline; a law number in a row is the old numbering, which
LAWS.md's appendix maps to the rule that carries it.

    uv run python scripts/count_ui_laws.py                      # the table
    uv run python scripts/count_ui_laws.py --values font-size   # the values behind one row
    uv run python scripts/count_ui_laws.py --source <dir>       # another workbench-source tree
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/alphalattice/interface/local_application/assets/workbench-source"

# Declarations counted by distinct value (a token or a var() reads as one value).
VALUE_ROWS = {
    "font-size": "the text styles + the brand mark",
    "line-height": "the styles' leadings",
    "letter-spacing": "the styles' trackings",
    "font-weight": "400 / 500 / 600",
    "border-radius": "three tokens + pill + circle",
    "box-shadow": "three levels",
    "grid-template-columns": "8 families + the roster",
}
# Spacing lengths written as pixels (gap / padding / margin): the scale is the --space tokens, so a
# literal is off the scale unless it is 0, 1px, or a layout offset (a dock reserve, an indent).
SPACING_LITERAL = re.compile(
    r"\b(?:gap|row-gap|column-gap|padding(?:-[a-z-]+)?|margin(?:-[a-z-]+)?)\s*:\s*([^;{}]*\b(?!0px)(?!1px)\d+px[^;{}]*);"
)
# @font-face descriptors (a weight range, a family name) are not product declarations.
FONT_FACE = re.compile(r"@font-face\s*\{[^}]*\}")
DURATION = re.compile(r"(?:transition|animation)(?:-duration)?\s*:\s*([^;{}]+);")
TIME = re.compile(r"\b(\d*\.?\d+m?s)\b")
IMPORTANT = re.compile(r"!important")
BUTTON_CLASS = re.compile(r'class="([^"]*\b(?:button|btn)\b[^"]*)"')
# the state table (round 71: `STATES` in status.js; `SEMANTICS` before it): one entry per state
STATES = re.compile(r"const (?:STATES|SEMANTICS)\s*=\s*\{(.*?)\n\};", re.S)
STATE_KEY = re.compile(r"\b([a-z_]+)\s*:\s*\{", re.M)
TONE = re.compile(r"tone:\s*'([^']*)'")
# the bars (law 89): one meter owns every bar; a chart's bars are exempt
BAR_CLASS = re.compile(
    r"\.(meter-bar|share-bar|tp-progress|tp-stage-progress"
    r"|sector-meter-bar|progress|fv-progress)\b"
)


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _css_text(source: Path) -> tuple[str, int]:
    lines = 0
    parts = []
    for path in sorted((source / "css").glob("*.css")):
        text = path.read_text(encoding="utf-8")
        lines += text.count("\n")
        parts.append(FONT_FACE.sub("", text))
    return "\n".join(parts), lines


def _js_text(source: Path) -> str:
    scripts = sorted((source / "js/app").glob("*.js"))
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in scripts)


def distinct_values(css: str, declaration: str) -> Counter[str]:
    pattern = declaration + r"\s*:\s*([^;{}]+?)\s*(?:!important)?\s*;"
    return Counter(_normalise(m) for m in re.findall(pattern, css))


def durations(css: str) -> Counter[str]:
    found: Counter[str] = Counter()
    for declaration in DURATION.findall(css):
        for stamp in TIME.findall(declaration):
            found[stamp] += 1
    return found


def counts(source: Path = SOURCE) -> dict[str, tuple[int, str, Counter[str] | None]]:
    css, lines = _css_text(source)
    js = _js_text(source)
    rows: dict[str, tuple[int, str, Counter[str] | None]] = {}
    for name, target in VALUE_ROWS.items():
        values = distinct_values(css, name)
        rows[f"distinct {name}"] = (len(values), target, values)
    stamps = durations(css)
    rows["distinct motion durations"] = (len(stamps), "3 (brand animation exempt)", stamps)
    literals = Counter(_normalise(m) for m in SPACING_LITERAL.findall(css))
    rows["spacing literals off the --space scale"] = (
        sum(literals.values()),
        "layout offsets only",
        literals,
    )
    rows["!important declarations"] = (len(IMPORTANT.findall(css)), "< 40", None)
    rows["css lines"] = (lines, "~14,000", None)
    buttons = Counter(_normalise(m) for m in BUTTON_CLASS.findall(js))
    rows["button class strings in js"] = (len(buttons), "4 (+ icon button)", buttons)
    bars = Counter(BAR_CLASS.findall(css))
    rows["bar classes in the sheet"] = (len(bars), "1 (meter-bar)", bars)
    states = STATES.search(js)
    body = states.group(1) if states else ""
    statuses = STATE_KEY.findall(body)
    tones = Counter(TONE.findall(body))
    rows["states in STATES"] = (len(statuses), "one table, a word each", Counter(statuses))
    rows["state tones in STATES"] = (len(tones), "6", tones)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--values", metavar="ROW", help="print the values behind one row (a substring of its name)"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE,
        help="a workbench-source tree (default: this checkout)",
    )
    args = parser.parse_args(argv)
    rows = counts(args.source)
    if args.values:
        matches = [name for name in rows if args.values in name]
        if not matches:
            print(f"no row matches {args.values!r}; rows: {', '.join(rows)}", file=sys.stderr)
            return 2
        for name in matches:
            _, _, values = rows[name]
            print(f"## {name}")
            if values is None:
                print("(a count, not a set)")
                continue
            for value, n in values.most_common():
                print(f"{n:5d}  {value}")
        return 0
    width = max(len(name) for name in rows)
    print(f"{'dimension':<{width}}  {'now':>7}  target")
    for name, (now, target, _) in rows.items():
        print(f"{name:<{width}}  {now:>7,}  {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
