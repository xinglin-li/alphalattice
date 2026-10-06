"""Refuse a design value written outside the parameter source (laws 105-108).

The Local Web workbench keeps every design value in
`workbench-source/design/parameters.json`; the build writes it as the first stylesheet. This
check fails when:

- the source is malformed (an unknown mode, a duplicate name, a value that is not one CSS value);
- a sheet declares a parameter the source owns (outside the overrides the source lists) or a
  custom property the source does not know (a component's local argument is declared in
  `arguments`);
- a script sets a custom property that is not an argument;
- a parameter nothing reads, or a `var()` that names nothing;
- a rule that sets a text style's size without the same style's leading (law 75; the body size
  may take the prose leading, and a text on a line grid of its own keeps the grid's line);
- a literal the ratchet has not seen: a length in px (other than 0 and 1), a raw colour, a
  duration, an opacity, a z-index, a mixed colour's strength or a length in em / rem / ch / vh /
  vw written in a sheet. Existing literals are listed in `design/literals-baseline.json` and may
  only go away; a category the gate newly counts may be recorded once (`--update-baseline`
  widens the list by that category's literals only);
- a breakpoint that is not a registered step, or one written as `min-`/`max-width`;
- a value of the glass (`material.glass`, law 150) without its `opaque` value -- the configuration
  Opaque controls switches to is recorded for every one, but for those only a glass rule reads.

    uv run python scripts/check_ui_parameters.py                    # exit 1 on a problem
    uv run python scripts/check_ui_parameters.py --update-baseline  # shrink it after a cleanup
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src/alphalattice/interface/local_application/assets/workbench-source"
PARAMETERS = SRC / "design/parameters.json"
BASELINE = SRC / "design/literals-baseline.json"

PX = re.compile(r"(?<![\w.-])-?\d*\.?\d+px\b")
COLOUR = re.compile(r"#[0-9a-fA-F]{3,8}\b|\b(?:rgba?|hsla?|oklch|oklab|lab|lch)\(")
DURATION = re.compile(r"(?<![\w.-])\d*\.?\d+m?s\b")
NUMBER = re.compile(r"-?\d*\.?\d+")
FONT_LENGTH = re.compile(r"(?<![\w.-])-?\d*\.?\d+(?:em|rem|ch|vh|vw|dvh|svh)\b")
MIX_STRENGTH = re.compile(r"color-mix\([^;]*?\d+%")
# the categories the ratchet counts; a literal's category is read from its hit
CATEGORIES = ("px", "colour", "duration", "opacity", "z", "tint", "length")
VAR = re.compile(r"var\(\s*--([A-Za-z0-9-]+)")
SET_PROPERTY = re.compile(r"setProperty\(\s*['\"]--([A-Za-z0-9-]+)")
# a text that sits on a grid of its own keeps the grid's line: the code editor's numbers
LINE_GRIDS = {"var(--editor-line)"}
# law 150: the glass's values that need no `opaque` value -- a shape, and the tints only a rule
# gated on the glass (`body:not(.opaque-controls)`) reads; any other records its opaque value
GLASS_WITHOUT_OPAQUE = {
    "glass-rim-mask": "the ring's mask is a shape, the same in every configuration",
    "glass-control": "read only while the glass is on; opaque, a control keeps its own face",
    "glass-control-edge": "read only while the glass is on; opaque, a control keeps its own edge",
    "glass-control-hover": "read only while the glass is on; opaque, a control keeps its own hover",
}


def _masked(text: str) -> str:
    return re.sub(r"/\*.*?\*/", lambda m: " " * len(m.group(0)), text, flags=re.S)


def _split_top(text: str, sep: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in text:
        depth += ch == "("
        depth -= ch == ")"
        if ch == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    return [*parts, cur]


def rules(text: str) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """(media, selector, declarations) for every style rule, at the top level or inside @media,
    @container, @supports or @layer; @font-face and @keyframes are not rules of the product's
    surfaces."""
    masked = _masked(text)
    out: list[tuple[str, str, list[tuple[str, str]]]] = []

    def walk(start: int, end: int, media: str) -> None:
        i = start
        while i < end:
            while i < end and masked[i].isspace():
                i += 1
            brace = masked.find("{", i, end)
            if brace == -1:
                return
            prelude = " ".join(masked[i:brace].split())
            depth, j = 0, brace
            while j < end:
                depth += masked[j] == "{"
                depth -= masked[j] == "}"
                if depth == 0:
                    break
                j += 1
            if prelude.startswith("@media"):
                walk(
                    brace + 1, j, " ".join(filter(None, (media, prelude[len("@media") :].strip())))
                )
            elif prelude.startswith(("@container", "@supports", "@layer")):
                # a conditional block holds rules of the surfaces like @media does (2026-09-22: the
                # walk skipped them, so no rule inside a container query was ever checked)
                walk(brace + 1, j, " ".join(filter(None, (media, prelude))))
            elif not prelude.startswith("@"):
                decls = []
                for part in _split_top(masked[brace + 1 : j], ";"):
                    if ":" in part:
                        prop, value = part.split(":", 1)
                        decls.append((prop.strip(), " ".join(value.split())))
                out.append((media, prelude, decls))
            i = j + 1

    walk(0, len(masked), "")
    return out


def load() -> dict:
    return json.loads(PARAMETERS.read_text(encoding="utf-8"))


def literals(css: dict[str, str]) -> Counter[str]:
    found: Counter[str] = Counter()
    for name, text in css.items():
        for media, selector, decls in rules(text):
            where = f"{name} | {media} | {selector}"
            for prop, value in decls:
                if prop.startswith("--"):
                    continue
                hits = [
                    m.group(0)
                    for m in PX.finditer(value)
                    if m.group(0) not in ("0px", "1px", "-1px")
                ]
                hits += [m.group(0) for m in COLOUR.finditer(value)]
                if prop.startswith(("transition", "animation")):
                    hits += [m.group(0) for m in DURATION.finditer(value)]
                if prop == "opacity" and NUMBER.fullmatch(value) and value not in ("0", "1"):
                    hits.append("opacity " + value)
                if prop == "z-index" and NUMBER.fullmatch(value) and value not in ("0", "-1"):
                    hits.append("z " + value)
                if MIX_STRENGTH.search(value):
                    hits.append("tint " + value)
                if not prop.startswith(("font", "letter-spacing", "line-height", "word-spacing")):
                    hits += [
                        "length " + m.group(0)
                        for m in FONT_LENGTH.finditer(re.sub(r"var\([^)]*\)", "", value))
                    ]
                for hit in hits:
                    found[f"{where} | {prop} | {hit}"] += 1
    return found


def category(key: str) -> str:
    hit = key.rsplit(" | ", 1)[-1]
    word = hit.split(" ", 1)[0]
    if word in ("opacity", "z", "tint", "length"):
        return word
    if hit.endswith("px"):
        return "px"
    if COLOUR.match(hit):
        return "colour"
    return "duration"


def problems(update_baseline: bool = False) -> list[str]:
    from scripts.build_local_web_ui import parameters_css

    out: list[str] = []
    p = load()
    modes = [m["name"] for m in p["modes"]]
    if len(set(modes)) != len(modes):
        out.append("parameters.json: a mode is named twice")
    tokens: dict[str, dict] = {}
    for body in p["groups"].values():
        for name, values in body["tokens"].items():
            if name in tokens:
                out.append(f"parameters.json: --{name} is in two groups")
            tokens[name] = values
            for mode, value in values.items():
                if mode == "about":
                    continue
                if mode not in modes:
                    out.append(f"parameters.json: --{name} names an unknown mode {mode!r}")
                if not isinstance(value, str) or not value.strip() or re.search(r"[;{}]", value):
                    out.append(
                        f"parameters.json: --{name} in {mode} is not one CSS value: {value!r}"
                    )
    arguments, overrides = p.get("arguments", {}), p.get("overrides", {})
    for name in arguments:
        if name in tokens:
            out.append(f"parameters.json: --{name} is both a parameter and an argument")
    for name in overrides:
        if name not in tokens:
            out.append(f"parameters.json: the override --{name} names no parameter")
    for name, values in p["groups"].get("material.glass", {}).get("tokens", {}).items():
        if "opaque" not in values and name not in GLASS_WITHOUT_OPAQUE:
            out.append(
                f"parameters.json: --{name} has no `opaque` value -- law 150 records the "
                "configuration without the glass for every value of it"
            )
    try:
        generated = parameters_css(p)
    except SystemExit as error:
        return [*out, str(error)]

    css = {
        path.name: path.read_text(encoding="utf-8") for path in sorted((SRC / "css").glob("*.css"))
    }
    js = {
        path.relative_to(SRC).as_posix(): path.read_text(encoding="utf-8", errors="replace")
        for path in sorted((SRC / "js").rglob("*.js"))
    }
    known = set(tokens) | set(arguments)

    for name, text in css.items():
        for _media, selector, decls in rules(text):
            for prop, _value in decls:
                if not prop.startswith("--"):
                    continue
                n = prop[2:]
                if n in tokens and name not in overrides.get(n, {}).get("files", []):
                    out.append(
                        f"{name}: `{selector}` declares --{n}, "
                        "which design/parameters.json owns (law 105)"
                    )
                elif n not in known:
                    out.append(
                        f"{name}: `{selector}` declares --{n}, "
                        "which design/parameters.json does not know -- a value belongs in its "
                        "groups, a component's local argument in its `arguments`"
                    )
    for rel, text in js.items():
        for m in SET_PROPERTY.finditer(text):
            if m.group(1) not in known:
                out.append(
                    f"{rel}: sets --{m.group(1)}, "
                    "which design/parameters.json does not know as an argument"
                )

    # a text style is a size and a leading together (law 75): a rule that sets a style's size sets
    # the same style's leading; the reading measure may pair the body size with the prose leading
    for name, text in css.items():
        for _media, selector, decls in rules(text):
            d = dict(decls)
            size = re.fullmatch(r"var\(--style-([a-z0-9-]+)-size\)", d.get("font-size", ""))
            if not size or "font" in d:
                continue
            if d.get("line-height") in LINE_GRIDS:
                continue
            lead = re.fullmatch(r"var\(--style-([a-z0-9-]+)-leading\)", d.get("line-height", ""))
            if not lead:
                out.append(
                    f"{name}: `{selector}` sets the {size.group(1)} size without its leading "
                    "-- a text style is a size and a leading together (law 75)"
                )
            elif lead.group(1) != size.group(1) and (size.group(1), lead.group(1)) != (
                "body",
                "prose",
            ):
                out.append(
                    f"{name}: `{selector}` pairs the {size.group(1)} size "
                    f"with the {lead.group(1)} leading"
                )

    reads: Counter[str] = Counter()
    for text in [*css.values(), generated]:
        reads.update(VAR.findall(_masked(text)))
    everything_js = "\n".join(js.values())
    # a value the build hands the scripts (`scripts`, `counts`, `tones`) is read where a script
    # names it
    scripted = set(p.get("scripts", [])) | set(p.get("counts", [])) | set(p.get("tones", []))
    for name in sorted(scripted - set(tokens)):
        out.append(
            f"parameters.json: `scripts`, `counts` or `tones` lists --{name}, "
            "which no group declares"
        )
    for name in tokens:
        named = name in scripted and re.search(r"['\"]" + re.escape(name) + r"['\"]", everything_js)
        if (
            not reads[name]
            and not named
            and not re.search(r"--" + re.escape(name) + r"(?![A-Za-z0-9-])", everything_js)
        ):
            out.append(f"parameters.json: nothing reads --{name}")
    for name in sorted(set(reads) - known):
        out.append(f"a sheet reads --{name}, which design/parameters.json does not know")

    # a breakpoint is a step of the ladder (Q1): every width or height a @media or @container
    # prelude names -- and a mode's media -- is registered, in range syntax
    steps = {
        kind: {v.get("width", v.get("height")) for v in body.values() if isinstance(v, dict)}
        for kind, body in p.get("breakpoints", {}).items()
        if kind != "about"
    }

    def refuse_prelude(where: str, kind: str, prelude: str) -> None:
        if re.search(r"\b(min|max)-(width|height)\s*:", prelude):
            out.append(
                f"{where}: `{prelude}` -- write a breakpoint in range syntax (`width < 900px`)"
            )
        for m in re.finditer(r"(\d+)px", prelude):
            near = prelude[max(0, m.start() - 12) : m.end() + 12]
            axis = "height" if "height" in near and "width" not in near else kind
            if int(m.group(1)) not in steps.get(axis, set()):
                out.append(
                    f"{where}: `{prelude}` names {m.group(0)}, "
                    f"which is not a registered {axis} breakpoint "
                    "(design/parameters.json `breakpoints`)"
                )

    for name, text in css.items():
        for m in re.finditer(r"@(media|container)\s+([^{]*)\{", _masked(text)):
            refuse_prelude(
                name,
                "window" if m.group(1) == "media" else "container",
                " ".join(m.group(2).split()),
            )
    for mode in p["modes"]:
        if mode.get("media"):
            refuse_prelude(f"parameters.json mode {mode['name']}", "window", mode["media"])

    found = literals(css)
    first = not BASELINE.exists()
    recorded = {} if first else json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline = Counter(recorded.get("literals", {}))
    counted = set(recorded.get("categories", ("px", "colour", "duration")))
    new = Counter() if first and update_baseline else found - baseline

    # a literal that only moved -- to another breakpoint, selector or sheet (C1's one owner per
    # selector folds rules together; its property and value did not change) -- is the same
    # literal, not a new one: the ratchet counts the values a person chose, not where they sit
    def unplaced(key: str) -> str:
        _file, _media, _selector, prop, hit = key.split(" | ")
        return f"{prop} | {hit}"

    gone = Counter()
    for key, n in (baseline - found).items():
        gone[unplaced(key)] += n
    for key in list(new):
        u = unplaced(key)
        moved = min(new[key], gone[u])
        if moved:
            new[key] -= moved
            gone[u] -= moved
    new = +new
    # every literal the sheets still write is a named exception (S1): its reason is in
    # design/literal-exceptions.json, and a reason no sheet uses any more goes with its literal
    exceptions_path = BASELINE.with_name("literal-exceptions.json")
    reasons = (
        json.loads(exceptions_path.read_text(encoding="utf-8")).get("reasons", {})
        if exceptions_path.exists()
        else {}
    )
    unnamed = [
        key for key in sorted(found) if key not in new and not str(reasons.get(key, "")).strip()
    ]
    stale = [key for key in sorted(reasons) if key not in found]
    for key in list(unnamed):  # a reason follows its literal when the literal only moved
        twin = next((s for s in stale if unplaced(s) == unplaced(key)), None)
        if twin:
            unnamed.remove(key)
            stale.remove(twin)
    for key in unnamed:
        out.append(
            f"{key} is an exception without a reason -- make it a parameter, "
            "or name it in design/literal-exceptions.json"
        )
    for key in stale:
        out.append(
            f"design/literal-exceptions.json names {key}, "
            "which no sheet writes any more -- remove it"
        )
    # a category the gate did not count when the list was written may be recorded once, whole
    widening = Counter({k: n for k, n in new.items() if category(k) not in counted})
    if update_baseline:
        new -= widening
    for key, n in sorted(new.items()):
        file, media, selector, prop, hit = key.split(" | ")
        where = f"@media {media} " if media else ""
        out.append(
            f"{file}: {where}`{selector}` {{ {prop}: … }} writes {hit} ({n} new) "
            "-- use a parameter from design/parameters.json or add one there"
        )
    if update_baseline:
        if new:
            out.append(
                "the ratchet only shrinks: "
                "parameterize the new literals above before updating the baseline"
            )
        else:
            BASELINE.write_text(
                json.dumps(
                    {
                        "about": (
                            "The literals the sheets still write (a px length other than 0 and 1, "
                            "a raw colour, a duration, an opacity, a z-index, a mixed colour's "
                            "strength, a length in em / rem / ch / vh / vw), by file, media, "
                            "selector and property. The list may only shrink: "
                            "scripts/check_ui_parameters.py refuses a literal that is not here; "
                            "a category the gate newly counts is recorded once."
                        ),
                        "categories": list(CATEGORIES),
                        "literals": dict(sorted(found.items())),
                    },
                    indent=1,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--update-baseline", action="store_true", help="rewrite the ratchet when no literal is new"
    )
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    found = problems(update_baseline=args.update_baseline)
    for line in found:
        print(line)
    literal_count = sum(
        literals(
            {p.name: p.read_text(encoding="utf-8") for p in (SRC / "css").glob("*.css")}
        ).values()
    )
    print(f"{len(found)} problem(s); {literal_count} literal(s) left in the sheets")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
