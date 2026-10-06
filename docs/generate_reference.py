"""Generate source-backed MkDocs stubs for public AlphaLattice modules."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/alphalattice"
REFERENCE = ROOT / "docs/reference"


def public(path: Path) -> bool:
    parts = path.relative_to(SOURCE).parts
    return all(not part.startswith("_") or part == "__init__.py" for part in parts)


def page_for(path: Path) -> Path:
    relative = path.relative_to(SOURCE)
    if path.name == "__init__.py":
        return (
            REFERENCE.joinpath(*relative.parts[:-1], "index.md")
            if len(relative.parts) > 1
            else REFERENCE / "alphalattice/index.md"
        )
    return REFERENCE / relative.with_suffix(".md")


def name_for(path: Path) -> str:
    parts = list(path.relative_to(SOURCE).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(["alphalattice", *parts])


def summary_for(path: Path, module: str) -> str:
    docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8-sig")))
    return docstring.splitlines()[0].strip() if docstring else f"Public API of `{module}`."


def generate() -> list[Path]:
    pages = []
    for path in sorted(SOURCE.rglob("*.py")):
        if not public(path):
            continue
        page = page_for(path)
        module = name_for(path)
        page.parent.mkdir(parents=True, exist_ok=True)
        content = f"# {module}\nDate: 2026-09-26\n\n{summary_for(path, module)}\n\n::: {module}\n"
        page.write_text(content, encoding="utf-8", newline="\n")
        pages.append(page)
    return pages


if __name__ == "__main__":
    print(f"Generated {len(generate())} reference stubs")
