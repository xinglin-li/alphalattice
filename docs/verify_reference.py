"""Verify Markdown twins and the generated llms.txt index hierarchy."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
ROOT_ENTRY = re.compile(r"^- \[[^]]+\]\(([^)]+/llms\.txt)\) \((\d+) pages\)$")
PAGE_ENTRY = re.compile(r"^- \[[^]]+\]\(([^)]+\.md)\)$")


def local_path(site: Path, url: str) -> Path:
    """Resolve a documentation URL inside the local build directory."""
    parsed = urlparse(url)
    path = (site / parsed.path.lstrip("/")).resolve()
    if not path.is_relative_to(site.resolve()):
        raise ValueError(f"index points outside site: {url}")
    return path


def verify(site: Path) -> tuple[int, int]:
    """Assert that each HTML page has one Markdown twin and one index entry."""
    root = site / "llms.txt"
    if not root.is_file() or not (site / "llms-full.txt").is_file():
        raise ValueError("root llms.txt or llms-full.txt missing")
    indexes = []
    indexed_pages = []
    for line in root.read_text(encoding="utf-8").splitlines():
        if not line.startswith("- "):
            continue
        match = ROOT_ENTRY.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed root entry: {line}")
        index = local_path(site, match.group(1))
        indexes.append(index)
        entries = []
        for entry in index.read_text(encoding="utf-8").splitlines():
            if not entry.startswith("- "):
                continue
            page_match = PAGE_ENTRY.fullmatch(entry)
            if page_match is None:
                raise ValueError(f"malformed page entry: {entry}")
            entries.append(local_path(site, page_match.group(1)))
        if len(entries) != int(match.group(2)):
            raise ValueError(f"page count differs in {index}")
        indexed_pages.extend(entries)
    if len(indexes) != len(set(indexes)):
        raise ValueError("duplicate section index in root llms.txt")
    if set(indexes) != set(site.rglob("llms.txt")) - {root}:
        raise ValueError("root llms.txt does not list every section index")
    duplicates = [path for path, count in Counter(indexed_pages).items() if count != 1]
    if duplicates:
        raise ValueError(f"pages indexed more than once: {duplicates[:5]}")
    twins = set(site.rglob("*.md"))
    if set(indexed_pages) != twins:
        raise ValueError(
            f"indexed Markdown pages differ from twins: "
            f"missing={len(twins - set(indexed_pages))}, "
            f"extra={len(set(indexed_pages) - twins)}"
        )
    html_pages = {path for path in site.rglob("index.html") if path != site / "404.html"}
    for html in html_pages:
        if html.with_suffix(".md") not in twins:
            raise ValueError(f"HTML page has no Markdown twin: {html}")
    if len(html_pages) != len(twins):
        raise ValueError(f"HTML page count {len(html_pages)} != twin count {len(twins)}")
    return len(indexes), len(twins)


def main() -> None:
    """Check one MkDocs output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", nargs="?", type=Path, default=ROOT / "site")
    args = parser.parse_args()
    sections, pages = verify(args.site.resolve())
    print(f"verified {pages} HTML/Markdown twins in {sections} llms.txt indexes")


if __name__ == "__main__":
    main()
