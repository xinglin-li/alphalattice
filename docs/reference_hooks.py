"""Build package llms.txt indexes from MkDocs' Markdown twins."""

from __future__ import annotations

from pathlib import Path


def on_post_build(config: dict) -> None:
    """Replace the root index with section indexes after llmstxt builds twins."""

    site = Path(config["site_dir"])
    base = config["site_url"].rstrip("/")
    pages = sorted(
        path for path in site.rglob("*.md") if path.is_file() and path.name != "llms-full.md"
    )
    sections: dict[str, list[Path]] = {}
    for page in pages:
        relative = page.relative_to(site)
        if relative.parts[0] == "reference" and len(relative.parts) > 2:
            section = relative.parts[1]
        else:
            section = "overview"
        sections.setdefault(section, []).append(page)

    root_lines = [
        "# AlphaLattice documentation",
        "",
        "> Local Markdown reference.",
        "",
        "## Section indexes",
        "",
    ]
    for section, entries in sorted(sections.items()):
        target = (
            site / "overview/llms.txt"
            if section == "overview"
            else site / "reference" / section / "llms.txt"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# {section} reference", "", f"> {len(entries)} Markdown pages.", ""]
        for page in entries:
            relative = page.relative_to(site).as_posix()
            text = page.read_text(encoding="utf-8")
            title = next(
                (
                    line.removeprefix("# ").strip()
                    for line in text.splitlines()
                    if line.startswith("# ")
                ),
                relative,
            )
            lines.append(f"- [{title}]({base}/{relative})")
        target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        index = target.relative_to(site).as_posix()
        root_lines.append(f"- [{section}]({base}/{index}) ({len(entries)} pages)")
    (site / "llms.txt").write_text("\n".join(root_lines) + "\n", encoding="utf-8", newline="\n")
