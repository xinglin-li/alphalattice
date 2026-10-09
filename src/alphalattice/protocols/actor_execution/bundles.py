"""What an agent reads: one index and a few Markdown files the Host prepared.

A bundle is an index (`README.md`) and material files packed by size, not one
per subject: sections -- an issuer each -- are filled into a file in order up
to the bound and the next file opens when it is full. A section is never split
across files unless it alone exceeds the bound; then it is split between its
blocks (a filing each), and a block that alone exceeds the bound between its
lines. No line is longer than the line bound: a long paragraph wraps at a
sentence. The index names every file with what it covers and its counts, and
how to read them (`READING_RULE`): each file whole, one read a file.

What a bundle says is the owning domain's; this module only packs it. What
the Host keeps about a bundle it wrote -- the directory, the role and the exact
submission an answer to it completes -- is an `AgentBundleRecord`, never a file
beside the bundle: an agent's files carry no hash, so its answer needs none.
An answer names the files its agent read whole (`ANSWER_READ_FIELD`): its own word,
kept beside the answer as provenance, never a condition of reading it (`answer_read`, OP11).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.actor_execution.answers import SPECIALIST_REFERENCE_COUNT

BUNDLE_FILE_BYTES: int = 32 * 1024
"""A file is one whole read on every agent host. Codex hands the model at most 10,000 tokens of
one tool call's output, counted at four bytes a token, and cuts the middle of the rest: a
140,285-byte bundle read in one call came back as 35,073 tokens cut to 10,000 (
2026-09-27). 32 KB leaves the reading command room under 40,000 bytes; Claude Code reads
2,000 lines a call."""
BUNDLE_FILE_LINES: int = 1500
BUNDLE_LINE_CHARACTERS: int = 1000
BUNDLE_INDEX = "README.md"
PACKING_RULE = (
    f"Files are packed by size: at most {BUNDLE_FILE_BYTES // 1024} KB or "
    f"{BUNDLE_FILE_LINES:,} lines a file, whichever comes first, and "
    f"{BUNDLE_LINE_CHARACTERS:,} characters a line, so each file is one whole read. A section "
    "is never split across files unless it alone exceeds that bound; then it is split between "
    "its filings."
)
READING_RULE = (
    "Read this index, then every file listed above whole, one tool call a file -- never two "
    "files in one call; calls may run together. A read that comes back shortened (cut in the "
    "middle, or missing its end) is not a read: read that file again in parts, by line range, "
    "until you have all of it. Decide only when every file is read."
)
ANSWER_READ_FIELD = "read"
"""Beside an external answer's own fields: the names of the bundle files its agent read whole,
this index included. The Host keeps the list with the answer as provenance, reads the answer
whatever it names (OP11), and keeps it out of what it submits."""

_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")

type AgentRole = Literal["ALPHA", "ANALYST", "CRO", "DATA", "FACTOR", "PORTFOLIO", "RISK"]


def bundle_directory_key(directory: str) -> str:
    """Normalize a bundle directory to one platform-specific key.

    Callers supply an absolute directory. Path spelling follows the platform's
    separator and case rules so equivalent names share one bundle key.
    """
    return os.path.normcase(os.path.normpath(directory))


def bundle_slot(directory: str) -> str:
    """Where the Host keeps what one bundle directory binds."""
    return str(canonical_hash({"bundle_directory": bundle_directory_key(directory)}))


class AgentBundleRecord(BaseModel):  # type: ignore[misc]
    """Record the Host's binding for one prepared bundle.

    The record binds the output directory, role, and exact submission
    completed by an answer. One directory holds one bundle.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["alphalattice.agent-bundle.v1"] = "alphalattice.agent-bundle.v1"
    role: AgentRole
    bundle_directory: str = Field(min_length=1, max_length=1024)
    submission: dict[str, str]
    """The domain completion request without its answer, or a generic Task id/version binding."""
    files: tuple[str, ...] = Field(min_length=1, max_length=256)
    allowed_references: tuple[str, ...] = Field(default=(), max_length=SPECIALIST_REFERENCE_COUNT)
    """Exact Task/artifact references this generic specialist bundle admits; never client text."""
    record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _slot_matches(self) -> Self:
        if self.bundle_directory != bundle_directory_key(self.bundle_directory):
            raise ValueError("agent_bundle.directory_not_normalized")
        if self.record_hash != bundle_slot(self.bundle_directory):
            raise ValueError("agent_bundle.record_slot_mismatch")
        return self


@dataclass(frozen=True, slots=True)
class BundleBlock:
    """Keep related lines and their index counts together.

    A block can hold one filing's excerpts or one finding, with its counts
    for the bundle index.
    """

    lines: tuple[str, ...]
    counts: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BundleSection:
    """One subject of the material (an issuer): its heading and its blocks."""

    key: str
    heading: str
    blocks: tuple[BundleBlock, ...]


@dataclass(frozen=True, slots=True)
class BundleFile:
    """Hold one packed material file and its index metadata."""

    name: str
    text: str
    covers: tuple[str, ...]
    """What the file holds, as the index names it (`AAPL`, `MSFT part 2 of 3`)."""
    counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class AgentBundle:
    """The files of one bundle, the index first."""

    files: tuple[tuple[str, str], ...]

    def text(self, name: str) -> str:
        """Read one named bundle file.

        Args:
            name: File name listed by the bundle index.

        Returns:
            The file's text.

        Raises:
            KeyError: If the bundle does not contain the name.

        """
        for file_name, text in self.files:
            if file_name == name:
                return text
        raise KeyError(name)

    def as_message(self) -> str:
        """Render the whole bundle as one text with the index first.

        The built-in agent reads this form in one call.
        """
        return "\n\n".join(
            text if name == BUNDLE_INDEX else f"<!-- file: {name} -->\n\n{text}"
            for name, text in self.files
        )

    def write(self, directory: Path) -> tuple[Path, ...]:
        """Write the bundle's files with LF into a directory.

        Args:
            directory: Destination for the index and material files.

        Returns:
            The paths written in bundle order.

        """
        directory.mkdir(parents=True, exist_ok=True)
        written = []
        for name, text in self.files:
            path = directory / name
            path.write_text(text, encoding="utf-8", newline="\n")
            written.append(path)
        return tuple(written)

    def measure(self) -> dict[str, object]:
        """Files, lines, bytes and a cl100k proxy token count, per file and in all."""
        from .projections import estimate_cl100k_proxy_tokens

        rows = [
            {
                "name": name,
                "lines": text.count("\n") + (0 if text.endswith("\n") else 1),
                "bytes": len(text.encode("utf-8")),
                "longest_line": max((len(line) for line in text.splitlines()), default=0),
                "proxy_tokens": estimate_cl100k_proxy_tokens(text),
            }
            for name, text in self.files
        ]
        return {
            "files": len(rows),
            "lines": sum(int(str(row["lines"])) for row in rows),
            "bytes": sum(int(str(row["bytes"])) for row in rows),
            "proxy_tokens": sum(int(str(row["proxy_tokens"])) for row in rows),
            "per_file": rows,
        }


def wrap_text(text: str, limit: int = BUNDLE_LINE_CHARACTERS) -> list[str]:
    """Wrap source text into lines no longer than ``limit``.

    Paragraphs remain separate. Within a paragraph, whitespace folds; long
    text wraps at sentences, then spaces, then by character for an oversized
    word.
    """
    lines: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        folded = " ".join(paragraph.split())
        if not folded:
            continue
        if lines:
            lines.append("")
        row = ""
        for sentence in _SENTENCE_END.split(folded):
            for piece in _fit(sentence, limit):
                if row and len(row) + 1 + len(piece) > limit:
                    lines.append(row)
                    row = piece
                else:
                    row = f"{row} {piece}" if row else piece
        if row:
            lines.append(row)
    return lines


def _fit(sentence: str, limit: int) -> Iterable[str]:
    if len(sentence) <= limit:
        yield sentence
        return
    row = ""
    for word in sentence.split(" "):
        while len(word) > limit:
            if row:
                yield row
                row = ""
            yield word[:limit]
            word = word[limit:]
        if row and len(row) + 1 + len(word) > limit:
            yield row
            row = word
        else:
            row = f"{row} {word}" if row else word
    if row:
        yield row


def pack_sections(sections: Sequence[BundleSection], *, stem: str) -> tuple[BundleFile, ...]:
    """Fill the sections into files in order, by size (`PACKING_RULE`)."""
    files: list[BundleFile] = []
    lines: list[str] = []
    covers: list[str] = []
    counts: dict[str, int] = {}

    def flush() -> None:
        nonlocal lines, covers, counts
        if lines:
            name = f"{stem}-{len(files) + 1:02d}.md"
            files.append(
                BundleFile(
                    name=name, text="\n".join(lines) + "\n", covers=tuple(covers), counts=counts
                )
            )
        lines, covers, counts = [], [], {}

    def fits(extra: Sequence[str]) -> bool:
        added = list(extra) if not lines else ["", *extra]
        return len(lines) + len(added) <= BUNDLE_FILE_LINES and _bytes([*lines, *added]) <= (
            BUNDLE_FILE_BYTES
        )

    def add(part: Sequence[str], label: str, part_counts: Mapping[str, int]) -> None:
        if lines:
            lines.append("")
        lines.extend(part)
        covers.append(label)
        for key, value in part_counts.items():
            counts[key] = counts.get(key, 0) + value

    for section in sections:
        whole = _section_lines(section.heading, section.blocks)
        whole_counts = _counts(section.blocks)
        if fits(whole):
            add(whole, section.key, whole_counts)
            continue
        if _within_bound(whole):
            flush()
            add(whole, section.key, whole_counts)
            continue
        parts = _split(section)
        flush()
        for index, blocks in enumerate(parts, start=1):
            heading = f"{section.heading} (part {index} of {len(parts)})"
            part = _section_lines(heading, blocks)
            if not fits(part):
                flush()
            add(part, f"{section.key} part {index} of {len(parts)}", _counts(blocks))
    flush()
    return tuple(files)


def answer_read(answer: Mapping[str, object]) -> tuple[dict[str, object], tuple[str, ...]]:
    """An external answer's own fields, and the bundle files it names as read whole.

    `ANSWER_READ_FIELD` lists the file names, each once in the order given; a list that is
    missing or holds no names names none. The list is the agent's own word, kept beside its
    answer as provenance: the Host reads the answer whatever it names (OP11).
    """
    named = answer.get(ANSWER_READ_FIELD)
    read = (
        tuple(dict.fromkeys(value for value in named if isinstance(value, str)))
        if isinstance(named, list)
        else ()
    )
    own = {key: value for key, value in answer.items() if key != ANSWER_READ_FIELD}
    return own, read


def compose_bundle(index_lines: Sequence[str], files: Sequence[BundleFile]) -> AgentBundle:
    """Compose a bundle with its index before the material files.

    The index lists its own lines, then each file's coverage and counts,
    followed by the packing rule and the reading rule. Material files follow.
    """
    listing = [
        f"- `{value.name}` -- {'; '.join(value.covers)}"
        + (
            ""
            if not value.counts
            else " (" + ", ".join(f"{count} {key}" for key, count in value.counts.items()) + ")"
        )
        for value in files
    ]
    index = [*index_lines, "", "## Files", "", *listing, "", PACKING_RULE, "", READING_RULE]
    wrapped = [
        row
        for line in index
        for row in ([line] if len(line) <= BUNDLE_LINE_CHARACTERS else wrap_text(line))
    ]
    return AgentBundle(
        files=(
            (BUNDLE_INDEX, "\n".join(wrapped).rstrip("\n") + "\n"),
            *((value.name, value.text) for value in files),
        )
    )


def _section_lines(heading: str, blocks: Sequence[BundleBlock]) -> list[str]:
    lines = [heading]
    for block in blocks:
        lines.append("")
        lines.extend(block.lines)
    return lines


def _counts(blocks: Iterable[BundleBlock]) -> dict[str, int]:
    total: dict[str, int] = {}
    for block in blocks:
        for key, value in block.counts.items():
            total[key] = total.get(key, 0) + value
    return total


def _bytes(lines: Sequence[str]) -> int:
    return sum(len(line.encode("utf-8")) + 1 for line in lines)


def _within_bound(lines: Sequence[str]) -> bool:
    return len(lines) <= BUNDLE_FILE_LINES and _bytes(lines) <= BUNDLE_FILE_BYTES


def _split(section: BundleSection) -> list[tuple[BundleBlock, ...]]:
    """Split a section that exceeds one file's bounds.

    Whole blocks stay together when possible. An oversized block splits by
    line. Each part reserves four lines and a kilobyte for its heading.
    """
    lines_bound, bytes_bound = BUNDLE_FILE_LINES - 4, BUNDLE_FILE_BYTES - 1024
    pieces: list[BundleBlock] = []
    for block in section.blocks:
        if len(block.lines) + 1 <= lines_bound and _bytes(block.lines) <= bytes_bound:
            pieces.append(block)
            continue
        chunk: list[str] = []
        first = True
        for line in block.lines:
            if chunk and (len(chunk) + 1 > lines_bound or _bytes([*chunk, line]) > bytes_bound):
                pieces.append(BundleBlock(tuple(chunk), block.counts if first else {}))
                chunk, first = [], False
            chunk.append(line)
        if chunk:
            pieces.append(BundleBlock(tuple(chunk), block.counts if first else {}))
    parts: list[list[BundleBlock]] = [[]]
    for piece in pieces:
        candidate = [*parts[-1], piece]
        size = _section_lines(section.heading, candidate)
        if parts[-1] and (len(size) > lines_bound or _bytes(size) > bytes_bound):
            parts.append([piece])
        else:
            parts[-1] = candidate
    return [tuple(part) for part in parts if part]


__all__ = [
    "ANSWER_READ_FIELD",
    "BUNDLE_FILE_BYTES",
    "BUNDLE_FILE_LINES",
    "BUNDLE_INDEX",
    "BUNDLE_LINE_CHARACTERS",
    "PACKING_RULE",
    "READING_RULE",
    "AgentBundle",
    "AgentBundleRecord",
    "AgentRole",
    "BundleBlock",
    "BundleFile",
    "BundleSection",
    "answer_read",
    "bundle_directory_key",
    "bundle_slot",
    "compose_bundle",
    "pack_sections",
    "wrap_text",
]
