"""Checked reference images for layout.plan; retire with the one-time mover unless still used."""

from __future__ import annotations

import io
import json
import re
import tokenize
from collections.abc import Iterable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any, cast

ERROR = "test_layout.registry_invalid: reference "


class _Object(dict[str, Any]):
    """Retain the current parser convention and remember ambiguous addressed members."""

    def __init__(self, pairs: list[tuple[str, Any]]) -> None:
        super().__init__()
        self.duplicates: set[str] = set()
        for key, value in pairs:
            if key in self:
                self.duplicates.add(key)
            self[key] = value


def _path(value: Any, field: str) -> None:
    """Keep each declared source identity canonical and within its immutable tree."""
    if not isinstance(value, str) or not value or value == ".":
        raise ValueError(ERROR + field + "; declare a canonical source path")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or str(parsed) != value
        or ".." in parsed.parts
        or any(c in value for c in "\\:\0\r\n")
    ):
        raise ValueError(ERROR + field + "; keep it inside the source tree")


def source_paths(value: Sequence[str]) -> set[str]:
    """Identify extra immutable inputs for the same snapshot, without editing declarations."""
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(ERROR + "source paths; provide a sequence of canonical paths")
    for path in value:
        _path(path, "source path")
    return set(value)


def declarations(value: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate reviewed addresses without guessing a missing field or replacement value."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(ERROR + "declarations must be a list; rebuild from the pinned source")
    result, seen = [], set()
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) - {"path", "target", "pointer", "key", "line"}:
            raise ValueError(ERROR + "fields; use path,target and pointer/key or line")
        row = dict(raw)
        for field in ("path", "target"):
            _path(row.get(field), field)
        if ("pointer" in row) == ("line" in row):
            raise ValueError(ERROR + "address; choose exactly one pointer or line")
        if "pointer" in row:
            if not row["path"].endswith(".json") or type(row.get("key", False)) is not bool:
                raise ValueError(ERROR + "JSON address; use a JSON pointer and boolean key flag")
            _tokens(row["pointer"])
        elif (
            "key" in row
            or type(row["line"]) is not int
            or row["line"] < 1
            or not row["path"].endswith((".py", ".md", ".rst", ".txt"))
        ):
            raise ValueError(ERROR + "line; use a positive source/text line number")
        identity = row["path"], row.get("pointer", row.get("line")), row.get("key", False)
        if identity in seen:
            raise ValueError("test_layout.registry_collision: duplicate reference address")
        seen.add(identity)
        result.append(row)
    return result


def provenance(value: Any) -> list[dict[str, Any]]:
    """Validate recorded source addresses without treating their payloads as permission."""
    if not isinstance(value, list) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("value"), str)
        or not isinstance(row.get("successor"), str)
        for row in value
    ):
        raise ValueError(ERROR + "provenance; restore its generated source records")
    return declarations(
        [{k: v for k, v in row.items() if k not in {"value", "successor"}} for row in value]
    )


def _tokens(pointer: Any) -> tuple[str, ...]:
    if (
        not isinstance(pointer, str)
        or (pointer and not pointer.startswith("/"))
        or re.search(r"~(?![01])", pointer)
    ):
        raise ValueError(ERROR + "pointer; use RFC6901 tokens")
    return (
        tuple(t.replace("~1", "/").replace("~0", "~") for t in pointer[1:].split("/"))
        if pointer
        else ()
    )


def _at(document: Any, tokens: tuple[str, ...]) -> Any:
    for token in tokens:
        if isinstance(document, _Object) and token in document.duplicates:
            raise ValueError(
                ERROR + "addressed member is duplicated; declare an unambiguous locator"
            )
        if isinstance(document, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token) or int(token) >= len(document):
                raise ValueError(ERROR + "pointer index; rebuild from the pinned source")
            document = document[int(token)]
        elif isinstance(document, dict) and token in document:
            document = document[token]
        else:
            raise ValueError(ERROR + "pointer absent; rebuild from the pinned source")
    return document


def _location(value: Any, target: str, moves: dict[str, str], modules: dict[str, str]) -> str:
    if isinstance(value, str):
        for old, new in (
            (target, moves[target]),
            (target[:-3].replace("/", "."), modules[target[:-3].replace("/", ".")]),
        ):
            if value == old or value.startswith(old + ":") or value.startswith(old + "*"):
                return new + value[len(old) :]
    raise ValueError(ERROR + "value is not the declared locator; rebuild its source address")


def _json(
    text: str, rows: list[dict[str, Any]], moves: dict[str, str], modules: dict[str, str]
) -> tuple[str, list[dict[str, Any]]]:
    try:
        document = json.loads(text, object_pairs_hook=_Object)
    except json.JSONDecodeError as error:
        raise ValueError(ERROR + "JSON unreadable; restore its valid source") from error
    values: dict[tuple[str, ...], str] = {}
    keys: dict[tuple[str, ...], str] = {}
    records: list[dict[str, Any]] = []
    for row in rows:
        tokens = _tokens(row["pointer"])
        before = _at(document, tokens)
        if row.get("key"):
            if not tokens or not isinstance(_at(document, tokens[:-1]), dict):
                raise ValueError(ERROR + "key; address an existing object member")
            before = tokens[-1]
        after = _location(before, row["target"], moves, modules)
        (keys if row.get("key") else values)[tokens] = after
        records.append(row | {"value": before, "successor": after})

    def rewrite(item: Any, path: tuple[str, ...] = ()) -> Any:
        if path in values:
            return values[path]
        if isinstance(item, dict):
            changed = {}
            for key, value in item.items():
                target = keys.get((*path, key), cast(str, key))
                if target in changed:
                    raise ValueError("test_layout.registry_collision:" + target)
                changed[target] = rewrite(value, (*path, key))
            return changed
        if isinstance(item, list):
            return [rewrite(value, (*path, str(index))) for index, value in enumerate(item)]
        return item

    return json.dumps(rewrite(document), indent=2, ensure_ascii=False) + "\n", records


def _text(
    text: str, rows: list[dict[str, Any]], moves: dict[str, str], modules: dict[str, str]
) -> tuple[str, list[dict[str, Any]]]:
    lines = text.splitlines(keepends=True)
    permitted = None
    if rows[0]["path"].endswith(".py"):
        try:
            permitted = [
                (t.start, t.end)
                for t in tokenize.generate_tokens(io.StringIO(text).readline)
                if t.type in {tokenize.STRING, tokenize.COMMENT}
            ]
        except (tokenize.TokenError, IndentationError) as error:
            raise ValueError(ERROR + "Python unreadable; restore its valid source") from error
    records = []
    for row in rows:
        number, target = row["line"], row["target"]
        if number > len(lines):
            raise ValueError(
                ERROR + "line must address a literal/comment, not code; prepare it first"
            )
        before = lines[number - 1]
        found: list[tuple[int, int, str]] = []
        for old, new in (
            (target, moves[target]),
            (target[:-3].replace("/", "."), modules[target[:-3].replace("/", ".")]),
        ):
            pattern = re.compile(
                r"(?<![\w])" + re.escape(old) + r"(?=$|[:\s`'\"),;>\]]|\.(?:\s|$))"
            )
            found.extend((m.start(), m.end(), new) for m in pattern.finditer(before))
        if len(found) != 1:
            raise ValueError(ERROR + "line absent or ambiguous; declare its exact source locator")
        start, end, successor = found[0]
        if permitted is not None and not any(
            a <= (number, start) and (number, end) <= b for a, b in permitted
        ):
            raise ValueError(
                ERROR + "line must address a literal/comment, not code; prepare it first"
            )
        lines[number - 1] = before[:start] + successor + before[end:]
        records.append(row | {"value": before[start:end], "successor": successor})
    return "".join(lines), records


def images(
    files: dict[str, str],
    moves: dict[str, str],
    modules: dict[str, str],
    references: list[dict[str, Any]],
    *,
    blocked: Iterable[str] = (),
) -> tuple[dict[str, list[str | None]], list[dict[str, Any]]]:
    """Compose validated source declarations; file images remain the sole write authority."""
    denied = set(blocked)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in references:
        path = row["path"]
        if path not in files or row["target"] not in moves or path in moves or path in denied:
            raise ValueError(
                ERROR + "source/target; prepare moving-body references before relocation"
            )
        groups.setdefault(path, []).append(row)
    changed: dict[str, list[str | None]] = {}
    records: list[dict[str, Any]] = []
    for path, rows in groups.items():
        transform = _json if "pointer" in rows[0] else _text
        if any(("pointer" in row) != ("pointer" in rows[0]) for row in rows):
            raise ValueError(ERROR + "file address kinds must agree")
        try:
            after, observed = transform(files[path], rows, moves, modules)
        except ValueError as error:
            raise ValueError(str(error) + "; source=" + path) from error
        changed[path] = [files[path], after]
        records.extend(observed)
    return changed, records
