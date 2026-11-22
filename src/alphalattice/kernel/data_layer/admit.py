"""The append boundary's form checks: what an event must be before the log keeps it forever.

`admit` judges one event against its type's registered spec. Fields a type does not know are kept
and never judged. A live append is held to every rule; replay (live=False) re-admits what was
admitted once, so a text later forgotten and a version later superseded do not refuse it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any, Final

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.chain import digest

NUMERIC: Final = (
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "cost_usd",
    "bytes_read",
    "bytes_written",
)
"""Fields a reader casts to a number: their form is checked here, once."""
SECRET_NAME: Final = re.compile(
    r"(?i)(secret|token|password|passwd|api[_-]?key|private[_-]?key|credential)"
)
SECRET_VALUE: Final = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:sk|rk)-[A-Za-z0-9_-]{16,}"
    r"|\bghp_[A-Za-z0-9]{20,}|\bAKIA[0-9A-Z]{16}\b"
)
PART_KEYS: Final = ("part", "min_known_asof_us", "max_known_asof_us")

Superseded = Callable[[str, Any, Any], bool]
"""Whether (family, name, version) is a superseded definition."""


def _refuse(code: str, message: str) -> DataError:
    return DataError(message, code=f"data_layer.{code}")


def admit(
    row: Mapping[str, Any],
    text: str | None,
    spec: Mapping[str, Any],
    superseded: Superseded,
    *,
    live: bool = True,
    body: bytes | None = None,
) -> dict[str, Any]:
    """Admit one event and return its parsed attributes."""
    attrs: dict[str, Any] = json.loads(row["attrs"])
    missing = [name for name in spec.get("required", ()) if name not in attrs]
    if missing:
        raise _refuse("event_attrs_missing", f"{row['type']} needs {', '.join(missing)}")
    _admit_text(row, spec, text, live=live)
    bad = [name for name in NUMERIC if name in attrs and not _number(attrs[name])]
    if bad or _parts_malformed(attrs.get("parts")):
        raise _refuse("event_attrs_form", f"{', '.join(bad) or 'parts'} malformed")
    hits = secrets_in(attrs) + (["text"] if text and SECRET_VALUE.search(text) else [])
    hits += _body_secret(body) + _setting_value(row["type"], attrs)
    if hits:
        raise _refuse(
            "event_secret",
            f"{', '.join(hits)} would make a credential permanent; use a keychain reference",
        )
    for family, name, version in _uses(row, attrs) if live else ():
        if superseded(family, name, version):
            raise _refuse(
                "definition_superseded",
                f"{family} {name} v{version} is defective; write under its successor",
            )

    return attrs


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _admit_text(
    row: Mapping[str, Any], spec: Mapping[str, Any], text: str | None, *, live: bool
) -> None:
    named, use = row.get("text_digest"), spec.get("text")
    if text is not None and named != digest(text.encode()):
        raise _refuse("event_text_unbound", "the text does not match the event's text_digest")
    if text is None and named is None and use == "required":
        raise _refuse("event_text_missing", f"{row['type']} carries its text")
    if text is None and named is not None and live:
        raise _refuse("event_text_missing", f"a live {row['type']} writes the text it names")
    if named is not None and use not in ("required", "optional"):
        raise _refuse("event_text_unexpected", f"{row['type']} carries no text")


def _parts_malformed(parts: Any) -> bool:
    """An admission's parts: each partition it touched with its rows' knowledge-time range."""
    if parts is None:
        return False
    return not isinstance(parts, list) or not all(
        isinstance(p, dict)
        and all(key in p for key in PART_KEYS)
        and type(p["min_known_asof_us"]) is int
        and type(p["max_known_asof_us"]) is int
        and p["min_known_asof_us"] <= p["max_known_asof_us"]
        for p in parts
    )


def secrets_in(value: Any, name: str = "") -> list[str]:
    """Each place in an attribute tree holding a credential.

    A place is a secret-named field with a value (a `*_ref` names a keychain entry, never a
    secret), or a key-shaped string.
    """
    if isinstance(value, dict):
        return [hit for key, item in value.items() for hit in secrets_in(item, str(key))]
    if isinstance(value, list):
        return [hit for item in value for hit in secrets_in(item, name)]
    named = bool(SECRET_NAME.search(name)) and not name.endswith("_ref") and bool(value)
    if isinstance(value, str) and (named or SECRET_VALUE.search(value)):
        return [name or "value"]
    return []


def _body_secret(body: bytes | None) -> list[str]:
    """An inline body is kept forever, so a body that is text is scanned like a text."""
    try:
        return ["body"] if body is not None and SECRET_VALUE.search(body.decode()) else []
    except UnicodeDecodeError:
        return []


def _setting_value(type_: str, attrs: Mapping[str, Any]) -> list[str]:
    """A credential setting is written as its secret_ref, never its value."""
    key = str(attrs.get("key", ""))
    secret = type_ == "setting.changed" and SECRET_NAME.search(key)
    return ["value"] if secret and attrs.get("value") is not None else []


def _uses(row: Mapping[str, Any], attrs: Mapping[str, Any]) -> list[tuple[str, Any, Any]]:
    """The definitions an event writes under: its type and its measures, artifact type or method."""
    uses: list[tuple[str, Any, Any]] = [("event_type", row["type"], row["type_version"])]
    if row["type"] == "result.recorded":
        uses += [("metric", measure[0], measure[1]) for measure in attrs.get("measures", ())]
    elif row["type"] == "artifact.produced":
        uses.append(("artifact_type", attrs.get("type"), attrs.get("type_version")))
    elif row["type"] == "recipe.defined":
        uses.append(("method", attrs.get("method"), attrs.get("method_version")))
    return uses
