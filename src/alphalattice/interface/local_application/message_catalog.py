"""The id catalog owns the checked English aliases used by legacy message readers."""

from collections.abc import Mapping
from functools import cache
from typing import Any

from alphalattice.kernel.shared_kernel.project_layout import runtime_json


def english_message_ids(entries: Mapping[str, Any]) -> dict[str, str]:
    """Resolve each MESSAGE's English form once; ambiguous aliases cannot be published."""
    aliases: dict[str, str] = {}
    for code, row in entries.items():
        if not isinstance(row, Mapping) or row.get("kind") != "MESSAGE":
            continue
        forms = row.get("words")
        english = forms.get("en") if isinstance(forms, Mapping) else None
        if not isinstance(english, str):
            continue
        if english in aliases or english in entries:
            raise ValueError(f"messages: {code} duplicate or conflicting English alias {english!r}")
        aliases[english] = code
    return aliases


@cache
def catalog_words() -> dict[str, dict[str, str]]:
    """Project ids and their checked English aliases for both CLI and Workbench readers."""
    entries = runtime_json("config/registries/messages.json")["entries"]
    ids = {code: code for code, row in entries.items() if row["kind"] == "MESSAGE"}
    return {
        key: entries[code]["words"] for key, code in (ids | english_message_ids(entries)).items()
    }
