"""What a committed request document is allowed to say, and nothing else.

A request selects; the Host resolves. That rule was written three times and
enforced once and a half: Alpha scanned its own *top-level* keys for
authority-shaped names, Sector scanned nothing, Portfolio scanned nothing -- so
``config/stage6-portfolio-campaign.yaml`` carried six resolved artifact
identities nested one level under ``request.upstream[]``, where a top-level key
scan cannot see them, in a file whose own header says there is no derived hash in
it.

So the rule lives here, in ``protocols``, rather than in the control plane. It is
a statement about the shape of a document, not a behaviour of a runtime, and
putting it in ``control`` made every Desk that enforces it depend on the control
plane to do so -- which widened the Alpha package's reusable closure the first
time it was tried.

The check is recursive through mappings *and sequences* at every depth, and it
looks at values as well as keys. A key scan alone is defeated by naming the field
something innocent -- ``sector_revision`` carries a sealed revision identity and
reads like a version label -- so the value rule is the one that actually holds: a
sixty-four character hex digest in a request is a resolved identity whatever it
is called, and an absolute path is a machine rather than a selection.

The key rule is deliberately narrow rather than a substring sweep. ``hash`` as a
substring also appears in ``data_snapshot_handle`` and ``publication_intent``,
which are exactly the handle and the intent a request is *supposed* to state; a
substring ban would refuse the two documents that were already right in order to
catch the two that were wrong.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from .contracts import AuthoringError

FORBIDDEN_KEYS = frozenset(
    {
        "python_module",
        "python_path",
        "module",
        "callable",
        "entry_point",
        "entrypoint",
        "import",
        "import_path",
        "factory",
        "class_path",
        "dotted_path",
        "publication_target",
    }
)

AUTHORITY_KEY_NAMES = frozenset({"hash", "uri", "url", "pointer", "credential", "path"})
"""Bare key names that are an identity rather than a selection."""

AUTHORITY_KEY_SUFFIXES = ("_hash", "_uri", "_url", "_pointer", "_credential", "_path")
"""Suffixes that declare the field's own content to be a resolved identity.

A ``*_handle`` is the opposite and stays admitted: a handle is what a caller is
entitled to name, and resolving it to an identity is the Host's job.
"""

_RESOLVED_IDENTITY = re.compile(r"^[0-9a-f]{64}$")
_ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|/|\\\\)")

MAXIMUM_DEPTH = 12


class _DeclarationLoader(yaml.SafeLoader):  # type: ignore[misc]
    """Safe YAML in the declaration dialect: YAML 1.2's core booleans, numbers and null, and
    YAML 1.1's dates.

    `true` and `false` are the booleans, where YAML 1.1 also read `yes`, `on` and their kin;
    a number with a leading zero is decimal, where 1.1 read it as octal; `0o` and `0x` name
    octal and hexadecimal; an exponent needs no point (`1e-10`). Dates stay dates: authored
    sessions are written as them. A key written twice in one mapping is refused, at its line
    and column, where PyYAML keeps the last value silently.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _value in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    "while reading a mapping",
                    node.start_mark,
                    f"found the key {key!r} twice",
                    key_node.start_mark,
                )
            seen.add(key)
        return dict(super().construct_mapping(node, deep=deep))


class _DeclarationDumper(yaml.SafeDumper):  # type: ignore[misc]
    """Safe YAML written in the loader's dialect: a string it would read as a number is quoted."""


_CORE_SCALARS: tuple[tuple[str, re.Pattern[str], list[str]], ...] = (
    (
        "tag:yaml.org,2002:bool",
        re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
        list("tTfF"),
    ),
    (
        "tag:yaml.org,2002:int",
        re.compile(r"^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$"),
        list("-+0123456789"),
    ),
    (
        "tag:yaml.org,2002:float",
        re.compile(
            r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?"
            r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
        ),
        list("-+0123456789."),
    ),
    ("tag:yaml.org,2002:null", re.compile(r"^(?:~|null|Null|NULL|)$"), ["~", "n", "N", ""]),
)
"""YAML 1.2's core scalars, in place of YAML 1.1's. Measured before the dialect was named:
of the 1,222 YAML documents in the QA store and the tree, none read differently but by its
unquoted dates, which the dialect keeps, so no sealed declaration's normalized form moves."""


def _core_int(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> int:
    text = str(loader.construct_scalar(node))
    if text.startswith(("0o", "0x")):
        return int(text[2:], 8 if text[1] == "o" else 16)
    return int(text, 10)


# Keep SafeLoader's and SafeDumper's global resolvers unchanged: other YAML documents keep their
# own dialect. Quoted scalars remain strings. The dumper resolves as the loader does, so what
# the Host writes reads back as it was.
for _dialect, _base in (
    (_DeclarationLoader, yaml.SafeLoader),
    (_DeclarationDumper, yaml.SafeDumper),
):
    replaced = {tag for tag, _pattern, _first in _CORE_SCALARS}
    _dialect.yaml_implicit_resolvers = {
        key: [(tag, pattern) for tag, pattern in value if tag not in replaced]
        for key, value in _base.yaml_implicit_resolvers.items()
    }
    for _tag, _pattern, _first in _CORE_SCALARS:
        _dialect.add_implicit_resolver(_tag, _pattern, _first)
_DeclarationLoader.add_constructor("tag:yaml.org,2002:int", _core_int)


def dump_declaration(value: Any) -> str:
    """Write a declaration so that the declaration loader reads the same values back.

    Args:
        value: The declaration, of JSON-like values.

    Returns:
        Its YAML, keys in their order, text unescaped.
    """
    return cast(
        str, yaml.dump(value, Dumper=_DeclarationDumper, sort_keys=False, allow_unicode=True)
    )


def rewrite_in_declaration_dialect(text: str) -> str:
    """Rewrite YAML that PyYAML's default dumper wrote, in the declaration dialect.

    The default dumper leaves a string such as ``"1e-10"`` unquoted, which the
    declaration loader reads as a number, so an exported declaration could not be
    submitted again. The default loader reads its own dumper's output
    exactly, and the declaration dumper writes those values back.

    Args:
        text: YAML written by ``yaml.safe_dump``.

    Returns:
        The same values, in the declaration dialect.
    """
    return dump_declaration(yaml.safe_load(text))


def load_safe_yaml_document(text: str) -> Any:
    """Interpret a declaration with safe constructors and consistent number types."""
    return yaml.load(text, Loader=_DeclarationLoader)


class UnparsableDocument(AuthoringError):
    """A declaration that does not parse, located where YAML stopped reading it.

    The code stays ``research_authoring.document_unparsable``; the line and column are
    ``document_location``, as the client locates a file it reads itself, so a declaration sent
    as text is located on every entry.
    """

    def __init__(self, error: yaml.YAMLError) -> None:
        """Refuse the declaration, located at the YAML error's mark when it has one.

        Args:
            error: What the declaration loader raised.
        """
        super().__init__("research_authoring.document_unparsable")
        mark = getattr(error, "problem_mark", None)
        self.document_location: dict[str, int] | None = (
            {"line": mark.line + 1, "column": mark.column + 1} if mark is not None else None
        )


def load_selection_only_document(text: str) -> Mapping[str, Any]:
    """Parse one request document, failing closed on anything the Host resolves."""
    try:
        parsed = load_safe_yaml_document(text)
    except yaml.YAMLError as error:
        raise UnparsableDocument(error) from error
    if not isinstance(parsed, dict):
        raise AuthoringError("research_authoring.document_not_a_mapping")
    require_selection_only_document(parsed)
    return parsed


def require_selection_only_document(payload: object) -> None:
    """Refuse a request that states what the Host is supposed to resolve.

    Raises :class:`AuthoringError` with one of four codes. Each Desk catches it
    at its own boundary and re-raises its own stable error type, so there is one
    rule and still one failure class per Desk.
    """
    _require_selection_only(payload, depth=0)


def _require_selection_only(node: object, *, depth: int) -> None:
    if depth > MAXIMUM_DEPTH:
        raise AuthoringError("research_authoring.document_too_deep")
    if isinstance(node, dict):
        for key, value in node.items():
            if not isinstance(key, str):
                raise AuthoringError("research_authoring.document_key_not_a_string")
            _require_selection_key(key)
            _require_selection_only(value, depth=depth + 1)
        return
    if isinstance(node, list | tuple):
        for value in node:
            _require_selection_only(value, depth=depth + 1)
        return
    if isinstance(node, str):
        _require_selection_value(node)


def _require_selection_key(key: str) -> None:
    lowered = key.lower()
    if lowered in FORBIDDEN_KEYS:
        raise AuthoringError("research_authoring.document_declares_python_path")
    if lowered in AUTHORITY_KEY_NAMES or lowered.endswith(AUTHORITY_KEY_SUFFIXES):
        raise AuthoringError("research_authoring.document_declares_authority_key")


def _require_selection_value(value: str) -> None:
    if looks_like_python_path(value):
        raise AuthoringError("research_authoring.document_declares_python_path")
    if _RESOLVED_IDENTITY.match(value) is not None:
        raise AuthoringError("research_authoring.document_states_resolved_identity")
    if _ABSOLUTE_PATH.match(value) is not None:
        raise AuthoringError("research_authoring.document_states_absolute_path")


def looks_like_python_path(value: str) -> bool:
    """Reject `package.module:callable` and `alphalattice.a.b.C` shaped strings."""
    if ":" in value and " " not in value:
        left, _, right = value.partition(":")
        if left.replace(".", "").replace("_", "").isalnum() and right.isidentifier():
            return True
    return value.startswith(("alphalattice.", "builtins.", "os.", "subprocess."))


__all__ = [
    "AUTHORITY_KEY_NAMES",
    "AUTHORITY_KEY_SUFFIXES",
    "FORBIDDEN_KEYS",
    "MAXIMUM_DEPTH",
    "UnparsableDocument",
    "dump_declaration",
    "load_safe_yaml_document",
    "load_selection_only_document",
    "looks_like_python_path",
    "require_selection_only_document",
    "rewrite_in_declaration_dialect",
]
