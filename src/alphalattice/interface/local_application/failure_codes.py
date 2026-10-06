"""What a failure may say to its caller: a typed product code, never exception text.

Owners raise codes -- dotted names (`research_experiment.input_not_admitted`) or enum names
(`TASK_EXECUTION_INTERRUPTED`) -- optionally suffixed after `:` with identifiers and bounds
(`:10<100`, `:2 of 1`). Anything else -- an exception message, a path, a quoted key, a
document -- is not served or persisted: its name has a space, or it has a quote or a path
separator, which a code never has. The Web handler, the Agent bridge, the activity ledger and the
recovery view say a typed code instead, and the Host's log keeps the detail under an
incident that code carries (binding plan, C1 rule 2): a caller learns that something failed
and where to look, never the Host's internals.
"""

from __future__ import annotations

import os
import re
import subprocess
from hashlib import sha256
from typing import Final

FAILURE_DETAIL_WITHHELD = "activity.failure_detail_withheld"
"""Served or persisted in place of any failure text that is not a typed product code."""

_FAILURE_CODE = re.compile(
    r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*(?::[A-Za-z0-9_.,:<>=+\- ]{1,160})?$"
)
_FIELD = re.compile(r"^[A-Za-z0-9_]{1,64}$")
_FAILURE_FIELDS = frozenset({"failure_code", "refused", "latest_failure_code"})


def safe_failure_code(value: object) -> str | None:
    """The value itself when it is a typed product failure code, else nothing."""
    if isinstance(value, str) and len(value) <= 200 and _FAILURE_CODE.match(value):
        return value
    return None


def owner_failure_code(error: BaseException) -> str | None:
    """Read an owner's code independently of its exception's inheritance.

    Args:
        error: A raised refusal or an unexpected failure.

    Returns:
        A bounded explicit code (``failure_code``, ``failure.code`` or ``code``), or
        the dotted/enum code a string-code owner raises. Prose, paths, HTTP integers
        and unrelated exception arguments are never treated as a refusal.
    """

    def admitted(value: object) -> str | None:
        code = safe_failure_code(value)
        if code is not None:
            return code
        # Some owners append diagnostic prose to a named refusal. Keep its dotted
        # identity, withholding an inadmissible suffix rather than serving that prose.
        if isinstance(value, str):
            head, separator, _detail = value.partition(":")
            if separator and "." in head:
                return safe_failure_code(head)
        return None

    failure = getattr(error, "failure", None)
    for value in (
        getattr(error, "failure_code", None),
        getattr(failure, "code", None),
        getattr(error, "code", None),
    ):
        code = admitted(value)
        if code is not None:
            return code
    # A code raised as text is one argument, not a formatted multi-argument failure.
    if len(error.args) == 1:
        code = admitted(error.args[0])
        if code is not None:
            name = code.partition(":")[0]
            if "." in name or name.isupper():
                return code
    return None


def setup_failure(error: Exception) -> dict[str, object]:
    """Word a setup failure without serving exception messages, URLs or headers.

    Args:
        error: The failed setup's exception, including a bounded chain of causes.

    Returns:
        The owner's code when present, otherwise a typed, sanitized cause and door words.
        The setup entry supplies its own input context and lawful continuation.
    """
    from .cli_contract import refusal_words

    causes: list[dict[str, object]] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = owner_failure_code(current)
        if code is not None:
            cause: dict[str, object] = {"kind": "PRODUCT", "code": code}
        else:
            name = type(current).__name__
            response = getattr(current, "response", None)
            status = getattr(response, "status_code", None)
            status = status if type(status) is int else getattr(current, "code", None)
            if type(status) is int and 100 <= status <= 599:
                code = f"setup.http_failed:{status}"
                cause = {"kind": "HTTP", "status": status}
            elif isinstance(current, subprocess.CalledProcessError):
                code = f"setup.command_failed:{current.returncode}"
                cause = {"kind": "COMMAND", "exit_code": current.returncode}
            elif isinstance(current, (ConnectionError, TimeoutError)) or any(
                cls.__name__ in {"RequestError", "URLError"} for cls in type(current).__mro__
            ):
                code = f"setup.network_failed:{name}"
                cause = {"kind": "NETWORK", "exception": name}
            elif isinstance(current, OSError):
                code = f"setup.file_unavailable:{name}"
                cause = {"kind": "OS", "exception": name}
                if current.errno is not None:
                    cause.update(errno=current.errno, message=os.strerror(current.errno))
                if getattr(current, "winerror", None) is not None:
                    cause["winerror"] = current.winerror
            elif isinstance(current, (ValueError, KeyError, TypeError)):
                code = f"setup.input_invalid:{name}"
                cause = {"kind": "INPUT", "exception": name}
            else:
                code = f"setup.unexpected_failure:{name}"
                cause = {"kind": "UNEXPECTED", "exception": name}
        causes.append({**cause, "failure_code": code})
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    code = str(causes[0]["failure_code"])
    words = refusal_words(code) or refusal_words(f"setup.owner_refused:{code}")
    return {"status": "REFUSED", "failure_code": code, **words, "causes": causes}


def public_failure(error: BaseException, fallback: str) -> str:
    """The code a caller may see for `error`, the same for the same failure at every entry.

    The owner's own code when it raised one. A request that fails validation names each
    field and the owner's code for it (`fallback:top_k=TOP_K_OUTSIDE_ADMITTED_RANGE`), never
    the value. Anything else is `fallback:<exception class>:<incident>`, the incident a digest
    of the failure, with the detail logged under it.
    """
    # The owners' models' failure; imported here, so the client's transport never loads them.
    from pydantic import ValidationError

    code = owner_failure_code(error)
    if code is not None:
        return code
    if isinstance(error, ValueError):
        code = safe_failure_code(str(error))
        if code is not None:
            return code
    if isinstance(error, ValidationError):
        fields = []
        for item in error.errors():
            location = ".".join(str(part) for part in item.get("loc", ()))
            cause = (item.get("ctx") or {}).get("error")
            reason = (
                owner_failure_code(cause)
                if isinstance(cause, BaseException)
                else safe_failure_code(str(cause))
                if cause is not None
                else None
            )
            fields.append(
                f"{location if all(_FIELD.match(p) for p in location.split('.')) else 'field'}"
                f"={reason or item.get('type', 'invalid')}"
            )
        # As many fields as the code holds, the rest counted: a document broken in many places
        # still refuses by its fields, never as a fingerprint (V449).
        for count in range(len(fields), 0, -1):
            rest = len(fields) - count
            shown = ",".join(fields[:count]) + (f",+{rest}" if rest else "")
            code = safe_failure_code(f"{fallback}:{shown}")
            if code is not None:
                return code
    name = type(error).__name__
    return _withheld(f"{name}: {error}", prefix=f"{fallback}:{name}")


_UNTYPED: Final = re.compile(r"^[a-z0-9_.]+:[A-Z][A-Za-z0-9_]*:[0-9a-f]{8}$")
"""A code `public_failure` gives an exception without a product code: `fallback:Type:digest`."""


def untyped_failure(code: str) -> bool:
    """Whether a code is an exception that reached an answer without a product code (V449).

    A crash, or an owner's failure that names no rule, which a caller cannot act on.

    Args:
        code: A failure code as answered.

    Returns:
        True for `fallback:<exception class>:<digest>`.
    """
    return bool(_UNTYPED.match(code))


def located_failure(error: BaseException, fallback: str) -> dict[str, object]:
    """The code an answer gives for `error`, and each field a contract refused (V248).

    One shape at every operation: a document that fails its contract answers the owner's code
    (`fallback`), its fields by location (`fields`, each a path), the reason for each by its
    dotted path (`reasons`: the owner's code where its validator raised one, else the
    contract's error type) and the contract's words (`message`), never a value. Any other
    failure answers the code `public_failure` gives.
    """
    from pydantic import ValidationError

    if not isinstance(error, ValidationError):
        return {"failure_code": public_failure(error, fallback)}
    from alphalattice.interface.local_application.cli_contract import refusal_words

    fields: list[list[str]] = []
    reasons: dict[str, str] = {}
    words: list[str] = []
    items = error.errors(include_input=False, include_url=False)
    located = [tuple(str(part) for part in item.get("loc", ())) for item in items]
    for item, place in zip(items, located, strict=True):
        kind = str(item.get("type", "invalid"))
        # A list whose item failed counts only the items that passed: its "too short" is that
        # failure again, never a reason of its own (V453).
        if kind == "too_short" and any(
            len(other) > len(place) and other[: len(place)] == place for other in located
        ):
            continue
        location = list(place)
        cause = (item.get("ctx") or {}).get("error")
        code = (
            owner_failure_code(cause)
            if isinstance(cause, BaseException)
            else safe_failure_code(str(cause))
            if cause is not None
            else None
        )
        # A validator's own text is the owner's code or withheld: it may carry the value.
        reason = code or kind
        fields.append(location)
        reasons[".".join(location)] = reason
        said = reason if kind == "value_error" else str(item.get("msg", kind))
        # An owner's code says what its rule expects, in its own words (V453).
        expected = refusal_words(code).get("detail") if code is not None else None
        words.append(f"{'.'.join(location)}: {said}" + (f" ({expected})" if expected else ""))
    return {
        "failure_code": fallback,
        "fields": fields,
        "reasons": reasons,
        "message": "; ".join(words)[:600],
    }


def typed_failures(body: dict[str, object]) -> dict[str, object]:
    """The answer with every failure field a typed code, at any depth.

    A field that should hold a code (`failure_code`, `refused`, `latest_failure_code`) and
    holds anything else -- an exception message, a path -- is `FAILURE_DETAIL_WITHHELD`, as
    the activity ledger persists it, and the text goes to the Host's log. One scrub at the
    operation boundary, so no site that answers with `str(error)` can hand its text to the
    Web, the CLI or an Agent.
    Unchanged parts are returned as they were, not copied.
    """

    def walk(value: object) -> object:
        if isinstance(value, dict):
            changed: dict[object, object] | None = None
            for key, item in value.items():
                new: object
                if key in _FAILURE_FIELDS and isinstance(item, str) and item:
                    new = item if safe_failure_code(item) else _withheld(item, incident=False)
                else:
                    new = walk(item)
                if new is not item:
                    changed = dict(value) if changed is None else changed
                    changed[key] = new
            return value if changed is None else changed
        if isinstance(value, list):
            items = [walk(item) for item in value]
            return value if all(a is b for a, b in zip(items, value, strict=True)) else items
        return value

    result = walk(body)
    assert isinstance(result, dict)
    return result


def _withheld(text: str, *, prefix: str = FAILURE_DETAIL_WITHHELD, incident: bool = True) -> str:
    """`prefix:<incident>`, the incident a digest of the text, which the Host's log keeps."""
    import logging  # only a withheld failure is logged; a call's imports are its cost (W12, V29)

    digest = sha256(text.encode("utf-8", "replace")).hexdigest()[:8]
    logging.getLogger("alphalattice.local_web").error("%s:%s %s", prefix, digest, text[:2000])
    return f"{prefix}:{digest}" if incident else prefix


__all__ = [
    "FAILURE_DETAIL_WITHHELD",
    "located_failure",
    "public_failure",
    "safe_failure_code",
    "setup_failure",
    "typed_failures",
    "untyped_failure",
]
