"""Which published causal execution outcome answers for a listing set (O3).

Selection is by matching identity, never by recency: two eligible snapshots are ambiguous
and reported, and a snapshot a person named is admitted only in the store's two spellings
and only when it answers for the listing set. Moved from the Host's research authoring,
which asks it for the outcome a development run binds. It sits outside `execution/`: it
decides no outcome value, and a study binds the outcome it selects by content (ID8), so it is
no part of the Alpha array closure that package's modules all join.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from alphalattice.foundation.causal_outcomes.execution.contracts import (
    CausalExecutionOutcomeManifest,
    DevelopmentOnlyExecutionOutcomeManifest,
)
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
    DevelopmentOnlyExecutionOutcomeReader,
)
from alphalattice.protocols.research_authoring.contracts import AuthoringError

_HASH_CHARACTERS = frozenset("0123456789abcdef")


def selected_snapshot_hash(handle: str, *, manifest_uri: Callable[[str], str]) -> str:
    """Admit a content-addressed outcome handle, in exactly two spellings.

    A handle *selects* among published evidence; it never supplies authority.
    Everything the Host would have checked about the resolved snapshot is still
    checked afterwards -- its listing set, its method seal, the recipe that seal
    names -- so naming one can only ever choose between candidates that would
    each independently pass. What it cannot do is smuggle a hash the Host has
    not re-derived from the artifact it names.

    Either the bare 64-hex snapshot hash or the store's own URI spelling of that
    manifest. A URI in any other shape is refused rather than parsed leniently:
    a second spelling of where artifacts live is a second definition of it.

    Args:
        handle: The named snapshot, its bare hash or its manifest URI.
        manifest_uri: The store's URI spelling for a snapshot hash.

    Returns:
        The snapshot hash.

    Raises:
        AuthoringError: The handle is neither spelling.
    """
    value = handle.rsplit("/", 1)[-1] if "/" in handle else handle
    if len(value) != 64 or not set(value).issubset(_HASH_CHARACTERS):
        raise AuthoringError("research_authoring.execution_outcome_handle_invalid")
    if "/" in handle and handle != manifest_uri(value):
        raise AuthoringError("research_authoring.execution_outcome_handle_invalid")
    return value


def eligible_published_execution_outcomes(
    *, artifact_root: Path, listing_set_hash: str
) -> tuple[CausalExecutionOutcomeManifest, ...]:
    """Every published outcome that answers for this listing set, in hash order.

    Exposed so an ambiguity can be *reported* with the handles that would
    resolve it. Enumerating candidates is not selecting one, and the order is
    for a reproducible message rather than a preference.

    Args:
        artifact_root: The artifacts root the outcomes are published under.
        listing_set_hash: The listing set they must answer for.

    Returns:
        The matching manifests, in hash order; none when nothing is published.
    """
    manifests = Path(artifact_root) / "data-operations" / "execution-outcomes" / "manifests"
    if not manifests.is_dir():
        return ()
    matched = []
    for path in sorted(manifests.glob("*.json")):
        manifest = CausalExecutionOutcomeManifest.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
        if manifest.listing_set_hash == listing_set_hash:
            matched.append(manifest)
    return tuple(matched)


def _resolve_execution_outcome[
    OutcomeManifest: (CausalExecutionOutcomeManifest, DevelopmentOnlyExecutionOutcomeManifest)
](
    *,
    listing_set_hash: str,
    snapshot_handle: str | None,
    manifest_uri: Callable[[str], str],
    load_manifest: Callable[[str], OutcomeManifest],
    eligible: Callable[[], tuple[OutcomeManifest, ...]],
) -> OutcomeManifest:
    if snapshot_handle is not None:
        selected = selected_snapshot_hash(snapshot_handle, manifest_uri=manifest_uri)
        try:
            manifest = load_manifest(selected)
        except FileNotFoundError as error:
            raise AuthoringError("research_authoring.execution_outcome_unavailable") from error
        if manifest.listing_set_hash != listing_set_hash:
            raise AuthoringError("research_authoring.execution_outcome_listing_set_mismatch")
        return manifest
    matched = eligible()
    if not matched:
        raise AuthoringError("research_authoring.execution_outcome_unavailable")
    if len(matched) > 1:
        raise AuthoringError("research_authoring.execution_outcome_ambiguous")
    return matched[0]


def resolve_published_execution_outcome(
    *, artifact_root: Path, listing_set_hash: str, snapshot_handle: str | None = None
) -> CausalExecutionOutcomeManifest:
    """Select the published causal outcome that answers for this listing set.

    Selection is by *matching identity*, never by recency. Reading a directory and
    taking the newest manifest succeeds on the machine that has that directory and
    means something different everywhere else, which is precisely the class of
    silent binding this milestone is closing.

    Two published outcomes over one listing set is genuinely ambiguous rather than
    a case to resolve by preference, so it is reported -- unless the caller named
    one explicitly, which is a selection a person made rather than a rule this
    code invented. A named snapshot still has to answer for this listing set.

    Args:
        artifact_root: The artifacts root the outcomes are published under.
        listing_set_hash: The listing set the outcome must answer for.
        snapshot_handle: A snapshot a person named, when one was.

    Returns:
        The selected manifest.

    Raises:
        AuthoringError: None answers, several do and none was named, or the named one is
            unavailable, misspelled or answers for another listing set.
    """
    reader = CausalExecutionOutcomeDevelopmentReader(Path(artifact_root))
    return _resolve_execution_outcome(
        listing_set_hash=listing_set_hash,
        snapshot_handle=snapshot_handle,
        manifest_uri=reader.manifest_uri,
        load_manifest=reader.load_manifest,
        eligible=lambda: eligible_published_execution_outcomes(
            artifact_root=artifact_root, listing_set_hash=listing_set_hash
        ),
    )


def eligible_development_only_execution_outcomes(
    *, artifact_root: Path, listing_set_hash: str
) -> tuple[DevelopmentOnlyExecutionOutcomeManifest, ...]:
    """Every development-only snapshot that answers for this listing set.

    Args:
        artifact_root: The artifacts root the snapshots are published under.
        listing_set_hash: The listing set they must answer for.

    Returns:
        The matching manifests; none when nothing is published.
    """
    reader = DevelopmentOnlyExecutionOutcomeReader(Path(artifact_root))
    try:
        published = reader.artifacts.development_only_manifests()
    except FileNotFoundError:
        return ()
    return tuple(
        manifest for manifest in published if manifest.listing_set_hash == listing_set_hash
    )


def resolve_development_only_execution_outcome(
    *, artifact_root: Path, listing_set_hash: str, snapshot_handle: str | None = None
) -> DevelopmentOnlyExecutionOutcomeManifest:
    """Select the development-only snapshot that answers for this listing set.

    Same selection discipline as the frozen resolver above: matching identity,
    never recency, two eligible snapshots reported as ambiguous rather than
    resolved by preference, and an explicitly named one admitted only after it
    is proven to answer for this listing set.

    Args:
        artifact_root: The artifacts root the snapshots are published under.
        listing_set_hash: The listing set the snapshot must answer for.
        snapshot_handle: A snapshot a person named, when one was.

    Returns:
        The selected manifest.

    Raises:
        AuthoringError: None answers, several do and none was named, or the named one is
            unavailable, misspelled or answers for another listing set.
    """
    reader = DevelopmentOnlyExecutionOutcomeReader(Path(artifact_root))
    return _resolve_execution_outcome(
        listing_set_hash=listing_set_hash,
        snapshot_handle=snapshot_handle,
        manifest_uri=reader.manifest_uri,
        load_manifest=reader.load_manifest,
        eligible=lambda: eligible_development_only_execution_outcomes(
            artifact_root=artifact_root, listing_set_hash=listing_set_hash
        ),
    )


__all__ = [
    "eligible_development_only_execution_outcomes",
    "eligible_published_execution_outcomes",
    "resolve_development_only_execution_outcome",
    "resolve_published_execution_outcome",
    "selected_snapshot_hash",
]
