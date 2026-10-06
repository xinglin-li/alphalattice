"""Load one frozen current-universe manifest from Front Desk eligibility."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid5

import yaml
from pydantic import BaseModel, ConfigDict, Field

from alphalattice.foundation.market_data_ops.sources.universe import (
    ORIGINAL_RESEARCH_WHITELIST_STANDARD,
    CurrentUniverseBootstrap,
    membership_fingerprint,
)
from alphalattice.protocols.research_authoring.selection import load_safe_yaml_document

_LISTING_NAMESPACE = UUID("1d8862b4-a419-4d1a-999d-cf70583f9957")
CURRENT_ACTIVE_SURVIVORS = "CURRENT_ACTIVE_SURVIVORS"
CURRENT_SURVIVOR_COMPOSITE = "CURRENT_SURVIVOR_COMPOSITE"
CURRENT_UNIVERSE_RESEARCH_ONLY = "CURRENT_UNIVERSE_RESEARCH_ONLY"
CURRENT_UNIVERSE_COMPONENTS = ("SP500", "NASDAQ100", "DJIA")


@dataclass(frozen=True)
class MarketProfile:
    """Carry the qualified market, provider and session conventions of a universe manifest.

    Attributes:
        market_profile_id: Stable configuration handle used by workspace tasks.
        display_name: Human-readable market profile name.
        market: Market code governing listing identity and acquisition rules.
        currency: Currency in which source prices are expressed.
        calendar_id: Exchange calendar governing session schedules.
        provider: Named source provider, without a callable path.
        daily_price_basis: Declared provider daily-price adjustment convention.
        manifest_as_of: Recorded market clock for the frozen manifest.
        data_validity_class: Research scope admitted for the source data.
    """

    market_profile_id: str
    display_name: str
    market: str
    currency: str
    calendar_id: str
    provider: str
    daily_price_basis: str
    manifest_as_of: date
    data_validity_class: str


@dataclass(frozen=True)
class ManifestListing:
    """Link an admitted workspace listing to its display, exchange and provider handles.

    Attributes:
        listing_id: Opaque stable listing identity used by workspace evidence.
        symbol: Normalized display symbol retained by the manifest.
        mic: Exchange or acquisition-market identifier.
        provider_symbol: Source-provider handle used for data reads.
        issuer_external_id: Optional external issuer handle for evidence joins.
    """

    listing_id: str
    symbol: str
    mic: str
    provider_symbol: str
    issuer_external_id: str | None = None


@dataclass(frozen=True)
class UniverseManifest:
    """Freeze admitted market membership and its research qualification commitments.

    The effective listing set is separate from the source membership fingerprint.
    Derived manifests retain satisfied qualification obligations; current-survivor
    research does not grant point-in-time historical validity.

    Attributes:
        manifest_id: Stable handle for this qualified or acquisition manifest.
        profile: Market/provider/session conventions carried by the snapshot.
        listings: Ordered admitted workspace listing records.
        revision_sha256: Recorded content identity of the manifest declaration.
        universe_membership_basis: Declared source-membership basis.
        is_point_in_time_historical: Whether historical membership is PIT-qualified.
        membership_fingerprint: Source candidate-set identity inherited by derivations.
        qualification_policy_hash: Identity of satisfied obligations, or a legacy opaque policy.
        universe_policy_type: Declared membership composition policy.
        universe_components: Source index components used by the composition.
        survivorship_bias_warning: Whether the declared membership carries survivor bias.
        research_use_class: Admitted research scope of the snapshot.
        qualification_obligations: Sorted unique obligations satisfied by this membership.
    """

    manifest_id: str
    profile: MarketProfile
    listings: tuple[ManifestListing, ...]
    revision_sha256: str
    universe_membership_basis: str = CURRENT_ACTIVE_SURVIVORS
    is_point_in_time_historical: bool = False
    membership_fingerprint: str | None = None
    qualification_policy_hash: str | None = None
    universe_policy_type: str = CURRENT_SURVIVOR_COMPOSITE
    universe_components: tuple[str, ...] = CURRENT_UNIVERSE_COMPONENTS
    survivorship_bias_warning: bool = True
    research_use_class: str = CURRENT_UNIVERSE_RESEARCH_ONLY
    qualification_obligations: tuple[str, ...] = ()
    """The obligations this membership satisfied, one entry each, sorted.

    ``qualification_policy_hash`` is their identity for a manifest derived
    with them (see ``qualification_policy_identity``); a manifest recorded
    before obligations were named carries only the hash, and a derivation
    from it keeps that hash as one opaque obligation.
    """

    def listing_for_symbol(self, symbol: str) -> ManifestListing | None:
        """Find an admitted listing by normalized display symbol.

        Args:
            symbol: Display symbol to normalize and match.

        Returns:
            Matching listing, or ``None`` when absent.

        """
        normalized = symbol.strip().upper()
        return next((item for item in self.listings if item.symbol == normalized), None)

    @property
    def effective_membership_hash(self) -> str:
        """The identity of the listings this manifest actually admits.

        Not ``membership_fingerprint``, which names the source candidate set a
        manifest was derived from and is inherited by every child of that
        candidate set whatever subset the child admits.
        """
        return _canonical_hash(
            {
                "kind": "UniverseMembership",
                "listing_ids": sorted(item.listing_id for item in self.listings),
            }
        )

    def obligations_for_derivation(self) -> tuple[str, ...]:
        """Return the satisfied obligations inherited by a derived manifest."""
        if self.qualification_obligations:
            return self.qualification_obligations
        if self.qualification_policy_hash == default_qualification_policy_hash():
            return (history_quality_obligation(),)
        if self.qualification_policy_hash:
            return (qualification_obligation("legacy_policy", self.qualification_policy_hash),)
        return ()


class MarketProfileDocument(BaseModel):  # type: ignore[misc]
    """A market profile as its YAML declares it: the keys a profile holds, typed (V275).

    An unknown key or a wrong type is refused where the profile is read, before any manifest
    carries it: a `provider` is a provider's name, never a callable path.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    profile_schema: Literal["alphalattice.playpen.market-profile"] = Field(alias="schema")
    """The profile's schema name."""
    market_profile_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    """The profile's stable identity."""
    display_name: str = Field(min_length=1)
    """Its name for a person."""
    market: str = Field(pattern=r"^[A-Z]{2}$")
    """The market, as its two-letter code."""
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    """The currency prices are in."""
    calendar_id: str = Field(pattern=r"^[A-Z_]+$")
    """The exchange calendar sessions come from."""
    provider: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    """The data provider, by its name."""
    daily_price_basis: str = Field(pattern=r"^[a-z_]+$")
    """What the provider's daily prices already hold (`split_adjusted`)."""
    data_validity_class: str = Field(pattern=r"^[A-Z_]+$")
    """What research the data may serve."""
    manifest_as_of: date | None = None
    """The day a fixed allowlist's manifest is as of."""
    eligibility_source: str | None = Field(default=None, min_length=1)
    """The capability file a fixed allowlist is read from, relative to the profile."""


def read_market_profile(path: Path) -> MarketProfileDocument:
    """The market profile at `path`, read by the one declaration loader and its contract."""
    try:
        return MarketProfileDocument.model_validate(
            load_safe_yaml_document(path.read_text(encoding="utf-8"))
        )
    except (ValueError, yaml.YAMLError) as error:
        raise ValueError("market_data.market_profile_invalid") from error


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def default_qualification_policy_hash() -> str:
    """Stable identity of the current-universe history-quality policy."""
    return _canonical_hash(asdict(ORIGINAL_RESEARCH_WHITELIST_STANDARD))


def qualification_obligation(kind: str, identity: str) -> str:
    """One named obligation a qualified membership satisfied."""
    if not kind or ":" in kind or not identity:
        raise ValueError("qualification obligation requires a kind and an identity")
    return f"{kind}:{identity}"


def history_quality_obligation() -> str:
    """Return the onboarding obligation for each research manifest."""
    return qualification_obligation("history_quality", default_qualification_policy_hash())


def qualification_policy_identity(obligations: Iterable[str]) -> str:
    """Hash satisfied obligations without regard to order or repetition.

    One spelling for every derivation owner: a child that re-satisfies an
    obligation its parent already carried keeps its parent's identity, and a
    child that adds one (the Feature Input policy, a Sector requirement) gets
    the identity of the larger set. Ancestry lives in the admission ledger.
    """
    ordered = tuple(sorted(set(obligations)))
    if not ordered:
        raise ValueError("qualification policy identity requires at least one obligation")
    return _canonical_hash({"kind": "QualificationObligations", "obligations": list(ordered)})


def _market_profile(config: MarketProfileDocument, *, manifest_as_of: date) -> MarketProfile:
    return MarketProfile(
        market_profile_id=config.market_profile_id,
        display_name=config.display_name,
        market=config.market,
        currency=config.currency,
        calendar_id=config.calendar_id,
        provider=config.provider,
        daily_price_basis=config.daily_price_basis,
        manifest_as_of=manifest_as_of,
        data_validity_class=config.data_validity_class,
    )


def current_index_profile_id(profile_path: Path) -> str:
    """Read only the stable profile identity needed to locate a resumable task."""
    return read_market_profile(profile_path).market_profile_id


def build_current_index_acquisition_manifest(
    profile_path: Path,
    bootstrap: CurrentUniverseBootstrap,
) -> UniverseManifest:
    """Bind one discovered current-index union to opaque workspace listings.

    This is an acquisition manifest, not the final research whitelist. It
    bounds the initial data-maintenance task while keeping every source label
    in the separate candidate-manifest provenance record.
    """
    config = read_market_profile(profile_path)
    profile = _market_profile(
        config,
        manifest_as_of=bootstrap.source_manifest.as_of_timestamp.date(),
    )
    if (
        profile.market != "US"
        or profile.data_validity_class != bootstrap.standard.data_validity_class
    ):
        raise ValueError("current-index acquisition profile does not match the bootstrap standard")
    listings = tuple(
        ManifestListing(
            listing_id=str(
                uuid5(_LISTING_NAMESPACE, f"{profile.market_profile_id}:{candidate.symbol}")
            ),
            symbol=candidate.symbol,
            mic="US_INDEX_CANDIDATE",
            provider_symbol=candidate.provider_symbol,
        )
        for candidate in bootstrap.source_manifest.candidates
    )
    content = {
        "kind": "current-index-acquisition",
        "candidate_manifest_hash": bootstrap.source_manifest.content_hash,
        "profile": {
            **profile.__dict__,
            "manifest_as_of": profile.manifest_as_of.isoformat(),
        },
        "listings": [item.__dict__ for item in listings],
        "universe_membership_basis": CURRENT_ACTIVE_SURVIVORS,
        "is_point_in_time_historical": False,
        "universe_policy_type": CURRENT_SURVIVOR_COMPOSITE,
        "universe_components": CURRENT_UNIVERSE_COMPONENTS,
        "survivorship_bias_warning": True,
        "research_use_class": CURRENT_UNIVERSE_RESEARCH_ONLY,
    }
    revision = _canonical_hash(content)
    return UniverseManifest(
        manifest_id=f"{profile.market_profile_id}:acquisition:{revision[:16]}",
        profile=profile,
        listings=listings,
        revision_sha256=revision,
        universe_membership_basis=CURRENT_ACTIVE_SURVIVORS,
        is_point_in_time_historical=False,
        membership_fingerprint=membership_fingerprint(bootstrap.source_manifest),
        qualification_policy_hash=default_qualification_policy_hash(),
    )


def build_quality_filtered_research_manifest(
    acquisition_manifest: UniverseManifest,
    *,
    eligible_listing_ids: tuple[str, ...],
    quality_admission_hash: str | None = None,
    qualification_policy_hash: str | None = None,
    qualification_obligations: Iterable[str] | None = None,
) -> UniverseManifest:
    """Freeze the quality-qualified subset without changing listing identity.

    ``qualification_obligations`` names what the subset satisfied and seals
    their identity as the policy hash; the parent's obligations are kept.
    ``qualification_policy_hash`` alone is the recorded form of manifests
    that predate named obligations and stays accepted for their readback.
    """
    known = {listing.listing_id for listing in acquisition_manifest.listings}
    requested = tuple(sorted(set(eligible_listing_ids)))
    if not requested or not set(requested).issubset(known):
        raise ValueError("quality-filtered manifest has an invalid eligible listing scope")
    listings = tuple(
        listing for listing in acquisition_manifest.listings if listing.listing_id in set(requested)
    )
    obligations: tuple[str, ...] = ()
    if qualification_obligations is not None:
        if qualification_policy_hash is not None:
            raise ValueError("pass qualification obligations or a policy hash, not both")
        obligations = tuple(
            sorted({*acquisition_manifest.obligations_for_derivation(), *qualification_obligations})
        )
        policy_hash = qualification_policy_identity(obligations)
    else:
        policy_hash = qualification_policy_hash or default_qualification_policy_hash()
    membership_identity = acquisition_manifest.membership_fingerprint or _canonical_hash(
        {
            "market_profile_id": acquisition_manifest.profile.market_profile_id,
            "listing_ids": sorted(item.listing_id for item in acquisition_manifest.listings),
        }
    )
    content = {
        "kind": "quality-filtered-current-index-research",
        "candidate_membership_fingerprint": membership_identity,
        "qualification_policy_hash": policy_hash,
        "profile": {
            "market_profile_id": acquisition_manifest.profile.market_profile_id,
            "market": acquisition_manifest.profile.market,
            "currency": acquisition_manifest.profile.currency,
            "calendar_id": acquisition_manifest.profile.calendar_id,
            "provider": acquisition_manifest.profile.provider,
            "daily_price_basis": acquisition_manifest.profile.daily_price_basis,
            "data_validity_class": acquisition_manifest.profile.data_validity_class,
        },
        "listing_ids": requested,
        "universe_membership_basis": acquisition_manifest.universe_membership_basis,
        "is_point_in_time_historical": acquisition_manifest.is_point_in_time_historical,
        "universe_policy_type": acquisition_manifest.universe_policy_type,
        "universe_components": acquisition_manifest.universe_components,
        "survivorship_bias_warning": acquisition_manifest.survivorship_bias_warning,
        "research_use_class": acquisition_manifest.research_use_class,
    }
    revision = _canonical_hash(content)
    return UniverseManifest(
        manifest_id=(
            f"{acquisition_manifest.profile.market_profile_id}:research-whitelist:{revision[:16]}"
        ),
        profile=replace(acquisition_manifest.profile),
        listings=listings,
        revision_sha256=revision,
        universe_membership_basis=acquisition_manifest.universe_membership_basis,
        is_point_in_time_historical=acquisition_manifest.is_point_in_time_historical,
        membership_fingerprint=membership_identity,
        qualification_policy_hash=policy_hash,
        universe_policy_type=acquisition_manifest.universe_policy_type,
        universe_components=acquisition_manifest.universe_components,
        survivorship_bias_warning=acquisition_manifest.survivorship_bias_warning,
        research_use_class=acquisition_manifest.research_use_class,
        qualification_obligations=obligations,
    )
