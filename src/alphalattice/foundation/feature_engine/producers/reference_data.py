"""Host-owned SPY and Yahoo-current-sector reference maintenance."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid5

from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.feature_engine.contracts import canonical_hash
from alphalattice.foundation.feature_engine.publication.sector_map_activation import (
    SectorRevisionMapActivationCoordinator,
)
from alphalattice.foundation.feature_engine.storage.repositories import FeatureStateRepository
from alphalattice.foundation.market_data_ops.sources.contracts import FailureEvidence
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    UniverseManifest,
)
from alphalattice.foundation.market_data_ops.sources.providers import (
    MarketDataProvider,
    ProviderFetchError,
    SectorReferenceProvider,
)
from alphalattice.foundation.market_data_ops.sources.sanitization import (
    CorruptedPayload,
    sanitize_payload,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import (
    ActionAuditScopeInsufficient,
    MarketDataRepository,
)

_REFERENCE_NAMESPACE = UUID("15bf0e17-4d4f-4a5e-8aae-026a01aa7f0d")
SPY_REFERENCE_ID = "SPY"
SECTOR_SOURCE = "YAHOO_CURRENT_SECTOR"
SECTOR_BUCKET_ID = "yahoo-current-sector"


@dataclass(frozen=True)
class MarketReference:
    """Name the isolated market-reference listing and its source manifest."""

    reference_id: str
    symbol: str
    listing_id: str
    manifest: UniverseManifest

    @classmethod
    def spy(cls, parent: UniverseManifest) -> MarketReference:
        """Derive the SPY reference manifest from the parent market profile."""
        listing_id = str(uuid5(_REFERENCE_NAMESPACE, f"{parent.profile.market_profile_id}:SPY"))
        listing = ManifestListing(
            listing_id=listing_id,
            symbol="SPY",
            mic="US_MARKET_REFERENCE",
            provider_symbol="SPY",
        )
        revision = canonical_hash(
            {
                "kind": "market-reference",
                "reference": SPY_REFERENCE_ID,
                "profile": parent.profile.market_profile_id,
                "provider": parent.profile.provider,
                "listing": listing.__dict__,
            }
        )
        manifest = UniverseManifest(
            manifest_id=f"{parent.profile.market_profile_id}:market-reference:SPY",
            profile=parent.profile,
            listings=(listing,),
            revision_sha256=revision,
            universe_membership_basis="MARKET_REFERENCE_ONLY",
            is_point_in_time_historical=False,
        )
        return cls(SPY_REFERENCE_ID, "SPY", listing_id, manifest)


@dataclass(frozen=True)
class ReferenceRefreshOutcome:
    """Report completion, deferral or refusal of a market-reference refresh."""

    status: str
    revision_hash: str | None
    failure_code: str | None = None
    retry_after_at: datetime | None = None


@dataclass
class MarketReferenceMaintainer:
    """Maintain SPY reference bars under the parent manifest's provider authority."""

    market_data: MarketDataRepository
    feature_state: FeatureStateRepository
    parent_manifest: UniverseManifest
    provider: MarketDataProvider
    retry_budget: int = 2

    def refresh(self, *, as_of_session: date, observed_at: datetime) -> ReferenceRefreshOutcome:
        """Maintain SPY using the same sanitizer/action-audit discipline as listings."""
        reference = MarketReference.spy(self.parent_manifest)
        self.market_data.bootstrap(reference.manifest)
        current = self.feature_state.verified_market_reference(
            reference.manifest,
            reference_id=SPY_REFERENCE_ID,
            requested_as_of=as_of_session,
        )
        if current is not None:
            if str(current["provider"]) != self.provider.name:
                return ReferenceRefreshOutcome(
                    "blocked", None, "market_reference.provider_identity_mismatch"
                )
            return ReferenceRefreshOutcome(
                "completed",
                str(current["revision_hash"]),
            )
        parent_range = self.market_data.manifest_raw_range(self.parent_manifest)
        if parent_range is None:
            return ReferenceRefreshOutcome("blocked", None, "feature.parent_raw_history_missing")
        start = parent_range[0]
        bars = self.market_data.raw_bars(reference.listing_id, through=as_of_session)
        full_backfill = not bars or bars[0].session_date > parent_range[0]
        if bars and not full_backfill:
            start = max(start, bars[-1].session_date - timedelta(days=45))
        payload = None
        hydration = None
        last_error: ProviderFetchError | None = None
        for _attempt in range(self.retry_budget):
            try:
                if hasattr(self.provider, "fetch_hydration"):
                    hydration = self.provider.fetch_hydration(
                        listing_id=reference.listing_id,
                        provider_symbol=reference.symbol,
                        start=start,
                        end=as_of_session,
                    )
                    payload = {reference.symbol: hydration.daily_rows}
                else:
                    payload = self.provider.fetch_daily(
                        (reference.symbol,), start=start, end=as_of_session
                    )
                break
            except ProviderFetchError as exc:
                last_error = exc
                if not exc.retryable:
                    break
        if payload is None:
            if last_error and last_error.code == "data.rate_limited":
                return ReferenceRefreshOutcome(
                    "deferred", None, last_error.code, observed_at + timedelta(minutes=5)
                )
            return ReferenceRefreshOutcome(
                "blocked", None, last_error.code if last_error else "data.provider_fetch_failed"
            )
        try:
            batch = sanitize_payload(
                reference.manifest, self.provider.name, payload, (reference.symbol,)
            )
            self.market_data.apply_validated_batch(
                reference.manifest,
                batch,
                ingestion_id=canonical_hash(
                    ["SPY", as_of_session.isoformat(), observed_at.isoformat(), "refresh"]
                ),
                observed_at=observed_at,
            )
            bars = self.market_data.raw_bars(reference.listing_id, through=as_of_session)
            if not bars:
                raise ValueError("SPY refresh persisted no bars")
            receipt = self.market_data.reusable_action_audit_receipt(
                reference.manifest,
                listing_id=reference.listing_id,
                provider=self.provider.name,
                requested_as_of=as_of_session,
                now=observed_at,
            )
            if receipt is None:
                scoped_sessions = {
                    item.session_date
                    for item in bars
                    if start <= item.session_date <= as_of_session
                }
                if (
                    hydration is not None
                    and {item.session_date for item in hydration.adjusted_closes} == scoped_sessions
                ):
                    actions = hydration.actions
                    adjusted = hydration.adjusted_closes
                else:
                    actions = self.provider.fetch_action_history(
                        listing_id=reference.listing_id,
                        provider_symbol=reference.symbol,
                        start=start,
                        end=as_of_session,
                    )
                    adjusted = self.provider.fetch_adjusted_close_history(
                        listing_id=reference.listing_id,
                        provider_symbol=reference.symbol,
                        start=start,
                        end=as_of_session,
                    )
                receipt, _audit_summary = self.market_data.complete_action_audit(
                    reference.manifest,
                    listing_id=reference.listing_id,
                    provider=self.provider.name,
                    observed_actions=actions,
                    observed_adjusted_closes=adjusted,
                    history_start=start,
                    history_end=as_of_session,
                    requested_as_of=as_of_session,
                    observed_at=observed_at,
                )
            revision = canonical_hash(
                {
                    "reference": SPY_REFERENCE_ID,
                    "raw_sessions": [(bar.session_date.isoformat(), bar.close) for bar in bars],
                    "action_receipt": receipt.receipt_hash,
                }
            )
            self.feature_state.upsert_market_reference(
                reference_id=SPY_REFERENCE_ID,
                listing_id=reference.listing_id,
                provider=self.provider.name,
                symbol=reference.symbol,
                revision_hash=revision,
                action_audit_receipt_hash=receipt.receipt_hash,
                latest_session=bars[-1].session_date,
                observed_at=observed_at,
            )
        except ActionAuditScopeInsufficient:
            self.market_data.record_failures(
                (
                    FailureEvidence(
                        reference.listing_id,
                        self.parent_manifest.profile.market_profile_id,
                        "market_reference.action_audit_scope_insufficient",
                        start,
                        as_of_session,
                        observed_at,
                    ),
                )
            )
            return ReferenceRefreshOutcome(
                "blocked", None, "market_reference.action_audit_scope_insufficient"
            )
        except (CorruptedPayload, ValueError):
            self.market_data.record_failures(
                (
                    FailureEvidence(
                        reference.listing_id,
                        self.parent_manifest.profile.market_profile_id,
                        "market_reference.invalid_payload",
                        start,
                        as_of_session,
                        observed_at,
                    ),
                )
            )
            return ReferenceRefreshOutcome("blocked", None, "market_reference.invalid_payload")
        except ProviderFetchError as exc:
            return ReferenceRefreshOutcome(
                "deferred" if exc.retryable else "blocked",
                None,
                exc.code,
                observed_at + timedelta(minutes=5) if exc.retryable else None,
            )
        return ReferenceRefreshOutcome("completed", revision)


@dataclass(frozen=True)
class SectorRefreshOutcome:
    """Report a staged sector refresh and its activation or failure receipt."""

    status: str
    sector_revision: str | None
    receipt_hash: str | None
    deferred_retry_id: str | None = None
    retry_after_at: datetime | None = None
    failure_code: str | None = None
    failed_listing_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SectorTransportPolicy:
    """Operational network policy; never part of sector truth identity."""

    max_workers: int = 1
    chunk_size: int = 20

    def __post_init__(self) -> None:
        """Reject transport settings outside the bounded worker and chunk policy."""
        if self.max_workers not in {1, 2}:
            raise ValueError("sector transport supports only one or two workers")
        if self.chunk_size != 20:
            raise ValueError("sector transport chunk size is fixed at twenty")

    @property
    def policy_hash(self) -> str:
        """Bind transport settings without making them sector-truth identity."""
        return canonical_hash(
            {
                "transport": "bounded-yfinance-current-sector",
                "max_workers": self.max_workers,
                "chunk_size": self.chunk_size,
                "provider_wide_failures": [
                    "data.rate_limited",
                    "data.provider_session_unstable",
                ],
                "same_turn_retry": False,
            }
        )


@dataclass
class SectorRefreshStager:
    """Acquire current-sector evidence without touching DuckDB.

    The JSON file is a private, run-resumable staging artifact (DA9, V268). It is never a
    research authority: only a complete, revalidated document may be committed atomically
    into the workspace DuckDB by :meth:`commit`. It binds the provider, the manifest and what
    it stages, never the transport: a staging deferred under two workers resumes under one,
    each chunk recording the workers that fetched it (PA2, PA3, V257).
    """

    manifest: UniverseManifest
    provider: SectorReferenceProvider
    artifact_root: Path
    staging_id: str
    activation_coordinator: SectorRevisionMapActivationCoordinator
    retry_budget: int = 2
    transport_policy: SectorTransportPolicy = field(default_factory=SectorTransportPolicy)

    @property
    def path(self) -> Path:
        """Return the private JSON artifact path for this staging identifier."""
        return self.artifact_root / f"{self.staging_id}.json"

    def acquire(self, *, observed_at: datetime) -> SectorRefreshOutcome:
        """Fetch and validate current-sector evidence into resumable JSON staging.

        Refuse a provider outside the active manifest and require a timezone-aware
        observation. The staged document is not yet sector truth.
        """
        if observed_at.tzinfo is None:
            raise ValueError("sector staging observed_at must be timezone-aware")
        if self.provider.name != self.manifest.profile.provider:
            raise ValueError("sector provider is not bound to the active manifest")
        payload = self._load_or_create(observed_at)
        if payload["status"] == "COMPLETED":
            return self._outcome(payload)
        if payload["status"] == "BLOCKED":
            return self._outcome(payload)
        if payload["status"] == "DEFERRED":
            retry_after = self._datetime(payload.get("retry_after_at"))
            if retry_after is not None and observed_at < retry_after:
                return self._outcome(payload)

        observations = {str(item["listing_id"]): item for item in payload.get("observations", [])}
        failures = {str(item["listing_id"]): item for item in payload.get("failures", [])}
        remaining = [
            listing
            for listing in self.manifest.listings
            if listing.listing_id not in observations and listing.listing_id not in failures
        ]
        workers = int(payload.get("next_workers") or self.transport_policy.max_workers)
        payload.update(
            status="RUNNING",
            failure_code=None,
            retry_after_at=None,
            deferred_retry_id=None,
            observed_workers=workers,
            next_workers=None,
        )
        self._write(payload)
        for offset in range(0, len(remaining), self.transport_policy.chunk_size):
            chunk = remaining[offset : offset + self.transport_policy.chunk_size]
            results = self._fetch_chunk(chunk, workers=workers)
            chunks = payload.setdefault("chunks", [])
            assert isinstance(chunks, list)
            chunks.append(
                {
                    "cursor_listing_id": chunk[-1].listing_id,
                    "listings": len(chunk),
                    "workers": workers,
                }
            )
            for listing, result in results:
                if isinstance(result, ProviderFetchError):
                    if result.code not in {
                        "data.rate_limited",
                        "data.provider_session_unstable",
                    }:
                        failures[listing.listing_id] = {
                            "listing_id": listing.listing_id,
                            "provider_symbol": listing.provider_symbol,
                            "failure_code": result.code,
                            "retryable": result.retryable,
                        }
                    continue
                if (
                    result.provider != self.provider.name
                    or result.provider_symbol != listing.provider_symbol
                ):
                    failures[listing.listing_id] = {
                        "listing_id": listing.listing_id,
                        "provider_symbol": listing.provider_symbol,
                        "failure_code": "sector.identity_mismatch",
                        "retryable": False,
                    }
                    continue
                observations[listing.listing_id] = {
                    "listing_id": listing.listing_id,
                    "provider": result.provider,
                    "provider_symbol": result.provider_symbol,
                    "sector_name": result.sector_name,
                    "sector_key": result.sector_key,
                    "payload_hash": result.payload_hash,
                    "evidence_hash": canonical_hash(
                        {
                            "source": SECTOR_SOURCE,
                            "payload": result.payload_hash,
                            "symbol": listing.provider_symbol,
                        }
                    ),
                }
            payload["observations"] = [observations[key] for key in sorted(observations)]
            payload["failures"] = [failures[key] for key in sorted(failures)]
            payload["cursor_listing_id"] = chunk[-1].listing_id
            self._write(payload)

            provider_wide = [
                error
                for _listing, error in results
                if isinstance(error, ProviderFetchError)
                and error.code in {"data.rate_limited", "data.provider_session_unstable"}
            ]
            if provider_wide:
                pending_ids = tuple(
                    item.listing_id
                    for item in remaining[offset:]
                    if item.listing_id not in observations
                )
                return self._defer(
                    payload,
                    error=provider_wide[0],
                    observed_at=observed_at,
                    workers=workers,
                    pending_ids=pending_ids,
                )
        payload.update(
            status="BLOCKED" if failures else "COMPLETED",
            failure_code="sector.partial_current_sector" if failures else None,
            retry_after_at=None,
            deferred_retry_id=None,
            cursor_listing_id=None,
            next_workers=None,
        )
        self._write(payload)
        return self._outcome(payload)

    def commit(
        self,
        *,
        store: FeatureStateRepository,
        mutation_gate: WorkspaceMutationGate,
        observed_at: datetime,
    ) -> SectorRefreshOutcome:
        """Import one validated staging artifact at the fan-in boundary."""
        payload = self._load()
        outcome = self._outcome(payload)
        if outcome.status != "completed":
            if outcome.status == "blocked" and outcome.failure_code == (
                "sector.partial_current_sector"
            ):
                # Preserve every verified success behind the DuckDB authority
                # boundary.  A host policy may then preflight a reduced
                # manifest without reopening Yahoo or trusting the staging file
                # directly.  Nothing is activated until that reduced scope is
                # complete and structurally admissible.
                mutation_gate.run(
                    store.stage_sector_observations,
                    manifest_revision=self.manifest.revision_sha256,
                    observations=tuple(payload["observations"]),
                    observed_at=observed_at,
                )
            failures = payload.get("failures", [])
            if (
                outcome.failure_code == "sector.partial_current_sector"
                and failures
                and all(
                    item["failure_code"] == "sector.missing_current_sector" for item in failures
                )
            ):
                prior = store.current_sector_state(self.manifest)
                if prior is not None:
                    # Retain the complete dated reference, not a partially fresh
                    # mix. The failed attempt remains staged and no new truth
                    # receipt, observation clock or revision is published.
                    outcome = replace(
                        outcome,
                        status="deferred",
                        sector_revision=prior.sector_revision,
                        failure_code="sector.refresh_unavailable_prior_reference_retained",
                    )
            mutation_gate.run(
                store.save_sector_progress,
                manifest_revision=self.manifest.revision_sha256,
                cursor_listing_id=(
                    str(payload["cursor_listing_id"]) if payload.get("cursor_listing_id") else None
                ),
                status=outcome.status.upper(),
                deferred_retry_id=outcome.deferred_retry_id,
                retry_after_at=outcome.retry_after_at,
                failure_code=outcome.failure_code,
                observed_workers=(
                    int(payload["observed_workers"])
                    if payload.get("observed_workers") is not None
                    else None
                ),
                next_workers=(
                    int(payload["next_workers"])
                    if payload.get("next_workers") is not None
                    else None
                ),
                transport_policy_hash=self.transport_policy.policy_hash,
                observed_at=observed_at,
            )
            return outcome
        revision, changed, receipt = self.activation_coordinator.activate(
            manifest=self.manifest,
            observations=tuple(payload["observations"]),
            observed_at=observed_at,
        )
        return SectorRefreshOutcome(
            "completed", revision, receipt, failure_code="sector.changed" if changed else None
        )

    def _load_or_create(self, observed_at: datetime) -> dict[str, object]:
        if self.path.exists():
            return self._load()
        payload: dict[str, object] = {
            "schema": "alphalattice-sector-refresh-staging",
            "staging_id": self.staging_id,
            "manifest_revision": self.manifest.revision_sha256,
            "provider": self.provider.name,
            "started_at": observed_at.astimezone(UTC).isoformat(),
            "status": "RUNNING",
            "cursor_listing_id": None,
            "deferred_retry_id": None,
            "retry_after_at": None,
            "failure_code": None,
            "observed_workers": self.transport_policy.max_workers,
            "next_workers": None,
            "chunks": [],
            "observations": [],
            "failures": [],
        }
        self._write(payload)
        return payload

    def _load(self) -> dict[str, object]:
        loaded = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("sector staging artifact must be a JSON mapping")
        content_hash = loaded.pop("content_hash", None)
        if content_hash != canonical_hash(loaded):
            raise ValueError("sector staging artifact content hash mismatch")
        expected = {
            "schema": "alphalattice-sector-refresh-staging",
            "staging_id": self.staging_id,
            "manifest_revision": self.manifest.revision_sha256,
            "provider": self.provider.name,
        }
        if any(loaded.get(key) != value for key, value in expected.items()):
            raise ValueError("sector staging artifact binding mismatch")
        chunks = loaded.get("chunks", [])
        if not isinstance(chunks, list) or not all(
            isinstance(item, dict) and type(item.get("workers")) is int for item in chunks
        ):
            raise ValueError("sector staging chunks must record their workers")
        listing_by_id = {item.listing_id: item for item in self.manifest.listings}
        allowed_ids = set(listing_by_id)
        observations = loaded.get("observations")
        failures = loaded.get("failures", [])
        if not isinstance(observations, list):
            raise ValueError("sector staging observations must be a list")
        if not all(isinstance(item, dict) for item in observations):
            raise ValueError("sector staging observation must be a mapping")
        if not isinstance(failures, list) or not all(isinstance(item, dict) for item in failures):
            raise ValueError("sector staging failures must be a list of mappings")
        observation_ids = {str(item.get("listing_id")) for item in observations}
        if len(observation_ids) != len(observations):
            raise ValueError("sector staging contains duplicate listings")
        if observation_ids - allowed_ids:
            raise ValueError("sector staging contains a listing outside the manifest")
        failure_ids = {str(item.get("listing_id")) for item in failures}
        if len(failure_ids) != len(failures) or failure_ids - allowed_ids:
            raise ValueError("sector staging failures contain invalid listing identity")
        if observation_ids & failure_ids:
            raise ValueError("sector staging listing cannot be both successful and failed")
        for item in observations:
            listing = listing_by_id[str(item["listing_id"])]
            if (
                item.get("provider") != self.provider.name
                or item.get("provider_symbol") != listing.provider_symbol
                or not str(item.get("sector_name") or "").strip()
                or not str(item.get("payload_hash") or "").strip()
                or not str(item.get("evidence_hash") or "").strip()
            ):
                raise ValueError("sector staging observation failed identity validation")
        if loaded.get("status") not in {"RUNNING", "DEFERRED", "BLOCKED", "COMPLETED"}:
            raise ValueError("sector staging status is invalid")
        if loaded.get("status") == "COMPLETED" and observation_ids != allowed_ids:
            raise ValueError("completed sector staging must cover the active manifest")
        return loaded

    def _write(self, payload: dict[str, object]) -> str:
        document = dict(payload)
        content_hash = canonical_hash(document)
        document["content_hash"] = content_hash
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(document, sort_keys=True, indent=1) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)
        return content_hash

    def _fetch_chunk(self, chunk, *, workers: int):
        if workers == 1:
            return [(listing, self._fetch_one(listing.provider_symbol)) for listing in chunk]
        by_future = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sector-fetch") as pool:
            for listing in chunk:
                by_future[pool.submit(self._fetch_one, listing.provider_symbol)] = listing
            completed = {
                listing.listing_id: (listing, future.result())
                for future, listing in by_future.items()
            }
        return [completed[listing.listing_id] for listing in chunk]

    def _fetch_one(self, provider_symbol: str):
        last_error: ProviderFetchError | None = None
        for _attempt in range(self.retry_budget):
            try:
                return self.provider.fetch_current_sector(provider_symbol=provider_symbol)
            except ProviderFetchError as exc:
                last_error = exc
                if not exc.retryable or exc.code in {
                    "data.rate_limited",
                    "data.provider_session_unstable",
                }:
                    break
        return last_error or ProviderFetchError(
            "sector.fetch_failed", "sector fetch produced no result", retryable=False
        )

    def _defer(
        self,
        payload: dict[str, object],
        *,
        error: ProviderFetchError,
        observed_at: datetime,
        workers: int,
        pending_ids: tuple[str, ...],
    ) -> SectorRefreshOutcome:
        retry_after = observed_at + timedelta(minutes=5)
        deferred_id = canonical_hash(
            {
                "staging_id": self.staging_id,
                "failure_code": error.code,
                "retry_after": retry_after.isoformat(),
                "pending_listing_set": canonical_hash(pending_ids),
                "transport_policy": self.transport_policy.policy_hash,
            }
        )
        payload.update(
            status="DEFERRED",
            failure_code=error.code,
            deferred_retry_id=deferred_id,
            retry_after_at=retry_after.astimezone(UTC).isoformat(),
            observed_workers=workers,
            next_workers=1,
        )
        self._write(payload)
        return self._outcome(payload)

    @staticmethod
    def _datetime(value: object) -> datetime | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            raise ValueError("sector staging timestamp must be timezone-aware")
        return parsed.astimezone(UTC)

    def _outcome(self, payload: dict[str, object]) -> SectorRefreshOutcome:
        return SectorRefreshOutcome(
            str(payload["status"]).casefold(),
            None,
            canonical_hash(payload),
            (str(payload["deferred_retry_id"]) if payload.get("deferred_retry_id") else None),
            self._datetime(payload.get("retry_after_at")),
            str(payload["failure_code"]) if payload.get("failure_code") else None,
            tuple(sorted(str(item["listing_id"]) for item in payload.get("failures", []))),
        )
