"""Durable advancement programs, lane outcomes and receipts, content-addressed.

Separate from the Portfolio ledger on purpose. That store holds one declared
path's descendants; this one holds the record of how the workspace's inputs got
to where they are, and the two answer different questions -- "what did this
configuration do" against "what has this workspace materialized". Merging them
would put an advancement receipt under a program hash that describes holdings.

The by-program index is what makes a repeated advancement cost nothing: the same
sealed program reopens its receipt instead of driving seven lanes again.
"""

from __future__ import annotations

import json
from pathlib import Path

from alphalattice.control.workspace_runtime.content_store import (
    ContentAddressedStore,
    ContentAddressedStoreError,
)
from alphalattice.investment.portfolio_strategy_lab.application.advancement import (
    AdvancementLaneId,
    DomainLaneReceipt,
    OperationalReceipt,
    PortfolioLedgerCoverage,
    WatermarkAdvancementProgram,
    WatermarkAdvancementReceipt,
)


class AdvancementLedgerStoreError(ValueError):
    """Stable advancement storage refusal."""


class AdvancementLedgerStore:
    """Content-addressed advancement artifacts plus one by-program receipt index."""

    def __init__(self, artifact_root: Path) -> None:
        """Bind content-addressed advancement evidence to the caller-owned artifact root.

        Args:
            artifact_root: Caller-owned workspace artifact directory.
        """
        self.root = artifact_root.resolve() / "portfolio-strategy-lab" / "advancement"
        self.content = ContentAddressedStore(
            self.root,
            uri_prefix="playpen://portfolio-strategy-lab/advancement",
        )

    def publish_program(self, value: WatermarkAdvancementProgram) -> str:
        """Publish exact advancement program evidence.

        Args:
            value: Validated sealed evidence contract.

        Returns:
            Content-addressed evidence URI.
        """
        return self.content.publish_model(
            category="programs", value=value, identity_field="program_hash"
        )

    def load_program(self, program_hash: str) -> WatermarkAdvancementProgram:
        """Reopen exact advancement program evidence.

        Args:
            program_hash: Exact retained content identity.

        Returns:
            Validated contract reopened through content-addressed storage.
        """
        return self.content.load_model(
            category="programs",
            content_hash=program_hash,
            model=WatermarkAdvancementProgram,
            identity_field="program_hash",
        )

    def publish_outcome(self, value: DomainLaneReceipt) -> str:
        """Publish exact advancement outcome evidence.

        Args:
            value: Validated sealed evidence contract.

        Returns:
            Content-addressed evidence URI; publication also refuses a contradictory retained
            program/lane index.
        """
        uri = self.content.publish_model(
            category="lane-outcomes", value=value, identity_field="receipt_hash"
        )
        index = self._outcome_index(program_hash=value.program_hash, lane=value.lane)
        payload = json.dumps(
            {
                "program_hash": value.program_hash,
                "lane": value.lane,
                "receipt_hash": value.receipt_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if index.is_file() and index.read_bytes() != payload:
            raise AdvancementLedgerStoreError("portfolio_advancement.lane_outcome_identity_reused")
        if not index.is_file():
            index.parent.mkdir(parents=True, exist_ok=True)
            index.write_bytes(payload)
        return uri

    def find_outcome_for(
        self, *, program_hash: str, lane: AdvancementLaneId
    ) -> DomainLaneReceipt | None:
        """One lane's durable answer for one program.

        This is what makes recovery resume a verified prefix rather than replay
        it: a resumed run reads the lanes that already finished from here instead
        of holding them in a process that no longer exists.
        """
        index = self._outcome_index(program_hash=program_hash, lane=lane)
        if not index.is_file():
            return None
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
            if payload["program_hash"] != program_hash or payload["lane"] != lane:
                raise ValueError
            outcome = self.load_outcome(str(payload["receipt_hash"]))
            program = self.load_program(program_hash)
            program.require_lane_receipt(lane=lane, receipt=outcome)
        except (ContentAddressedStoreError, KeyError, ValueError, json.JSONDecodeError) as error:
            if (
                isinstance(error, ContentAddressedStoreError)
                and str(error).partition(":")[0] == "content_store.artifact_missing"
            ):
                raise
            raise AdvancementLedgerStoreError(
                "portfolio_advancement.lane_outcome_index_tampered"
            ) from error
        if outcome.program_hash != program_hash or outcome.lane != lane:
            raise AdvancementLedgerStoreError("portfolio_advancement.lane_outcome_index_tampered")
        return outcome

    def load_outcome(self, outcome_hash: str) -> DomainLaneReceipt:
        """Reopen exact advancement outcome evidence.

        Args:
            outcome_hash: Exact retained content identity.

        Returns:
            Validated contract reopened through content-addressed storage.
        """
        return self.content.load_model(
            category="lane-outcomes",
            content_hash=outcome_hash,
            model=DomainLaneReceipt,
            identity_field="receipt_hash",
        )

    def publish_operational(self, value: OperationalReceipt) -> str:
        """Publish exact advancement operational evidence.

        Args:
            value: Validated sealed evidence contract.

        Returns:
            Content-addressed evidence URI.
        """
        return self.content.publish_model(
            category="operational-receipts", value=value, identity_field="receipt_hash"
        )

    def load_operational(self, receipt_hash: str) -> OperationalReceipt:
        """Reopen exact advancement operational evidence.

        Args:
            receipt_hash: Exact retained content identity.

        Returns:
            Validated contract reopened through content-addressed storage.
        """
        return self.content.load_model(
            category="operational-receipts",
            content_hash=receipt_hash,
            model=OperationalReceipt,
            identity_field="receipt_hash",
        )

    def publish_ledger_coverage(self, value: PortfolioLedgerCoverage) -> str:
        """Publish exact advancement ledger coverage evidence.

        Args:
            value: Validated sealed evidence contract.

        Returns:
            Content-addressed evidence URI.
        """
        return self.content.publish_model(
            category="ledger-coverage", value=value, identity_field="coverage_hash"
        )

    def load_ledger_coverage(self, coverage_hash: str) -> PortfolioLedgerCoverage:
        """Reopen exact advancement ledger coverage evidence.

        Args:
            coverage_hash: Exact retained content identity.

        Returns:
            Validated contract reopened through content-addressed storage.
        """
        return self.content.load_model(
            category="ledger-coverage",
            content_hash=coverage_hash,
            model=PortfolioLedgerCoverage,
            identity_field="coverage_hash",
        )

    def publish_receipt(self, value: WatermarkAdvancementReceipt) -> str:
        """Publish exact advancement receipt evidence.

        Args:
            value: Validated sealed evidence contract.

        Returns:
            Content-addressed evidence URI; publication also refuses a contradictory retained
            program/lane index.
        """
        uri = self.content.publish_model(
            category="receipts", value=value, identity_field="receipt_hash"
        )
        index = self._receipt_index(value.program_hash)
        payload = json.dumps(
            {"program_hash": value.program_hash, "receipt_hash": value.receipt_hash},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if index.is_file() and index.read_bytes() != payload:
            # One program, one receipt. A second value under this key would mean
            # the same sealed advancement produced two different histories.
            raise AdvancementLedgerStoreError("portfolio_advancement.receipt_identity_reused")
        if not index.is_file():
            index.parent.mkdir(parents=True, exist_ok=True)
            index.write_bytes(payload)
        return uri

    def load_receipt(self, receipt_hash: str) -> WatermarkAdvancementReceipt:
        """Reopen exact advancement receipt evidence.

        Args:
            receipt_hash: Exact retained content identity.

        Returns:
            Validated contract reopened through content-addressed storage.
        """
        return self.content.load_model(
            category="receipts",
            content_hash=receipt_hash,
            model=WatermarkAdvancementReceipt,
            identity_field="receipt_hash",
        )

    def find_receipt_for(self, program_hash: str) -> WatermarkAdvancementReceipt | None:
        """The published receipt for one sealed program, if it has already run."""
        index = self._receipt_index(program_hash)
        if not index.is_file():
            return None
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
            if payload["program_hash"] != program_hash:
                raise ValueError
            receipt = self.load_receipt(str(payload["receipt_hash"]))
            program = self.load_program(program_hash)
            program.require_receipt(receipt)
        except (ContentAddressedStoreError, KeyError, ValueError, json.JSONDecodeError) as error:
            if (
                isinstance(error, ContentAddressedStoreError)
                and str(error).partition(":")[0] == "content_store.artifact_missing"
            ):
                raise
            raise AdvancementLedgerStoreError(
                "portfolio_advancement.receipt_index_tampered"
            ) from error
        return receipt

    def _receipt_index(self, program_hash: str) -> Path:
        return self.root / "index" / "by-program" / f"{program_hash}.json"

    def _outcome_index(self, *, program_hash: str, lane: str) -> Path:
        return self.root / "index" / "by-program-lane" / program_hash / f"{lane}.json"


__all__ = ["AdvancementLedgerStore", "AdvancementLedgerStoreError"]
