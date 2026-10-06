"""Durable, local Human-issued Data decision grants and historical receipt readback.

A grant is issued only by the same-origin Human operation and stored by the
Data decision repository. Its digest identifies exact scope; external requests
and self-hashed documents are never issuance evidence. The local Host and
workspace are the trust boundary, not a multi-user identity system.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from alphalattice.foundation.feature_engine.inputs.gateway import FeatureInputAgentCase
from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
from alphalattice.foundation.market_data_ops.runtime.remediation import (
    RemediationOption,
    canonical_hash,
)

# An older conversation-approved grant remains recognizable in historical Task receipts.
_HISTORICAL_REVIEW_SHA256 = "7ddb12e50553a8e54f9c631119418d6e28995b6800657367872c1202069567a0"
_HISTORICAL_PREPARATION_TASK_ID = "2eda7f2d-b9d5-4860-8828-c083ccdab0ab"


class DataIssueDelegation:
    """One owner for issuing, validating and revoking exact local grants."""

    def __init__(self, workspace: Path) -> None:
        """Resolve the workspace owning the Feature input delegation ledger.

        Args:
            workspace: Workspace root owning the local records.
        """
        self.workspace = workspace.resolve()
        self.ledger = PanelStateRepository(self.workspace)

    def issue(
        self,
        *,
        case: FeatureInputAgentCase,
        option: RemediationOption,
        task_id: str,
        plan_hash: str,
        now: datetime,
    ) -> dict[str, Any]:
        """Persist an exact Human-issued Data decision grant or reuse its active equivalent.

        Args:
            case: Current Feature input case with its evidence and policy bindings.
            option: Exact admitted remediation option and its typed policy arguments.
            task_id: Preparation task identifier bound by the delegation.
            plan_hash: Exact preparation or update plan identity being bound.
            now: Clock used to assess the active delegation or deadline.

        Returns:
            The sealed grant bound to the task, plan, case, evidence, policy and option.

        Raises:
            ValueError: The bounded grant deadline has already passed.
        """
        deadline = now.astimezone(UTC) + timedelta(hours=24)
        recheck = getattr(option.policy_args, "recheck_after_at", None)
        if recheck is not None:
            deadline = min(deadline, recheck.astimezone(UTC))
        if deadline <= now.astimezone(UTC):
            raise ValueError("feature_input.case_expired_reassess_required")
        body: dict[str, Any] = {
            "kind": "DataIssueDelegation.v1",
            "workspace": str(self.workspace),
            "issued_by": "LOCAL_WEB_HUMAN",
            "actual_executor": "EXTERNAL_AUTOMATION",
            "issued_at": now.astimezone(UTC).isoformat(),
            "expires_at": deadline.isoformat(),
            "preparation_task_id": task_id,
            "preparation_plan_hash": plan_hash,
            "case_token": case.case_token,
            "evidence_hash": case.evidence_hash,
            "policy_hash": case.policy_hash,
            "option_id": option.option_id,
            "option_hash": option.option_hash,
            "policy_args": option.policy_args.model_dump(mode="json"),
            "target_listing_ids": list(option.target_listing_ids),
        }
        # Repeated Human clicks on the same active scope return its existing grant.
        # Issuance is still a Human operation; this scan does not admit any new caller.
        for prior, revoked in self.ledger.feature_input_delegation_documents():
            if revoked:
                continue
            try:
                prior = self._validate(prior, revoked=revoked, now=now)
            except ValueError:
                continue
            if all(
                prior.get(key) == body[key]
                for key in (
                    "workspace",
                    "actual_executor",
                    "preparation_task_id",
                    "preparation_plan_hash",
                    "case_token",
                    "evidence_hash",
                    "policy_hash",
                    "option_id",
                    "option_hash",
                    "policy_args",
                    "target_listing_ids",
                )
            ):
                return prior
        grant_hash = canonical_hash(body)
        document = {**body, "grant_hash": grant_hash}
        self.ledger.record_feature_input_delegation(document)
        return document

    def read(self, grant_hash: str, *, now: datetime | None = None) -> dict[str, Any]:
        """Load and validate one stored delegation against scope, revocation and expiry.

        Args:
            grant_hash: Canonical identity of the delegation grant.
            now: Clock used to assess the active delegation or deadline.

        Returns:
            The validated stored grant.

        Raises:
            ValueError: Identity is malformed, the record is absent, or grant validation fails.
        """
        if len(grant_hash) != 64 or any(c not in "0123456789abcdef" for c in grant_hash):
            raise ValueError("feature_input.delegation_invalid")
        stored = self.ledger.feature_input_delegation_documents(grant_hash)
        if len(stored) != 1:
            raise ValueError("feature_input.delegation_not_found")
        document, revoked = stored[0]
        return self._validate(document, revoked=revoked, now=now)

    def _validate(
        self, document: dict[str, Any], *, revoked: bool, now: datetime | None
    ) -> dict[str, Any]:
        grant_hash = document.get("grant_hash")
        if not isinstance(grant_hash, str):
            raise ValueError("feature_input.delegation_invalid")
        body = {key: value for key, value in document.items() if key != "grant_hash"}
        if (
            canonical_hash(body) != grant_hash
            or document.get("kind") != "DataIssueDelegation.v1"
            or document.get("workspace") != str(self.workspace)
            or document.get("issued_by") != "LOCAL_WEB_HUMAN"
            or document.get("actual_executor") != "EXTERNAL_AUTOMATION"
        ):
            raise ValueError("feature_input.delegation_invalid")
        if now is not None:
            if revoked:
                raise ValueError("feature_input.delegation_revoked")
            if now.astimezone(UTC) >= datetime.fromisoformat(document["expires_at"]):
                raise ValueError("feature_input.delegation_expired")
        return document

    def active_grants(self, *, now: datetime) -> list[dict[str, Any]]:
        """Collect nonrevoked grants that validate at the supplied clock.

        Args:
            now: Clock used to assess the active delegation or deadline.

        Returns:
            Active validated grant documents; invalid or expired records are omitted.
        """
        active: list[dict[str, Any]] = []
        for document, revoked in self.ledger.feature_input_delegation_documents():
            if revoked:
                continue
            try:
                active.append(self._validate(document, revoked=revoked, now=now))
            except ValueError:
                continue
        return active

    def revoke(self, grant_hash: str, *, now: datetime) -> dict[str, object]:
        """Validate the grant and record its durable revocation.

        Args:
            grant_hash: Canonical identity of the delegation grant.
            now: Clock used to assess the active delegation or deadline.

        Returns:
            A REVOKED status with the grant identity.

        Raises:
            ValueError: The grant cannot be read and validated.
        """
        self.read(grant_hash)
        self.ledger.revoke_feature_input_delegation(grant_hash, now)
        return {"status": "REVOKED", "grant_hash": grant_hash}

    def permits_choice(
        self,
        grant_hash: str,
        case: FeatureInputAgentCase,
        option: RemediationOption,
        *,
        now: datetime,
    ) -> bool:
        """Validate an active grant and compare its exact remediation choice.

        Args:
            grant_hash: Canonical identity of the delegation grant.
            case: Current Feature input case with its evidence and policy bindings.
            option: Exact admitted remediation option and its typed policy arguments.
            now: Clock used to assess the active delegation or deadline.

        Returns:
            Whether the case, policy arguments, option and listing scope match.

        Raises:
            ValueError: The grant cannot be read and validated at the supplied clock.
        """
        grant = self.read(grant_hash, now=now)
        return self.matches_choice(grant, case, option)

    @staticmethod
    def matches_choice(
        grant: dict[str, Any], case: FeatureInputAgentCase, option: RemediationOption
    ) -> bool:
        """Compare all case, evidence, policy, option and listing bindings.

        Args:
            grant: Already validated delegation document to compare.
            case: Current Feature input case with its evidence and policy bindings.
            option: Exact admitted remediation option and its typed policy arguments.

        Returns:
            Whether the supplied validated grant authorizes the exact choice.
        """
        return bool(
            grant["case_token"] == case.case_token
            and grant["evidence_hash"] == case.evidence_hash
            and grant["policy_hash"] == case.policy_hash
            and grant["option_id"] == option.option_id
            and grant["option_hash"] == option.option_hash
            and grant["policy_args"] == option.policy_args.model_dump(mode="json")
            and grant["target_listing_ids"] == list(option.target_listing_ids)
        )

    def recorded_human_authorization(
        self, actor_id: str, case: FeatureInputAgentCase, option: RemediationOption
    ) -> bool:
        """Recognize an already confirmed choice by its exact stored Human grant.

        Revocation prevents new confirmations and continuations; it does not
        erase an immutable decision receipt already accepted by the owner.
        """
        prefix = "user-delegated:"
        if not actor_id.startswith(prefix):
            return False
        grant_hash = actor_id[len(prefix) :]
        try:
            grant = self.read(grant_hash)
        except ValueError:
            return False
        return self.matches_choice(grant, case, option)

    def permits_resume(
        self,
        grant_hash: str,
        *,
        task_id: str,
        plan_hash: str,
        lifecycle: str,
        failure_code: str | None,
        now: datetime,
    ) -> bool:
        """Validate an active grant and compare the blocked preparation scope.

        Args:
            grant_hash: Canonical identity of the delegation grant.
            task_id: Preparation task identifier bound by the delegation.
            plan_hash: Exact preparation or update plan identity being bound.
            lifecycle: Current preparation task lifecycle.
            failure_code: Stable failure code retained for diagnosis.
            now: Clock used to assess the active delegation or deadline.

        Returns:
            Whether the grant matches a truth-review-blocked preparation task and plan.

        Raises:
            ValueError: The grant cannot be read and validated at the supplied clock.
        """
        grant = self.read(grant_hash, now=now)
        return self.matches_resume(
            grant,
            task_id=task_id,
            plan_hash=plan_hash,
            lifecycle=lifecycle,
            failure_code=failure_code,
        )

    @staticmethod
    def matches_resume(
        grant: dict[str, Any],
        *,
        task_id: str,
        plan_hash: str,
        lifecycle: str,
        failure_code: str | None,
    ) -> bool:
        """Compare preparation bindings and the required truth-review block.

        Args:
            grant: Already validated delegation document to compare.
            task_id: Preparation task identifier bound by the delegation.
            plan_hash: Exact preparation or update plan identity being bound.
            lifecycle: Current preparation task lifecycle.
            failure_code: Stable failure code retained for diagnosis.

        Returns:
            Whether task/plan match and the lifecycle and failure code permit this resume.
        """
        return (
            grant["preparation_task_id"] == task_id
            and grant["preparation_plan_hash"] == plan_hash
            and lifecycle == "BLOCKED"
            and failure_code == "data.truth_review_required"
        )

    @staticmethod
    def actor_id(grant_hash: str) -> str:
        """Derive the delegated actor label from the grant identity.

        Args:
            grant_hash: Canonical identity of the delegation grant.

        Returns:
            The user-delegated label carrying the supplied grant hash.
        """
        return "user-delegated:" + grant_hash

    @staticmethod
    def rationale(grant_hash: str) -> str:
        """Describe the Human-issued grant and the external executor it authorizes.

        Args:
            grant_hash: Canonical identity of the delegation grant.

        Returns:
            A bounded grant-specific rationale for the execution receipt.
        """
        return (
            "The local Human issued an exact Data decision grant; executor "
            "EXTERNAL_AUTOMATION; grant SHA-256 " + grant_hash + "."
        )

    @staticmethod
    def known_successor_receipt(payload: dict[str, Any]) -> bool:
        """Recognize the exact retained historical automation receipt bindings.

        Args:
            payload: Recorded actor receipt to classify.

        Returns:
            Whether caller, historical delegation commitment and source task match.
        """
        return (
            payload.get("caller") == "EXTERNAL_AUTOMATION"
            and payload.get("user_delegation_sha256") == _HISTORICAL_REVIEW_SHA256
            and payload.get("source_task_id") == _HISTORICAL_PREPARATION_TASK_ID
        )
