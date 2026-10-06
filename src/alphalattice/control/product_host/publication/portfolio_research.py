"""Thin Product Host manifest over one verified public Portfolio result."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.control.workspace_runtime.content_store import (
    ContentAddressedStore,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


class PortfolioResearchPipelineManifest(BaseModel):  # type: ignore[misc]
    """Seal completed portfolio task, program, result and report lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["PortfolioResearchPipelineManifest"] = "PortfolioResearchPipelineManifest"
    workspace_id: str = Field(min_length=1, max_length=128)
    task_id: UUID
    task_record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed_at: datetime
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal explicit completed pipeline fields with the fixed manifest kind.

        Args:
            values: Explicit manifest fields excluding the generated identity and fixed kind.

        Returns:
            Validated manifest with canonical manifest_hash.
        """
        identity = cls.model_construct(
            kind="PortfolioResearchPipelineManifest",
            **values,
            manifest_hash="0" * 64,
        ).model_dump(mode="json", exclude={"manifest_hash"})
        return cls(**identity, manifest_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require an aware completion timestamp and exact pipeline manifest identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Completion timestamp is naive or canonical manifest binding differs.
        """
        if self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None:
            raise ValueError("portfolio_application.pipeline_manifest_clock_invalid")
        if self.manifest_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"manifest_hash"})
        ):
            raise ValueError("portfolio_application.pipeline_manifest_identity_invalid")
        return self


class PortfolioResearchPipelineStore:
    """Own immutable pipeline manifests and exact result/task lookup associations."""

    def __init__(self, artifact_root: Path) -> None:
        """Open the product-host portfolio lineage namespace beneath an explicit artifact root.

        Args:
            artifact_root: Caller-owned artifact root.
        """
        self.content = ContentAddressedStore(
            artifact_root.resolve() / "product-host" / "portfolio-research",
            uri_prefix="playpen://product-host/portfolio-research",
        )

    def publish(self, value: PortfolioResearchPipelineManifest) -> str:
        """Publish exact pipeline lineage and preserve first-result and every-task associations.

        The result index keeps its first publisher; the task index names every task that published
        or reused the result.

        Args:
            value: Explicit sealed completed pipeline manifest.

        Returns:
            Published content-addressed manifest URI.
        """
        uri = self.content.publish_model(
            category="manifests",
            value=value,
            identity_field="manifest_hash",
        )
        index = self._result_index(value.result_hash)
        payload = json.dumps(
            {"result_hash": value.result_hash, "manifest_hash": value.manifest_hash},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if not index.is_file():
            index.parent.mkdir(parents=True, exist_ok=True)
            index.write_bytes(payload)
        # Every Task that published is named, a reuse of an earlier result
        # included (V189): the by-result index keeps the first, this keeps each.
        used = self._task_index(value.task_id)
        if not used.is_file():
            used.parent.mkdir(parents=True, exist_ok=True)
            used.write_bytes(
                json.dumps(
                    {"task_id": str(value.task_id), "manifest_hash": value.manifest_hash},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )
        return uri

    def find_for_task(self, task_id: UUID) -> PortfolioResearchPipelineManifest | None:
        """The manifest a Task published, whether its result was new or reused.

        A manifest published before the by-Task index was kept is found among the
        manifests by its Task.

        Args:
            task_id: The Task that ran.

        Returns:
            Its manifest, or None when it published none.
        """
        index = self._task_index(task_id)
        if index.is_file():
            payload = json.loads(index.read_text(encoding="utf-8"))
            if payload["task_id"] != str(task_id):
                raise ValueError("portfolio_application.pipeline_task_index_tampered")
            value = self.load(str(payload["manifest_hash"]))
            if value.task_id != task_id:
                raise ValueError("portfolio_application.pipeline_task_index_tampered")
            return value
        return next((value for value in self._all_manifests() if value.task_id == task_id), None)

    def tasks_for_result(self, result_hash: str) -> tuple[UUID, ...]:
        """Every Task that published this result, first to last.

        Args:
            result_hash: The numeric result.

        Returns:
            The Tasks, by completion, the one that first produced it first.
        """
        found = [value for value in self._all_manifests() if value.result_hash == result_hash]
        return tuple(
            value.task_id
            for value in sorted(found, key=lambda value: (value.completed_at, str(value.task_id)))
        )

    def _all_manifests(self) -> tuple[PortfolioResearchPipelineManifest, ...]:
        root = self.content.root / "manifests"
        if not root.is_dir():
            return ()
        return tuple(self.load(path.stem) for path in sorted(root.glob("*.json")))

    def _task_index(self, task_id: UUID) -> Path:
        return self.content.root / "index" / "by-task" / f"{task_id}.json"

    def find_for_result(self, result_hash: str) -> PortfolioResearchPipelineManifest | None:
        """Which task produced this result.

        A report and an export have to be able to name their originating run, and
        the researcher only ever holds a result hash. Without this index the
        answer exists but is unreachable, which is the same as not existing.

        The first manifest wins. A result reopened later is the same numerical
        object under a new task, and repointing would rewrite which run produced
        it.
        """
        index = self._result_index(result_hash)
        if not index.is_file():
            return None
        payload = json.loads(index.read_text(encoding="utf-8"))
        if payload["result_hash"] != result_hash:
            raise ValueError("portfolio_application.pipeline_result_index_tampered")
        return self.load(str(payload["manifest_hash"]))

    def manifests(self) -> tuple[PortfolioResearchPipelineManifest, ...]:
        """Every completed run this workspace has published, newest first.

        A local UI has to be able to offer "the runs you have" without the user
        already holding a hash, and the by-result index is the durable record of
        exactly that. Sorted by completion so the list reads the way a session
        happened; ties keep result order so it is stable across calls.
        """
        root = self.content.root / "index" / "by-result"
        if not root.is_dir():
            return ()
        found: list[PortfolioResearchPipelineManifest] = []
        for path in sorted(root.glob("*.json")):
            manifest = self.find_for_result(path.stem)
            if manifest is not None:
                found.append(manifest)
        return tuple(
            sorted(found, key=lambda value: (value.completed_at, value.result_hash), reverse=True)
        )

    def manifest_collection(
        self,
    ) -> tuple[
        tuple[PortfolioResearchPipelineManifest, ...],
        tuple[dict[str, str], ...],
    ]:
        """Discover readable result rows while naming index records this read cannot verify.

        This partial read is for a person-facing collection only. The scientific and lineage
        selectors (`manifests`, `find_for_result`, `find_for_task`, and `tasks_for_result`) stay
        strict: a refusal here never proves that a Task did not publish or that a result is absent.

        Returns:
            Readable manifests newest first, and one typed refusal for each unreadable by-result
            index or sealed manifest. A valid `result_hash` is included only when the index
            filename itself is a full lowercase content hash; no Task identity is inferred from
            an index that could not be read.
        """
        root = self.content.root / "index" / "by-result"
        found: list[PortfolioResearchPipelineManifest] = []
        refused: list[dict[str, str]] = []
        paths: list[Path] = []
        try:
            for path in root.iterdir():
                if path.name.endswith(".json"):
                    paths.append(path)
        except FileNotFoundError:
            if not paths:
                return (), ()
            refused.append(
                {
                    "status": "REFUSED",
                    "failure_code": "portfolio_application.pipeline_manifest_unreadable",
                    "index_file": "by-result",
                    "entry_id": "result-index:by-result",
                }
            )
        except OSError:
            refused.append(
                {
                    "status": "REFUSED",
                    "failure_code": "portfolio_application.pipeline_manifest_unreadable",
                    "index_file": "by-result",
                    "entry_id": "result-index:by-result",
                }
            )
        for path in sorted(paths):
            stem = path.stem
            valid_hash = len(stem) == 64 and all(
                character in "0123456789abcdef" for character in stem
            )
            try:
                manifest = self.find_for_result(stem)
            except (OSError, ValueError, KeyError, TypeError):
                row = {
                    "status": "REFUSED",
                    "failure_code": "portfolio_application.pipeline_manifest_unreadable",
                    "index_file": path.name,
                }
                if valid_hash:
                    row.update(result_hash=stem, entry_id=f"result:{stem}")
                else:
                    row["entry_id"] = f"result-index:{path.name}"
                refused.append(row)
                continue
            if manifest is not None:
                found.append(manifest)
        return (
            tuple(
                sorted(
                    found, key=lambda value: (value.completed_at, value.result_hash), reverse=True
                )
            ),
            tuple(refused),
        )

    def _result_index(self, result_hash: str) -> Path:
        return self.content.root / "index" / "by-result" / f"{result_hash}.json"

    def load(self, manifest_hash: str) -> PortfolioResearchPipelineManifest:
        """Reopen one exact pipeline manifest through identity-validating content storage.

        Args:
            manifest_hash: Exact retained manifest identity.

        Returns:
            Validated completed pipeline manifest.
        """
        return self.content.load_model(
            category="manifests",
            content_hash=manifest_hash,
            model=PortfolioResearchPipelineManifest,
            identity_field="manifest_hash",
        )


__all__ = ["PortfolioResearchPipelineManifest", "PortfolioResearchPipelineStore"]
