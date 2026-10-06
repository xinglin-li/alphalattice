"""Exact Foundation label metadata over the existing sealed-publication reader.

This projection validates the recorded admission and binding identities only. Selected
Foundation readback, export and admission continue to verify their full source graph.
"""

from __future__ import annotations

import re
from pathlib import Path

from alphalattice.foundation.research_foundation.publication.sponsorship import (
    read_foundation_admission,
)
from alphalattice.interface.local_application.failure_codes import public_failure
from alphalattice.protocols.research_authoring.contracts import AuthoringError


def read_foundation_summary(workspace: Path, foundation_admission_hash: str) -> dict[str, object]:
    """Read one exact Foundation's names without opening its parent study.

    Args:
        workspace: Workspace whose sealed Foundation publication store owns the record.
        foundation_admission_hash: Required exact admission identity; no latest fallback.

    Returns:
        Label metadata, its limited verification scope and the exact full-read request.

    Raises:
        AuthoringError: The exact identity is invalid, unavailable or contradicts its binding.
    """
    if (
        not isinstance(foundation_admission_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", foundation_admission_hash) is None
    ):
        raise AuthoringError("research_foundation.admission_identity_invalid")
    try:
        admission = read_foundation_admission(workspace / "artifacts", foundation_admission_hash)
    except OSError as error:
        raise AuthoringError("research_foundation.admission_artifact_unavailable") from error
    except ValueError as error:
        raise AuthoringError(
            public_failure(error, "research_foundation.admission_identity_invalid")
        ) from error
    return {
        "status": "FOUNDATION_SUMMARY",
        "foundation_admission_hash": admission.admission_hash,
        "foundation_hash": admission.foundation.foundation_hash,
        "factor_task_id": admission.factor_task_id,
        "input_id": admission.input_id,
        "input_binding_hash": admission.input_binding_hash,
        "ordered_factor_ids": list(admission.foundation.ordered_factor_ids),
        "market_as_of": admission.foundation.execution_outcome.market_as_of,
        "evidence_verification": "METADATA_ONLY_SOURCE_GRAPH_NOT_CHECKED",
        "numerical_call_count": 0,
        "next_requests": {
            "verify": {
                "operation": "EXPERIMENT_FOUNDATION_READBACK",
                "foundation_admission_hash": admission.admission_hash,
            }
        },
        "limitations": ["SAVED_SUMMARY_NOT_FULL_GRAPH_VERIFICATION"],
    }
