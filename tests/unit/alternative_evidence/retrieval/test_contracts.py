"""A hybrid-v4 generation names the machine class of an int8 recipe's numbers, in its identity.

The field is absent at None, so every generation sealed before it, and every one of a recipe a
sealed canary fixes, keeps its hash; a generation that names another machine is another
generation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceReadingDepth,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.retrieval.contracts import (
    HybridV4RetrievalGeneration,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

TEST_SOURCE = "src/alphalattice/evidence/alternative_evidence/retrieval/contracts.py"
FIELDS: dict[str, object] = {
    "document_set_hash": "1" * 64,
    "workspace_snapshot_id": str(UUID(int=1)),
    "workspace_snapshot_hash": "2" * 64,
    "retrieval_binding_hash": "3" * 64,
    "reading_depth": AlternativeEvidenceReadingDepth.FULL_FILING,
    "corpus_hash": "4" * 64,
    "index_id": "5" * 64,
    "index_spec_hash": "6" * 64,
    "index_manifest_hash": "7" * 64,
    "vector_payload_sha256": "8" * 64,
    "document_count": 3,
    "chunk_count": 120,
    "built_at": datetime(2026, 11, 15, 4, tzinfo=UTC),
}


@pytest.mark.proves("alternative_evidence.int8_generation_names_its_machine_class")
def test_the_machine_class_is_identity_when_named_and_absent_when_not() -> None:
    def seal(**extra: object) -> HybridV4RetrievalGeneration:
        return seal_contract(HybridV4RetrievalGeneration, "generation_hash", **FIELDS, **extra)

    retained = seal()
    payload = retained.model_dump(mode="json", exclude={"generation_hash"})
    assert "execution_class" not in payload
    # The hash a generation sealed before the field had is the one it still has.
    assert retained.generation_hash == canonical_hash(payload)

    here = seal(execution_class="AMD64; Intel64 Family 6 Model 183; X86_V2+X86_V3")
    there = seal(execution_class="arm64; arm; NEON+ASIMD+ASIMDDP")
    assert len({retained.generation_hash, here.generation_hash, there.generation_hash}) == 3
    assert HybridV4RetrievalGeneration.model_validate_json(here.model_dump_json()) == here
