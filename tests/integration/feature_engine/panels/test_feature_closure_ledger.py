import pytest

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
    FeatureClosureLedger,
)

TEST_SOURCE = "src/alphalattice/foundation/feature_engine/panels/feature_closure_ledger.py"


@pytest.mark.proves("feature_engine.closure_pointer_requires_a_full_content_identity")
def test_a_head_lookup_refuses_a_partial_or_noncanonical_catalog_identity(tmp_path):
    ledger = FeatureClosureLedger(PanelClosureArtifactStore(ArtifactResolver(tmp_path)))
    assert ledger.current_head("a" * 64) is None
    for malformed in ("a" * 63, "A" * 64, "../head"):
        with pytest.raises(ValueError, match="Feature closure pointer identity is invalid"):
            ledger.current_head(malformed)
