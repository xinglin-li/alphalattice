from collections.abc import Iterator

import pytest

from tests._support.alternative_evidence_desk_fixtures import built_http_book as built_http_book
from tests._support.alternative_evidence_desk_fixtures import http_book as http_book
from tests.alternative_evidence_desk.planted_corpus import review_store


@pytest.fixture(autouse=True)
def _review_store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Every review here runs on the Host's store, with the planted models."""
    with review_store(tmp_path_factory.mktemp("review-store")):
        yield
