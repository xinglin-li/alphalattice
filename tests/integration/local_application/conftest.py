import pytest

from tests._support.alternative_evidence_desk_fixtures import built_http_book as built_http_book
from tests._support.alternative_evidence_desk_fixtures import http_book as http_book
from tests._support.portfolio_strategy_lab_fixtures import completed_host as completed_host
from tests._support.portfolio_strategy_lab_fixtures import fake_host as fake_host
from tests._support.portfolio_strategy_lab_fixtures import live as live
from tests._support.portfolio_strategy_lab_fixtures import read_only_live as read_only_live


@pytest.fixture(autouse=True)
def own_codex_home(tmp_path_factory, monkeypatch):
    """Each test reads its own empty Codex configuration, never the person's: a Codex session's
    first use reads the project's trust, and native usage its sessions, from `CODEX_HOME`. A test
    that needs another home sets its own."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path_factory.mktemp("codex-home")))
