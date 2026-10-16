"""The workspace network control reads closed on anything its writer does not write."""

from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from alphalattice.control.workspace_runtime.network_access import (
    CONTROL_PATH,
    network_access,
    set_network_access,
)


def test_the_written_control_opens_and_closes_the_network(tmp_path: Path) -> None:
    """The written network control obeys the operator's switch."""

    now = datetime.now(UTC)
    assert network_access(tmp_path, environment={}, now=now).decided_by == "DEFAULT"
    set_network_access(tmp_path, enabled=True, now=now)
    assert network_access(tmp_path, environment={}, now=now).allowed
    switch = {"ALPHALATTICE_NETWORK_DISABLED": "1"}
    assert not network_access(tmp_path, environment=switch, now=now).allowed
    set_network_access(tmp_path, enabled=False, now=now)
    assert not network_access(tmp_path, environment={}, now=now).allowed


@pytest.mark.parametrize(
    "document",
    [
        "[]",
        '"open"',
        "7",
        "null",
        json.dumps({"schema": "network-access", "version": 999, "network_enabled": True}),
        json.dumps({"schema": "network-access", "version": True, "network_enabled": True}),
        json.dumps({"schema": "network-access", "version": "1", "network_enabled": True}),
        json.dumps({"schema": "network-access", "network_enabled": True}),
        json.dumps({"schema": "network-access", "version": 1, "network_enabled": 1}),
        json.dumps({"schema": "other", "version": 1, "network_enabled": True}),
        json.dumps({"schema": "network-access", "version": 2, "network_enabled": True}),
        json.dumps(
            {
                "schema": "network-access",
                "version": 2,
                "network_enabled": True,
                "delegation": "first-use-goal:x",
                "until": "2999-01-01T00:00:00",
            }
        ),
        json.dumps(
            {"schema": "network-access", "version": 1, "network_enabled": True, "hosts": ["*"]}
        ),
        "{not json",
    ],
)
def test_anything_but_a_version_one_mapping_with_a_boolean_reads_closed(
    tmp_path: Path, document: str
) -> None:
    """regression (OP5): a JSON array raised AttributeError and version 999 with
    ``network_enabled: true`` opened the network; each reads as the control, closed."""

    path = tmp_path / CONTROL_PATH
    path.parent.mkdir(parents=True)
    path.write_text(document, encoding="utf-8")
    access = network_access(tmp_path, environment={}, now=datetime.now(UTC))
    assert access.decided_by == "WORKSPACE_CONTROL" and not access.allowed


def test_a_delegated_setting_holds_until_its_end_with_no_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """regression (a QA Host's past clock): a delegation holds until its end on the Host's clock."""

    monkeypatch.delenv("ALPHALATTICE_NETWORK_DISABLED", raising=False)
    end = datetime(2026, 9, 11, tzinfo=UTC)
    host = end - timedelta(hours=1)
    set_ = set_network_access(
        tmp_path, enabled=True, delegation="first-use-goal:g", until=end, now=host
    )
    before = network_access(tmp_path, environment={}, now=end - timedelta(seconds=1))
    after = network_access(tmp_path, environment={}, now=end + timedelta(hours=1))
    assert set_.allowed and before.allowed and not after.allowed
    with pytest.raises(TypeError):
        network_access(tmp_path, environment={})
    assert before.body()["set_by"] == {"delegation": "first-use-goal:g", "until": end.isoformat()}
    # The operator's switch decides, and who set the control is still named.
    held = network_access(tmp_path, environment={"ALPHALATTICE_NETWORK_DISABLED": "1"}, now=end)
    assert held.decided_by == "OPERATOR_OFFLINE_SWITCH" and held.body()["set_by"]
    set_network_access(tmp_path, enabled=True, now=host)
    assert network_access(tmp_path, environment={}, now=end + timedelta(days=9)).allowed
    assert "set_by" not in network_access(tmp_path, environment={}, now=host).body()


@pytest.mark.parametrize("state", ["operator", "run", "closed", "open", "default", "unlocated"])
def test_every_network_way_on_reads_the_effective_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    """Network advice follows the effective owner and cannot lift an operator or run hold."""

    import alphalattice.interface.local_application as local_application
    from alphalattice.control.product_host.composition.plain_refusals import explain
    from alphalattice.control.workspace_runtime.network_access import set_network_access
    from alphalattice.interface.local_application.cli_contract import (
        NETWORK_ACCESS_REFUSALS,
        refusal_words,
        worded_refusal,
    )
    from alphalattice.kernel.shared_kernel.environment import held_offline

    table = json.loads(Path(local_application.__file__).with_name("refusal_words.json").read_text())
    network_codes = {
        code
        for code, words in table.items()
        if words["next_action"] == "ASK_A_PERSON_TO_ALLOW_NETWORK_ACCESS"
    }
    assert (
        network_codes | {"evidence_review.workspace_network_not_allowed"} == NETWORK_ACCESS_REFUSALS
    )
    assert {
        code for code, words in table.items() if "network set" in words["detail"]
    } <= NETWORK_ACCESS_REFUSALS
    monkeypatch.delenv("ALPHALATTICE_NETWORK_DISABLED", raising=False)
    now = datetime.now(UTC)
    if state not in {"default", "unlocated"}:
        set_network_access(tmp_path, enabled=state != "closed", now=now)
    if state == "operator":
        monkeypatch.setenv("ALPHALATTICE_NETWORK_DISABLED", "1")
    workspace = None if state == "unlocated" else tmp_path
    with held_offline() if state == "run" else nullcontext():
        for base in NETWORK_ACCESS_REFUSALS:
            code = base + (":2026-09-11,DATA" if base.startswith("research_update.") else "")
            owner = explain(code, workspace=workspace, now=now)
            fresh = worded_refusal(
                {
                    "status": "REFUSED",
                    "failure_code": code,
                    "detail": owner["detail"],
                    "network_access": {"decided_by": "WORKSPACE_CONTROL", "network_allowed": False},
                    "next_requests": {
                        "set": {"operation": "NETWORK_ACCESS_SET", "network_enabled": True}
                    },
                },
                workspace=workspace,
                now=now,
            )
            words = refusal_words(code, workspace=workspace, now=now)
            assert fresh["next_action"] == words["next_action"]
            assert fresh["next_requests"] == {"network": {"operation": "NETWORK_ACCESS"}}
            if state in {"operator", "run"}:
                access = fresh["network_access"]
                assert access["decided_by"] == (
                    "OPERATOR_OFFLINE_SWITCH" if state == "operator" else "RUN_HELD_OFFLINE"
                )
                assert not access["network_allowed"] and not access["next_requests"]
                assert "network set" not in fresh["detail"]
                assert (
                    "ALPHALATTICE_NETWORK_DISABLED=1" in fresh["detail"]
                    if state == "operator"
                    else fresh["detail"]
                )
                if state == "operator":
                    assert fresh["detail"]
                else:
                    assert fresh["next_action"] == "WAIT_FOR_THE_OFFLINE_RUN_TO_FINISH"
            elif state in {"closed", "default"}:
                assert (
                    "set the workspace control"
                    if base == "evidence_review.workspace_network_not_allowed"
                    else "network set"
                ) in fresh["detail"]
                assert fresh["network_access"]["next_requests"]["set"]["network_enabled"]
            elif state == "open":
                assert fresh["network_access"]["network_allowed"]
                assert fresh["next_action"] == "RETRY_THE_REFUSED_STEP"
                assert fresh["detail"]
                assert "network set" not in fresh["detail"]
            else:
                # No located control: a read is actionable without claiming its state.
                assert fresh["next_action"] == "READ_NETWORK_ACCESS"
                assert "network show" in fresh["detail"] and "network set" not in fresh["detail"]
                assert "network_access" not in refusal_words(code)
                assert "network_access" not in fresh
            if base.startswith("research_update."):
                assert "2026-09-11" in fresh["detail"]
                assert "2026-09-11,DATA" not in fresh["detail"]


def test_admitted_network_work_names_whose_setting_allowed_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (the review of a36e210c4): an answer that admits network work names whose
    setting allowed it and the line said for it; its activity row keeps it, so a Host-run
    update is disclosed too."""
    from alphalattice.interface.local_application.activity import returned_subject
    from alphalattice.interface.local_application.cli_contract import disclosed

    monkeypatch.delenv("ALPHALATTICE_NETWORK_DISABLED", raising=False)
    now = datetime.now(UTC)
    admitted = {"status": "ADMITTED", "task_id": "t1"}
    control = lambda: {"workspace": tmp_path, "now": now}  # noqa: E731
    delegated = {"delegation": "first-use-goal:g", "until": now + timedelta(hours=1)}
    for setting, under in (
        ({"relayed_by": "claude-code:s"}, "RELAYED_YES"),
        (delegated, "FIRST_USE_DELEGATION"),
        ({}, "PERSON"),
    ):
        (tmp_path / CONTROL_PATH).unlink(missing_ok=True)
        set_network_access(tmp_path, enabled=True, now=now, **setting)
        answer = disclosed(admitted, "RESEARCH_UPDATE_RUN", control)
        assert answer["network_use"]["under"] == under
        assert answer["ask_now"] == [answer["network_use"]["words"]]  # said, as the guide says
        assert returned_subject(answer)["network_used_under"] == under
    assert disclosed(admitted, "RECOVER", control)["network_use"]
    assert "network_use" not in disclosed(admitted, "STATUS", lambda: 1 / 0)  # type: ignore
    set_network_access(tmp_path, enabled=False, now=now)
    closed = disclosed(admitted, "RESEARCH_UPDATE_RUN", control)
    assert "network_use" not in closed


def test_the_persons_stop_closes_the_network_before_it_revokes_sec_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """regression (the review of a28b30c73): the stop wrote the consent first, so a failed
    consent write left the network on."""
    from alphalattice.evidence.alternative_evidence.sources import admission

    monkeypatch.delenv("ALPHALATTICE_NETWORK_DISABLED", raising=False)
    now = datetime.now(UTC)
    set_network_access(tmp_path, enabled=True, now=now)

    def unwritable(*_: object) -> None:
        raise OSError("consent unwritable")

    monkeypatch.setattr(admission, "record_evidence_source_consent", unwritable)
    with pytest.raises(OSError):
        admission.set_workspace_network(tmp_path, enabled=False, now=now)
    assert not network_access(tmp_path, now=now).allowed
