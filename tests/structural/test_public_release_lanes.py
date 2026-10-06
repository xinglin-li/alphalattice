"""The public checkout refuses unavailable private verification by name."""

import sys

from scripts import check_playpen as gate


def test_public_full_gate_names_missing_private_guard(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    guard = tmp_path / "tests/structural/test_structural_guards.py"
    monkeypatch.setattr(gate, "STRUCTURAL_TESTS", (guard,))
    monkeypatch.setattr(sys, "argv", ["check_playpen.py", "--all-python"])
    assert gate.main(ensure_environment=lambda: None) == 1
    message = capsys.readouterr().err
    assert "playpen.private_structural_inputs_unavailable" in message
    assert guard.relative_to(tmp_path).as_posix() in message
    assert "private records" in message and "--all-python --fast" in message


def test_public_evidence_gate_names_missing_private_root_declaration(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["check_playpen.py", "--evidence"])
    assert gate.main(ensure_environment=lambda: None) == 1
    message = capsys.readouterr().err
    assert "devtools.private_evidence_lane_unavailable" in message
    assert "maintainers' private QA roots" in message
    assert "not part of the public checkout" in message
