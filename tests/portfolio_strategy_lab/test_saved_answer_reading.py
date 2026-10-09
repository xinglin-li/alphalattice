"""Full saved answers are local snapshots, never requests or current verification."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from alphalattice.interface.local_application import cli, cli_contract, client
from alphalattice.interface.local_application.cli_contract import envelope

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_alphalattice.py"
CLAIM = "HISTORICAL_SAVED_ANSWER_NOT_REVERIFIED"


def _answer() -> dict[str, Any]:
    return {
        "status": "VERIFIED",
        "verification": {
            "verified_at": "2026-09-14T15:30:00+00:00",
            "basis": "The retained publication and its recorded implementation.",
            "implementation_identity": "f" * 64,
        },
        "source_context": {
            "workspace_id": "saved-workspace",
            "agent_session": "original-session",
            "recorded_at": "2026-09-14T15:00:00+00:00",
        },
        "limits": {
            "unread": ["source-still-unread"],
            "coverage": "Only the recorded prepared packet was read.",
        },
        "exact_references": ["a" * 64, "00000000-0000-4000-8000-000000000001"],
        "items": [
            {"id": "a" * 64, "label": "first"},
            {"id": "b" * 64, "label": "second"},
            {"id": "c" * 64, "label": "third"},
        ],
        "files": {"manifest.json": {"sha256": "d" * 64}},
        "position": {"weight": 0.25},
        "metric_units": {"position.weight": "portfolio fraction"},
        "awkward": ["yes", "007", "2026-09-14", "1e-10", "两行\n原话"],
        "large_read": "中" * 50000,
        "next_requests": {
            "open": {"operation": "GOAL_OPEN", "goal_declaration": {"title": "Never execute"}}
        },
    }


def _save(path: Path, answer: dict[str, Any], format_: str) -> bytes:
    from alphalattice.protocols.research_authoring.selection import dump_declaration

    text = (
        json.dumps(answer, ensure_ascii=False)
        if format_ == "json"
        else dump_declaration(json.loads(json.dumps(answer)))
    )
    payload = text.encode("utf-8")
    path.write_bytes(payload)
    return payload


def _snapshot(path: Path) -> dict[str, str]:
    return {"file": str(path.resolve()), "claim": CLAIM}


def _tree_bytes(path: Path) -> dict[str, bytes | None]:
    return {
        ("directory/" if entry.is_dir() else "file/") + entry.relative_to(path).as_posix(): (
            None if entry.is_dir() else entry.read_bytes()
        )
        for entry in path.rglob("*")
    }


@pytest.fixture
def local_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[..., Any]:
    """Public entry sentinels hold the no-Host, no-observation and no-network boundary."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("A saved-answer reading reached a live entry.")

    monkeypatch.setattr(client, "LocalResearchClient", forbidden)
    monkeypatch.setattr(client, "lead_readings", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    return forbidden


def _read(
    capsys: pytest.CaptureFixture[str], forbidden: Callable[..., Any], *arguments: str
) -> tuple[int, dict[str, Any]]:
    code = cli.main(
        ["answer", "show", *arguments], serve=forbidden, restore=forbidden, sandbox=forbidden
    )
    printed = capsys.readouterr()
    lines = printed.out.strip().splitlines()
    assert len(lines) == 1, printed
    return code, json.loads(lines[0])


@pytest.mark.parametrize("format_", ["json", "yaml"])
def test_saved_answers_keep_their_values_and_read_the_existing_section_class(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    local_only: Callable[..., Any],
    format_: str,
) -> None:
    """Full metadata and whole references survive every supported section shape unchanged."""
    answer = _answer()
    source = tmp_path / f"answer.{format_}"
    original = _save(source, answer, format_)
    before = _tree_bytes(tmp_path)
    code, read = _read(capsys, local_only, "--file", str(source))
    assert (code, read["operation"], read["outcome"], read["status"]) == (
        0,
        "SAVED_ANSWER_SHOW",
        "OK",
        "READ_SAVED_SNAPSHOT",
    )
    assert read["data"] == {"snapshot": _snapshot(source), "answer": answer}
    sections = {
        "verification": answer["verification"],
        "source_context": answer["source_context"],
        "limits.unread": answer["limits"]["unread"],
        "items": answer["items"],
        "items.1.id": "b" * 64,
        "items.1:3": answer["items"][1:3],
        "items.2:": answer["items"][2:],
        "files.manifest.json.sha256": "d" * 64,
        "position.weight": 0.25,
        "large_read": answer["large_read"],
        "next_requests": answer["next_requests"],
    }
    for section, value in sections.items():
        code, read = _read(capsys, local_only, "--file", str(source), "--section", section)
        expected = {"snapshot": _snapshot(source), "section": section, "value": value}
        if section == "position.weight":
            expected["unit"] = "portfolio fraction"
        assert (code, read["outcome"]) == (0, "OK"), read
        assert read["data"] == expected
        assert read["next_requests"] is None
        assert "representation" not in read and "omitted_sections" not in read
    assert source.read_bytes() == original
    assert _tree_bytes(tmp_path) == before


@pytest.mark.parametrize("status", ["REFUSED", "PENDING"])
def test_a_saved_owner_outcome_is_historical_while_the_local_read_succeeds(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    local_only: Callable[..., Any],
    status: str,
) -> None:
    answer = _answer()
    answer.update(status=status, failure_code="saved.owner_refusal", detail="Recorded words.")
    source = tmp_path / "historical.json"
    _save(source, answer, "json")
    code, read = _read(capsys, local_only, "--file", str(source))
    assert (code, read["outcome"], read["status"], read["failure_code"]) == (
        0,
        "OK",
        "READ_SAVED_SNAPSHOT",
        None,
    )
    assert read["data"] == {"snapshot": _snapshot(source), "answer": answer}
    assert read["next_requests"] is None


@pytest.mark.parametrize("format_", ["json", "yaml"])
def test_the_real_cli_reads_without_a_workspace_session_or_server_and_writes_nothing(
    tmp_path: Path, format_: str
) -> None:
    """The shipped entry works from an empty project without creating binding or runtime state."""
    answer = _answer()
    source = tmp_path / f"answer.{format_}"
    _save(source, answer, format_)
    before = _tree_bytes(tmp_path)
    environment = dict(os.environ, ALPHALATTICE_NETWORK_DISABLED="1", PYTHONDONTWRITEBYTECODE="1")
    for name in ("CODEX_THREAD_ID", "CLAUDE_CODE_SESSION_ID", "PYTHONPATH"):
        environment.pop(name, None)
    for extra in ([], ["--section", "exact_references"]):
        ran = subprocess.run(
            [sys.executable, str(SCRIPT), "answer", "show", "--file", str(source), *extra],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert ran.returncode == 0, (ran.stdout, ran.stderr)
        lines = ran.stdout.strip().splitlines()
        assert len(lines) == 1
        read = json.loads(lines[0])
        assert read["outcome"] == "OK"
        assert read["data"]["snapshot"] == _snapshot(source)
        if extra:
            assert read["data"]["value"] == answer["exact_references"]
        else:
            assert read["data"]["answer"] == answer
        assert read["next_requests"] is None
        assert "context" not in read and "native_observation" not in read
        assert _tree_bytes(tmp_path) == before


@pytest.mark.parametrize("format_", ["json", "yaml"])
def test_list_sections_lists_the_saved_answers_own_root_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    local_only: Callable[..., Any],
    format_: str,
) -> None:
    answer = _answer()
    source = tmp_path / f"answer.{format_}"
    _save(source, answer, format_)
    code, read = _read(capsys, local_only, "--file", str(source), "--list-sections")
    assert (code, read["outcome"]) == (0, "OK")
    assert read["data"] == {"snapshot": _snapshot(source), "sections": list(answer)}


@pytest.mark.parametrize("format_", ["json", "yaml"])
@pytest.mark.parametrize("mode", ["whole", "section", "list"])
def test_output_keeps_the_full_snapshot_when_stdout_selects_only_a_part(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    local_only: Callable[..., Any],
    format_: str,
    mode: str,
) -> None:
    from alphalattice.protocols.research_authoring.selection import load_safe_yaml_document

    answer = _answer()
    source = tmp_path / "answer.json"
    original = _save(source, answer, "json")
    output = tmp_path / f"reading.{format_}"
    selection = (
        ["--section", "limits.unread"]
        if mode == "section"
        else ["--list-sections"]
        if mode == "list"
        else []
    )
    code, read = _read(
        capsys,
        local_only,
        "--file",
        str(source),
        "--output",
        str(output),
        "--format",
        format_,
        *selection,
    )
    assert (code, read["outcome"]) == (0, "OK"), read
    assert read["output_file"] == str(output.resolve())
    saved = (
        json.loads(output.read_text(encoding="utf-8"))
        if format_ == "json"
        else load_safe_yaml_document(output.read_text(encoding="utf-8"))
    )
    assert saved == {"snapshot": _snapshot(source), "answer": answer}
    assert source.read_bytes() == original
    if mode == "section":
        assert read["data"] == {
            "snapshot": _snapshot(source),
            "section": "limits.unread",
            "value": answer["limits"]["unread"],
        }
    elif mode == "list":
        assert read["data"] == {"snapshot": _snapshot(source), "sections": list(answer)}
    else:
        assert read["data"] == saved


def test_output_collision_preserves_both_files_even_for_a_section_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], local_only: Callable[..., Any]
) -> None:
    source, output = tmp_path / "answer.json", tmp_path / "reading.json"
    _save(source, _answer(), "json")
    output.write_bytes(b"retained output\n")
    before = _tree_bytes(tmp_path)
    code, read = _read(
        capsys,
        local_only,
        "--file",
        str(source),
        "--section",
        "verification",
        "--output",
        str(output),
    )
    assert (code, read["outcome"], read["failure_code"]) == (
        1,
        "INVALID_INPUT",
        "local_client.output_exists_choose_another_path",
    )
    assert _tree_bytes(tmp_path) == before


@pytest.mark.parametrize(
    ("path", "reached", "sections"),
    [
        ("items.0.absent", "items.0.absent", ["items.0.id", "items.0.label"]),
        ("items.99", "items.99", ["items.0", "items.1", "items.2"]),
        ("data.verification", "data", list(_answer())),
    ],
)
def test_unknown_sections_name_the_available_saved_paths_without_live_fallback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    local_only: Callable[..., Any],
    path: str,
    reached: str,
    sections: list[str],
) -> None:
    source = tmp_path / "answer.json"
    _save(source, _answer(), "json")
    before = _tree_bytes(tmp_path)
    code, read = _read(capsys, local_only, "--file", str(source), "--section", path)
    assert (code, read["outcome"], read["failure_code"]) == (
        1,
        "INVALID_INPUT",
        "local_client.saved_answer_section_unknown:" + reached,
    )
    assert read["sections"] == sections
    assert read["detail"] and read["next_action"]
    assert _tree_bytes(tmp_path) == before


@pytest.mark.parametrize(
    "tail",
    [
        pytest.param("²", id="superscript-index"),
        pytest.param("①", id="circled-index"),
        pytest.param("9" * 5000, id="long-index"),
        pytest.param("0" * 5000, id="long-zero-index"),
        pytest.param("9" * 5000 + ":2", id="long-slice-start"),
        pytest.param("1:" + "9" * 5000, id="long-slice-stop"),
        pytest.param("9" * 5000 + ":" + "9" * 5000, id="long-both-bounds"),
        pytest.param("0" * 5000 + ":", id="long-zero-slice-start"),
    ],
)
def test_numeric_section_conversion_failures_have_the_local_refusal(
    tmp_path: Path,
    tail: str,
) -> None:
    """Invalid numeric paths keep one envelope, sibling paths and the no-write boundary."""
    source, output = tmp_path / "answer.json", tmp_path / "reading.json"
    _save(source, _answer(), "json")
    before = _tree_bytes(tmp_path)
    path = "items." + tail
    environment = dict(os.environ, ALPHALATTICE_NETWORK_DISABLED="1", PYTHONDONTWRITEBYTECODE="1")
    for name in ("CODEX_THREAD_ID", "CLAUDE_CODE_SESSION_ID", "PYTHONPATH"):
        environment.pop(name, None)
    ran = subprocess.run(
        [
            sys.executable,
            "-X",
            "int_max_str_digits=4300",
            str(SCRIPT),
            "answer",
            "show",
            "--file",
            str(source),
            "--section",
            path,
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert ran.stderr == ""
    lines = ran.stdout.strip().splitlines()
    assert len(lines) == 1
    read = json.loads(lines[0])
    assert (ran.returncode, read["operation"], read["outcome"], read["failure_code"]) == (
        1,
        "SAVED_ANSWER_SHOW",
        "INVALID_INPUT",
        "local_client.saved_answer_section_unknown",
    )
    assert read["sections"] == ["items.0", "items.1", "items.2"]
    words = cli_contract.client_refusal("local_client.saved_answer_section_unknown:items.99")
    assert (read["detail"], read["next_action"]) == (words.detail, words.next_action)
    assert read["next_requests"] is None
    assert _tree_bytes(tmp_path) == before


@pytest.mark.parametrize(
    ("tail", "selected"),
    [
        pytest.param("1", 1, id="decimal-index"),
        pytest.param("\u0661", 1, id="arabic-decimal-index"),
        pytest.param("\uff11", 1, id="fullwidth-decimal-index"),
        pytest.param("0001", 1, id="zero-padded-index"),
        pytest.param("1:3", slice(1, 3), id="decimal-slice"),
        pytest.param("\u0661:\u0663", slice(1, 3), id="arabic-decimal-slice"),
        pytest.param(":", slice(None), id="open-slice"),
        pytest.param(":" + "9" * 100, slice(None), id="large-valid-stop"),
        pytest.param("9" * 100 + ":", slice(3, None), id="large-valid-start"),
    ],
)
def test_numeric_sections_keep_the_existing_decimal_and_slice_reading(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    local_only: Callable[..., Any],
    tail: str,
    selected: int | slice,
) -> None:
    answer = _answer()
    source = tmp_path / "answer.json"
    _save(source, answer, "json")
    before = _tree_bytes(tmp_path)
    path = "items." + tail
    code, read = _read(capsys, local_only, "--file", str(source), "--section", path)
    assert (code, read["outcome"]) == (0, "OK")
    assert read["data"] == {
        "snapshot": _snapshot(source),
        "section": path,
        "value": answer["items"][selected],
    }
    assert read["next_requests"] is None
    assert _tree_bytes(tmp_path) == before


@pytest.mark.parametrize(
    "payload",
    [
        b'{"status":',
        b"answer:\n  - [unfinished",
        b"",
        b"[]",
        b'"not an answer"',
        b"null",
        b"status: AVAILABLE\n1: value\n",
        b'{"value": NaN}',
        b"loop: &loop\n  child: *loop\n",
        b"first: &shared {kept: true}\nsecond: *shared\n",
        b"long: &value repeated\nitems: [*value, *value]\n",
    ],
    ids=[
        "broken-json",
        "broken-yaml",
        "empty",
        "list",
        "scalar",
        "null",
        "non-string-key",
        "non-finite-number",
        "cyclic-yaml",
        "shared-yaml-container",
        "shared-yaml-scalar",
    ],
)
def test_invalid_saved_documents_are_local_failures_without_writes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    local_only: Callable[..., Any],
    payload: bytes,
) -> None:
    source = tmp_path / "invalid.txt"
    source.write_bytes(payload)
    before = _tree_bytes(tmp_path)
    code, read = _read(capsys, local_only, "--file", str(source))
    assert (code, read["outcome"], read["failure_code"]) == (
        1,
        "INVALID_INPUT",
        "local_client.saved_answer_invalid",
    )
    assert read["detail"] and read["next_action"]
    assert _tree_bytes(tmp_path) == before


def test_a_printed_cli_envelope_is_not_a_full_saved_owner_answer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], local_only: Callable[..., Any]
) -> None:
    printed = envelope(operation="REPORT", outcome="OK", body=_answer(), elapsed_seconds=0.125)
    source = tmp_path / "printed.json"
    _save(source, printed, "json")
    code, read = _read(capsys, local_only, "--file", str(source))
    assert (code, read["failure_code"]) == (1, "local_client.saved_answer_invalid")


def test_an_html_report_names_the_unsupported_saved_format(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], local_only: Callable[..., Any]
) -> None:
    source = tmp_path / "report.html"
    source.write_bytes(b"<html><body>Saved report</body></html>")
    before = _tree_bytes(tmp_path)
    code, read = _read(capsys, local_only, "--file", str(source))
    assert (code, read["outcome"], read["failure_code"]) == (
        1,
        "INVALID_INPUT",
        "local_client.saved_answer_format_unavailable:html",
    )
    assert read["detail"] and read["next_action"]
    assert _tree_bytes(tmp_path) == before


@pytest.mark.parametrize(
    "marker",
    ["representation", "omitted-sections", "root-omitted", "nested-omitted", "nested-keys"],
)
def test_compact_input_is_refused_even_when_the_selected_part_looks_whole(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    local_only: Callable[..., Any],
    marker: str,
) -> None:
    answer = _answer()
    if marker == "representation":
        answer["representation"] = "COMPACT_DISPLAY_NOT_AN_AUTHORITY_DOCUMENT"
    elif marker == "omitted-sections":
        answer["omitted_sections"] = ["items.1:"]
    elif marker == "root-omitted":
        answer["_omitted"] = True
    elif marker == "nested-omitted":
        answer["items"][1]["_omitted"] = True
    else:
        answer["items"][1]["_omitted_keys"] = ["exact_reference"]
    source = tmp_path / "compact.json"
    _save(source, answer, "json")
    before = _tree_bytes(tmp_path)
    code, read = _read(
        capsys, local_only, "--file", str(source), "--section", "verification.verified_at"
    )
    assert (code, read["outcome"], read["failure_code"]) == (
        1,
        "INVALID_INPUT",
        "local_client.saved_answer_incomplete",
    )
    assert read["detail"] and read["next_action"]
    assert _tree_bytes(tmp_path) == before


@pytest.mark.parametrize("kind", ["missing", "unreadable", "over-bound"])
def test_saved_file_failures_keep_the_existing_file_owner_contract(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    local_only: Callable[..., Any],
    kind: str,
) -> None:
    source = tmp_path / "answer.json"
    limit = 4 * 1024 * 1024
    if kind == "unreadable":
        source.write_bytes(b"\xff\xfe not UTF-8")
    elif kind == "over-bound":
        source.write_bytes(b"x" * (limit + 1))
    before = _tree_bytes(tmp_path)
    code, read = _read(capsys, local_only, "--file", str(source))
    expected = {
        "missing": "local_client.document_missing",
        "unreadable": "local_client.document_unreadable",
        "over-bound": "local_client.document_too_large",
    }[kind]
    assert (code, read["outcome"], read["failure_code"]) == (1, "INVALID_INPUT", expected)
    if kind == "over-bound":
        assert read["document_size"] == {"bytes": limit + 1, "limit_bytes": limit}
    else:
        assert read["document_location"]["file"] == str(source)
    assert _tree_bytes(tmp_path) == before


def test_the_saved_reader_accepts_only_json_or_yaml_output_format(tmp_path: Path) -> None:
    source = tmp_path / "answer.json"
    _save(source, _answer(), "json")
    before = _tree_bytes(tmp_path)
    environment = dict(os.environ, ALPHALATTICE_NETWORK_DISABLED="1", PYTHONDONTWRITEBYTECODE="1")
    for name in ("CODEX_THREAD_ID", "CLAUDE_CODE_SESSION_ID", "PYTHONPATH"):
        environment.pop(name, None)
    ran = subprocess.run(
        [sys.executable, str(SCRIPT), "answer", "show", "--file", str(source), "--format", "html"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert ran.returncode == 1
    read = json.loads(ran.stdout.strip().splitlines()[-1])
    assert read["failure_code"] == "local_client.usage_invalid"
    assert _tree_bytes(tmp_path) == before
