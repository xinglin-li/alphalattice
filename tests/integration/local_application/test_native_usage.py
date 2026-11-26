"""Native binding reads only bounded session headers, never conversation usage."""

import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import pytest

from alphalattice.interface.local_application.native_bridge import pin_differs
from alphalattice.interface.local_application.native_usage import ModelUsage

TEST_SOURCE = "src/alphalattice/interface/local_application/native_usage.py"


def test_a_codex_thread_names_its_spawn_in_its_first_record_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Codex thread names its spawn in its first record alone."""

    from alphalattice.interface.local_application import native_usage
    from alphalattice.interface.local_application.native_usage import codex_thread_spawn

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    day = tmp_path / "sessions" / "2026" / "10" / "03"
    day.mkdir(parents=True)
    lead, child, other = (str(uuid4()) for _ in range(3))

    def rollout(thread: str, *records: dict[str, object]) -> None:
        name = f"rollout-2026-10-03T00-00-00-{thread}.jsonl"
        (day / name).write_text("".join(json.dumps(r) + "\n" for r in records), "utf-8")

    spawn = {"parent_thread_id": lead, "agent_role": "alphalattice_cro", "agent_path": "SECRET"}
    source = {"subagent": {"thread_spawn": spawn}}
    meta = {"id": child, "base_instructions": "SECRET", "source": source}
    rollout(child, {"type": "session_meta", "payload": meta}, {"payload": {"text": "SECRET"}})
    rollout(lead, {"type": "session_meta", "payload": {"id": lead, "source": "cli"}})
    found = codex_thread_spawn(child)
    assert found is not None
    assert asdict(found) == {"parent_thread_id": lead, "agent_role": "alphalattice_cro"}
    assert codex_thread_spawn(lead) is None and codex_thread_spawn(other) is None
    # A first record naming another thread is not this thread's.
    rollout(other, {"type": "session_meta", "payload": {**meta, "id": lead}})
    assert codex_thread_spawn(other) is None
    monkeypatch.setattr(native_usage, "FIRST_RECORD_BYTES", 64)
    assert codex_thread_spawn(child) is None


@pytest.mark.parametrize(
    ("pins", "model", "efforts", "differs"),
    [
        (
            {"role_model": "claude-sonnet-5-5", "role_effort": "high"},
            "claude-sonnet-5-5",
            ("high",),
            [],
        ),
        ({"role_model": "claude-sonnet-5-5"}, "claude-sonnet-5", ("high",), ["model"]),
        ({"role_model": "sonnet"}, "claude-sonnet-5-5", ("high",), []),
        ({"role_model": "sonnet"}, "claude-opus-5-5", ("high",), ["model"]),
        ({"role_model": "gpt-6-luna", "role_effort": "xhigh"}, "gpt-6-luna", ("high",), ["effort"]),
        ({"role_model": "inherit", "role_effort": "inherit"}, "claude-opus-5-5", ("max",), []),
        ({}, "claude-opus-5-5", ("max",), []),
    ],
    ids=[
        "exact",
        "different-id",
        "matching-alias",
        "different-alias",
        "different-effort",
        "inherit",
        "unconfigured",
    ],
)
def test_a_reading_differs_from_its_card_by_an_exact_id_an_alias_or_an_effort(
    pins, model, efforts, differs
):
    reading = ModelUsage(model, efforts, 1, 1, 0, 0, 1, None, None)
    assert pin_differs(pins, reading) == differs
