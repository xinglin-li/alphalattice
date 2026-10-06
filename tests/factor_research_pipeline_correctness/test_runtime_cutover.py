"""Focused gates for the single active one-session Factor Research runtime."""

from __future__ import annotations

from pathlib import Path


def test_factor_cutover_has_no_cli_writer_or_legacy_mandate_layer() -> None:
    playpen_root = Path(__file__).resolve().parents[2]
    runner = (playpen_root / "scripts" / "run_factor_research_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert "--admit-live-review" not in runner
    assert "--promote-verified-evaluation" not in runner
    assert ".publish_current(" not in runner
    # The front desk's agent terminal, which rendered the confirmation, retired with AG2.
    assert not (playpen_root / "src/alphalattice/interface/front_desk/agent_interface").exists()
