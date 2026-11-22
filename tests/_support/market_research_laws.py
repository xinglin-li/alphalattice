"""The research laws shared by the store and daily provider contract suites."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from uuid import UUID

import pyarrow as pa

from alphalattice.kernel.data_layer import create_data_root
from alphalattice.kernel.data_layer.chain import canonical
from alphalattice.kernel.data_layer.log import HostLog
from alphalattice.kernel.data_layer.store import as_columns, stage

LISTING = str(UUID(int=7))
SOURCE = "contract_fixture"
PIT = (
    "SELECT session_date, close, version_id, admission_id, epoch_us(known_asof) AS known_us "
    "FROM raw_bar_asof(?) WHERE listing_id = ? ORDER BY session_date, version_id"
)


@dataclass(frozen=True)
class Readbacks:
    before: pa.Table
    initial: pa.Table
    duplicate: pa.Table
    before_correction: pa.Table
    at_correction: pa.Table
    after_restart: tuple[pa.Table, pa.Table]
    first_close: float
    restated_close: float
    known: int
    corrected: int
    duplicate_rows: int


def store_readbacks(root: Path, open_store, bar: dict) -> Readbacks:
    """Admit one price, its exact duplicate, then a later restatement of that session."""
    create_data_root(root)
    first = bar | {
        "listing_id": LISTING,
        "source_observed_at": "2026-07-01 23:00:00+00",
    }
    restated = first | {"close": first["close"] + 1, "source_observed_at": "2026-07-02 23:00:00+00"}
    with HostLog.open(root) as log:
        definition = {"family": "provider", "name": SOURCE, "version": 1}
        log.record(
            "definition.registered",
            ("definition", f"provider/{SOURCE}"),
            definition,
            text=canonical(definition | {"text": "The generated contract bars."}).decode(),
        )
        provider_id = log.require_lineage("provider", SOURCE)
        with closing(open_store(root, log)) as port:

            def admit(values):
                rows = pa.table(as_columns([values]))
                with stage(root, "raw_bar", provider_id, rows) as staged:
                    committed = port.commit(staged)
                return committed.rows, staged.admission_id

            def read(when):
                return port.lake.execute(PIT, (when, LISTING)).to_arrow_table()

            def known_at(admission_id):
                return log.read(
                    "SELECT known_us FROM admission WHERE admission_id = ?", (admission_id,)
                )[0][0]

            _, first_id = admit(first)
            known = known_at(first_id)
            before, initial = read(known - 1), read(known)
            repeated_rows, _ = admit(first)
            duplicate = read(log.clock())
            _, correction_id = admit(restated)
            corrected = known_at(correction_id)
            before_correction, at_correction = read(corrected - 1), read(corrected)
    with HostLog.open(root) as log, closing(open_store(root, log)) as port:
        restart = tuple(
            port.lake.execute(PIT, (when, LISTING)).to_arrow_table() for when in (known, corrected)
        )
    return Readbacks(
        before,
        initial,
        duplicate,
        before_correction,
        at_correction,
        restart,
        first["close"],
        restated["close"],
        known,
        corrected,
        repeated_rows,
    )


def assert_provider_research_laws(whole, cutoff: date, held, replay) -> None:
    """Compare bounded and reopened observations with the provider's own full read."""
    held = list(held)
    expected = [row for row in whole if date.fromisoformat(str(row["session_date"])) <= cutoff]
    assert held == expected, "provider-session-cutoff"
    assert list(replay) == held, "provider-replay-all-equal"


def known_time(seen: Readbacks) -> bool:
    return (
        seen.before.num_rows == 0
        and seen.initial["close"].to_pylist() == [seen.first_close]
        and seen.initial["known_us"].to_pylist() == [seen.known]
    )


def no_lookahead(seen: Readbacks) -> bool:
    return (
        seen.corrected > seen.known
        and seen.before_correction.equals(seen.initial)
        and seen.at_correction["close"].to_pylist() == [seen.restated_close]
        and seen.at_correction["known_us"].to_pylist() == [seen.corrected]
        and seen.initial["version_id"].to_pylist() != seen.at_correction["version_id"].to_pylist()
    )


def evidence_identity(seen: Readbacks) -> bool:
    return seen.duplicate_rows == 0 and seen.duplicate.equals(seen.initial)


def equal_replay(seen: Readbacks) -> bool:
    return all(
        actual.equals(expected, check_metadata=True)
        for actual, expected in zip(
            seen.after_restart, (seen.initial, seen.at_correction), strict=True
        )
    )


# Each double changes just the observation this law must refuse. The same checks run on the ports.
RESEARCH_CASES = (
    ("known-time", known_time, lambda s: replace(s, initial=s.before)),
    ("no-lookahead", no_lookahead, lambda s: replace(s, before_correction=s.at_correction)),
    ("same-evidence-identity", evidence_identity, lambda s: replace(s, duplicate=s.at_correction)),
    ("replay-all-equal", equal_replay, lambda s: replace(s, after_restart=(s.initial, s.initial))),
)


def assert_research_laws(seen: Readbacks) -> None:
    for name, law, _ in RESEARCH_CASES:
        assert law(seen), name
