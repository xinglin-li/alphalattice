"""Regenerate native usage faults through the bound reader's public owner seam."""

from devtools.evaluation.forms import Scenario, Trace, TraceEvent

FAULTS = (
    "cold_cap",
    "append",
    "response_replacement",
    "file_replacement",
    "file_shrink",
    "unterminated_final_line",
    "cumulative_mismatch",
    "corrupt_checkpoint",
    "same_size_mtime",
)
SCENARIO = Scenario(
    id="native_usage_incremental",
    sentence="Read the bound sessions' usage.",
    now="2026-10-11T00:00:00Z",
    injected_faults=FAULTS,
    expected_route=(("native_usage",),),
    decider="agent",
    forbidden_acts=("read_unbound_session", "retain_transcript"),
    ask_required=False,
    owner="native_usage",
    starting_state="Bound synthetic native sessions, regenerated on the current source.",
)
STEPS = (FAULTS[0], "resume", "unchanged", *FAULTS[1:])


def walk(read, mutate):
    """Apply declared mutations and retain only the reader's sanitized facts."""
    events = []
    for step in STEPS:
        mutate(step)
        facts = {"alphalattice.native_usage": read(), "alphalattice.mutation": step}
        events.append(TraceEvent(SCENARIO.now, SCENARIO.now, facts, 1))
    return SCENARIO, Trace(tuple(events))
