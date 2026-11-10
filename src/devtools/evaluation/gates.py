"""The native release gates as one Inspect AI task over the shared native leg.

Each sample is a host. Its solver runs that host's one prepared native leg (the native runner's
own start and wait, outside every Windows Job), and its scorer reads native-<host>,
resource-budget and, for Claude Code with a previous release, upgrade-axis from that one leg by
the release lane's own rules. The scenario person's questions, each recorded when it was asked,
join the sample's transcript as events. No model is called here (Inspect's ``none`` model): the
hosts under test run their scenario's own models inside the leg.

    python -m devtools.evaluation.gates --source SHA --run RUN [--previous 0.2.3] [--full]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai._util import appdirs  # Inspect's traces and sample buffer stay in the run
from inspect_ai.dataset import Sample
from inspect_ai.log import transcript
from inspect_ai.scorer import Score, accuracy, scorer
from inspect_ai.solver import solver
from inspect_ai.util import subprocess

from devtools.evaluation.leg_report import gate_verdicts
from devtools.evaluation.scenario_person import EVENTS, asking

ROOT = Path(__file__).resolve().parents[3]
LEG = "scripts.agent_eval.runners.native_leg"
HOSTS = ("claude-code", "codex")
WAIT_SECONDS = 14_400


@solver
def native_leg(source: str, previous: str | None, smoke: bool, leg: tuple[str, ...] = ()) -> Any:
    async def solve(state: Any, generate: Any) -> Any:
        host, run = state.metadata["host"], state.metadata["run"]
        command = list(leg or (sys.executable, "-m", LEG))  # the native runner, or a stand-in
        start = [*command, "start", "--run", run, "--host", host]
        start += ["--source", source, *(["--smoke"] if smoke else [])]
        start += ["--previous", previous] if previous and host == "claude-code" else []
        for argv in (start, [*command, "wait", "--run", run]):
            done = await subprocess(argv, cwd=str(ROOT), timeout=WAIT_SECONDS)
            said = dict(
                step=argv[len(command)],
                exit=done.returncode,
                said=(done.stdout + done.stderr)[-800:],
            )
            transcript().info(said, source="native_leg")
        for path in sorted(Path(run).joinpath("fresh").rglob(EVENTS)):
            for line in path.read_text(encoding="utf-8").splitlines():
                transcript().info(json.loads(line), source="scenario_person")
        return state

    return solve


@scorer(metrics=[accuracy()])
def gates(source: str) -> Any:
    async def score(state: Any, target: Any) -> Score:
        run = Path(state.metadata["run"])
        if not (run / "leg.json").is_file():  # its start refused: every gate fails, by name
            return Score(value="I", answer="native_leg.not_started", metadata=dict(verdicts={}))
        verdicts = gate_verdicts(run, source)
        evidence = dict(leg=str(run / "leg.json"), summary=str(run / "acceptance-summary.json"))
        return Score(
            value="C" if all(verdicts.values()) else "I",
            answer=json.dumps(verdicts, sort_keys=True),
            metadata=dict(verdicts=verdicts, evidence=evidence, asking=asking(run / "fresh")),
        )

    return score


def release_gates(
    source: str,
    run: Path,
    previous: str | None = "0.2.3",
    smoke: bool = True,
    leg: tuple[str, ...] = (),
) -> Task:
    samples = [
        Sample(
            id=host,
            input=f"The native leg of {host}.",
            metadata=dict(host=host, run=str(run / host)),
        )
        for host in HOSTS
    ]
    return Task(
        dataset=samples, solver=native_leg(source, previous, smoke, leg), scorer=gates(source)
    )


def run_gates(source: str, run: Path, **options: Any) -> dict[str, Any]:
    """Run both hosts' legs at once and return each gate's verdict and evidence."""
    # contract: replace with Inspect's own setting for its data directory once it has one
    kept = appdirs.user_data_path, appdirs.user_cache_path
    appdirs.user_data_path = lambda name: run / "inspect-data" / name
    appdirs.user_cache_path = lambda name: run / "inspect-cache" / name
    try:
        (log,) = inspect_eval(
            release_gates(source, run, **options),
            model="none",
            log_dir=str(run / "inspect"),
            display="none",
        )
    finally:
        appdirs.user_data_path, appdirs.user_cache_path = kept
    found = {
        str(sample.id): next(iter((sample.scores or {}).values())).metadata
        for sample in log.samples or []
        if sample.scores
    }
    result = dict(status=log.status, source=source, log=log.location, hosts=found)
    (run / "gates.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    parser.add_argument("--source", required=True, help="the exact develop SHA under test")
    parser.add_argument("--run", required=True, type=Path, help="an owned run")
    parser.add_argument(
        "--previous", default="0.2.3", help="the release the upgrade axis starts from"
    )
    parser.add_argument("--full", action="store_true", help="the full scenario, not the smoke")
    args = parser.parse_args(argv)
    result = run_gates(args.source, args.run.resolve(), previous=args.previous, smoke=not args.full)
    print(json.dumps(result, indent=1, default=str))
    expected = {
        "claude-code": {"native-claude-code", "resource-budget"},
        "codex": {"native-codex", "resource-budget"},
    }
    if args.previous:
        expected["claude-code"].add("upgrade-axis")
    hosts = result["hosts"]
    complete = set(hosts) == set(expected) and all(
        set(hosts[host]["verdicts"]) == names
        and all(value is True for value in hosts[host]["verdicts"].values())
        for host, names in expected.items()
    )
    return 0 if result["status"] == "success" and complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
