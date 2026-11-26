"""U0: open every saved object of harvested workspaces with one source tree.

  probe:   <tree>/scripts/u0_probe.py probe <harvest-root> <work-dir> <out.json> [--keep]
  compare: u0_probe.py compare <old.json> <new.json>

The probe reads with the tree it lives in: it binds that tree's `src` and its internal tests as
its import root, so a card probes its parent with the parent tree's own copy and its own change
with its own. Every research
workspace under <harvest-root> (a directory holding research-workspace.json) is copied to
<work-dir> and opened with the product's own LocalPortfolioWebSession.from_workspace. Then every
saved object is read back through the Host's routes; nothing is planned, run or frozen:
  TASKS; each succeeded research experiment: EXPERIMENT_READBACK, EXPERIMENT_EXPORT,
  EXPERIMENT_DRAFT and EXPERIMENT_REPLAY (which recomputes nothing: it reuses sealed evidence or
  refuses); each feature trial: FEATURE_TRIAL_READBACK; each Portfolio result: REPORT; each frozen
  candidate: FINALIZATION; each published CRO review: EVIDENCE_CRO_EXPORT.
Two harness shapes hold the kinds a research workspace never saves in the suite:
  harness-candidate/<test>/workspace  a Portfolio workspace with frozen candidates, opened the
      way the suite's `live` fixture opened it; FINALIZATION per candidate;
  harness-review/<test>/{evidence,portfolio}  a published Evidence/CRO review, reopened the way
      test_a_published_review_reopens_with_no_provider_credential reopens it (no model authority,
      so a read cannot become work); EVIDENCE_CRO per result and per publication, FINALIZATION.
Each read is OPENS (with method_standing and the evidence disposition when present), REFUSES
<code>, or ERROR <text>; an opened read also records its numbers, each section's numeric leaves
hashed in order with their count (V277). `compare` prints every read whose verdict differs between
two probes, and every read that opens on both whose numbers moved, with the sections that moved.
Each copy keeps its source's hard links and is deleted once its reads are done, and the
work dir goes with the probe (`--keep` keeps it), so a run holds one workspace's copy at a
time instead of both corpora's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# The tree the probe reads with: its source and the internal tests its harness shapes use.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


from alphalattice.control.product_host.composition.saved_object_readback import (  # noqa: E402
    _copy,
    numbers,
    verdict,
)
from alphalattice.control.product_host.composition.saved_object_readback import (  # noqa: E402
    probe as product_probe,
)


def probe(harvest: Path, work: Path, out: Path, *, keep: bool = False) -> None:
    from tests.portfolio_strategy_lab.local_web_support import _resolved, _Resolver

    from alphalattice.control.product_host.composition.local_web_session import (
        LocalPortfolioWebSession,
    )
    from alphalattice.control.product_host.composition.research_workspace import (
        read_research_workspace_manifest,
    )

    def fallback(copy: Path) -> LocalPortfolioWebSession:
        return LocalPortfolioWebSession(
            workspace=copy,
            workspace_manifest=read_research_workspace_manifest(copy),
            resolver=_Resolver(_resolved()),
        )

    product_probe(harvest, work, out, keep=keep, fallback=fallback, harness=probe_harness)


def probe_harness(harvest: Path, work: Path, rows: list[dict[str, Any]]) -> None:
    from tests.alternative_evidence_desk.planted_corpus import _NOW
    from tests.alternative_evidence_desk.review_http_support import (
        _workspace_manifest,
        build_authority,
        start_service,
    )
    from tests.portfolio_strategy_lab.local_web_support import (
        _manifest,
        _request,
        _resolved,
        _Resolver,
    )

    from alphalattice.control.product_host.composition.local_web_session import (
        LocalPortfolioWebSession,
    )

    def reader(live: Any, name: str) -> Any:
        def read(kind: str, key: str, op: str, path: str) -> Any:
            try:
                status, _headers, body = _request(live, path, timeout=120.0)
                rows.append(
                    {
                        "workspace": name,
                        "kind": kind,
                        "key": key,
                        "op": op,
                        "verdict": verdict(status, body),
                        "numbers": numbers(status, body),
                    }
                )
                return json.loads(body) if status == 200 else None
            except Exception as error:
                rows.append(
                    {
                        "workspace": name,
                        "kind": kind,
                        "key": key,
                        "op": op,
                        "verdict": f"ERROR {type(error).__name__}: {str(error)[:120]}",
                    }
                )
                return None

        return read

    def candidates(workspace: Path) -> list[str]:
        return sorted(
            p.stem
            for p in workspace.glob(
                "runtime/artifacts/portfolio-strategy-lab/finalization/candidates/*.json"
            )
        )

    def results(read: Any) -> list[str]:
        body = read("results", "-", "RESULTS", "/api/results") or {}
        return [
            r["result_hash"]
            for r in body.get("results") or []
            if isinstance(r, dict) and r.get("result_hash")
        ]

    for source in sorted((harvest / "harness-candidate").glob("*/workspace")):
        name = f"harness-candidate/{source.parent.name}"
        copy = work / name
        _copy(source, copy)
        session = LocalPortfolioWebSession(
            workspace=copy,
            workspace_manifest=_manifest("qa-local-web"),
            resolver=_Resolver(_resolved()),
        )
        try:
            with session as live:
                read = reader(live, name)
                for rh in results(read):
                    read("result", rh[:12], "REPORT", f"/api/report?result_hash={rh}")
                for ch in candidates(copy):
                    read(
                        "frozen candidate",
                        ch[:12],
                        "FINALIZATION",
                        f"/api/finalization?candidate_hash={ch}",
                    )
        except Exception as error:
            rows.append(
                {
                    "workspace": name,
                    "kind": "workspace",
                    "key": "-",
                    "op": "OPEN",
                    "verdict": f"ERROR {type(error).__name__}: {str(error)[:140]}",
                }
            )

    for source in (
        sorted((harvest / "harness-review").iterdir())
        if (harvest / "harness-review").is_dir()
        else ()
    ):
        name = f"harness-review/{source.name}"
        root = work / name
        _copy(source, root)
        workspace = root / "portfolio" / "workspace"
        try:
            plain = LocalPortfolioWebSession(
                workspace=workspace,
                workspace_manifest=_workspace_manifest("qa-gate-9c5-http"),
                resolver=_Resolver(_resolved()),
                clock=lambda: _NOW,
            )
            with plain as live:
                hashes = results(reader(live, name))
                report = plain.application.report(hashes[0])  # type: ignore[union-attr]
            service = start_service(
                workspace,
                build_authority(tmp_path=root, report=report, model_authority_admitted=False),
                root,
            )
        except Exception as error:
            rows.append(
                {
                    "workspace": name,
                    "kind": "workspace",
                    "key": "-",
                    "op": "OPEN",
                    "verdict": f"ERROR {type(error).__name__}: {str(error)[:140]}",
                }
            )
            continue
        try:
            read = reader(service.session, name)
            for rh in hashes:
                read("review", rh[:12], "EVIDENCE_CRO", f"/api/evidence-cro?result_hash={rh}")
            for publication in sorted(
                root.glob("evidence/artifacts/alternative-evidence/cro-review-publications/*.json")
            ):
                ph = publication.stem
                read(
                    "review",
                    ph[:12],
                    "EVIDENCE_CRO_PUBLICATION",
                    f"/api/evidence-cro?review_publication_hash={ph}",
                )
                read(
                    "review",
                    ph[:12],
                    "EVIDENCE_CRO_EXPORT",
                    f"/api/evidence-cro/export?review_publication_hash={ph}",
                )
            for ch in candidates(workspace):
                read(
                    "frozen candidate",
                    ch[:12],
                    "FINALIZATION",
                    f"/api/finalization?candidate_hash={ch}",
                )
        finally:
            service.session.stop()


def moved_numbers(before: dict[str, str] | None, after: dict[str, str] | None) -> list[str] | None:
    """The sections whose numbers differ between two reads of one object; None when either probe
    recorded none (a tree before V277), so a read is never called unchanged unseen."""

    if before is None or after is None:
        return None
    return sorted(
        section for section in set(before) | set(after) if before.get(section) != after.get(section)
    )


def compare(old: Path, new: Path) -> None:
    def index(path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
        return {
            (r["workspace"], r["kind"], r["key"], r["op"]): r
            for r in json.load(open(path, encoding="utf-8"))["rows"]
        }

    before, after = index(old), index(new)
    keys = sorted(set(before) | set(after))

    def verdict_of(rows: dict[tuple[str, str, str, str], dict[str, Any]], key: Any) -> Any:
        return rows[key]["verdict"] if key in rows else None

    changed = [
        (k, verdict_of(before, k), verdict_of(after, k))
        for k in keys
        if verdict_of(before, k) != verdict_of(after, k)
    ]
    print(f"{len(changed)} reads changed verdict (of {len(keys)})")
    for (workspace, kind, key, op), b, a in changed:
        print(f"  {kind:17} {op:20} {key:14} {b}  ->  {a}   [{workspace[:40]}]")
    same = [
        k
        for k in keys
        if k in before and k in after and before[k]["verdict"] == after[k]["verdict"]
    ]
    moves = {k: moved_numbers(before[k].get("numbers"), after[k].get("numbers")) for k in same}
    unseen = sum(1 for value in moves.values() if value is None)
    numeric = {k: value for k, value in moves.items() if value}
    print(
        f"{len(numeric)} reads with the same verdict moved numbers (of {len(same)}"
        + (f"; {unseen} without numbers on one side" if unseen else "")
        + ")"
    )
    for (workspace, kind, key, op), sections in numeric.items():
        print(f"  {kind:17} {op:20} {key:14} {', '.join(sections)}   [{workspace[:40]}]")


def main(argv: list[str] | None = None) -> None:
    """Run one command: read a harvest with this tree, or compare two probes' reads.

    Args:
        argv: The command line after the program name; the process's own when None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    reading = commands.add_parser("probe", help="read every saved object of a harvest")
    reading.add_argument("harvest", type=Path, help="the directory holding the workspaces")
    reading.add_argument("work", type=Path, help="where each workspace is copied to be read")
    reading.add_argument("out", type=Path, help="the verdicts, as JSON")
    reading.add_argument("--keep", action="store_true", help="keep the work directory")
    comparing = commands.add_parser(
        "compare", help="print every read whose verdict or numbers moved"
    )
    comparing.add_argument("old", type=Path, help="the parent's verdicts")
    comparing.add_argument("new", type=Path, help="the change's verdicts")
    args = parser.parse_args(argv)
    if args.command == "probe":
        probe(args.harvest, args.work, args.out, keep=args.keep)
    else:
        compare(args.old, args.new)


if __name__ == "__main__":
    main()
