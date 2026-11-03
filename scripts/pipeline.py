"""The pipeline's command over its one definition, config/pipeline.toml.

``show`` lists the stages in order, or one stage's rows: what each proves, its kind, owner and
hosts. ``run`` runs one stage on this host into its own folder and prints its receipt's summary;
it exits 0 only when the stage passed. ``verify`` prints a stage receipt's summary as JSON, or
refuses it. ``health`` prints the day's test-health report from the checkout and the commit-stage
receipts it is given.
"""

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from devtools.pipeline.definition import Definition, load  # noqa: E402
from devtools.pipeline.health import report  # noqa: E402
from devtools.pipeline.receipt import verify  # noqa: E402
from devtools.pipeline.reuse import Options  # noqa: E402
from devtools.pipeline.stage import run  # noqa: E402


def show(definition: Definition, name: str | None) -> str:
    """The stages in order, or one stage's rows in the order it runs them."""
    if name is None:
        lines = [
            f"{s.order} {s.id:<12} {s.trigger:<11} {s.bound_minutes:>4g} min  {' '.join(s.rows)}"
            for s in definition.stages.values()
        ]
        return "\n".join([f"config/pipeline.toml sha256:{definition.sha256[:12]}", *lines])
    stage = definition.stage(name)
    head = (
        f"{stage.id}: {stage.reason} ({stage.trigger}, {stage.range}, {stage.bound_minutes:g} min"
    )
    lines = [head + (f", lane {stage.lane} {'/'.join(stage.sizes)})" if stage.lane else ")")]
    for row in (definition.rows[r] for r in stage.rows):
        lines.append(
            f"  {row.id:<20} {row.kind:<13} {','.join(row.hosts):<16} {row.owner}: {row.reason}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    showing = commands.add_parser("show", help="the stages in order, or one stage's rows")
    showing.add_argument("stage", nargs="?", help="a stage, such as commit")
    running = commands.add_parser("run", help="run one stage into its own folder")
    running.add_argument("stage", help="a stage, such as commit")
    running.add_argument("--out", type=Path, required=True, help="a new folder for this run")
    running.add_argument("--host", choices=("workshop", "public"), default="workshop")
    running.add_argument("--base", default="develop", help="the branch or commit it is based on")
    running.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    running.add_argument("--plugin", type=Path, action="append", default=[], help="a host's rows")
    running.add_argument(
        "--required-tests", nargs="+", default=[], help="required cost-card nodes, Large included"
    )
    running.add_argument("--line", default="pipeline", help="the cache writer line")
    running.add_argument("--result-cache", help="verified result store; absent disables it")
    running.add_argument("--cache", choices=("off", "record", "shadow", "on"))
    running.add_argument("--result-cache-bytes", help="positive derived-store byte cap")
    checking = commands.add_parser("verify", help="a receipt's summary, refused unless complete")
    checking.add_argument("receipt", type=Path)
    for name in ("stage", "host", "head"):
        checking.add_argument(f"--{name}", default="", help=f"refuse a receipt of another {name}")
    checking.add_argument("--passing", action="store_true", help="refuse one that did not pass")
    checking.add_argument(
        "--required-tests", nargs="+", default=[], help="passing nodes or families"
    )
    checking.add_argument("--within-limits", action="store_true", help="require bounded test times")
    checking.add_argument(
        "--cost-standing", action="store_true", help="require verified measured cost standing"
    )
    checking.add_argument(
        "--cost-if-present",
        action="store_true",
        help="verify cost standing when the receipt carries it",
    )
    checking.add_argument(
        "--cost-nodes", nargs="+", default=[], help="require standing for these exact nodes"
    )
    checking.add_argument(
        "--cost-root", type=Path, help="Git checkout holding source and target commits"
    )
    checking.add_argument(
        "--cost-source",
        nargs=2,
        action="append",
        default=[],
        metavar=("SHA256", "PATH"),
        help="digest and retained source receipt, repeat for distinct sources",
    )
    health = commands.add_parser("health", help="the day's test-health report as JSON")
    health.add_argument("receipts", type=Path, nargs="*", help="commit-stage receipts")
    args = parser.parse_args(argv)
    if args.command == "run" and args.workers < 1:
        parser.error("--workers is at least 1")
    try:
        if args.command == "run":
            summary = run(
                ROOT,
                args.stage,
                host=args.host,
                out=args.out.resolve(),
                base=args.base,
                workers=args.workers,
                paths=args.plugin,
                command=["scripts/pipeline.py", *(sys.argv[1:] if argv is None else argv)],
                required_tests=args.required_tests,
                cache_options=Options.read(args.result_cache, args.cache, args.result_cache_bytes),
                line=args.line,
            )
            print(json.dumps(summary, indent=2))
            return 0 if summary["result"] == "PASS" else 1
        if args.command == "verify":
            asked = vars(args).copy()
            path = asked.pop("receipt")
            asked.pop("command")
            pairs = asked.pop("cost_source")
            if pairs and not (
                asked["cost_standing"] or asked["cost_if_present"] or asked["cost_nodes"]
            ):
                raise ValueError("cost sources require a cost standing request")
            sources = {}
            for digest, source in pairs:
                if not re.fullmatch(r"[0-9a-f]{64}", digest) or digest in sources:
                    raise ValueError("cost source digest is malformed or duplicate")
                sources[digest] = Path(source)
            asked["cost_sources"] = sources or None
            print(json.dumps(verify(path, **asked), indent=2))
        elif args.command == "health":
            print(json.dumps(report(ROOT, args.receipts, date.today()), indent=2))
        else:
            print(show(load(ROOT), args.stage))
    except ValueError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
