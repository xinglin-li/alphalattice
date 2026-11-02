"""The pipeline's command over its one definition, config/pipeline.toml.

``show`` lists the stages in order, or one stage's rows: what each proves, its kind, owner and
hosts. ``run`` runs one stage on this host into its own folder and prints its receipt's summary;
it exits 0 only when the stage passed. ``verify`` prints a stage receipt's summary as JSON, or
refuses it.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from devtools.pipeline.definition import Definition, DefinitionError, load  # noqa: E402
from devtools.pipeline.receipt import verify  # noqa: E402
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
    checking = commands.add_parser("verify", help="a receipt's summary, refused unless complete")
    checking.add_argument("receipt", type=Path)
    for name in ("stage", "host", "head"):
        checking.add_argument(f"--{name}", default="", help=f"refuse a receipt of another {name}")
    checking.add_argument("--passing", action="store_true", help="refuse one that did not pass")
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
            )
            print(json.dumps(summary, indent=2))
            return 0 if summary["result"] == "PASS" else 1
        if args.command == "verify":
            asked = {key: getattr(args, key) for key in ("stage", "host", "head", "passing")}
            print(json.dumps(verify(args.receipt, **asked), indent=2))
        else:
            print(show(load(ROOT), args.stage))
    except DefinitionError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
