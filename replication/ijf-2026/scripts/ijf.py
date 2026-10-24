"""In-container orchestrator of the SARIMAX.jl IJF replication package.

Called by ../../../reproduce.sh inside the replication image (or with --native on a host
that has the toolchains). Every command writes a Markdown/CSV/JSON report under
<output>/reports and exits non-zero on any failure: a failed or skipped computation is never
reported as a success.

    python ijf.py quick              check + smoke + tables
    python ijf.py check | smoke | tables | list
    python ijf.py full --experiment ID [--jobs N] [--resume] [--limit N | --sample N] [--timeout S]
    python ijf.py fetch artifacts|m4|pjme|scip|all [--if-available] [--if-needed] [--from FILE]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ijflib import common  # noqa: E402
from ijflib.common import EXIT_FAIL, EXIT_INPUT, EXIT_OK, EXIT_USAGE  # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="ijf.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("quick", help="check, smoke and tables in sequence")
    sub.add_parser("check", help="verify runtime, image identity, inputs and checksums")
    sub.add_parser("smoke", help="small real computations compared with fixtures")
    t = sub.add_parser("tables", help="rebuild tables from archived results and compare")
    t.add_argument("--group", action="append", help="only this table group (repeatable)")
    sub.add_parser("list", help="experiments and their replication readiness")
    f = sub.add_parser("full", help="re-run experiments from the original input data")
    f.add_argument("--experiment", action="append", default=[])
    f.add_argument("--all", action="store_true")
    f.add_argument("--jobs", type=int, default=1)
    f.add_argument("--resume", action="store_true")
    f.add_argument("--limit", type=int, default=0,
                   help="first N series per task: an execution check, not a scientific result")
    f.add_argument("--sample", type=int, default=0,
                   help="N series drawn at random per frequency (same for every arm): an execution "
                        "check on a spread of series, not a scientific result")
    f.add_argument("--seed", type=int, default=20261018, help="seed of --sample")
    f.add_argument("--timeout", type=float, default=None)
    f.add_argument("--run-id", default=None)
    f.add_argument("--frequency", action="append", help="only these M4 frequencies (repeatable)")
    f.add_argument("--arm", action="append", help="only these arms (repeatable)")
    f.add_argument("--keep-going", action="store_true", help="continue with the next experiment after a failure")
    g = sub.add_parser("fetch", help="download and verify archived inputs")
    g.add_argument("what", nargs="?", default="all", choices=["artifacts", "m4", "pjme", "scip", "all"])
    g.add_argument("--if-available", action="store_true",
                   help="do nothing (successfully) when no download location is recorded yet")
    g.add_argument("--if-needed", action="store_true",
                   help="only what the selected experiments need")
    g.add_argument("--from", dest="source", action="append", default=None,
                   help="use a local copy of a bundle part tarball instead of downloading it (repeatable)")
    g.add_argument("--experiment", action="append", default=[])
    g.add_argument("--all", action="store_true")
    g.add_argument("--frequency", action="append", default=None)
    g.add_argument("--arm", action="append", default=None, help=argparse.SUPPRESS)
    for name in ("--jobs", "--limit", "--sample", "--seed", "--timeout", "--run-id"):
        g.add_argument(name, default=None, help=argparse.SUPPRESS)
    g.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    g.add_argument("--keep-going", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    paths = common.detect_paths()
    paths.output.mkdir(parents=True, exist_ok=True)
    print(f"SARIMAX.jl IJF replication — {args.command} "
          f"(runtime {paths.runtime}, outputs in {paths.output})", flush=True)

    if args.command == "quick":
        from ijflib import check, smoke, tables
        codes = {}
        for name, mod in (("check", check), ("smoke", smoke), ("tables", tables)):
            print(f"\n######## {name} ########", flush=True)
            codes[name] = mod.run(paths, argparse.Namespace(group=None))
        from ijflib.summary import write_summary
        return write_summary(paths, codes)
    if args.command == "check":
        from ijflib import check
        return check.run(paths, args)
    if args.command == "smoke":
        from ijflib import smoke
        return smoke.run(paths, args)
    if args.command == "tables":
        from ijflib import tables
        return tables.run(paths, args)
    if args.command == "list":
        from ijflib import manifest
        return manifest.list_experiments(paths)
    if args.command == "full":
        from ijflib import full
        return full.run(paths, args)
    if args.command == "fetch":
        from ijflib import fetch
        return fetch.run(paths, args)
    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
