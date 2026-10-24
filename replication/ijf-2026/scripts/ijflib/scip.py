"""The SCIP binaries, which the image does not ship.

SCIP 8.0.0 and SoPlex 6.0 (the artifacts of SCIP_jll and SCIP_PaPILO_jll) are under the ZIB
Academic License, whose terms for redistribution a public container image does not meet. The
image build records their identity in IJF_SCIP_RECORD and removes them in the layer that
installs them (scripts/checks/scip_artifacts.jl); `fetch scip` installs them, on the user's
machine, into the depot IJF_SCIP_DEPOT (a Docker volume that persists between runs) through
Julia's package manager, which verifies each one against its content hash. Sarimax.jl itself
loads SCIP (the default solver of its lower-bounding step), so no Julia computation runs before
they are installed.

Outside the image (the native developer mode) there is no record, and SCIP is whatever the
host's Julia depot holds.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from .common import EXIT_INPUT, EXIT_OK, Paths, julia_cmd, run_logged, tail

RECORD = Path(os.environ.get("IJF_SCIP_RECORD", "/opt/ijf-env/scip-artifacts.toml"))
DEPOT = Path(os.environ.get("IJF_SCIP_DEPOT", "/ijf/julia"))
NOTICE = ("SCIP 8.0.0 and SoPlex 6.0 are distributed under the ZIB Academic License "
          "(https://scipopt.org/academic.txt): free for academic, non-commercial use")
FETCH_HINT = ("the SCIP binaries, which Sarimax.jl loads, are not shipped in the image; "
              "run ./reproduce.sh fetch scip (needs network once)")


def status() -> tuple[bool | None, list[str]]:
    """(installed, missing packages); installed is None when the runtime does not manage SCIP."""
    if not RECORD.exists():
        return None, []
    record = tomllib.loads(RECORD.read_text(encoding="utf-8"))
    depots = [Path(d) for d in os.environ.get("JULIA_DEPOT_PATH", "").split(os.pathsep) if d]
    missing = [a["package"] for a in record["artifact"]
               if not any((d / "artifacts" / a["git-tree-sha1"]).is_dir() for d in depots)]
    return not missing, missing


def fetch(paths: Paths, offline: bool) -> int:
    installed, missing = status()
    if installed is None:
        print("  scip: not managed by this runtime (native mode uses the host's Julia depot)")
        return EXIT_OK
    if installed:
        print("  scip: SCIP 8.0.0 binaries already installed and verified")
        return EXIT_OK
    if offline:
        print(f"  scip: {', '.join(missing)} missing (offline: not downloaded)")
        return EXIT_INPUT
    print(f"  scip: {NOTICE}")
    log = paths.output / "fetch" / "scip.log"
    script = paths.repl / "scripts" / "checks" / "scip_artifacts.jl"
    env = {"JULIA_DEPOT_PATH": os.pathsep.join([str(DEPOT), "/opt/julia-depot"]), "JULIA_PKG_OFFLINE": "false"}
    code, secs = run_logged(julia_cmd(paths) + [str(script), "fetch"], log, timeout=1800, env=env)
    if code != 0 or not status()[0]:
        print(f"  ERROR scip: download or verification failed after {secs:.0f}s; see {log}: {tail(log, 3)}")
        return EXIT_INPUT
    print(f"  scip: installed and verified in {secs:.0f}s ({tail(log, 2)})")
    return EXIT_OK
