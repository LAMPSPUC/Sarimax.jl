"""Check the archived 2026-10-21 re-run of the August 2026 certification batch.

results/reruns/certification-2026-08-rerun-2026-10-21/ holds the re-run the batch's author made
of its 46 instances with the batch's harness and environment (RUN.md). This script verifies,
without the artifact bundle:

* every archived file of received/ against received/SHA256SUMS;
* that every file is byte-identical to what was received (received/SHA256SUMS.received, the
  author's list), except the logs listed in received_transforms.csv, whose received hash is the
  one that list gives;
* that the instances the re-run fitted (received/instancias/) are the package's
  (experiments/configs/certification_2026_08_instances.csv): the synthetic series value for
  value, the three M4 series by identifier and length;
* that raw/certification-2026-08/records.csv follows from the received files
  (import_certification_rerun_2026_10.py), and comparison.csv and summary.md from the records
  and the batch's record (compare_certification_2026_08.py).

It re-runs no solver. Exit 0 when everything agrees, 1 otherwise.
"""

from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPL = Path(__file__).resolve().parents[2]
RUN = REPL / "results" / "reruns" / "certification-2026-08-rerun-2026-10-21"
REC = RUN / "received"
PREFIX = "scip_rerun_replicacao_21-10/"
INSTANCES = REPL / "experiments" / "configs" / "certification_2026_08_instances.csv"
GLOBAL = REPL / "experiments" / "runners" / "global"
OUTPUTS = ["records.csv", "comparison.csv", "summary.md"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sums(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        h, name = line.split(maxsplit=1)
        out[name.lstrip("*")] = h
    return out


def check_files(problems: list[str]) -> int:
    archived = sums(REC / "SHA256SUMS")
    for name, h in archived.items():
        p = REC / name
        if not p.exists() or sha(p) != h:
            problems.append(f"received/{name}: missing or SHA-256 differs from received/SHA256SUMS")
    on_disk = {str(p.relative_to(REC)).replace(os.sep, "/") for p in REC.rglob("*") if p.is_file()}
    extra = on_disk - set(archived) - {"SHA256SUMS", "SHA256SUMS.received"}
    if extra:
        problems.append(f"received/ holds files that received/SHA256SUMS does not list: {sorted(extra)}")
    with open(RUN / "received_transforms.csv", newline="", encoding="utf-8") as fh:
        changed = {r["file"].removeprefix("received/"): r for r in csv.DictReader(fh)}
    received = {k.removeprefix(PREFIX): v for k, v in sums(REC / "SHA256SUMS.received").items()}
    received.pop("SHA256SUMS", None)  # the author's list hashes itself
    if set(received) != set(archived):
        problems.append("the archived files are not the received ones")
    for name, h in received.items():
        if name in changed:
            t = changed[name]
            if t["sha256_received"] != h or t["sha256_archived"] != archived.get(name):
                problems.append(f"received/{name}: the transform record does not match the hashes")
        elif archived.get(name) != h:
            problems.append(f"received/{name}: changed but not listed in received_transforms.csv")
    return len(archived)


def check_instances(problems: list[str]) -> int:
    n = 0
    for c in csv.DictReader(open(INSTANCES, newline="", encoding="utf-8")):
        p = REC / "instancias" / f"caso_{c['id']}.csv"
        if not p.exists():
            continue  # ids 34, 35, 38 were killed before the re-run wrote their series
        lines = p.read_text(encoding="utf-8").splitlines()
        y = [float(x) for x in lines[2:] if x.strip()]
        head = dict(kv.split("=", 1) for kv in lines[0].lstrip("# ").split() if "=" in kv)
        same = head.get("cel") == c["cell"] and head.get("sid") == c["sid"] and head.get("modo") == c["mode"]
        if c["y"]:
            same &= y == [float(x) for x in c["y"].split(";")]
        else:
            same &= int(head.get("T", -1)) == len(y)  # M4 series: standardized at run time
        if not same:
            problems.append(f"instance {c['id']}: the re-run's series is not the package's")
        n += 1
    return n


def check_outputs(problems: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / RUN.name  # the summary is titled with the run's name
        env = dict(os.environ, IJF_RUN_DIR=str(out))
        for cmd in ([sys.executable, "-I", str(GLOBAL / "import_certification_rerun_2026_10.py"), str(RUN), str(out)],
                    [sys.executable, "-I", str(GLOBAL / "compare_certification_2026_08.py")]):
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if r.returncode != 0:
                problems.append(f"{Path(cmd[2]).name} exited {r.returncode}: {r.stdout[-300:]}{r.stderr[-300:]}")
                return
        for name in OUTPUTS:
            a, b = RUN / "raw" / "certification-2026-08" / name, out / "raw" / "certification-2026-08" / name
            if not a.exists() or a.read_bytes() != b.read_bytes():
                problems.append(f"raw/certification-2026-08/{name} does not follow from the received files")


def main() -> int:
    problems: list[str] = []
    files = check_files(problems)
    inst = check_instances(problems)
    check_outputs(problems)
    for p in problems:
        print("FAIL", p)
    print(f"re-run 2026-10-21: {files} files checked; {inst} instances identical to the package's; "
          f"records, comparison and summary: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
