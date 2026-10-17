"""`check`: verify the runtime, the image identity, the inputs and their checksums.

Nothing here computes science; it establishes that the computations that follow run on the
pinned environment, on this checkout's code and on intact inputs.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from .common import (EXIT_FAIL, EXIT_INPUT, EXIT_OK, FAIL, INFO, MISSING, PASS, SKIPPED, WARN,
                     Paths, Report, julia_cmd, read_sha256_list, run_logged, sha256_file, tail,
                     which)


# ---------------------------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------------------------
def parse_requirements_lock(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip().rstrip("\\").strip()
        if not line or line.startswith("--"):
            continue
        name, sep, version = line.partition("==")
        if sep:
            pins[name.strip().lower()] = version.split(";")[0].split()[0].strip()
    return pins


def check_python(paths: Paths, report: Report) -> None:
    lock = paths.repl / "environments" / "python" / "requirements.lock"
    report.add("runtime", "python", INFO, generated=sys.version.split()[0])
    if not lock.exists():
        report.add("runtime", "python lockfile", MISSING, expected=str(lock))
        return
    bad = []
    for name, version in parse_requirements_lock(lock).items():
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = "not installed"
        if have != version:
            bad.append(f"{name} {have} (locked {version})")
    if bad:
        report.add("runtime", "python packages vs requirements.lock", FAIL, note="; ".join(bad))
    else:
        report.add("runtime", "python packages vs requirements.lock", PASS,
                   note=f"{len(parse_requirements_lock(lock))} pinned packages")


def _json_lines(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _relay(report: Report, stage: str, records: list[dict]) -> None:
    for r in records:
        if "status" in r and "check" in r:
            report.add(stage, r["check"], r["status"], expected=r.get("expected", ""),
                       generated=r.get("generated", ""), note=r.get("note", ""))


def check_julia(paths: Paths, report: Report) -> None:
    exe = os.environ.get("IJF_JULIA", "julia")
    if not which(exe):
        report.add("runtime", "julia", MISSING, note="julia is not on PATH")
        return
    log = paths.output / "check" / "julia_env.log"
    code, secs = run_logged(julia_cmd(paths) + [str(paths.repl / "scripts" / "checks" / "julia_env.jl")],
                            log, timeout=1800)
    _relay(report, "runtime", _json_lines(log))
    if code != 0:
        report.add("runtime", "julia environment", FAIL,
                   note=f"exit {code} after {secs:.0f}s; see {log}: {tail(log, 3)}")


def check_r(paths: Paths, report: Report) -> None:
    if not which("Rscript"):
        report.add("runtime", "R", MISSING, note="Rscript is not on PATH")
        return
    log = paths.output / "check" / "r_env.log"
    lock = paths.repl / "environments" / "r" / "renv.lock"
    code, secs = run_logged(["Rscript", "--vanilla", str(paths.repl / "scripts" / "checks" / "r_env.R"),
                             str(lock)], log, timeout=600)
    _relay(report, "runtime", _json_lines(log))
    if code != 0:
        report.add("runtime", "R environment", FAIL,
                   note=f"exit {code} after {secs:.0f}s; see {log}: {tail(log, 3)}")


# ---------------------------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------------------------
def compute_fingerprint(paths: Paths) -> str | None:
    bash = which("bash")
    if not bash:
        return None
    out = subprocess.run([bash, str(paths.repl / "scripts" / "fingerprint.sh"), str(paths.repo)],
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def check_identity(paths: Paths, report: Report) -> None:
    own = compute_fingerprint(paths)
    host = os.environ.get("IJF_HOST_FINGERPRINT", "")
    baked = os.environ.get("IJF_IMAGE_FINGERPRINT", "")
    report.add("identity", "source revision", INFO,
               generated=os.environ.get("IJF_SOURCE_REVISION", "unknown"))
    if paths.runtime == "docker":
        if not baked or own != baked:
            report.add("identity", "image content vs its build-time fingerprint", FAIL,
                       expected=baked, generated=own or "",
                       note="the code inside the image differs from what was fingerprinted at build time")
        else:
            report.add("identity", "image content vs its build-time fingerprint", PASS, generated=own)
        if host and host != own:
            status = WARN if os.environ.get("IJF_IMAGE_STATUS") == "UNVERIFIED" else FAIL
            report.add("identity", "image code vs host checkout", status, expected=host,
                       generated=own, note="the image was not built from this checkout")
        elif host:
            report.add("identity", "image code vs host checkout", PASS, generated=own)
    else:
        report.add("identity", "native run (no image)", WARN, generated=own or "unavailable",
                   note="host toolchains are used; results are labelled NATIVE")


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------
def check_committed(paths: Paths, report: Report) -> bool:
    lst = paths.repl / "checksums.sha256"
    if not lst.exists():
        report.add("inputs", "checksums.sha256", MISSING)
        return False
    bad, n = [], 0
    for digest, rel in read_sha256_list(lst):
        n += 1
        f = paths.repl / rel
        binary = f.suffix in (".gz", ".zip", ".png", ".pdf")
        if not f.exists():
            bad.append(f"{rel} (missing)")
        elif sha256_file(f, strip_cr=not binary) != digest:
            bad.append(f"{rel} (checksum mismatch)")
    if bad:
        report.add("inputs", "committed replication files", FAIL, expected=f"{n} files",
                   generated=f"{n - len(bad)} intact", note="; ".join(bad[:20]))
        return False
    report.add("inputs", "committed replication files", PASS, generated=f"{n} files intact")
    return True


def artifact_inventory(paths: Paths) -> list[tuple[str, str]]:
    lst = paths.repl / "artifacts.sha256"
    return read_sha256_list(lst) if lst.exists() else []


def check_artifacts(paths: Paths, report: Report, required: bool = True) -> bool:
    inventory = artifact_inventory(paths)
    if not inventory:
        report.add("inputs", "artifact inventory (artifacts.sha256)", MISSING)
        return False
    missing, corrupt = [], []
    for digest, rel in inventory:
        f = paths.artifacts / rel
        if not f.exists():
            missing.append(rel)
        elif sha256_file(f) != digest:
            corrupt.append(rel)
    if corrupt:
        report.add("inputs", "archived artifact bundle", FAIL, expected=f"{len(inventory)} files",
                   generated=f"{len(corrupt)} corrupted", note="; ".join(corrupt[:20]))
    if missing:
        status = MISSING if required else SKIPPED
        report.add("inputs", "archived artifact bundle", status, expected=f"{len(inventory)} files",
                   generated=f"{len(inventory) - len(missing)} present",
                   note=("missing: " + "; ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
                         + " — run ./reproduce.sh fetch artifacts, or pass --artifacts DIR"))
    if not missing and not corrupt:
        report.add("inputs", "archived artifact bundle", PASS,
                   generated=f"{len(inventory)} files verified against artifacts.sha256")
        return True
    return False


def m4_inventory(paths: Paths) -> list[dict]:
    with open(paths.repl / "data" / "external_files.csv", newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["dataset"] == "m4"]


def check_m4(paths: Paths, report: Report, required: bool, frequencies: set[str] | None = None) -> bool:
    ok = True
    for row in m4_inventory(paths):
        if frequencies and row["frequency"] not in frequencies:
            continue
        f = paths.m4 / row["file"]
        if not f.exists():
            report.add("inputs", f"M4 {row['file']}", MISSING if required else SKIPPED,
                       note="run ./reproduce.sh fetch m4" if required else "needed only by `full`")
            ok = ok and not required
            continue
        digest = sha256_file(f, strip_cr=True)
        if digest != row["sha256"]:
            report.add("inputs", f"M4 {row['file']}", FAIL, expected=row["sha256"], generated=digest,
                       note="not the official M4 file (Mcompetitions/M4-methods)")
            ok = False
        else:
            report.add("inputs", f"M4 {row['file']}", PASS, generated="official M4 content")
    return ok


# ---------------------------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------------------------
def _memory_gib() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1048576
    except OSError:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1073741824
    except (ValueError, OSError, AttributeError):
        return None


def check_resources(paths: Paths, report: Report) -> None:
    paths.output.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(paths.output).free / 1073741824
    mem = _memory_gib()
    report.add("resources", "free disk at output", PASS if free >= 2 else WARN, generated=f"{free:.1f} GiB",
               note="quick reproduction needs < 1 GiB; full M4 campaigns a few GiB")
    report.add("resources", "memory", PASS if (mem or 0) >= 4 else WARN,
               generated=f"{mem:.1f} GiB" if mem else "unknown",
               note="smoke tests need ~3 GiB; each full-run worker ~2 GiB")
    report.add("resources", "CPUs", INFO, generated=str(os.cpu_count()),
               note=f"{platform.system()} {platform.machine()}")


# ---------------------------------------------------------------------------------------------
def run(paths: Paths, args) -> int:
    report = Report("check", paths)
    print("== check: runtime")
    check_python(paths, report)
    check_julia(paths, report)
    check_r(paths, report)
    print("== check: identity")
    check_identity(paths, report)
    print("== check: inputs")
    committed_ok = check_committed(paths, report)
    artifacts_ok = check_artifacts(paths, report, required=True)
    check_m4(paths, report, required=False)
    print("== check: resources")
    check_resources(paths, report)
    out = report.write()
    print(f"check report: {out}")
    if not committed_ok or not artifacts_ok:
        return EXIT_INPUT if not report.failed or _only_input_failures(report) else EXIT_FAIL
    return EXIT_FAIL if report.failed else EXIT_OK


def _only_input_failures(report: Report) -> bool:
    return all(r.stage == "inputs" for r in report.rows if r.status in (FAIL, MISSING))
