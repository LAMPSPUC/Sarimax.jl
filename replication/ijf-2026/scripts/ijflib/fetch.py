"""`fetch`: download and verify the archived artifact bundle and the third-party data.

* artifacts - the Zenodo data record (location and SHA-256 in artifacts.lock); `--from FILE`
              uses a local copy of the tarball instead. Every extracted file is verified
              against artifacts.sha256 before the bundle is marked complete.
* m4        - the official M4 files at a pinned commit of Mcompetitions/M4-methods
              (data/external_files.csv). M4 carries no licence, so it is fetched rather than
              redistributed; the SHA-256 proves the content is the official one.
* pjme      - the PJM hourly load file of the July 2026 battery, from this repository's
              history at a pinned commit (licence not established; not redistributed).

Downloads are versioned (pinned commits or a versioned DOI record) and checksummed; a file
whose checksum does not match is deleted and reported, never used.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import tarfile
import tempfile
import tomllib
import urllib.request
from pathlib import Path

from .check import artifact_inventory
from .common import EXIT_FAIL, EXIT_INPUT, EXIT_OK, Paths, sha256_file

USER_AGENT = "sarimax-ijf-replication/1 (+https://github.com/LAMPSPUC/Sarimax.jl)"


def _download(url: str, dest: Path, expected: str | None, strip_cr: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f"  downloading {url}", flush=True)
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    digest = sha256_file(tmp, strip_cr=strip_cr)
    if expected and digest != expected:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{dest.name}: SHA-256 {digest} does not match the pinned {expected}")
    os.replace(tmp, dest)
    print(f"  verified {dest.name} ({dest.stat().st_size} bytes, sha256 {digest[:16]}...)", flush=True)


def external_files(paths: Paths, dataset: str) -> list[dict]:
    with open(paths.repl / "data" / "external_files.csv", newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["dataset"] == dataset]


def fetch_external(paths: Paths, dataset: str, frequencies: set[str] | None, offline: bool) -> int:
    target = paths.m4 if dataset == "m4" else paths.pjme
    code = EXIT_OK
    for row in external_files(paths, dataset):
        if frequencies and row["frequency"] not in frequencies:
            continue
        dest = target / row["file"]
        if dest.exists() and sha256_file(dest, strip_cr=True) == row["sha256"]:
            print(f"  {dataset}: {row['file']} already present and verified")
            continue
        if offline:
            print(f"  {dataset}: {row['file']} missing (offline: not downloaded)")
            code = EXIT_INPUT
            continue
        try:
            _download(row["url"], dest, row["sha256"], strip_cr=True)
        except (OSError, ValueError) as exc:
            print(f"  ERROR {dataset} {row['file']}: {exc}")
            code = EXIT_INPUT
    return code


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> int:
    """Extract regular files only, stripping the top-level directory; refuses path traversal."""
    n = 0
    root = dest.resolve()
    for m in tar.getmembers():
        if not m.isfile():
            continue
        parts = Path(m.name).parts[1:]
        if not parts or ".." in parts or Path(m.name).is_absolute():
            raise ValueError(f"unsafe path in bundle: {m.name}")
        out = (dest / Path(*parts)).resolve()
        if root not in out.parents:
            raise ValueError(f"unsafe path in bundle: {m.name}")
        out.parent.mkdir(parents=True, exist_ok=True)
        with tar.extractfile(m) as src, open(out, "wb") as f:
            shutil.copyfileobj(src, f, length=1 << 20)
        n += 1
    return n


def fetch_artifacts(paths: Paths, source: str | None, if_available: bool, offline: bool) -> int:
    lock = tomllib.loads((paths.repl / "artifacts.lock").read_text(encoding="utf-8"))["bundle"]
    marker = paths.artifacts / ".complete"
    inventory = artifact_inventory(paths)
    if marker.exists() and marker.read_text().strip() == lock.get("inventory_sha256", ""):
        print("  artifact bundle already present and verified")
        return EXIT_OK
    with tempfile.TemporaryDirectory() as td:
        if source:
            tarball = Path(source)
            if not tarball.exists():
                print(f"  ERROR: {tarball} does not exist")
                return EXIT_INPUT
        else:
            if not lock.get("url"):
                msg = ("  the artifact bundle has no published location yet (artifacts.lock: url is "
                       "empty; the Zenodo record is created at release time). Provide a local copy with "
                       "`./reproduce.sh fetch artifacts --from FILE` or `--artifacts DIR`.")
                print(msg)
                return EXIT_OK if if_available else EXIT_INPUT
            if offline:
                print("  artifact bundle missing (offline: not downloaded)")
                return EXIT_INPUT
            tarball = Path(td) / "bundle.tar.gz"
            try:
                _download(lock["url"], tarball, lock.get("sha256") or None)
            except (OSError, ValueError) as exc:
                print(f"  ERROR: {exc}")
                return EXIT_INPUT
        if lock.get("sha256") and sha256_file(tarball) != lock["sha256"]:
            print(f"  ERROR: {tarball} does not match the SHA-256 recorded in artifacts.lock")
            return EXIT_INPUT
        print(f"  extracting {tarball.name} into {paths.artifacts}", flush=True)
        paths.artifacts.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball, "r:gz") as tar:
            n = _safe_extract(tar, paths.artifacts)
    bad = [rel for digest, rel in inventory
           if not (paths.artifacts / rel).exists() or sha256_file(paths.artifacts / rel) != digest]
    if bad:
        print(f"  ERROR: {len(bad)} of {len(inventory)} files missing or corrupted after extraction, e.g. {bad[:3]}")
        return EXIT_INPUT
    marker.write_text(lock.get("inventory_sha256", "") + "\n")
    print(f"  artifact bundle verified: {n} files, all {len(inventory)} inventory checksums match")
    return EXIT_OK


def run(paths: Paths, args) -> int:
    offline = os.environ.get("IJF_OFFLINE", "0") == "1"
    what = args.what
    codes = []
    if what in ("artifacts", "all"):
        print("== fetch: artifact bundle")
        codes.append(fetch_artifacts(paths, args.source, args.if_available, offline))
    if what in ("m4", "all"):
        freqs = set(args.frequency or [])
        if args.if_needed:
            freqs |= _frequencies_needed(paths, args)
            if not freqs:
                print("== fetch: M4 not needed by the selected experiments")
        if not args.if_needed or freqs:
            print("== fetch: M4 data")
            codes.append(fetch_external(paths, "m4", freqs or None, offline))
    if what in ("pjme", "all") or (args.if_needed and _needs_pjme(args)):
        print("== fetch: PJME data")
        codes.append(fetch_external(paths, "pjme", None, offline))
    return max(codes) if codes else EXIT_OK


def _targets(paths: Paths, args) -> list[str]:
    from .manifest import load_manifest
    rows = load_manifest(paths)
    if getattr(args, "all", False):
        return [r["experiment_id"] for r in rows if r["full_runner"] not in ("", "none")]
    return list(getattr(args, "experiment", []) or [])


def _frequencies_needed(paths: Paths, args) -> set[str]:
    from .manifest import load_manifest
    rows = {r["experiment_id"]: r for r in load_manifest(paths)}
    freqs: set[str] = set()
    for t in _targets(paths, args):
        r = rows.get(t)
        if not r or r["full_runner"] in ("", "none"):
            continue
        cfg = tomllib.loads((paths.repl / r["full_runner"]).read_text(encoding="utf-8"))
        if cfg.get("kind") == "m4":
            fs = set(cfg["frequencies"])
            if args.frequency:
                fs &= set(args.frequency)
            freqs |= fs
    return freqs


def _needs_pjme(args) -> bool:
    # battery_energy and battery_global_value (its third instance is PJME daily load)
    return (any(("energy" in t or "global_value" in t) for t in (getattr(args, "experiment", []) or []))
            or getattr(args, "all", False))
