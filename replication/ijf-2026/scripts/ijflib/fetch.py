"""`fetch`: download and verify the archived artifact bundle and the third-party data.

* artifacts - the Zenodo data record: one tarball per bundle part (location and SHA-256 of
              each in artifacts.lock); `--from FILE` (repeatable) uses local copies instead.
              Every extracted file is verified against its part's inventory (artifacts.sha256,
              artifacts-part2.sha256) before the part is marked complete.
* m4        - the official M4 files at a pinned commit of Mcompetitions/M4-methods
              (data/external_files.csv). M4 carries no licence, so it is fetched rather than
              redistributed; the SHA-256 proves the content is the official one.
* pjme      - the PJM hourly load file of the July 2026 battery, from this repository's
              history at a pinned commit (licence not established; not redistributed).
* scip      - the SCIP 8.0.0 binaries (ZIB Academic License), which the image does not ship:
              installed through Julia's package manager, verified by content hash (scip.py).

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

from . import scip
from .check import part_inventory
from .common import EXIT_FAIL, EXIT_INPUT, EXIT_OK, BundlePart, Paths, bundle_parts, sha256_file

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


def _part_for(tarball: Path, parts: list[BundlePart]) -> BundlePart | None:
    """The bundle part a local tarball is: by its SHA-256 when recorded, else by its file name."""
    digest = sha256_file(tarball)
    for p in parts:
        if p.sha256 and digest == p.sha256:
            return p
    for p in parts:
        if not p.sha256 and tarball.name == p.file:
            return p
    return None


def fetch_part(paths: Paths, part: BundlePart, tarball: Path | None, if_available: bool, offline: bool) -> int:
    """Obtain one bundle part (from `tarball`, else from its recorded URL), extract it and verify
    every file of its inventory. A part whose marker matches its inventory is left as it is."""
    marker = paths.artifacts / part.marker
    inventory = part_inventory(paths, part)
    label = f"{part.name}-v{part.version}"
    if tarball is None and marker.exists() and marker.read_text().strip() == part.inventory_sha256:
        print(f"  {label}: already present and verified")
        return EXIT_OK
    with tempfile.TemporaryDirectory() as td:
        if tarball is None:
            if not part.url:
                print(f"  {label}: no published location yet (artifacts.lock: url is empty; the Zenodo "
                      f"record is created at release time). Provide a local copy with "
                      f"`./reproduce.sh fetch artifacts --from {part.file}` or `--artifacts DIR`.")
                return EXIT_OK if if_available else EXIT_INPUT
            if offline:
                print(f"  {label}: missing (offline: not downloaded)")
                return EXIT_INPUT
            tarball = Path(td) / part.file
            try:
                _download(part.url, tarball, part.sha256 or None)
            except (OSError, ValueError) as exc:
                print(f"  ERROR {label}: {exc}")
                return EXIT_INPUT
        if part.sha256 and sha256_file(tarball) != part.sha256:
            print(f"  ERROR: {tarball} does not match the SHA-256 recorded in artifacts.lock for {label}")
            return EXIT_INPUT
        print(f"  extracting {tarball.name} into {paths.artifacts}", flush=True)
        paths.artifacts.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball, "r:gz") as tar:
            n = _safe_extract(tar, paths.artifacts)
    bad = [rel for digest, rel in inventory
           if not (paths.artifacts / rel).exists() or sha256_file(paths.artifacts / rel) != digest]
    if bad:
        print(f"  ERROR {label}: {len(bad)} of {len(inventory)} files missing or corrupted after extraction, "
              f"e.g. {bad[:3]}")
        return EXIT_INPUT
    marker.write_text(part.inventory_sha256 + "\n")
    print(f"  {label} verified: {n} files, all {len(inventory)} inventory checksums match ({part.inventory})")
    return EXIT_OK


def fetch_artifacts(paths: Paths, sources: list[str] | None, if_available: bool, offline: bool) -> int:
    parts = bundle_parts(paths.repl)
    given: dict[str, Path] = {}
    for s in sources or []:
        tarball = Path(s)
        if not tarball.exists():
            print(f"  ERROR: {tarball} does not exist")
            return EXIT_INPUT
        part = _part_for(tarball, parts)
        if part is None:
            print(f"  ERROR: {tarball.name} is none of the bundle parts recorded in artifacts.lock "
                  f"({', '.join(p.file for p in parts)}): unknown name or SHA-256")
            return EXIT_INPUT
        given[part.name] = tarball
    codes = [fetch_part(paths, p, given.get(p.name), if_available, offline) for p in parts]
    # One marker for the whole bundle, which reproduce.sh checks before deciding to fetch.
    all_marker = paths.artifacts / ".complete-all"
    if all((paths.artifacts / p.marker).exists() and
           (paths.artifacts / p.marker).read_text().strip() == p.inventory_sha256 for p in parts):
        all_marker.write_text("".join(f"{p.name} {p.inventory_sha256}\n" for p in parts))
    elif all_marker.exists():
        all_marker.unlink()
    return max(codes)


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
    if what in ("pjme", "all") or (what != "scip" and args.if_needed and _needs_pjme(args)):
        print("== fetch: PJME data")
        codes.append(fetch_external(paths, "pjme", None, offline))
    if what in ("scip", "all"):          # every Julia computation needs it: Sarimax.jl loads SCIP
        print("== fetch: SCIP binaries")
        codes.append(scip.fetch(paths, offline))
    return max(codes) if codes else EXIT_OK


def _targets(paths: Paths, args) -> list[str]:
    from .manifest import load_manifest
    if getattr(args, "all", False):
        from .full import all_targets     # the same selection as `full --all`
        return all_targets(paths, {r["experiment_id"]: r for r in load_manifest(paths)})[0]
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
