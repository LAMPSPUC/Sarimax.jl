"""Stage the files of the Zenodo data record inside the replication package (maintainers only).

    python stage_release.py [--bundle TARBALL] [--image TARBALL]
    python stage_release.py --check

The bundle tarball is copied into replication/ijf-2026/dist/bundle/ after its SHA-256 is
checked against artifacts.lock and every file it contains against the committed inventory
(artifacts.sha256). Next to it go the inventory and the bundle json that zenodo_deposit.py
reads. An exported image (`docker save | gzip`) can be staged in dist/image/ with its SHA-256.
dist/ is ignored by git and by the Docker build context.

The tarball is the one build_artifact_bundle.py wrote (it reads the experiment workspaces);
once staged, the deposit reads nothing outside this package:

    python zenodo_deposit.py --bundle-dir dist/bundle [--extra dist/image/FILE]

`--check` verifies what is staged. Exit 0 when everything agrees, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tomllib
from pathlib import Path

REPL = Path(__file__).resolve().parents[2]
DIST = REPL / "dist"
FOLDER = DIST / "bundle"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bundle() -> dict:
    return tomllib.loads((REPL / "artifacts.lock").read_text(encoding="utf-8"))["bundle"]


def inventory(lock: dict) -> dict[str, str]:
    out = {}
    for line in (REPL / lock["inventory"]).read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        out[name.strip().lstrip("*")] = digest
    return out


def verify_tarball(tarball: Path, lock: dict) -> list[str]:
    """Problems with the tarball: its SHA-256, and every file against the inventory."""
    if sha256(tarball) != lock["sha256"]:
        return [f"{tarball.name}: SHA-256 differs from artifacts.lock ({lock['sha256'][:16]}...)"]
    problems, want, seen = [], inventory(lock), {}
    with tarfile.open(tarball, "r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            rel = "/".join(Path(m.name).parts[1:])         # the top-level directory is stripped
            with tar.extractfile(m) as f:
                seen[rel] = hashlib.sha256(f.read()).hexdigest()
    missing = sorted(set(want) - set(seen))
    extra = sorted(set(seen) - set(want))
    differ = sorted(r for r in set(want) & set(seen) if want[r] != seen[r])
    for label, rows in (("missing", missing), ("not in the inventory", extra), ("SHA-256 differs", differ)):
        if rows:
            problems.append(f"{tarball.name}: {len(rows)} files {label}, e.g. {rows[:3]}")
    return problems


def stage(source: Path, lock: dict) -> list[str]:
    problems = verify_tarball(source, lock)
    if problems:
        return problems
    FOLDER.mkdir(parents=True, exist_ok=True)
    dest = FOLDER / lock["file"]
    if not dest.exists() or sha256(dest) != lock["sha256"]:
        shutil.copyfile(source, dest)
    shutil.copyfile(REPL / lock["inventory"], FOLDER / Path(lock["inventory"]).name)
    with tarfile.open(dest, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
    meta = {"bundle": lock["name"], "version": str(lock["version"]), "files": len(members),
            "bytes": sum(m.size for m in members), "tarball": lock["file"], "tarball_sha256": lock["sha256"],
            "tarball_bytes": dest.stat().st_size, "inventory_sha256": lock["inventory_sha256"]}
    (FOLDER / "bundle.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"staged bundle: {dest.relative_to(REPL)}, {len(members)} files verified")
    return []


def stage_image(source: Path) -> None:
    folder = DIST / "image"
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / source.name
    shutil.copyfile(source, dest)
    digest = sha256(dest)
    (folder / "SHA256SUMS").write_text(f"{digest}  {dest.name}\n", encoding="utf-8")
    print(f"staged image: {dest.relative_to(REPL)} (SHA-256 {digest[:16]}...)")


def check(lock: dict) -> list[str]:
    problems = []
    tarball = FOLDER / lock["file"]
    if not tarball.exists():
        problems.append(f"{tarball.relative_to(REPL)} not staged")
    else:
        problems += verify_tarball(tarball, lock)
        meta = json.loads((FOLDER / "bundle.json").read_text(encoding="utf-8"))
        if meta["tarball_sha256"] != lock["sha256"] or meta["inventory_sha256"] != lock["inventory_sha256"]:
            problems.append("bundle.json does not match artifacts.lock")
        if (FOLDER / Path(lock["inventory"]).name).read_bytes() != (REPL / lock["inventory"]).read_bytes():
            problems.append("staged inventory differs from the committed one")
        if not problems:
            print(f"bundle: {lock['file']} and its {meta['files']} files verified")
    image = DIST / "image" / "SHA256SUMS"
    if image.exists():
        digest, name = image.read_text(encoding="utf-8").split()
        if sha256(DIST / "image" / name) != digest:
            problems.append(f"image: {name} differs from its SHA256SUMS")
        else:
            print(f"image: {name} verified")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    lock = bundle()
    problems = []
    if a.bundle:
        problems += stage(a.bundle, lock)
    if a.image:
        stage_image(a.image)
    if a.check or not (a.bundle or a.image):
        problems += check(lock)
    for p in problems:
        print("PROBLEM", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
