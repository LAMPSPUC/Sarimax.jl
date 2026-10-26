"""Stage the files of the Zenodo data record inside the replication package (maintainers only).

    python stage_release.py [--part1 TARBALL] [--part2 TARBALL] [--image TARBALL]
    python stage_release.py --check

Each bundle part's tarball is copied into replication/ijf-2026/dist/<part>/ after its SHA-256
is checked against artifacts.lock and every file it contains against the part's committed
inventory (artifacts.sha256, artifacts-part2.sha256). Next to it go the inventory and the
bundle json that zenodo_deposit.py reads. An exported image (`docker save | gzip`) can be
staged in dist/image/ with its SHA-256. dist/ is ignored by git and by the Docker build
context.

The tarballs are the ones the builders wrote (build_artifact_bundle.py, build_artifact_part2.py,
which read the experiment workspaces); once staged, the deposit reads nothing outside this
package:

    python zenodo_deposit.py --bundle-dir dist/part1 --part2-dir dist/part2 [--extra dist/image/FILE]

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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parts() -> list[dict]:
    """Part 1 is [bundle] of artifacts.lock, part 2 its [[parts]] entry, with their staging names."""
    lock = tomllib.loads((REPL / "artifacts.lock").read_text(encoding="utf-8"))
    p1, p2 = lock["bundle"], lock["parts"][0]
    return [dict(p1, key="part1", json="bundle.json"), dict(p2, key="part2", json="bundle-part2.json")]


def inventory(part: dict) -> dict[str, str]:
    out = {}
    for line in (REPL / part["inventory"]).read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        out[name.strip().lstrip("*")] = digest
    return out


def verify_tarball(tarball: Path, part: dict) -> list[str]:
    """Problems with a part's tarball: its SHA-256, and every file against the inventory."""
    problems = []
    if sha256(tarball) != part["sha256"]:
        return [f"{tarball.name}: SHA-256 differs from artifacts.lock ({part['sha256'][:16]}...)"]
    want, seen = inventory(part), {}
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


def stage(source: Path, part: dict) -> list[str]:
    problems = verify_tarball(source, part)
    if problems:
        return problems
    folder = DIST / part["key"]
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / part["file"]
    if not dest.exists() or sha256(dest) != part["sha256"]:
        shutil.copyfile(source, dest)
    shutil.copyfile(REPL / part["inventory"], folder / Path(part["inventory"]).name)
    with tarfile.open(dest, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
    meta = {"bundle": part["name"], "version": str(part["version"]), "files": len(members),
            "bytes": sum(m.size for m in members), "tarball": part["file"], "tarball_sha256": part["sha256"],
            "tarball_bytes": dest.stat().st_size, "inventory_sha256": part["inventory_sha256"]}
    (folder / part["json"]).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"staged {part['key']}: {dest.relative_to(REPL)}, {len(members)} files verified")
    return []


def stage_image(source: Path) -> None:
    folder = DIST / "image"
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / source.name
    shutil.copyfile(source, dest)
    digest = sha256(dest)
    (folder / "SHA256SUMS").write_text(f"{digest}  {dest.name}\n", encoding="utf-8")
    print(f"staged image: {dest.relative_to(REPL)} (SHA-256 {digest[:16]}...)")


def check() -> list[str]:
    problems = []
    for part in parts():
        folder = DIST / part["key"]
        tarball = folder / part["file"]
        if not tarball.exists():
            problems.append(f"{part['key']}: {tarball.relative_to(REPL)} not staged")
            continue
        problems += verify_tarball(tarball, part)
        meta = json.loads((folder / part["json"]).read_text(encoding="utf-8"))
        if meta["tarball_sha256"] != part["sha256"] or meta["inventory_sha256"] != part["inventory_sha256"]:
            problems.append(f"{part['key']}: {part['json']} does not match artifacts.lock")
        if (folder / Path(part["inventory"]).name).read_bytes() != (REPL / part["inventory"]).read_bytes():
            problems.append(f"{part['key']}: staged inventory differs from the committed one")
        if not problems:
            print(f"{part['key']}: {part['file']} and its {meta['files']} files verified")
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
    ap.add_argument("--part1", type=Path)
    ap.add_argument("--part2", type=Path)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    p1, p2 = parts()
    problems = []
    if a.part1:
        problems += stage(a.part1, p1)
    if a.part2:
        problems += stage(a.part2, p2)
    if a.image:
        stage_image(a.image)
    if a.check or not (a.part1 or a.part2 or a.image):
        problems += check()
    for p in problems:
        print("PROBLEM", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
