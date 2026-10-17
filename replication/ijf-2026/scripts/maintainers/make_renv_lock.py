"""Generate environments/r/renv.lock from a dated Posit Package Manager CRAN snapshot.

The lockfile pins every package the R baselines need, plus their whole dependency closure
(Depends, Imports, LinkingTo), at the versions of ONE dated CRAN snapshot. Base and
recommended packages ship with the R build of the base image and are not listed.

Usage (maintainers only; needs network):
    python make_renv_lock.py --snapshot 2026-06-29 --r-version 4.6.1 forecast jsonlite > renv.lock
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

BASE = {"R", "base", "compiler", "datasets", "graphics", "grDevices", "grid", "methods",
        "parallel", "splines", "stats", "stats4", "tcltk", "tools", "utils"}
RECOMMENDED = {"KernSmooth", "MASS", "Matrix", "boot", "class", "cluster", "codetools",
               "foreign", "lattice", "mgcv", "nlme", "nnet", "rpart", "spatial", "survival"}


def parse_dcf(text: str) -> dict[str, dict[str, str]]:
    packages: dict[str, dict[str, str]] = {}
    for block in re.split(r"\n\s*\n", text):
        fields: dict[str, str] = {}
        key = None
        for line in block.splitlines():
            if line[:1].isspace() and key:
                fields[key] += " " + line.strip()
            elif ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                fields[key] = value.strip()
        if "Package" in fields:
            packages[fields["Package"]] = fields
    return packages


def requirements(fields: dict[str, str]) -> list[str]:
    out = []
    for key in ("Depends", "Imports", "LinkingTo"):
        for item in fields.get(key, "").split(","):
            name = re.sub(r"\(.*\)", "", item).strip()
            if name and name not in BASE:
                out.append(name)
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--r-version", required=True)
    ap.add_argument("packages", nargs="+")
    args = ap.parse_args()

    repo = f"https://packagemanager.posit.co/cran/{args.snapshot}"
    with urllib.request.urlopen(f"{repo}/src/contrib/PACKAGES") as r:
        index = parse_dcf(r.read().decode("utf-8"))

    closure: dict[str, dict] = {}
    todo = list(args.packages)
    while todo:
        name = todo.pop()
        if name in closure or name in BASE or name in RECOMMENDED:
            continue
        if name not in index:
            raise SystemExit(f"{name} is not in the {args.snapshot} snapshot")
        fields = index[name]
        reqs = requirements(fields)
        closure[name] = {"Package": name, "Version": fields["Version"], "Source": "Repository",
                         "Repository": "CRAN", "Requirements": reqs}
        todo.extend(reqs)

    lock = {"R": {"Version": args.r_version,
                  "Repositories": [{"Name": "CRAN", "URL": repo}]},
            "Packages": {k: closure[k] for k in sorted(closure, key=str.lower)}}
    json.dump(lock, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
