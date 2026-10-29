#!/usr/bin/env python
"""First step of the battery configuration that reads the PJME load data (battery_global_value):
check that $IJF_PJME_DIR/PJME_hourly.csv is present and is the file the July 2026 battery used.

The reference SHA-256 is the one archived with the original battery
(artifacts/original-scripts/battery-2026-07/SHA256SUMS, entry
results/raw/energy/data/PJME_hourly.csv). The digest is taken over the bytes as stored; a copy
whose only difference is CRLF line endings (e.g. a Windows checkout) is accepted on the digest
with CR bytes removed, the convention `./reproduce.sh fetch` uses, and that is said explicitly.

Exit codes: 0 verified; 4 (missing or corrupted input) otherwise. Writes nothing.
New in the port (the original reproduce.sh ran `shasum -c SHA256SUMS` inline).
"""
import hashlib
import os
import sys

PJME_SHA256 = "4eb2b16d42bf07ec41ab55cb842191594cb69452725a6d3c0991658a628fde84"
FILE = "PJME_hourly.csv"
EXIT_INPUT = 4
HOWTO = "run ./reproduce.sh fetch pjme (it downloads and verifies the file)"


def digest(path, strip_cr=False):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk.replace(b"\r", b"") if strip_cr else chunk)
    return h.hexdigest()


def main():
    pjme_dir = os.environ.get("IJF_PJME_DIR", "")
    if not pjme_dir:
        print(f"ERROR: IJF_PJME_DIR is not set, so {FILE} cannot be located; {HOWTO}.")
        return EXIT_INPUT
    path = os.path.join(pjme_dir, FILE)
    if not os.path.isfile(path):
        print(f"ERROR: {path} not found. The PJME data are not redistributed with the package; {HOWTO}.")
        return EXIT_INPUT
    got = digest(path)
    if got == PJME_SHA256:
        print(f"PJME input verified: {path} sha256 {got} (matches the archived SHA256SUMS)")
        return 0
    got_lf = digest(path, strip_cr=True)
    if got_lf == PJME_SHA256:
        print(f"PJME input verified: {path} matches the archived SHA-256 {PJME_SHA256} once CR bytes "
              f"are removed (the file has CRLF line endings; raw-byte sha256 {got}).")
        return 0
    print(f"ERROR: {path} has sha256 {got} (CR-stripped {got_lf}); the July 2026 battery used "
          f"{PJME_SHA256}. Refusing to run on different data; {HOWTO}.")
    return EXIT_INPUT


if __name__ == "__main__":
    sys.exit(main())
