"""Prepare (and only on request publish) the Zenodo data record of the IJF artifact bundle.

Default target is the Zenodo SANDBOX (https://sandbox.zenodo.org), which mints test DOIs
that are discarded. The script creates a draft deposition, uploads the files and sets the
metadata; it publishes ONLY with --publish, and refuses --publish against production unless
--i-have-approval is also given. It never reads a token from anywhere but the environment.

    ZENODO_SANDBOX_TOKEN=... python zenodo_deposit.py --bundle-dir DIR            # sandbox draft
    ZENODO_SANDBOX_TOKEN=... python zenodo_deposit.py --bundle-dir DIR --publish  # sandbox publish
    ZENODO_TOKEN=... python zenodo_deposit.py --production --bundle-dir DIR         # production draft

DIR is the output directory of build_artifact_bundle.py (tarball, artifacts.sha256,
bundle.json). Optional extra files (for example the exported image) via --extra FILE.
After a successful publication, copy the printed DOI and file URL into artifacts.lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

DESCRIPTION = (
    "<p>Archived experimental results of the article <em>SARIMAX.jl: Open-Source Optimization-Based "
    "SARIMAX Modeling in Julia</em> (International Journal of Forecasting, special issue Advances in "
    "Open Source Forecasting Software): original per-series results and logs of every campaign, the "
    "derived datasets and historical tables used for the article's numbers, the original runners and "
    "notes, and a provenance record of every file.</p>"
    "<p>It is the input of <code>./reproduce.sh tables</code> in the Sarimax.jl repository "
    "(replication/ijf-2026), which rebuilds every table from these files and compares each cell with "
    "its historical value. M4 competition data and PJM load data are not redistributed (their "
    "licences are not established); the replication package downloads them from pinned sources and "
    "verifies them. See README.md inside the bundle and LIMITATIONS.md in the repository.</p>"
)


def api(base: str, token: str, method: str, path: str, body=None, data: bytes | None = None,
        content_type: str = "application/json"):
    url = path if path.startswith("http") else f"{base}{path}"
    payload = data if data is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=payload, method=method,
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": content_type})
    with urllib.request.urlopen(req, timeout=600) as r:
        text = r.read().decode()
        return json.loads(text) if text else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument("--extra", type=Path, action="append", default=[])
    ap.add_argument("--production", action="store_true", help="use zenodo.org instead of the sandbox")
    ap.add_argument("--publish", action="store_true", help="publish the deposition (irreversible on zenodo.org)")
    ap.add_argument("--i-have-approval", action="store_true",
                    help="required with --production --publish: the authors approved this publication")
    ap.add_argument("--version", default="1")
    ap.add_argument("--related-software-doi", default="", help="version DOI of the archived software release")
    args = ap.parse_args()

    if args.production and args.publish and not args.i_have_approval:
        raise SystemExit("refusing to publish on zenodo.org without --i-have-approval")
    base = "https://zenodo.org/api" if args.production else "https://sandbox.zenodo.org/api"
    token = os.environ.get("ZENODO_TOKEN" if args.production else "ZENODO_SANDBOX_TOKEN")
    if not token:
        raise SystemExit("set ZENODO_SANDBOX_TOKEN (or ZENODO_TOKEN with --production)")

    meta = json.loads((args.bundle_dir / "bundle.json").read_text())
    files = [args.bundle_dir / meta["tarball"], args.bundle_dir / "artifacts.sha256", *args.extra]
    for f in files:
        if not f.exists():
            raise SystemExit(f"missing {f}")
    tar_sha = hashlib.sha256(files[0].read_bytes()).hexdigest()
    if tar_sha != meta["tarball_sha256"]:
        raise SystemExit("tarball SHA-256 differs from bundle.json; rebuild the bundle")

    dep = api(base, token, "POST", "/deposit/depositions", {})
    print(f"draft deposition {dep['id']} created on {base}")
    bucket = dep["links"]["bucket"]
    for f in files:
        print(f"  uploading {f.name} ({f.stat().st_size} bytes)")
        api(base, token, "PUT", f"{bucket}/{f.name}", data=f.read_bytes(), content_type="application/octet-stream")
    metadata = {
        "title": "SARIMAX.jl IJF replication: archived experimental artifacts",
        "upload_type": "dataset",
        "description": DESCRIPTION,
        "version": args.version,
        "creators": [{"name": "Duarte, Luiz Fernando Cunha",
                      "affiliation": "LAMPS, Department of Electrical Engineering, Pontifical Catholic University of Rio de Janeiro (PUC-Rio)"}],
        "keywords": ["M4 competition", "SARIMAX", "forecasting", "reproducibility", "optimization"],
        "license": "cc-by-4.0",
        "access_right": "open",
        "notes": f"Bundle SHA-256 {tar_sha}; {meta['files']} files; inventory in artifacts.sha256.",
        "related_identifiers": [{"identifier": "https://github.com/LAMPSPUC/Sarimax.jl", "relation": "isSupplementTo",
                                 "scheme": "url", "resource_type": "software"}],
    }
    if args.related_software_doi:
        metadata["related_identifiers"].append({"identifier": args.related_software_doi, "relation": "isSupplementTo",
                                                "scheme": "doi", "resource_type": "software"})
    api(base, token, "PUT", f"/deposit/depositions/{dep['id']}", {"metadata": metadata})
    print("  metadata set (creators and licence TO CONFIRM by the authors before publishing)")
    if not args.publish:
        print(f"draft ready for review: {dep['links']['html']}")
        return 0
    pub = api(base, token, "POST", f"/deposit/depositions/{dep['id']}/actions/publish")
    doi = pub.get("doi") or pub.get("metadata", {}).get("doi")
    tar_url = next((f["links"]["self"] for f in pub.get("files", []) if f["filename"] == files[0].name), "")
    print(f"published: DOI {doi}\n  tarball URL {tar_url}\n  record {pub['links']['html']}")
    print("Record doi, url and sha256 in replication/ijf-2026/artifacts.lock.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
