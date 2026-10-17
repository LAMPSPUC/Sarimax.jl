#!/usr/bin/env bash
# Collects the licences of the third-party software inside the replication image into
# /opt/licenses, so that the image documents what it redistributes:
#   julia-artifacts/<artifact>/...   licence files shipped with each binary dependency (JLL)
#   r-packages.csv                   R packages: name, version, licence (from DESCRIPTION)
#   python-packages.csv              Python packages: name, version, licence metadata
# See ARTIFACT_GUIDE.md ("Licences") for the summary, including the ZIB Academic License of
# the SCIP 8.0.0 / SoPlex binaries that SCIP.jl 0.11 bundles.
set -euo pipefail
out=/opt/licenses
mkdir -p "$out/julia-artifacts"

for dir in "$JULIA_DEPOT_PATH"/artifacts/*/share/licenses/*; do
    [ -d "$dir" ] || continue
    name="$(basename "$dir")"
    mkdir -p "$out/julia-artifacts/$name"
    cp -r "$dir"/. "$out/julia-artifacts/$name/"
done

Rscript --vanilla -e 'ip <- installed.packages(fields = "License"); write.csv(ip[, c("Package", "Version", "License")], "/opt/licenses/r-packages.csv", row.names = FALSE)'

/opt/venv/bin/python - <<'EOF'
import csv
from importlib import metadata
with open("/opt/licenses/python-packages.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Package", "Version", "License"])
    for d in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
        lic = d.metadata.get("License-Expression") or d.metadata.get("License") or ""
        w.writerow([d.metadata["Name"], d.version, lic.splitlines()[0] if lic else ""])
EOF
cp /opt/julia/LICENSE.md "$out/julia-LICENSE.md" 2>/dev/null || true
echo "licences collected in $out"
