#!/usr/bin/env bash
# Bundle the experiments/ directory into a self-contained, distributable artifact tarball.
# The tarball INCLUDES the Julia lockfiles (env/*.toml) even though the package .gitignore would
# ignore them -- tar does not consult .gitignore. The package source is NOT included: the artifact
# pins Sarimax.jl by commit and fetches it at reproduce time (see scripts/setup/julia_setup.jl).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(dirname "$HERE")"
NAME="sarimax_experiments_artifact_$(date +%Y%m%d).tar.gz"
mkdir -p "$HERE/dist"

tar czf "$HERE/dist/$NAME" \
  --exclude='experiments/.venv-benchmarks' \
  --exclude='experiments/dist' \
  --exclude='experiments/out' \
  --exclude='*/__pycache__' \
  --exclude='*.log' \
  --exclude='*.tar.gz' \
  -C "$PARENT" experiments

echo "wrote $HERE/dist/$NAME"
echo "contents (top level):"
tar tzf "$HERE/dist/$NAME" | sed 's#experiments/##' | awk -F/ 'NF<=2' | sort -u | head -40
