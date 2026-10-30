#!/usr/bin/env bash
# Source fingerprint of a Sarimax.jl checkout, as used by the IJF replication package.
#
# The fingerprint is a SHA-256 over the content of every file that determines what the
# replication container computes: the package source, the pinned environments, the
# runners, the table generators, the committed fixtures and the expected values.
# Documentation and the release-time lock files (image.lock, artifacts.lock) are left out
# on purpose, so that recording a published image digest does not change the fingerprint
# of the image it describes.
#
# The same script runs on the host (reproduce.sh) and inside the image (Dockerfile), so a
# match proves that the container was built from this checkout's code. Carriage returns
# are stripped before hashing, so a Windows checkout with core.autocrlf=true produces the
# same fingerprint as a Linux one.
#
# Usage: fingerprint.sh [REPO_ROOT]        prints the fingerprint
#        fingerprint.sh --list [REPO_ROOT] prints the per-file hashes it is built from
set -euo pipefail

list=0
if [[ "${1:-}" == "--list" ]]; then
    list=1
    shift
fi
root="${1:-.}"
cd "$root"

paths=(
    Project.toml
    src
    replication/ijf-2026/docker
    replication/ijf-2026/environments
    replication/ijf-2026/experiments
    replication/ijf-2026/scripts
    replication/ijf-2026/data
    replication/ijf-2026/results
    replication/ijf-2026/experiment_manifest.csv
    replication/ijf-2026/artifacts.sha256
    replication/ijf-2026/checksums.sha256
)

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | cut -d' ' -f1
    else
        shasum -a 256 | cut -d' ' -f1
    fi
}

existing=()
for p in "${paths[@]}"; do
    [[ -e "$p" ]] && existing+=("$p")
done

per_file() {
    find "${existing[@]}" -type f \
        ! -path 'replication/ijf-2026/data/external/*' \
        ! -name '*.pyc' ! -path '*/__pycache__/*' ! -name '.DS_Store' \
        ! -name '*.md' -print \
        | LC_ALL=C sort \
        | while IFS= read -r f; do
            printf '%s  %s\n' "$(tr -d '\r' < "$f" | sha256)" "$f"
        done
}

if [[ $list -eq 1 ]]; then
    per_file
else
    per_file | sha256
fi
