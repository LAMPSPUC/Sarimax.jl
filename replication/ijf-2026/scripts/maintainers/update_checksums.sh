#!/usr/bin/env bash
# Regenerates replication/ijf-2026/checksums.sha256: the SHA-256 of every committed file of the
# replication package that a computation reads (environments, configurations, runners,
# fixtures, expected values, the manifest and the artifact inventory). Documentation, the
# release-time lock files and this list itself are excluded.
#
# Checksums are taken over LF content (carriage returns stripped), the bytes .gitattributes
# guarantees on every platform. Verify with:  sha256sum --check checksums.sha256
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum | cut -d' ' -f1; else shasum -a 256 | cut -d' ' -f1; fi
}

list_files() {
    # In a git checkout, the files in the index (committed or staged); untracked scratch files
    # are not part of the package. Outside git, every file under the listed paths.
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git ls-files -- docker environments experiments scripts data results experiment_manifest.csv artifacts.sha256
    else
        find docker environments experiments scripts data results experiment_manifest.csv artifacts.sha256 -type f
    fi
}

list_files \
    | grep -v -e '^data/external/' -e '\.pyc$' -e '/__pycache__/' -e '\.DS_Store$' -e '\.md$' \
    | LC_ALL=C sort \
    | while IFS= read -r f; do
        case "$f" in
            *.gz|*.zip|*.png|*.pdf|*.bundle) printf '%s  %s\n' "$(sha256 < "$f")" "$f" ;;      # binary: raw bytes
            *) printf '%s  %s\n' "$(tr -d '\r' < "$f" | sha256)" "$f" ;;
        esac
    done > checksums.sha256
echo "$(wc -l < checksums.sha256) files listed in $(pwd)/checksums.sha256"
