#!/usr/bin/env bash
#
# Single entry point: regenerates every result and every table in this directory.
#
# ---------------------------------------------------------------------------------------
# EXPECTED WALL TIME: about 21 hours at 10 worker processes.
#
# HARDWARE ASSUMED: the published runs were produced on a 6-core / 12-thread Intel
# i7-5820K at 3.30 GHz with 64 GB of RAM, running 10 worker processes, each serial inside
# (BLAS pinned to one thread). Total cost was about 203 CPU-hours. On fewer cores the wall
# time scales roughly inversely; on fewer than ~16 GB of RAM the daily frequency will page
# and throughput collapses by more than an order of magnitude.
#
# Per campaign, at 10 workers:
#
#   A  M4 benchmark, six frequencies      ~8.2 h wall   (~82 CPU-h)
#        monthly 5.1 h · quarterly 1.5 h · daily 0.9 h · hourly 0.5 h
#        · yearly 0.15 h · weekly 0.05 h
#   B  objectives, monthly                ~8.7 h wall   (~87 CPU-h)
#   C  multistart, monthly                ~0.7 h wall   (~7 CPU-h)
#   D  same-commit isolation              ~0.6 h wall   (~6 CPU-h)
#   E  axes 2x2, weekly                   ~0.5 h wall   (~5 CPU-h)
#   F  stable objective                   ~1.7 h wall   (~17 CPU-h)
#
# These are measured totals from the published runs, not estimates.
# ---------------------------------------------------------------------------------------
#
# PREREQUISITES
#   - Julia 1.9.4 (the manifests in env/ are resolved for it)
#   - R with the `forecast` package, reachable through R_HOME (RCall needs it)
#   - the M4 dataset, reachable through the harness package
#   - REPLICATION_HARNESS_REPO pointing at the harness repository checkout
#
# USAGE
#   R_HOME=/path/to/R HARNESS=/path/to/harness ./reproduce.sh [campaign ...]
#
# With no arguments it runs every campaign. Named campaigns run only those, e.g.
#   ./reproduce.sh A F
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS="${HARNESS:-$(cd "$HERE/../.." && pwd)}"
JULIA="${JULIA:-julia}"
WORKERS="${WORKERS:-10}"
OUT="$HERE/results/raw"
SCRIPTS="$HERE/scripts"

: "${R_HOME:?set R_HOME to the R installation directory}"
export R_HOME
export REPLICATION_HARNESS_REPO="$HARNESS"

mkdir -p "$OUT" "$HERE/tables/generated"
cd "$HARNESS"

run() { echo; echo "### $*"; "$JULIA" --project=. "$@"; }

want() {
  [ "$#" -eq 0 ] && return 0
  for c in "${CAMPAIGNS[@]}"; do [ "$c" = "$1" ] && return 0; done
  return 1
}
CAMPAIGNS=("$@")

# --- A: M4 benchmark ---------------------------------------------------------------
# One frequency per invocation, deliberately: the runner holds one frequency's data at a
# time and releases it before the next. Arguments are
#   <initialization> <frequencies> <workers> <output> <capSeconds> <tailLength>
# with capSeconds = 0 meaning NO CAP, which is what the published rows used.
if want A; then
  for f in yearly weekly quarterly hourly daily monthly; do
    run "$SCRIPTS/run_m4.jl" innovations "$f" "$WORKERS" \
        "$OUT/m4_innovations_${f}.csv" 0 0
  done
fi

# --- B: objectives -----------------------------------------------------------------
# The `_pen` cells do NOT reproduce the archived rows; see the banner in run_objectives.jl
# and campaign B in REPRODUCE.md before using them.
if want B; then
  run "$SCRIPTS/run_objectives.jl" 4000 1500 "$WORKERS" \
      "$OUT/objectives_monthly.csv" mse,mse_zeroed,ridge,mae,huber
  run "$SCRIPTS/run_objectives.jl" 0 1500 "$WORKERS" \
      "$OUT/objectives_monthly_penalized.csv" ridge_pen,mae_pen,huber_pen
fi

# --- C: multistart -----------------------------------------------------------------
if want C; then
  run "$SCRIPTS/run_multistart.jl" 400 "$WORKERS" "$OUT/multistart_random.csv"
fi

# --- D: same-commit isolation ------------------------------------------------------
if want D; then
  run "$SCRIPTS/run_isolation.jl" weekly,yearly "$WORKERS" "$OUT/isolation.csv"
fi

# --- E: axes 2x2 -------------------------------------------------------------------
if want E; then
  run "$SCRIPTS/run_axes.jl" weekly "$WORKERS" "$OUT/axes_weekly.csv"
fi

# --- F: stable objective -----------------------------------------------------------
if want F; then
  run "$SCRIPTS/run_stable.jl" weekly,yearly "$WORKERS" "$OUT/stable_weekly_yearly.csv" 0
fi

# --- tables ------------------------------------------------------------------------
echo; echo "### tables"
"$JULIA" "$HERE/tables/make_table_m4.jl" "$OUT"/m4_innovations_*.csv \
    | tee "$HERE/tables/generated/table_m4.txt"

# --- checksums ---------------------------------------------------------------------
echo; echo "### checksums"
cd "$HERE"
find results/raw tables/generated -type f -name '*.csv' -o -type f -name '*.txt' \
    | sort | xargs sha256sum > SHA256SUMS.regenerated
echo "wrote SHA256SUMS.regenerated"
echo
echo "Compare against the published SHA256SUMS with:"
echo "    diff SHA256SUMS SHA256SUMS.regenerated"
echo "Rows will differ where REPRODUCE.md says they are expected to. Read it before"
echo "treating a difference as a defect."
