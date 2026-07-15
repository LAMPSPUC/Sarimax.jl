#!/usr/bin/env bash
# Reproduce the SARIMAX.jl empirical battery and regenerate all tables.
#
# Usage:
#   ./reproduce.sh setup        # create the Julia / Python / R environments
#   ./reproduce.sh run          # run the full battery + build tables (includes slow global solves)
#   ./reproduce.sh run --quick  # skip the slow SCIP global-solver blocks
#   ./reproduce.sh all          # setup then run
#
# Interpreter overrides (defaults shown). On a juliaup machine use: JULIA="julia +1.11"
#   JULIA=julia  PY=python  RSCRIPT=Rscript
#
# Expected wall time (full): a few minutes for validation/forecasting/architecture, plus
# ~10-15 min for the global-solver blocks. The optional SCIP scaling study (run_solver_scip_scaling.jl)
# can add up to ~20 more minutes and is gated behind RUN_SCIP_SCALING=1.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

JULIA="${JULIA:-julia}"
PY="${PY:-python}"
RSCRIPT="${RSCRIPT:-Rscript}"
ENVDIR="env"
VENV=".venv-benchmarks"
PJME="results/raw/energy/data/PJME_hourly.csv"
PJME_URL="https://raw.githubusercontent.com/archd3sai/Hourly-Energy-Consumption-Prediction/master/PJME_hourly.csv"

jl() { $JULIA --project="$ENVDIR" "$@"; }

setup() {
  echo ">> Julia environment"
  jl scripts/setup/julia_setup.jl
  echo ">> Python environment"
  $PY -m venv "$VENV"
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
  pip install --quiet --upgrade pip wheel
  pip install --quiet -r requirements.txt
  echo ">> R environment"
  $RSCRIPT scripts/setup/install_r.R
  echo ">> setup done"
}

run() {
  local quick="${1:-}"
  # PJME raw data ships with the artifact; download only if absent.
  if [ ! -f "$PJME" ]; then echo ">> downloading PJME"; curl -sL "$PJME_URL" -o "$PJME"; fi
  echo ">> verifying input-data checksums"; shasum -c SHA256SUMS

  # shellcheck disable=SC1091
  [ -d "$VENV" ] && . "$VENV/bin/activate"

  echo ">> data generation"
  $PY scripts/setup/gen_sim_data.py
  $PY scripts/setup/gen_energy_data.py

  echo ">> Block 1: validation"
  jl scripts/run_validation_julia.jl
  $PY scripts/run_validation_python.py
  $RSCRIPT scripts/run_validation_r.R
  $PY scripts/diagnose_exog.py

  echo ">> Block 2: forecasting (rolling-origin)"
  jl scripts/run_forecasting_julia.jl
  $PY scripts/run_forecasting_python.py
  $RSCRIPT scripts/run_forecasting_r.R

  echo ">> Block 3: architecture & extensibility"
  jl scripts/run_architecture_extensions.jl

  echo ">> External validation: PJME energy"
  jl scripts/run_energy_forecasting_julia.jl
  $PY scripts/run_energy_forecasting_python.py
  $RSCRIPT scripts/run_energy_forecasting_r.R
  jl scripts/run_energy_architecture.jl

  if [ "$quick" = "--quick" ]; then
    echo ">> Block 4: solver -- SKIPPED (--quick)"
  else
    echo ">> Block 4: solver diagnostics (slow: direct SCIP global solves)"
    jl scripts/run_solver_diagnostics.jl
    jl scripts/run_solver_jump_diagnostics.jl
    jl scripts/run_solver_scip_cert.jl        # direct SCIP -> exact global certificates (gap 0)
    jl scripts/run_solver_global_value.jl     # Ipopt vs certified SCIP: does optimality matter for forecasts?
    jl scripts/run_energy_solver.jl
    if [ "${RUN_SCIP_SCALING:-}" = "1" ]; then
      echo ">> Block 4b: SCIP scaling study (very slow: certifies up to the frontier, ~20 min)"
      jl scripts/run_solver_scip_scaling.jl
    fi
  fi

  echo ">> building tables"
  $PY scripts/combine_results.py
  echo ">> done. Tables in tables/ ; raw records in results/raw/"
}

case "${1:-}" in
  setup) setup ;;
  run)   run "${2:-}" ;;
  all)   setup; run "${2:-}" ;;
  *) echo "usage: $0 {setup|run [--quick]|all [--quick]}"; exit 2 ;;
esac
