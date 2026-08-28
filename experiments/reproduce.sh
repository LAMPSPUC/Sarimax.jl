#!/usr/bin/env bash
# =======================================================================================
# Replication entry point — M4 objective-function campaigns, host ICARAI.
#
# HARDWARE ASSUMED (the machine the recorded results came from)
#   Intel Core i7-12700, 12 physical / 20 logical cores, 64 GB RAM
#   Windows 10 Pro 10.0.19045, x86_64-w64-mingw32
#   Julia 1.10.12, 10 worker processes, one BLAS thread per worker
#
# EXPECTED WALL-CLOCK
#   verify        seconds        checksums only
#   tables        ~1 minute      regenerates the tables from the stored raw outputs
#   campaigns     ~59 hours      refits everything: 37 campaigns, 210 CPU-hours over 10
#                                workers. The single longest campaign is
#                                obj_huber_innov_monthly at 10.5 hours. This is not a
#                                figure of speech: budget two and a half days.
#
# Running the campaigns additionally requires the M4 datasets, which are not redistributed
# in this package. See REPRODUCE.md, "Inputs not produced here".
# =======================================================================================
set -euo pipefail
cd "$(dirname "$0")"

STEP="${1:-tables}"

case "$STEP" in
verify)
    sha256sum -c SHA256SUMS
    ;;

tables)
    # Refits nothing. Reads results/raw and results/baselines only.
    python scripts/make_tables.py
    ;;

campaigns)
    : "${SARIMAX_SRC:?set SARIMAX_SRC to a Sarimax.jl checkout}"
    : "${M4_DATASETS:?set M4_DATASETS to the directory holding <Freq>-train.csv/-test.csv}"
    mkdir -p rerun
    JL="julia --project=$SARIMAX_SRC"

    # Family A — objective isolated, initialization = :zeroed, uniform 600 s model cap.
    for freq in monthly quarterly yearly weekly; do
        for obj in mse huber mae; do
            $JL scripts/run_cell.jl 0 10 "rerun/cel_${obj}_zeroed_${freq}.csv" \
                "$obj" "$freq" 900 zeroed 600 true
        done
    done

    # Family B — production configuration, initialization = :innovations, no model cap.
    for freq in monthly quarterly yearly weekly; do
        for obj in mse huber mae ridge; do
            $JL scripts/run_cell.jl 0 10 "rerun/obj_${obj}_innov_${freq}.csv" \
                "$obj" "$freq" 3600 innovations 0 true
        done
    done

    # Over-differencing guard, both arms. The `true` arm of monthly/quarterly/yearly IS the
    # mse cell of family B; it is run once and used in both roles.
    for freq in monthly quarterly yearly; do
        for arm in true false; do
            $JL scripts/run_cell.jl 0 10 "rerun/req_${arm}_${freq}.csv" \
                mse "$freq" 3600 innovations 0 "$arm"
        done
    done

    # Ridge census under the production cap rule, and the two 1000-series smoke runs.
    for freq in monthly quarterly yearly weekly; do
        $JL scripts/run_cell.jl 0 10 "rerun/censo_ridge_${freq}.csv" \
            ridge "$freq" 600 innovations -1 true
    done
    $JL scripts/run_cell.jl 1000 10 rerun/smoke_innov_monthly.csv mse monthly 600 innovations -1 true
    $JL scripts/run_cell.jl 1000 10 rerun/smoke_ridge_monthly.csv ridge monthly 600 innovations -1 true
    ;;

*)
    echo "usage: $0 {verify|tables|campaigns}" >&2
    exit 2
    ;;
esac
