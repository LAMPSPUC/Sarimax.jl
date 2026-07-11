# Reproducibility Artifact — SARIMAX.jl Empirical Battery

This directory is a self-contained replication package for the computational experiments of the
SARIMAX.jl paper. It regenerates every raw result and every manuscript table.

- **Package under study:** `Sarimax.jl`, commit `144fb6e86c2743ff726c9716364407e6f2db12ba`
  (v0.1.3), repository <https://github.com/LAMPSPUC/Sarimax.jl>.
- **Toolchain used to produce the reported results:** Julia 1.11.9, Python 3.9.6
  (numpy 2.0.2, pandas 2.3.3, scipy 1.13.1, statsmodels 0.14.6), R 4.6.1 (forecast 9.0.2,
  jsonlite 2.0.0). Platform: macOS 15 / arm64.
- **Fixed seeds:** 1234 (fits, multistart) and 20240627 (simulated data).

The package source is **not** shipped here. The artifact pins Sarimax.jl by commit and fetches it at
setup time, so the artifact is decoupled from the package working tree.
> Before release, tag and push the tested commit so it is publicly fetchable:
> `git tag ijf-artifact-v1 144fb6e && git push origin ijf-artifact-v1`.

## Two ways to reproduce

### A. Native (lockfile-based)

Requirements on the host: Julia 1.11 (e.g. via `juliaup`), Python 3.9, R 4.6.x, and `curl`.

```bash
cd experiments
JULIA="julia +1.11" ./reproduce.sh all          # setup envs, then run the full battery
# or, faster, skipping the slow global-solver blocks:
JULIA="julia +1.11" ./reproduce.sh all --quick
```

Interpreter overrides: `JULIA`, `PY`, `RSCRIPT` (defaults `julia`, `python`, `Rscript`).
To run alongside a local package checkout instead of the pinned GitHub commit, export
`SARIMAX_LOCAL_PATH=/path/to/Sarimax.jl` before `./reproduce.sh setup`.

### B. Docker (frozen environment)

```bash
cd experiments
docker build -f Dockerfile -t sarimax-artifact .
docker run --rm -v "$PWD/out:/artifact/results" sarimax-artifact
```

The image pins R (rocker `4.6.1`), Julia (`1.11.9`), and Python (`3.9.6` via `uv`), restores the three
lockfiles, and runs `reproduce.sh run`. **Caveat:** the image is linux/amd64 while the reported numbers
were produced on macOS/arm64; the image freezes the *environment*, but last-digit values and solver
timing/termination can differ across platform/BLAS. The paper's claims are bounded and do not depend on
exact digits.

## What gets produced

- `results/raw/<block>/*_results.jsonl` — one JSON record per run (status, estimates, metrics, runtime).
- `tables/table_*.{csv,tex}` — the five manuscript tables.
- `results/raw/environment/` — captured tool/package versions and git commit.

Mapping to the manuscript tables:

| Manuscript table | Source |
|---|---|
| `tab:validation_implementations` | `tables/table_validation.tex` |
| `tab:forecast_oos` | `tables/table_forecasting.tex` |
| `tab:architecture_extensibility` | `tables/table_architecture_checks.tex` |
| `tab:solver_comparison` | `tables/table_solver_comparison.tex` |
| `tab:energy_forecasting` | `tables/table_energy_forecasting.tex` |

## Datasets and licenses

- **AirPassengers, GDPC1, NROU** — bundled with the pinned Sarimax.jl package (loaded via
  `loadDataset`); their integrity is guaranteed by the commit pin.
- **PJME (PJM East hourly electricity demand)** — public ISO operational data, distributed as the
  Kaggle "Hourly Energy Consumption" dataset (CC0 1.0, R. Mulla). The raw file ships here
  (`results/raw/energy/data/PJME_hourly.csv`); `reproduce.sh` re-downloads it only if absent and
  verifies it against `SHA256SUMS`. Provenance and cleaning stats:
  `results/raw/energy/data/pjme_preprocessing.json`.

## Expected runtime (host, warm)

Validation, forecasting, architecture, and energy forecasting are minutes total. The Julia runtimes
reported in the tables are **warm/steady-state**: Julia compiles the estimation code on first use (a
one-time cost of a few seconds, excluded by an untimed warm-up fit). The solver block is the slow part:
Alpine global solves are time-limited (~3-4 min each) and do **not** return an optimality certificate;
use `--quick` to skip them.

## Known non-determinism (reported honestly, not hidden)

- **Platform / BLAS:** numerical values at the last digits and solver timings can differ across
  architectures; the Docker image is amd64.
- **Global solver:** Alpine+SCIP runs under a time limit and terminates without a certificate; the
  incumbent matches the local (Ipopt) optimum but the exact termination/time can vary.
- **JIT:** reported Julia runtimes exclude one-time compilation (warm-up); the raw records also store
  `jit_warmup_s` for transparency.
- **Package test suite:** 42 pass, 1 fails — a pre-existing strict `atol=1e-3` parameter-recovery
  assertion, documented in `benchmark_status.md`; it does not affect the experiments.

## Files in this artifact

```
REPRODUCE.md            this file
reproduce.sh            one-command orchestrator (setup | run [--quick] | all)
Dockerfile, .dockerignore   frozen-environment build
make_artifact.sh        bundle experiments/ into a distributable tarball
requirements.txt        pinned Python deps (full pin: results/raw/environment/python_freeze.txt)
SHA256SUMS              checksum of the shipped external input (PJME raw)
env/Project.toml, env/Manifest.toml   as-run Julia lockfiles (exact package versions)
scripts/                all run_*/gen_*/combine scripts; setup/{julia_setup.jl,install_r.R,gen_*}
tables/                 reference outputs (regenerated by a run)
results/raw/            reference raw records + input data + environment capture
experiment_report.md, final_claims_for_manuscript.md, benchmark_status.md,
experiment_protocol.md, exog_discrepancy.md, README.md   documentation
```
