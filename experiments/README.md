# Replication material

This directory holds the scripts, environment pins and raw results behind the empirical
results reported for `Sarimax.jl`. It is self-contained: nothing here depends on state
outside the repository except the M4 dataset and the reference forecasts, whose retrieval
is described below.

**Read `REPRODUCE.md` before citing any number.** It is the contract: what each script
produces, the exact configuration of each campaign, the package commit each result was
produced under, the hardware and wall time, and — explicitly — what is known not to be
bit-for-bit reproducible and why.

## Layout

```
experiments/
  README.md                  this file
  REPRODUCE.md               the reproduction contract; start here
  reproduce.sh               single entry point, regenerates everything
  env/
    Project.toml             the harness environment
    Manifest.toml            fully resolved pin (MathOptInterface 1.48.0)
    Manifest-moi131.toml     the earlier pin (MathOptInterface 1.31.0), see REPRODUCE.md
  scripts/
    provenance.jl            provenance stamp; checks BOTH repository trees
    wrapper_v0_6.jl          frozen production wrapper (reconstructed — read its banner)
    run_m4.jl                campaign A
    run_objectives.jl        campaign B
    run_multistart.jl        campaign C
    run_isolation.jl         campaign D
    run_axes.jl              campaign E
    run_stable.jl            campaign F
    backfill_provenance.jl   adds provenance columns to the archived raw results
    verify_reproduction.jl   re-fits a seeded sample and compares against the archive
  results/raw/               raw per-series output, one row per (series, arm)
  results/verification/      output of verify_reproduction.jl; the evidence behind the
                             reproduction claims in REPRODUCE.md section 3
  tables/                    table generators; consume results/, fit nothing
  tables/generated/          the tables as published, regenerated from results/raw/
  SHA256SUMS                 covers every file in this directory; `sha256sum -c SHA256SUMS`
```

## Checking it before trusting it

```bash
cd experiments && sha256sum -c SHA256SUMS
```

That establishes the material is intact, not that it reproduces. For the latter, run
`scripts/verify_reproduction.jl` against the package version you care about; it re-fits a
seeded sample and reports what matches. It takes minutes, where re-running a campaign takes
hours, and REPRODUCE.md section 3 records what it returned here and against which commit.
Run it again after any release: the recorded answer is specific to one commit.

## What is not in this directory

**The M4 dataset.** Obtain it from the M4 competition repository and point the harness at
it. The runners read it through the harness package, not directly.

**The reference forecasts** (`auto.arima` and `Naive2`, per series, per horizon block).
These are third-party outputs; regenerating them requires R with the `forecast` package.
Point `M4_REFERENCE_DIR` at the directory holding them. The table generators degrade
gracefully and print `no reference` for blocks they cannot find.

**The environment itself.** `env/` pins it; `~/.julia`, build artifacts and caches are
deliberately absent and reinstall from the manifest.

## Conventions worth knowing before reading the code

**The runs store vectors, not just aggregates.** Every result row carries the forecast, the
realised values, and the MASE denominator of the whole series. Any metric at any horizon —
including metrics not yet chosen — is therefore recomputable without re-fitting. An earlier
run stored sMAPE alone, ended up unable to produce OWA, and had to be repeated in full.

**OWA is a ratio of means**, the published M4 convention. A mean of ratios gives a
different number that is not comparable to any published figure.

**No script inherits a package default.** Every keyword that can move an estimate is stated
explicitly in the configuration function of each runner, even where the value coincides
with the current default. The package defaults have changed between campaigns; a script
that inherits them reports a different number when re-run later, silently. The two
deliberate exceptions are marked as such in `run_axes.jl`, where "package default versus
wrapper value" is the experimental treatment itself.

**Parallelism is between processes, each serial inside.** Every runner calls
`BLAS.set_num_threads(1)`. Launching N worker processes while BLAS opens its own thread
pool oversubscribes the machine and has produced NaNs and false timeouts here.
