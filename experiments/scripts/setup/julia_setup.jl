#!/usr/bin/env julia
# Reproducible Julia environment for the SARIMAX.jl experiment artifact.
#
# Run with the artifact env active, e.g.:
#     julia +1.11 --project=experiments/env experiments/scripts/setup/julia_setup.jl
#
# Package source resolution:
#   * default (self-contained / Docker): add Sarimax pinned to the tested commit from GitHub;
#   * local development (running alongside the working tree): set SARIMAX_LOCAL_PATH to the
#     package root and it will be `dev`-ed instead.
#
# NOTE: the pinned commit must be reachable on the public repository. Before releasing the
# artifact, tag and push it, e.g.:  git tag ijf-artifact-v1 144fb6e && git push origin ijf-artifact-v1
using Pkg

const SARIMAX_URL = "https://github.com/LAMPSPUC/Sarimax.jl"
const SARIMAX_REV = "144fb6e86c2743ff726c9716364407e6f2db12ba"   # tested commit (package v0.1.3)

# Script dependencies (the experiment scripts `using` these). HiGHS is included only to exercise
# the Alpine sub-solver warning path; it is NOT a dependency of the Sarimax package itself.
const DEPS = ["JSON", "CSV", "DataFrames", "TimeSeries", "Distributions", "Dates",
              "Random", "Statistics", "LinearAlgebra", "JuMP", "Ipopt", "Alpine", "SCIP", "HiGHS"]

localpath = get(ENV, "SARIMAX_LOCAL_PATH", "")
if isempty(localpath)
    @info "Adding Sarimax pinned to commit $SARIMAX_REV from $SARIMAX_URL"
    Pkg.add(PackageSpec(url = SARIMAX_URL, rev = SARIMAX_REV))
else
    @info "Developing local Sarimax at $localpath"
    Pkg.develop(PackageSpec(path = localpath))
end

Pkg.add(DEPS)
Pkg.instantiate()
Pkg.precompile()
Pkg.status()
@info "Julia environment ready."
