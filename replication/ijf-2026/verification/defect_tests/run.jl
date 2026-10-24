# Targeted tests of two behaviours of the automatic order search at a given package revision:
# the stepwise incumbent (C1) and the drift level in the ranking likelihood (C2), plus the
# ranking likelihood under :armax with regressors. Each correct property is a @test_broken and
# the value the revision computes is pinned by a @test beside it, so a change either way shows.
#
#     julia +1.10.12 run.jl PATH_TO_A_CHECKOUT_OF_THE_REVISION
#
# The tests are not part of the package's test suite: at the current source the :armax case is
# fixed, and its @test_broken would then fail as an unexpected pass. Result at 14dd581:
# result_14dd581.txt.
using Pkg
Pkg.activate(; temp = true)
Pkg.develop(path = ARGS[1])
Pkg.add(["CSV", "DataFrames", "TimeSeries", "JSON"])
using CSV, DataFrames, Dates, Sarimax, Statistics, Test, Random, TimeSeries, JSON, LinearAlgebra
println("Sarimax at ", pkgdir(Sarimax))
@testset "targeted defect tests" begin
    include(joinpath(@__DIR__, "search_incumbent.jl"))
    include(joinpath(@__DIR__, "criterion_scoring.jl"))
end
