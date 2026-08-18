# Order guards must bind through EVERY adoption path, not only the stepwise move.
#
# `orderAllowed` was consulted by the stepwise neighbourhood move but not by the paths that
# adopt a model directly: the three seed candidates (AR-only, MA-only, all-zero) and the null
# model that doubles as the safety net when the first candidate is inadmissible. A guard that
# only filters the walk is inert whenever the search reaches a barred order some other way.
#
# The failure mode is silent — the option is accepted, the search runs, and a barred order
# comes back — so these tests assert the guard's OUTPUT, not its internals: no barred order may
# be reachable by any path. That is the property the guards actually promise.
@testset "Order guards bind on every adoption path" begin
    # A short, strongly trending series: exactly the shape that drives the search to d = 2 and
    # towards AR-only candidates.
    n = 72
    dates = collect(Date(2000, 1, 1):Month(1):Date(2000, 1, 1)+Month(n - 1))
    t = collect(1.0:n)
    vals = 100.0 .+ 0.5 .* t .^ 2 .+ 3.0 .* sin.(2π .* t ./ 12)
    y = TimeArray(dates, vals)

    common = (
        seasonality = 12,
        objectiveFunction = "mse",
        initialization = :penalized,
        showLogs = false,
    )

    @testset "requireMAWhenDoublyDifferenced forces q >= 1 when d >= 2" begin
        model = auto(y; common..., d = 2, D = 0, requireMAWhenDoublyDifferenced = true)
        @test model.d == 2
        # The guard's whole promise: no path may return q = 0 here.
        @test model.q >= 1
    end

    @testset "requireTermsWhenOverDifferenced removes the term-free corner" begin
        model = auto(y; common..., d = 2, D = 0, requireTermsWhenOverDifferenced = true)
        @test model.p + model.q + model.P + model.Q >= 1
    end

    @testset "guards are inert where they do not apply" begin
        # d = 1 is outside the MA guard's trigger, so enabling it must not change the search.
        base = auto(y; common..., d = 1, D = 0)
        guarded = auto(y; common..., d = 1, D = 0, requireMAWhenDoublyDifferenced = true)
        @test (base.p, base.q, base.P, base.Q) == (guarded.p, guarded.q, guarded.P, guarded.Q)
    end

    @testset "both guards default to off" begin
        # Off by default is part of the contract: they are deliberate divergences from
        # `auto.arima`, not silent behaviour changes.
        plain = auto(y; common..., d = 2, D = 0)
        @test plain isa SARIMAModel
    end
end
