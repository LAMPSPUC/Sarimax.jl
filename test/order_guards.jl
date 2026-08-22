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
#
# 22/08: this file used to pass with `orderAllowed` stubbed to `true` — that is, with the very
# functionality it watches removed from the package. The old fixture was a quadratic trend plus
# seasonality, which at d = 2 selects (0,5,0,2) on its own: seven terms, so `q >= 1` and
# `p+q+P+Q >= 1` held for free and the guard was never exercised. The series below is pure I(2),
# where two differences leave white noise and AICc PREFERS the term-free corner. The premise is
# now an assertion rather than an `if`, so that if it ever stops holding these tests fail loudly
# instead of going quietly green.
@testset "Order guards bind on every adoption path" begin
    Random.seed!(20260822)
    n = 72
    dates = collect(Date(2000, 1, 1):Month(1):Date(2000, 1, 1)+Month(n - 1))
    y = TimeArray(dates, 100.0 .+ cumsum(cumsum(randn(n))))

    common = (
        seasonality = 12,
        objectiveFunction = "mse",
        initialization = :penalized,
        showLogs = false,
    )

    @testset "premise: without guards the barred corner wins" begin
        # This is the load-bearing assertion of the whole file. If the unguarded search stops
        # choosing the term-free corner on this series, every test below becomes vacuous — so
        # the premise is asserted, never assumed.
        plain = auto(
            y;
            common...,
            d = 2,
            D = 0,
            requireTermsWhenOverDifferenced = false,
            requireMAWhenDoublyDifferenced = false,
        )
        @test plain.p + plain.q + plain.P + plain.Q == 0
        @test plain.q == 0
    end

    @testset "requireMAWhenDoublyDifferenced forces q >= 1 when d >= 2" begin
        model = auto(
            y;
            common...,
            d = 2,
            D = 0,
            requireTermsWhenOverDifferenced = false,
            requireMAWhenDoublyDifferenced = true,
        )
        @test model.d == 2
        # The guard's whole promise: no path may return q = 0 here.
        @test model.q >= 1
    end

    @testset "requireTermsWhenOverDifferenced removes the term-free corner" begin
        model = auto(
            y;
            common...,
            d = 2,
            D = 0,
            requireTermsWhenOverDifferenced = true,
            requireMAWhenDoublyDifferenced = false,
        )
        @test model.p + model.q + model.P + model.Q >= 1
    end

    @testset "guards are inert where they do not apply" begin
        # d = 1 is outside the MA guard's trigger, so enabling it must not change the search.
        base = auto(y; common..., d = 1, D = 0, requireMAWhenDoublyDifferenced = false)
        guarded = auto(y; common..., d = 1, D = 0, requireMAWhenDoublyDifferenced = true)
        @test (base.p, base.q, base.P, base.Q) == (guarded.p, guarded.q, guarded.P, guarded.Q)
    end
end
