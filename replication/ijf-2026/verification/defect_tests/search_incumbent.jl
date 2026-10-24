# The stepwise search must move from its incumbent, not from a rejected starting model.
#
# When the first candidate (2,d,2) is inadmissible, `stepwiseSearch` adopts the null model as
# the incumbent (`bestModel = considerModel ? bestModel : fitModel`) without moving the order
# that centres the neighbourhood (`p, q, P, Q`), which stays at the rejected start. Without the
# over-differencing guard this is usually repaired by accident: the next initial candidate
# (1,d,0) beats the null (0,d,0) and resets `p, q`. With `requireTermsWhenOverDifferenced = true`
# the null IS (1,d,0), the next candidate is the same model, it is never strictly better, and the
# search explores only the neighbours of the rejected (2,d,2). `forecast::auto.arima` gives an
# inadmissible fit an infinite criterion, so its null model replaces it and resets the orders.
#
# The series below reproduces the pattern of the M4 weekly series that carry most of the
# Table 3 / Table 5 difference (audit report weekly_mse_reconciliation.md): a random walk fitted
# with d = 2, so every candidate with an MA term has an MA root at the unit circle and is
# rejected, and only the pure-AR direction is admissible. Recorded, not fixed: a fix changes
# model selection.
@testset "stepwise search moves from its incumbent" begin
    rng = MersenneTwister(4242)
    n = 240
    e = randn(rng, n + 200)
    y = cumsum(e[201:end]) .+ 50.0
    ta = TimeArray(collect(Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))), y)
    settings = (d = 2, D = 0, seasonality = 1, maxp = 5, maxq = 5, maxOrder = 5,
                informationCriteria = "aicc", objectiveFunction = "mse", initialization = :innovations,
                stationary = false, invertible = false, assertStationarity = true,
                assertInvertibility = true, rootMargin = 1e-2, searchMethod = "stepwise")
    direct(p) = (m = SARIMA(ta, p, 2, 0; allowMean = false);
                 fit!(m; objectiveFunction = "mse", initialization = :innovations,
                      stationary = false, invertible = false); m)
    aicc220, aicc120 = aicc(direct(2)), aicc(direct(1))
    @test aicc220 < aicc120 - 5                   # premise: (2,2,0) is clearly better than (1,2,0)

    free = auto(ta; settings..., requireTermsWhenOverDifferenced = false)
    @test free.p >= 2 && free.q == 0              # control: the AR direction is explored

    guarded = auto(ta; settings..., requireTermsWhenOverDifferenced = true)
    # (2,2,0) satisfies the guard and is an immediate neighbour of the guarded incumbent (1,2,0):
    # a stepwise search that moved from its incumbent could not stop at a worse model.
    @test_broken aicc(guarded) <= aicc220 + 1e-6
    @test (guarded.p, guarded.q) == (1, 0)        # current behaviour, pinned
end
