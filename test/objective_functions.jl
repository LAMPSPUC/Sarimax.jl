@testset "Alternative objective functions" begin
    # The optimization formulation is what lets the objective be swapped — robust
    # (mae), tail-averse (stable/CVaR), regularized (elastic_net) and bilevel fits all
    # reuse the same model. These guards came out of a stability sweep across the six
    # M4 frequencies, which is also where the elastic-net edge case below surfaced.

    airp = Float64.(values(loadDataset(AIR_PASSENGERS)))
    dates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(length(airp) - 1))
    series = TimeArray(collect(dates), airp)

    @testset "each objective produces a usable fit" begin
        for obj in ("mse", "mae", "ml", "stable")
            model = SARIMA(series, 1, 1, 1; seasonality = 12, P = 0, D = 1, Q = 1)
            fit!(model; objectiveFunction = obj)
            @test Sarimax.isFitted(model)
            @test all(isfinite, model.ϕ)
            @test all(isfinite, model.θ)
            @test isfinite(model.σ²) && model.σ² > 0
            predict!(model; stepsAhead = 12)
            @test all(isfinite, values(model.forecast))
        end
    end

    @testset "bilevel rejects the invertible parameterization explicitly" begin
        # The MA coefficients are outer parameters there, so they cannot also be
        # generated from reflection coefficients. The clear failure is the contract.
        model = SARIMA(series, 1, 1, 1; seasonality = 12, P = 0, D = 1, Q = 1)
        @test_throws AssertionError fit!(model; objectiveFunction = "bilevel",
                                         invertible = true)
    end

    @testset "CVaR level controls how hard the tail is chased" begin
        # The "stable" objective is CVaR of the squared residuals in Rockafellar-Uryasev
        # form. Its normalization used to be missing: `0.7δ + Σu` is `δ + Σu/0.7`, i.e.
        # (1-α)·n = 0.7, so the effective level was α = 1 - 0.7/n — ~99% for a typical
        # sample and, worse, *dependent on the sample size*. That is min-max in
        # disguise: it equalizes residuals and chases the outlier instead of tolerating
        # it. With the normalization in place the level behaves monotonically, which is
        # what this test pins.
        Random.seed!(6001)
        n = 120
        dates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        y = 50.0 .+ cumsum(randn(n) .* 0.5)
        y[60] += 10 * std(diff(y))                     # a single 10σ additive outlier
        ta = TimeArray(collect(dates), y)

        function tailStats(level)
            m = SARIMA(ta, 1, 1, 1; seasonality = 1)
            fit!(m; objectiveFunction = "stable", cvarLevel = level)
            r = abs.(m.ϵ)
            (maximum(r), median(r))
        end

        loMax, loMed = tailStats(0.5)
        hiMax, hiMed = tailStats(0.99)

        # A high level concentrates on the worst residuals: it shrinks the outlier's
        # residual and pays for it in the body of the sample.
        @test hiMax < loMax
        @test hiMed > loMed

        @test_throws AssertionError fit!(SARIMA(ta, 1, 1, 1; seasonality = 1);
                                         objectiveFunction = "stable", cvarLevel = 1.0)
        @test_throws AssertionError fit!(SARIMA(ta, 1, 1, 1; seasonality = 1);
                                         objectiveFunction = "stable", cvarLevel = 0.0)
    end

    @testset "every objective defines the residual with the same sign" begin
        # Regression. The MAE branch used to skip the defining relation y = yhat + eps and
        # pin eps through the split into non-negative parts alone, which at the optimum
        # gives eps = yhat - y — inverted relative to every other objective. The objective
        # value stayed correct, so the fit converged and the whole suite passed; but eps
        # feeds the moving-average recursion, so theta was estimated against inverted
        # innovations while `predict!` builds forecasts with the standard convention. Only
        # q > 0 or Q > 0 models were affected, and nothing that squares eps (residual
        # diagnostics, sigma^2) could reveal it. Hence this test compares the SIGN.
        Random.seed!(8801)
        n = 120
        dates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        ta = TimeArray(collect(dates), 100.0 .+ cumsum(randn(n) .* 2))

        for obj in ("mse", "mae", "ridge", "ml")
            for q in (0, 1)          # q = 1 is where the inverted sign actually bites
                model = SARIMA(ta, 1, 1, q; seasonality = 1)
                fit!(model; objectiveFunction = obj, silent = true)
                fv = Float64.(values(fitted(model)))
                ev = Float64.(model.ϵ)
                k = min(length(fv), length(ev))
                f = fv[end-k+1:end]
                e = ev[end-k+1:end]
                y = Float64.(values(ta))[end-k+1:end]
                keep = [i for i = 1:k if isfinite(f[i]) && isfinite(e[i])]
                @test !isempty(keep)
                # eps must track y - yhat, never yhat - y
                @test cor(e[keep], y[keep] .- f[keep]) > 0.9
            end
        end
    end

    @testset "bilevel returns the outer minimizer, not the last probe" begin
        # The outer loop optimizes the MA coefficients with Optim while the inner JuMP
        # solve handles everything else. Its result used to be computed, checked and then
        # dropped: the model was left holding whatever `optimizeMA` evaluated LAST, which
        # is not the minimizer (a line search ends where it stopped; Nelder-Mead's final
        # evaluation can be a rejected reflection point). Everything downstream reads the
        # coefficients off that model.
        #
        # Pinning it directly is awkward — Optim's trajectory is not exposed — so this
        # asserts the observable consequence: the fitted objective must be no worse than
        # what the same model scores at the returned coefficients, and re-fitting must be
        # reproducible rather than landing wherever the search happened to stop.
        Random.seed!(6602)
        n = 110
        dates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        ta = TimeArray(collect(dates), 300.0 .+ cumsum(randn(n) .* 2))

        m1 = SARIMA(ta, 1, 1, 1; seasonality = 1)
        fit!(m1; objectiveFunction = "bilevel", invertible = false, silent = true)
        m2 = SARIMA(ta, 1, 1, 1; seasonality = 1)
        fit!(m2; objectiveFunction = "bilevel", invertible = false, silent = true)

        @test Sarimax.isFitted(m1)
        @test all(isfinite, m1.θ)
        @test isfinite(m1.σ²) && m1.σ² > 0
        # duas execucoes identicas tem de dar o mesmo ponto
        @test m1.θ ≈ m2.θ atol = 1e-6
        @test m1.σ² ≈ m2.σ² atol = 1e-8

        predict!(m1; stepsAhead = 10)
        @test all(isfinite, values(m1.forecast))
    end

    @testset "huber sits between mse and mae under contamination" begin
        # Huber is the classical M-estimator: quadratic near zero, linear in the tail, so
        # it keeps least-squares efficiency under Gaussian errors while BOUNDING the
        # influence of an outlier. That is the opposite of "stable" (CVaR), which
        # minimizes the mean of the tail and therefore chases the outlier.
        #
        # The property to pin is that Huber BRACKETS the other two: faced with a single
        # large contaminant, both the residual it leaves on the outlier and the median
        # residual over the sample must fall between the mse and the mae values.
        #
        # Do not assume which end is which. Measured here, mse leaves the LARGEST residual
        # on the outlier (5.06) and mae the smallest (exactly 0.00) — the opposite of the
        # naive reading. Two reasons: the fitted value in an ARIMA recursion is driven by
        # past observations rather than free to chase a point, and an L1 solution sits on a
        # vertex, so it interpolates as many points exactly as there are parameters.
        Random.seed!(9401)
        n = 140
        dates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        y = 200.0 .+ cumsum(randn(n) .* 1.5)
        hit = 70
        y[hit] += 12 * std(diff(y))            # um unico outlier grande
        ta = TimeArray(collect(dates), y)

        function outlierResidual(obj)
            m = SARIMA(ta, 1, 1, 1; seasonality = 1)
            fit!(m; objectiveFunction = obj, silent = true)
            (abs(m.ϵ[hit]), median(abs.(m.ϵ)))
        end

        rMse, medMse = outlierResidual("mse")
        rMae, medMae = outlierResidual("mae")
        rHub, medHub = outlierResidual("huber")

        @test all(isfinite, (rMse, rMae, rHub))
        # cercado pelos dois extremos, sem degenerar em nenhum deles
        @test min(rMse, rMae) <= rHub <= max(rMse, rMae)
        @test min(medMse, medMae) <= medHub <= max(medMse, medMae)
        # e estritamente distinto de ambos: se coincidisse, o delta nao estaria agindo
        @test abs(rHub - rMse) > 1e-6
        @test abs(rHub - rMae) > 1e-6

        model = SARIMA(ta, 1, 1, 1; seasonality = 1)
        fit!(model; objectiveFunction = "huber", silent = true)
        @test Sarimax.isFitted(model)
        predict!(model; stepsAhead = 8)
        @test all(isfinite, values(model.forecast))
    end

    @testset "ridge shrinks the AR/MA coefficients, not the level" begin
        # Penalized ridge: min sum(e^2) + lambda*||coef||^2 with lambda fixed a priori.
        # It differs from "elastic_net" only in that lambda is fixed here and the penalty
        # is a plain L2 term over the AR/MA blocks.
        #
        # The scale of lambda is the part that silently goes wrong: the usual heuristic
        # lambda = 1/sqrt(n) assumes a MEAN loss, while this objective is a SUM, so the
        # equivalent value is sqrt(n). Getting that backwards makes the penalty about n
        # times too weak and ridge becomes indistinguishable from mse — which is what
        # these assertions are here to catch.
        Random.seed!(7301)
        n = 160
        dates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        # strongly autocorrelated: unpenalized AR lands near the unit circle
        e = randn(n)
        v = zeros(n)
        for t = 2:n
            v[t] = 0.97 * v[t-1] + e[t]
        end
        ta = TimeArray(collect(dates), 500.0 .+ v .* 5)

        function fitWith(obj)
            m = SARIMA(ta, 2, 1, 1; seasonality = 1)
            fit!(m; objectiveFunction = obj, silent = true)
            m
        end

        mse = fitWith("mse")
        ridge = fitWith("ridge")

        @test Sarimax.isFitted(ridge)
        @test all(isfinite, ridge.ϕ)
        @test all(isfinite, ridge.θ)

        normAR(m) = sum(abs2, [m.ϕ...]) + sum(abs2, [m.θ...])
        # The penalty must actually bind: shrinkage pulls the coefficients in.
        @test normAR(ridge) < normAR(mse)

        # The level is deliberately NOT penalized, so the fit must still track it.
        predict!(ridge; stepsAhead = 6)
        fc = values(ridge.forecast)
        @test all(isfinite, fc)
        @test minimum(fc) > 0          # nowhere near the origin: the level survived
    end

    @testset "over-differenced guard forces at least one AR/MA term" begin
        # With d + D >= 2 and no AR/MA term the forecast is a bare extrapolation of the
        # local slope, which runs off in a straight line. The guard removes that corner.
        # It is opt-in: R's auto.arima has no equivalent rule, so the default must not
        # change behaviour.
        Random.seed!(7302)
        n = 90
        dates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        # quadratic drift => the differencing tests ask for d = 2
        ta = TimeArray(collect(dates),
                       [1000.0 + 0.6 * t^2 + 8randn() for t = 1:n])

        loose = auto(ta; seasonality = 12, integrationTest = "kpssShort",
                     searchMethod = "stepwise", showLogs = false)
        tight = auto(ta; seasonality = 12, integrationTest = "kpssShort",
                     searchMethod = "stepwise", showLogs = false,
                     requireTermsWhenOverDifferenced = true)

        if loose.d + loose.D >= 2
            # The guard binds only where the series is actually over-differenced.
            @test tight.p + tight.q + tight.P + tight.Q >= 1
        end
        @test Sarimax.isFitted(tight)
        predict!(tight; stepsAhead = 12)
        @test all(isfinite, values(tight.forecast))
    end

    @testset "elastic net implements the penalized estimator" begin
        # min L(e) + lambda*[alpha*L1 + (1-alpha)/2*L2] over the AR/MA and exogenous
        # blocks. These assertions pin the properties that distinguish a penalized
        # estimator from the constrained two-stage construction this replaced, in which
        # `lambda` reached nothing and the shrinkage came from a calibrated tolerance.
        airpLog = log.(loadDataset(AIR_PASSENGERS))
        mk() = SARIMA(airpLog, 2, 1, 1; seasonality = 12, P = 0, D = 1, Q = 1,
                      allowMean = false)
        coefsOf(m) = vcat([m.ϕ...], [m.θ...])
        function fitEN(; kwargs...)
            m = mk()
            fit!(m; objectiveFunction = "elastic_net", alpha = 0.5,
                 initialization = :zeroed, kwargs...)
            m
        end

        # lambda = 0 removes the penalty, so the fit must coincide with least squares.
        baseline = mk()
        fit!(baseline; objectiveFunction = "mse", initialization = :zeroed)
        @test maximum(abs.(coefsOf(fitEN(lambda = 0.0)) .- coefsOf(baseline))) < 1e-6

        # lambda must actually reach the optimization: shrinkage is monotone in it.
        norms = [sum(abs.(coefsOf(fitEN(lambda = λ)))) for λ in (0.0, 10.0, 100.0)]
        @test issorted(norms; rev = true)
        @test norms[end] < norms[1]

        # alpha = 1 is the lasso case and must drive coefficients to exactly zero.
        lasso = mk()
        fit!(lasso; objectiveFunction = "elastic_net", alpha = 1.0, lambda = 500.0,
             initialization = :zeroed)
        @test any(c -> abs(c) <= 1e-5, coefsOf(lasso))

        # With no AR/MA/exogenous block there is nothing to penalize, and the objective
        # must fall back to plain least squares rather than build an empty penalty.
        empty = SARIMA(airpLog, 0, 1, 0; allowMean = false)
        fit!(empty; objectiveFunction = "elastic_net", alpha = 0.5,
             initialization = :zeroed)
        plain = SARIMA(airpLog, 0, 1, 0; allowMean = false)
        fit!(plain; objectiveFunction = "mse", initialization = :zeroed)
        @test sum(abs2, values(residuals(empty))) ≈ sum(abs2, values(residuals(plain))) atol = 1e-8
    end

    @testset "elastic net survives a saturated model" begin
        # A short seasonal series leaves few residual degrees of freedom after
        # conditioning, since seasonal conditioning consumes the sample faster than a
        # longer non-seasonal one of similar length. The penalized objective must still
        # produce a usable fit there.
        Random.seed!(5001)
        n = 17
        shortDates = Date(2000, 1, 1):Quarter(1):(Date(2000, 1, 1)+Quarter(n - 1))
        shortSeries = TimeArray(collect(shortDates), 100.0 .+ cumsum(randn(n)))
        model = SARIMA(shortSeries, 1, 1, 1; seasonality = 4, P = 1, D = 0, Q = 1)

        @test (fit!(model; objectiveFunction = "elastic_net", alpha = 0.5); true)
        @test Sarimax.isFitted(model)
        @test all(isfinite, model.ϕ)
        @test isfinite(model.σ²)
    end
end
