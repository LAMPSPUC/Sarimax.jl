# Runner minimo sobre a M4 monthly do ForecastTester.
#
# NAO usa o modulo `ForecastTester`: ele faz `using RCall` no topo (src/ForecastTester.jl:3),
# entao o harness inteiro exige R instalado, e aqui nao ha R. Os dados, a config do v0_8 e as
# metricas sao lidos/replicados direto — as definicoes de smape/mase sao as do `v0_8_worker.jl`,
# copiadas verbatim para os numeros serem comparaveis com `v0_8_monthly.csv`.
#
# Roda com o Sarimax do branch de telemetria:
#   julia --project=/Users/davi/dev/Sarimax.jl m4_runner.jl <lista> <n> <saida>
using Sarimax, CSV, DataFrames, TimeSeries, Dates, Statistics, Printf
using LinearAlgebra, Random
BLAS.set_num_threads(1)   # ver armadilha de oversubscription do briefing

const FT = "/Users/davi/dev/ForecastTester"
const H = 18          # horizonte da M4 monthly
const S = 12

smape(f, a) = mean(2 .* abs.(f .- a) ./ (abs.(f) .+ abs.(a) .+ 1e-12)) * 100
function mase(f, a, y, m)
    den = m < length(y) ? mean(abs.(y[m+1:end] .- y[1:end-m])) : mean(abs.(diff(y)))
    den <= 0 && (den = 1e-9)
    mean(abs.(f .- a)) / den
end

rowvals(row) = Float64[x for x in row[2:end] if !ismissing(x)]

function runSeries(sid::Int, y::Vector{Float64}, yt::Vector{Float64})
    h = min(H, length(yt))
    t0 = time()
    try
        dates = collect(Date(2000, 1, 1):Month(1):Date(2000, 1, 1)+Month(length(y) - 1))
        m = auto(TimeArray(dates, y); seasonality = S,
                 integrationTest = "kpssShort", seasonalIntegrationTest = "seas",
                 searchMethod = "stepwise", objectiveFunction = "mse",
                 initialization = :penalized, warmStartFromBox = true,
                 maxTimeSeconds = 120.0, requireTermsWhenOverDifferenced = true,
                 stationary = false, invertible = false,
                 assertStationarity = true, assertInvertibility = true,
                 rootMargin = 1e-2)
        predict!(m; stepsAhead = H)
        f = Float64.(TimeSeries.values(m.forecast))

        tel = get(m.metadata, "searchTelemetry", Dict{String,Dict{String,Any}}())
        nc = length(tel)
        gv(k) = collect(skipmissing([get(v, k, missing) for v in values(tel)]))
        build = sum(gv("buildTimeSec"); init = 0.0)
        solve = sum(gv("solveTimeSec"); init = 0.0)
        iters = gv("solverIterations")
        fits  = sum(gv("fitCount"); init = 0)
        fb    = count(v -> get(v, "criterionFallback", missing) === true, values(tel))
        ks    = [get(v, "K", missing) for v in values(tel)]
        ks    = collect(skipmissing(ks))

        (sid = sid, T = length(y), p = m.p, d = m.d, q = m.q, P = m.P, D = m.D, Q = m.Q,
         smape = round(smape(f[1:h], yt[1:h]); digits = 6),
         mase = round(mase(f[1:h], yt[1:h], y, S); digits = 6),
         wall = round(time() - t0; digits = 3),
         cands = nc, fits = fits,
         build = round(build; digits = 4), solve = round(solve; digits = 4),
         iterMax = isempty(iters) ? -1 : maximum(iters),
         iterMed = isempty(iters) ? -1 : sort(iters)[cld(length(iters), 2)],
         fallback = fb,
         kMax = isempty(ks) ? -1 : maximum(ks),
         status = "OK")
    catch e
        (sid = sid, T = length(y), p = -1, d = -1, q = -1, P = -1, D = -1, Q = -1,
         smape = NaN, mase = NaN, wall = round(time() - t0; digits = 3),
         cands = 0, fits = 0, build = 0.0, solve = 0.0, iterMax = -1, iterMed = -1,
         fallback = 0, kMax = -1,
         status = "ERRO:" * first(replace(sprint(showerror, e), '\n' => ' ', ',' => ' '), 60))
    end
end

function main()
    listfile = ARGS[1]                       # "estrato_pior.txt" ou "aleatorio"
    n        = parse(Int, ARGS[2])
    out      = ARGS[3]

    @info "lendo M4 monthly..."
    train = CSV.read(joinpath(FT, "datasets/Monthly-train.csv"), DataFrame)
    test  = CSV.read(joinpath(FT, "datasets/Monthly-test.csv"), DataFrame)
    @info "series no treino: $(nrow(train))"

    ids = if listfile == "aleatorio"
        Random.seed!(20260814); sort(randperm(nrow(train))[1:n])
    else
        parse.(Int, readlines(joinpath(FT, listfile)))[1:min(n, end)]
    end

    open(out, "w") do io
        println(io, "sid,T,p,d,q,P,D,Q,smape,mase,wall,cands,fits,build,solve,iterMax,iterMed,fallback,kMax,status")
        for (k, sid) in enumerate(ids)
            y  = rowvals(train[sid, :])
            yt = rowvals(test[sid, :])
            r  = runSeries(sid, y, yt)
            println(io, join(values(r), ","))
            flush(io)
            k % 5 == 0 && @info "  $k/$(length(ids)) — ultima: sid=$sid wall=$(r.wall)s status=$(r.status)"
        end
    end
    @info "pronto: $out"
end

main()
