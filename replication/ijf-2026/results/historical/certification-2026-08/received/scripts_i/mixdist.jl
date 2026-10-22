# [RECUPERADO DO TRANSCRIPT DA SESSAO EM 21/10/2026 — escrito 16/08 04:33]
# A distancia CSS->ML na classe MISTA e o elo final do mecanismo. A extensao sazonal corrigida
# a fecha? Medido contra o loglik de ML do R (alvo externo), nos mesmos ajustes.
using Sarimax, TimeSeries, Dates, CSV, DataFrames, Printf, Statistics
function main()
    sp = ARGS[1] == "sim"
    tr = CSV.read("/Users/davi/dev/ForecastTester/datasets/Monthly-train.csv", DataFrame)
    rv(row) = Float64[x for x in row[2:end] if !ismissing(x)]
    R = CSV.read("/tmp/m4s/rmix.csv", DataFrame)
    ds = Float64[]
    for r in eachrow(R)
        y = rv(tr[r.sid,:])
        ta = TimeArray(collect(Date(2000,1,1):Month(1):Date(2000,1,1)+Month(length(y)-1)), y)
        try
            m = SARIMA(ta, 1, 1, 1; P=1, D=0, Q=1, seasonality=12, allowMean=false, allowDrift=false)
            kw = sp ? (seasonalPresamplePrior=true,) : NamedTuple()
            fit!(m; objectiveFunction="mse", initialization=:penalized,
                 stationary=true, invertible=false, kw...)
            ll = Sarimax.exactLoglike(m)
            isnothing(ll) || push!(ds, r.loglik - ll)
        catch e; end
    end
    @printf("seasonalPresamplePrior=%s  n=%d  distancia CSS->ML: mediana %.2f  media %.2f  max %.2f\n",
            sp, length(ds), median(ds), mean(ds), maximum(ds))
end
main()
