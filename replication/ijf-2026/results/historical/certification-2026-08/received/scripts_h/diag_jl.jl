# [RECUPERADO DO TRANSCRIPT DA SESSAO EM 21/10/2026 — escrito 20/08 13:50]
using Sarimax, TimeSeries, Dates, DelimitedFiles, Printf, LinearAlgebra
BLAS.set_num_threads(1)
const T, H, S, K = 180, 12, 12, 3
mkT(v)  = TimeArray(collect(Date(2000,1,1):Month(1):Date(2000,1,1)+Month(length(v)-1)), v)
mkTX(X) = TimeArray(collect(Date(2000,1,1):Month(1):Date(2000,1,1)+Month(size(X,1)-1)), X)
for tipo in ("erros_arima","ma_puro","armax")
    mp = tipo == "ma_puro"
    o  = mp ? (0,0,1,0,0,1) : (1,0,1,1,0,0)
    R  = fill(NaN, 20, H+K)
    for s = 1:20
        A = readdlm("/tmp/m4s/diag/$tipo/" * @sprintf("rep%02d.csv", s), ',', Float64)
        try
            mo = SARIMA(mkT(A[1:T,1]), mkTX(A[:,2:end]), o[1],o[2],o[3];
                        P=o[4], D=o[5], Q=o[6], seasonality=S, allowMean=false, allowDrift=false)
            fit!(mo; objectiveFunction="mse", initialization=:free, stationary=true, invertible=false)
            predict!(mo; stepsAhead=H)
            R[s,1:H] = Float64.(vec(values(mo.forecast)))
            R[s,H+1:end] = Float64.([mo.exogCoefficients...])
        catch e; end
    end
    writedlm("/tmp/m4s/diag/jl_$tipo.csv", R, ',')
end
println("bracos Julia ok")
