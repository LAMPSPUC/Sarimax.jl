# [RECUPERADO DO TRANSCRIPT DA SESSAO EM 21/10/2026 — escrito 20/08 13:49, versao RCall]
# ============================================================================
# DISCRIMINADOR: a equacao do pacote e ARMAX, nao regressao-com-erros-ARIMA?
#
#   erros-ARIMA (o que o forecast::Arima(xreg=) ajusta):
#       y_t = X_t'b + eta_t ,   phi(B)Phi(B^s) eta_t = theta(B) e_t
#   ARMAX (o que sarima.jl:1489 escreve):
#       phi(B)Phi(B^s) y_t = X_t'b + theta(B) e_t
#
# Os dois coincidem SSE o polinomio AR e 1 (p = P = 0).
# Tres DGPs, mesmo arnes:  se a leitura do codigo estiver certa,
#   (1) erros-ARIMA com AR  -> R ganha larga
#   (2) MA puro             -> empatam (identidade algebrica)
#   (3) ARMAX com AR        -> nos ganhamos larga
# Um so desses padroes e compativel com a hipotese. Os outros a matam.
# ============================================================================
using Sarimax, TimeSeries, Dates, Random, Statistics, Printf, LinearAlgebra, RCall
BLAS.set_num_threads(1)
const T, H, S, K = 180, 12, 12, 3
const BETA = [2.0, -1.5, 0.8]

function gera(seed, tipo)
    rng = MersenneTwister(seed); burn = 400; n = burn + T + H
    X = randn(rng, n, K); e = 0.9846 .* randn(rng, n)
    ϕ, θ, Φ, Θ = 0.6, 0.4, 0.8, 0.0
    Xb = X * BETA
    y = zeros(n)
    if tipo == :erros_arima            # y = Xb + eta,  eta ~ ARMA(1,1)x(1,0,0)
        η = zeros(n)
        for t = 14:n; η[t] = ϕ*η[t-1] + Φ*η[t-12] - ϕ*Φ*η[t-13] + e[t] + θ*e[t-1]; end
        y = Xb .+ η
    elseif tipo == :ma_puro            # y = Xb + eta,  eta ~ MA(1)x(0,0,1)  (AR = 1)
        Θ = -0.5; η = zeros(n)
        for t = 14:n; η[t] = e[t] + θ*e[t-1] + Θ*e[t-12] + θ*Θ*e[t-13]; end
        y = Xb .+ η
    else                               # ARMAX: phi(B)Phi(B^s) y = Xb + theta(B) e
        for t = 14:n
            y[t] = ϕ*y[t-1] + Φ*y[t-12] - ϕ*Φ*y[t-13] + Xb[t] + e[t] + θ*e[t-1]
        end
    end
    rg = (n-T-H+1):n
    (y[rg], X[rg, :], tipo == :ma_puro)
end

mkT(v)  = TimeArray(collect(Date(2000,1,1):Month(1):Date(2000,1,1)+Month(length(v)-1)), v)
mkTX(X) = TimeArray(collect(Date(2000,1,1):Month(1):Date(2000,1,1)+Month(size(X,1)-1)), X)

function nosso(y, X, mapuro)
    ordem = mapuro ? (0,0,1,0,0,1) : (1,0,1,1,0,0)
    mo = SARIMA(mkT(y[1:T]), mkTX(X), ordem[1], ordem[2], ordem[3];
                P=ordem[4], D=ordem[5], Q=ordem[6], seasonality=S,
                allowMean=false, allowDrift=false)
    fit!(mo; objectiveFunction="mse", initialization=:free, stationary=true, invertible=false)
    predict!(mo; stepsAhead=H)
    (Float64.(vec(values(mo.forecast))), Float64.([mo.exogCoefficients...]))
end

function doR(y, X, mapuro)
    ord = mapuro ? [0,0,1] : [1,0,1];  sord = mapuro ? [0,0,1] : [1,0,0]
    @rput y X ord sord T H S
    R"""
    suppressPackageStartupMessages(library(forecast))
    Xtr <- X[1:T, , drop=FALSE]; Xte <- X[(T+1):(T+H), , drop=FALSE]
    fit <- Arima(ts(y[1:T], frequency=S), order=ord,
                 seasonal=list(order=sord, period=S), xreg=Xtr,
                 include.mean=FALSE, method="ML")
    fr <- as.numeric(forecast(fit, xreg=Xte, h=H)$mean)
    bb <- as.numeric(tail(coef(fit), ncol(Xtr)))
    """
    (Float64.(rcopy(R"fr")), Float64.(rcopy(R"bb")))
end

rmse(f, v) = sqrt(mean(abs2, f .- v))
println("DISCRIMINADOR — 20 replicas por DGP, m=0, ordem fixada na verdade nos dois lados\n")
@printf("%-14s %11s %11s %11s   %-28s %s\n","DGP","RMSE nosso","RMSE R","dlog medio","beta nosso","beta R")
for tipo in (:erros_arima, :ma_puro, :armax)
    dn = Float64[]; dr = Float64[]; bn = zeros(K); br = zeros(K); nk = 0
    for s = 1:20
        y, X, mp = gera(9000+s, tipo)
        v = y[T+1:T+H]
        try
            fn, b1 = nosso(y, X, mp); fr, b2 = doR(y, X, mp)
            push!(dn, rmse(fn,v)); push!(dr, rmse(fr,v)); bn .+= b1; br .+= b2; nk += 1
        catch e; end
    end
    nk == 0 && (println("  $tipo FALHOU"); continue)
    dl = mean(log.(dn) .- log.(dr))
    @printf("%-14s %11.4f %11.4f %+11.4f   %-28s %s\n", tipo, mean(dn), mean(dr), dl,
            string(round.(bn./nk, digits=3)), string(round.(br./nk, digits=3)))
    flush(stdout)
end
println("\n  dlog < 0 => nosso melhor;  dlog > 0 => R melhor;  ~0 => equivalentes")
