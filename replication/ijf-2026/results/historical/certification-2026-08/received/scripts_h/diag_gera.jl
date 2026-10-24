# [RECUPERADO DO TRANSCRIPT DA SESSAO EM 21/10/2026 — escrito 20/08 13:50]
# Tres DGPs, m=0, 20 replicas cada. Escreve CSVs que os dois lados consomem.
using Random, DelimitedFiles, Printf
const T, H, S, K = 180, 12, 12, 3
const BETA = [2.0, -1.5, 0.8]
function gera(seed, tipo)
    rng = MersenneTwister(seed); burn = 400; n = burn + T + H
    X = randn(rng, n, K); e = 0.9846 .* randn(rng, n)
    ϕ, θ, Φ = 0.6, 0.4, 0.8
    Xb = X * BETA; y = zeros(n)
    if tipo == "erros_arima"          # y = Xb + eta,  eta ~ ARMA(1,1)x(1,0,0)
        η = zeros(n)
        for t = 14:n; η[t] = ϕ*η[t-1] + Φ*η[t-12] - ϕ*Φ*η[t-13] + e[t] + θ*e[t-1]; end
        y = Xb .+ η
    elseif tipo == "ma_puro"          # y = Xb + eta,  eta ~ MA(1)x(0,0,1)   (poli AR = 1)
        Θ = -0.5; η = zeros(n)
        for t = 14:n; η[t] = e[t] + θ*e[t-1] + Θ*e[t-12] + θ*Θ*e[t-13]; end
        y = Xb .+ η
    else                              # armax: phi(B)Phi(B^s) y = Xb + theta(B) e
        for t = 14:n; y[t] = ϕ*y[t-1] + Φ*y[t-12] - ϕ*Φ*y[t-13] + Xb[t] + e[t] + θ*e[t-1]; end
    end
    rg = (n-T-H+1):n
    hcat(y[rg], X[rg, :])
end
for tipo in ("erros_arima","ma_puro","armax")
    d = "/tmp/m4s/diag/$tipo"; mkpath(d)
    for s = 1:20; writedlm(joinpath(d, @sprintf("rep%02d.csv", s)), gera(9000+s, tipo), ','); end
end
println("gerado: 3 DGPs x 20 replicas")
