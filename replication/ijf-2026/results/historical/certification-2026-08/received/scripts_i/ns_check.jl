# [RECUPERADO DO TRANSCRIPT DA SESSAO EM 21/10/2026 — escrito 16/08 03:40]
# Item (3): o mesmo vies de expoente existe no caminho NAO-SAZONAL ja publicado?
# Gera AR nao-sazonal puro, ajusta com o pacote (P=0 => so o bloco nao-sazonal atua) e guarda
# as series para o R. O enigma a resolver: a validacao de 3e-5 daquele bloco aparentemente
# passou COM 1/nEf. Aqui nEf = T + p, entao o vies e T/(T+p) ~ 1,1% contra os 6-12% do sazonal.
using Sarimax, TimeSeries, Dates, Random, Printf
function main()
    Random.seed!(20260817)
    for rep = 1:6, p in (1,2)
        n = 180
        ϕt = p == 1 ? [0.6] : [0.5, 0.25]
        y = zeros(n + 60)
        for t = (p+1):length(y); y[t] = sum(ϕt[k]*y[t-k] for k=1:p) + randn(); end
        y = y[61:end]
        ta = TimeArray(collect(Date(2000,1,1):Month(1):Date(2000,1,1)+Month(length(y)-1)), y)
        m = SARIMA(ta, p, 0, 0; P=0, D=0, Q=0, seasonality=12, allowMean=false)
        try
            fit!(m; objectiveFunction="mse", initialization=:penalized,
                 stationary=true, invertible=false)
            @printf("rep=%d p=%d phi=%s\n", rep, p, string(round.(Float64.([m.ϕ...]);digits=6)))
        catch e; println("rep=$rep p=$p ERRO") end
        open("/tmp/m4s/nsser_$(rep)_$(p).csv","w") do io
            println(io,"y"); for v in y; println(io,v); end
        end
    end
end
main()
