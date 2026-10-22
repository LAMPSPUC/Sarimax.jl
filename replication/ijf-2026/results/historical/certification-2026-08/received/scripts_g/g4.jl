# [RECUPERADO DO TRANSCRIPT DA SESSAO EM 21/10/2026 — escrito 21/08 00:05]
# ============================================================================
# G4 — mecanismo com verdade conhecida.
# O braco com o termo cruzado FECHA a distancia de estimacao ate a ML?
# Alvo externo: coeficientes do arima(method="ML") do R nos MESMOS dados.
# DIRECAO PREDITA (escrita antes): fecha. Referencia atual: distancia mediana
# CSS->ML de 21,6 no misto sazonal, ~0 nos blocos puros.
# MATA: termo com sinal trocado — produziria numeros plausiveis e conclusao invertida.
# ============================================================================
using Sarimax, TimeSeries, Dates, Random, Statistics, Printf, DelimitedFiles, LinearAlgebra
BLAS.set_num_threads(1)
const s = 12
mkT(v)=TimeArray(collect(Date(2000,1,1):Month(1):Date(2000,1,1)+Month(length(v)-1)), v)
function gera(seed, T)
    rng=MersenneTwister(seed); ϕ,θ,Φ,Θ=0.6,0.4,0.7,-0.4; n=T+600
    e=randn(rng,n); y=zeros(n)
    for t=15:n
        y[t]=ϕ*y[t-1]+Φ*y[t-s]-ϕ*Φ*y[t-s-1]+e[t]+θ*e[t-1]+Θ*e[t-s]+θ*Θ*e[t-s-1]
    end
    y[end-T+1:end]
end
function ajusta(y, cross)
    m=SARIMA(mkT(y),1,0,1;P=1,D=0,Q=1,seasonality=s,allowMean=false,allowDrift=false)
    fit!(m; objectiveFunction="mse", initialization=:penalized, stationary=true,
         invertible=false, exactCrossTerm=cross, crossTermMinObs=0)
    [m.ϕ[1], m.θ[1], m.Φ[1], m.Θ[1]]
end
function main()
    T=200; N=10
    mkpath("/tmp/m4s/g4")
    verdade=[0.6,0.4,0.7,-0.4]
    dsem=Float64[]; dcom=Float64[]; nR=0
    println("G4 — distancia aos coeficientes de ML do R, T=$T, verdade=", verdade, "\n")
    @printf("%4s %34s %34s %10s %10s\n","rep","sem cruzado","com cruzado","d_sem","d_com")
    for r=1:N
        y=gera(500+r,T); writedlm("/tmp/m4s/g4/y$r.csv", y, ',')
        bs = try ajusta(y,false) catch; nothing end
        bc = try ajusta(y,true)  catch; nothing end
        (isnothing(bs)||isnothing(bc)) && continue
        run(pipeline(`Rscript -e "
          y <- scan('/tmp/m4s/g4/y$r.csv', quiet=TRUE)
          f <- stats::arima(ts(y,frequency=12), order=c(1,0,1),
                            seasonal=list(order=c(1,0,1),period=12),
                            include.mean=FALSE, method='ML')
          cat(unname(coef(f)), sep=',')
        "`, stdout="/tmp/m4s/g4/r$r.txt"), wait=true)
        bR = vec(readdlm("/tmp/m4s/g4/r$r.txt", ',', Float64))
        length(bR)!=4 && continue
        bR = [bR[1],bR[2],bR[3],bR[4]]
        ds=norm(bs.-bR); dc=norm(bc.-bR); push!(dsem,ds); push!(dcom,dc); nR+=1
        @printf("%4d %34s %34s %10.4f %10.4f\n", r, string(round.(bs,digits=3)),
                string(round.(bc,digits=3)), ds, dc)
        flush(stdout)
    end
    if nR>0
        @printf("\n  distancia mediana ao R: sem %.4f -> com %.4f  (%d replicas)\n",
                median(dsem), median(dcom), nR)
        melhora = median(dcom) < median(dsem)
        @printf("  fecha em %d/%d replicas\n", count(dcom .< dsem), nR)
        println("\n  ", melhora ? ">>> G4 PASSA: o termo fecha a distancia, direcao predita." :
                                  ">>> G4 FALHA: o termo AFASTA — verificar sinal.")
    else
        println("\n  *** nenhuma replica comparavel — R falhou ***")
    end
end
main()
