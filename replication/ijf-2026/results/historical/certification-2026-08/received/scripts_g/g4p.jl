# [RECUPERADO DO TRANSCRIPT DA SESSAO EM 21/10/2026 — escrito 21/08 01:09]
# G4' — bracos C e D, pre-registrado em GATE_TERMO_CRUZADO_20-08.md
#   A = 0,2076 (S antiga, sem det)   B = 0,4294 (S antiga, com det)   -- ja medidos
#   C = S nova (:innovations), sem det      D = S nova, com det
# PRIMARIO: D fecha se mediana < 0,2076 E os 4 coeficientes estritamente interiores.
using Sarimax, TimeSeries, Dates, Random, Statistics, Printf, DelimitedFiles, LinearAlgebra
BLAS.set_num_threads(1)
const s=12
mkT(v)=TimeArray(collect(Date(2000,1,1):Month(1):Date(2000,1,1)+Month(length(v)-1)), v)
function gera(seed,T)
    rng=MersenneTwister(seed); ϕ,θ,Φ,Θ=0.6,0.4,0.7,-0.4; n=T+600
    e=randn(rng,n); y=zeros(n)
    for t=15:n; y[t]=ϕ*y[t-1]+Φ*y[t-s]-ϕ*Φ*y[t-s-1]+e[t]+θ*e[t-1]+Θ*e[t-s]+θ*Θ*e[t-s-1]; end
    y[end-T+1:end]
end
function aj(y, cross)
    m=SARIMA(mkT(y),1,0,1;P=1,D=0,Q=1,seasonality=s,allowMean=false,allowDrift=false)
    fit!(m; objectiveFunction="mse", initialization=:innovations, stationary=true,
         invertible=false, exactCrossTerm=cross, crossTermMinObs=0)
    [m.ϕ[1], m.θ[1], m.Φ[1], m.Θ[1]]
end
interior(b; lim=1-1e-6, tol=1e-7) = all(abs.(b) .< lim - tol)
function main()
    T=200; N=10; mkpath("/tmp/m4s/g4")
    println("G4' — bracos C e D, T=$T, verdade=[0.6,0.4,0.7,-0.4]   (A=0,2076  B=0,4294)")
    println("PACOTE @ ", strip(read(`git -C /Users/davi/dev/Sarimax.jl rev-parse --short HEAD`,String)),
            isempty(strip(read(`git -C /Users/davi/dev/Sarimax.jl status --porcelain src`,String))) ?
            "  (arvore limpa)" : "  *** ARVORE SUJA ***", "\n")
    @printf("%4s %30s %30s %8s %8s %6s\n","rep","C (inov, sem det)","D (inov, com det)","d_C","d_D","int D")
    dC=Float64[]; dD=Float64[]; intD=Bool[]
    for r=1:N
        y=gera(500+r,T); writedlm("/tmp/m4s/g4/y$r.csv", y, ',')
        bc = try aj(y,false) catch ex; println("   C: ",ex); nothing end
        bd = try aj(y,true)  catch ex; println("   D: ",ex); nothing end
        (isnothing(bc)||isnothing(bd)) && continue
        run(pipeline(`Rscript -e "
          y <- scan('/tmp/m4s/g4/y$r.csv', quiet=TRUE)
          f <- stats::arima(ts(y,frequency=12), order=c(1,0,1),
                 seasonal=list(order=c(1,0,1),period=12), include.mean=FALSE, method='ML')
          cat(unname(coef(f)), sep=',')"`, stdout="/tmp/m4s/g4/r$r.txt"), wait=true)
        bR=vec(readdlm("/tmp/m4s/g4/r$r.txt", ',', Float64)); length(bR)!=4 && continue
        c_,d_ = norm(bc.-bR), norm(bd.-bR)
        push!(dC,c_); push!(dD,d_); push!(intD, interior(bd))
        @printf("%4d %30s %30s %8.4f %8.4f %6s\n", r, string(round.(bc,digits=3)),
                string(round.(bd,digits=3)), c_, d_, interior(bd) ? "sim" : "NAO")
        flush(stdout)
    end
    isempty(dD) && (println("\n  *** nenhuma replica ***"); return)
    @printf("\n  mediana:  C %.4f   D %.4f     (A 0,2076   B 0,4294)   [%d replicas]\n",
            median(dC), median(dD), length(dD))
    @printf("  C - A = %+.4f  (efeito da parametrizacao)\n", median(dC)-0.2076)
    @printf("  D - C = %+.4f  (efeito do determinante sob pareamento correto)\n", median(dD)-median(dC))
    @printf("  coeficientes estritamente interiores em D: %d/%d\n", count(intD), length(intD))
    passa = median(dD) < 0.2076 && all(intD)
    println("\n  ", passa ? ">>> G4' PASSA — D fecha abaixo de A, com todos interiores." :
                            ">>> G4' FALHA.")
end
main()
