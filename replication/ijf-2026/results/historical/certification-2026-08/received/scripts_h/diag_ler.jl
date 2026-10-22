# [RECUPERADO DO TRANSCRIPT DA SESSAO EM 21/10/2026 — escrito 20/08 13:50]
using DelimitedFiles, Statistics, Printf
const T,H,S,K = 180,12,12,3
println("DISCRIMINADOR — a equacao do pacote e ARMAX, e nao regressao-com-erros-ARIMA?\n")
println("  erros-ARIMA:  y_t = X_t'b + eta_t,      phi(B)Phi(B^s) eta_t = theta(B) e_t   [o que o R ajusta]")
println("  ARMAX:        phi(B)Phi(B^s) y_t = X_t'b + theta(B) e_t                        [sarima.jl:1489]")
println("  coincidem SSE o polinomio AR e 1  (p = P = 0)\n")
println("  PREDICAO:  erros_arima -> R ganha larga;  ma_puro -> empatam;  armax -> nos ganhamos larga\n")
@printf("%-13s %11s %11s %12s   %-24s %-24s\n","DGP","RMSE nosso","RMSE R","dlog medio","beta nosso","beta R")
for tipo in ("erros_arima","ma_puro","armax")
    J = readdlm("/tmp/m4s/diag/jl_$tipo.csv", ',', Float64)
    R = readdlm("/tmp/m4s/diag/R_$tipo.csv",  ',', Float64)
    dn=Float64[]; dr=Float64[]; bn=zeros(K); br=zeros(K); nk=0
    for s = 1:20
        A = readdlm("/tmp/m4s/diag/$tipo/" * @sprintf("rep%02d.csv", s), ',', Float64)
        v = A[T+1:T+H, 1]
        (any(isnan, J[s,1:H]) || any(isnan, R[s,1:H])) && continue
        push!(dn, sqrt(mean(abs2, J[s,1:H] .- v)))
        push!(dr, sqrt(mean(abs2, R[s,1:H] .- v)))
        bn .+= J[s,H+1:end]; br .+= R[s,H+1:end]; nk += 1
    end
    @printf("%-13s %11.4f %11.4f %+12.4f   %-24s %-24s\n", tipo, mean(dn), mean(dr),
            mean(log.(dn).-log.(dr)), string(round.(bn./nk,digits=3)), string(round.(br./nk,digits=3)))
end
println("\n  dlog < 0 => nosso melhor;  dlog > 0 => R melhor;  ~0 => equivalentes")
println("  beta verdadeiro = [2.0, -1.5, 0.8]")
