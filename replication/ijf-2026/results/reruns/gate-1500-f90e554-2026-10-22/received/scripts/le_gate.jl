# Leitor do gate (le_gate.jl de 21/08); unica troca: os dois CSVs vem por ARGS: julia le_gate.jl m1500_pen.csv m1500_inov.csv
# [RECUPERADO DO TRANSCRIPT DA SESSAO EM 21/10/2026 — escrito 21/08 11:04]
using DelimitedFiles, Statistics, Printf, Random
A=readdlm(ARGS[1], ',', Any); B=readdlm(ARGS[2], ',', Any)
ha=Dict(A[i,1]=>i for i=2:size(A,1)); hb=Dict(B[i,1]=>i for i=2:size(B,1))
ids=sort(collect(intersect(keys(ha),keys(hb))))
sa=[Float64(A[ha[i],9]) for i in ids]; sb=[Float64(B[hb[i],9]) for i in ids]
ma=[Float64(A[ha[i],10]) for i in ids]; mb=[Float64(B[hb[i],10]) for i in ids]
wa=[Float64(A[ha[i],11]) for i in ids]; wb=[Float64(B[hb[i],11]) for i in ids]
ita=[Float64(A[ha[i],16]) for i in ids]; itb=[Float64(B[hb[i],16]) for i in ids]
oka=count(x->A[ha[x],20]=="OK",ids); okb=count(x->B[hb[x],20]=="OK",ids)
function boot(v;B_=10_000,seed=20260821)
    rng=MersenneTwister(seed); n=length(v); m=zeros(B_)
    for b=1:B_; a=0.0; for _=1:n; a+=v[rand(rng,1:n)]; end; m[b]=a/n; end
    (mean(v), quantile(m,0.025), quantile(m,0.975))
end
println("GATE DAS 1.500 — pareado, mesmas series, commit f90e554\n")
@printf("%-16s %10s %10s %10s %12s %10s\n","braco","sMAPE","MASE","wall (s)","teto 200 (%)","status OK")
@printf("%-16s %10.4f %10.4f %10.3f %11.1f%% %9d\n",":penalized",mean(sa),mean(ma),mean(wa),100*count(>=(200),ita)/length(ita),oka)
@printf("%-16s %10.4f %10.4f %10.3f %11.1f%% %9d\n",":innovations",mean(sb),mean(mb),mean(wb),100*count(>=(200),itb)/length(itb),okb)
ds=sb.-sa; dm=mb.-ma
for (nm,d) in (("sMAPE",ds),("MASE",dm))
    mu,lo,hi=boot(d)
    @printf("\n  delta %s:  media %+.4f  IC95 [%+.4f, %+.4f]  mediana %+.4f  melhora %d/%d (%.1f%%)\n",
            nm,mu,lo,hi,median(d),count(<(0),d),length(d),100*count(<(0),d)/length(d))
end
@printf("\n  n pareado = %d   |  tempo total: %.0fs vs %.0fs\n", length(ids), sum(wa), sum(wb))
