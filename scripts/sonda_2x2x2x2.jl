# SONDA das celulas FATOR ATIVO do 2x2x2x2, com CONTROLE para separar duas causas.
# Ver comentario extenso no REPRODUCE.md. Em resumo: compara o arquivado (5 workers,
# commit 4e8cf11) com um reajuste em PROCESSO UNICO no ambiente indicado por ARM_ENV.
# Rodando a mesma sonda nos dois ambientes (4e8cf11 e v1.0.0) separa contencao de
# mudanca de comportamento.
# As celulas com fator = 1 NAO entram: o gancho `SARIMAX_PROBE_FATOR1` era patch do
# worktree `wt_probe`, destruido na limpeza do /tmp; nao existe em 4e8cf11 nem na v1.0.0.
using Sarimax, Dates, TimeSeries, Printf, Statistics, DataFrames, JuMP, Ipopt, LinearAlgebra, Random
BLAS.set_num_threads(1)
const FT="/Users/davi/dev/ForecastTester"; const ARQ="/Users/davi/dev/paper/dados_2x2x2x2"
const CEL=ENV["CEL"]; const STAT=ENV["STAT"]=="1"; const INV=ENV["INV"]=="1"
const N=parse(Int,get(ENV,"SONDA_N","20")); const TAG=ENV["TAG"]
num(x)=(s=strip(x,['"',' ']); isempty(s) ? nothing : tryparse(Float64,s))
serie(l)=(v=Float64[]; for x in split(l,","); y=num(x); y===nothing || push!(v,y); end; v)
tr=readlines("$FT/datasets/Monthly-train.csv")[2:end]; te=readlines("$FT/datasets/Monthly-test.csv")[2:end]
smape(a,f)=200/length(a)*sum(abs.(a.-f)./(abs.(a).+abs.(f)))
arq=Dict{Int,Tuple{Float64,NTuple{6,Int}}}()
for fn in filter(x->occursin("/$(CEL)_", x) && endswith(x,".csv") && !occursin("_prev",x), readdir(ARQ; join=true))
    for l in readlines(fn)
        p=split(l,","); (isempty(p) || p[1]=="sid" || length(p)<20 || p[9]=="NaN") && continue
        arq[parse(Int,p[1])] = (parse(Float64,p[9]), Tuple(parse(Int,p[i]) for i in 3:8))
    end
end
Random.seed!(20260829)
sids=sort(rand(collect(keys(arq)), min(N,length(arq))))
OPT=optimizer_with_attributes(Ipopt.Optimizer)
@printf("SONDA celula %s (stationary=%s, invertible=%s) — ambiente %s, processo unico, n=%d\n",
        CEL, STAT, INV, TAG, length(sids))
function roda(sids,tr,te,arq)
  ok=0; tot=0
  for sid in sids
    y=serie(tr[sid]); a=serie(te[sid]); (isempty(y)||isempty(a)) && continue
    df=DataFrame(y=y); ds=Sarimax.loadDataset(df)
    try
        m=Sarimax.auto(ds; seasonality=12, objectiveFunction="mse",
            initialization=:innovations, seasonalForm=:multiplicative,
            integrationTest="kpssShort", seasonalIntegrationTest="seas",
            searchMethod="stepwise", informationCriteria="aicc",
            stationary=STAT, stationarityMargin=1e-6,
            invertible=INV, invertibilityMargin=1e-6,
            assertStationarity=true, assertInvertibility=true,
            warmStartFromBox=true, maxTimeSeconds=120.0, rootMargin=1e-2,
            requireTermsWhenOverDifferenced=true, requireMAWhenDoublyDifferenced=false,
            optimizer=OPT, multistart=false, outlierDetection=false,
            maxp=5, maxq=5, maxP=2, maxQ=2, maxOrder=5, maxd=2, maxD=1,
            parallel=false, showLogs=false,
            # `exogDynamics` NAO EXISTE em 4e8cf11 — foi acrescentado depois (9607556).
            # Consequencia para a regra "declare todos os argumentos": a lista nao e
            # portavel entre commits; um argumento obrigatorio hoje e um MethodError ontem.
            (TAG == "v1.0.0" ? (; exogDynamics=:armax) : (;))...)
        Sarimax.predict!(m; stepsAhead=18)
        s2=smape(a[1:min(18,length(a))], Float64.(TimeSeries.values(m.forecast)))
        o2=(m.p,m.d,m.q,m.P,m.D,m.Q); s1,o1=arq[sid]
        bate = isapprox(s1,s2; atol=1e-4) && o1==o2
        ok+=bate; tot+=1
        @printf("%-7d %12.6f %12.6f  %-16s %-16s %s\n", sid, s1, s2, string(o1), string(o2),
            bate ? "IGUAL" : (o1==o2 ? "sMAPE difere" : "ORDEM difere"))
    catch e
        tot+=1; @printf("%-7d ERRO: %s\n", sid, first(replace(string(e),"\n"=>" "),55))
    end
    flush(stdout)
  end
  return ok,tot
end
ok,tot = roda(sids,tr,te,arq)
@printf("\nRESULTADO %s / %s: %d/%d reproduzem (%.1f%%)\n", CEL, TAG, ok, tot, 100*ok/max(tot,1))
