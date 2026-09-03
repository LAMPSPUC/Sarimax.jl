# SONDA DE REPRODUTIBILIDADE — responde a pergunta do Luiz por MEDICAO, nao lendo diff:
#   "Rodando o script com TODOS os argumentos explicitos, sob a tag v1.0.0, ele reproduz
#    os numeros arquivados?"
#
# Desenho:
#  - amostra SEMEADA (Random.seed! fixo) das series ja arquivadas, por frequencia e braco;
#  - reajuste em PROCESSO UNICO. A outra maquina mediu que uma serie reproduz sozinha e
#    diverge sob dez workers, entao contencao nao pode entrar na comparacao;
#  - compara sMAPE (6 casas) e ORDEM selecionada, que e o que a tabela usa;
#  - TODOS os argumentos que afetam estimacao vao explicitos, mesmo os que coincidem com o
#    default de hoje. Nada aqui pode depender de default.
using Sarimax, Dates, TimeSeries, Printf, Statistics, DataFrames, JuMP, Ipopt, LinearAlgebra, Random
BLAS.set_num_threads(1)
const D    = "/Users/davi/dev/ForecastTester/datasets"
const ARQ  = "/Users/davi/dev/campanha_stable"
const N    = parse(Int, get(ENV,"SONDA_N","40"))
const FREQ = ENV["SONDA_FREQ"]; const BRACO = ENV["SONDA_BRACO"]; const S = parse(Int, ENV["SONDA_S"])
num(x)=(s=strip(x,['"',' ']); isempty(s) ? nothing : tryparse(Float64,s))
serie(l)=(v=Float64[]; for x in split(l,","); y=num(x); y===nothing || push!(v,y); end; v)
cap(f)=uppercase(f[1:1])*f[2:end]
tr=readlines("$D/$(cap(FREQ))-train.csv")[2:end]; te=readlines("$D/$(cap(FREQ))-test.csv")[2:end]
smape(a,f)=200/length(a)*sum(abs.(a.-f)./(abs.(a).+abs.(f)))
arq=Dict{Int,Tuple{Float64,NTuple{6,Int},String}}()
for fn in filter(x->occursin("$(FREQ)_$(BRACO)_", x) && endswith(x,".csv"), readdir(ARQ; join=true))
    for l in readlines(fn)
        p=split(l,","); (isempty(p) || p[1]=="commit" || length(p)<16 || p[13]=="NaN") && continue
        arq[parse(Int,p[5])] = (parse(Float64,p[13]), Tuple(parse(Int,p[i]) for i in 7:12), p[16])
    end
end
Random.seed!(20260829)
sids = sort(rand(collect(keys(arq)), min(N, length(arq))))
OPT = optimizer_with_attributes(Ipopt.Optimizer, "mumps_mem_percent" => 1000)
@printf("SONDA %s/%s — tag v1.0.0, processo unico, %d series semeadas de %d arquivadas\n\n",
        FREQ, BRACO, length(sids), length(arq))
println("sid     sMAPE arquivado   sMAPE agora     ordem arq        ordem agora      veredito")
# Laco dentro de FUNCAO, nao no topo: o escopo de topo do Julia ja me mordeu tres vezes
# nesta sessao — `ok += 1` num `for` de topo nao enxerga o `ok` global.
function roda(sids, tr, te, arq, S, BRACO, OPT)
ok=0; tot=0
for sid in sids
    y=serie(tr[sid]); a=serie(te[sid]); (isempty(y)||isempty(a)) && continue
    df=DataFrame(y=y); ds=Sarimax.loadDataset(df)
    try
        m = Sarimax.auto(ds;
            # --- TODOS explicitos, por ordem da lista do Luiz ---
            seasonality = S, objectiveFunction = (BRACO=="stable" ? "stable" : "mse"),
            initialization = :innovations, seasonalForm = :multiplicative,
            stationary = true, stationarityMargin = 1e-6,   # DEFAULT_DOMAIN_MARGIN, nao 1e-2 (esse e o rootMargin)
            invertible = false, invertibilityMargin = 1e-6,
            assertStationarity = true, assertInvertibility = true,
            searchMethod = "stepwise", informationCriteria = "aicc",
            integrationTest = "kpssShort", seasonalIntegrationTest = "seas",
            optimizer = OPT, cvarLevel = 0.5, multistart = false, rootMargin = 1e-2,
            warmStartFromBox = false, maxTimeSeconds = nothing,
            outlierDetection = false,
            requireTermsWhenOverDifferenced = false, requireMAWhenDoublyDifferenced = false,
            maxp = 5, maxq = 5, maxP = 2, maxQ = 2, maxOrder = 5, maxd = 2, maxD = 1,
            exogDynamics = :armax, parallel = false, showLogs = false)
        Sarimax.predict!(m; stepsAhead=length(a))
        fc=Float64.(TimeSeries.values(m.forecast))
        s2=smape(a,fc); o2=(m.p,m.d,m.q,m.P,m.D,m.Q)
        s1,o1,_ = arq[sid]
        bate = isapprox(s1,s2; atol=1e-4) && o1==o2
        ok += bate; tot += 1
        @printf("%-7d %14.6f %14.6f   %-16s %-16s %s\n", sid, s1, s2, string(o1), string(o2),
                bate ? "IGUAL" : (o1==o2 ? "sMAPE difere" : "ORDEM difere"))
    catch e
        tot += 1
        @printf("%-7d ERRO: %s\n", sid, first(replace(string(e),"\n"=>" "),60))
    end
    flush(stdout)
end
    return ok, tot
end
ok, tot = roda(sids, tr, te, arq, S, BRACO, OPT)
@printf("\nRESULTADO: %d/%d reproduzem (%.1f%%)\n", ok, tot, 100*ok/max(tot,1))
