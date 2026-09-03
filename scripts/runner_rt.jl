# Runner v2 — campanha daily sob a TAG v1.0.0.
# Diferencas para o v1: (a) grava o sid EM ANDAMENTO antes de chamar `auto`, para que o
# supervisor saiba qual serie envenenou o worker; (b) le uma lista de censurados e pula.
# O deadlock e INTRA-SERIE: o crash do MUMPS acontece entre candidatos dentro do mesmo
# `auto`, entao o sid marcado e o culpado. Ver ACHADO_MUMPS_REPRODUZ_28-08.md.
using Sarimax, Dates, TimeSeries, Printf, Statistics, DataFrames, JuMP, Ipopt, LinearAlgebra
BLAS.set_num_threads(1)
const D=ENV["ARM_DATA"]; const OUT=ENV["ARM_OUT"]; const MARK=ENV["ARM_MARK"]; const CENS=ENV["ARM_CENS"]
const A=parse(Int,ENV["ARM_A"]); const B=parse(Int,ENV["ARM_B"])
const FREQ=ENV["ARM_FREQ"]; const BRACO=ENV["ARM_BRACO"]; const S=parse(Int,ENV["ARM_S"])
const COMMIT="v1.0.0@a1ebc0d+reqTerms"; const ENVID="MOI1.53.0|Ipopt1.11.0|JuMP1.31.2|MUMPSseq5.4.1"
num(x)=(s=strip(x,['"',' ']); isempty(s) ? nothing : tryparse(Float64,s))
serie(l)=(v=Float64[]; for x in split(l,","); y=num(x); y===nothing || push!(v,y); end; v)
cap(f)=uppercase(f[1:1])*f[2:end]
tr=readlines("$D/$(cap(FREQ))-train.csv")[2:end]; te=readlines("$D/$(cap(FREQ))-test.csv")[2:end]
smape(a,f)=200/length(a)*sum(abs.(a.-f)./(abs.(a).+abs.(f)))
function mase(a,f,y,m)
    den = length(y) > m ? mean(abs.(y[m+1:end] .- y[1:end-m])) : mean(abs.(diff(y)))
    den <= 0 && (den = 1e-9); return mean(abs.(a .- f))/den
end
isfile(OUT) || open(OUT,"w") do io
    println(io,"commit,env,freq,braco,sid,T,p,d,q,P,D,Q,smape,mase,wall,status")
end
feitos=Set{Int}()
for l in readlines(OUT)[2:end]; p=split(l,","); length(p)>5 && push!(feitos,parse(Int,p[5])); end
censurados=Set{Int}()
if isfile(CENS)
    for l in readlines(CENS); p=split(l,","); length(p)>1 && (v=tryparse(Int,p[2]); v===nothing || push!(censurados,v)); end
end
OPT = optimizer_with_attributes(Ipopt.Optimizer, "mumps_mem_percent"=>1000)
function trabalha(tr,te,feitos,censurados)
  for i in A:min(B,length(tr))
    (i in feitos || i in censurados) && continue
    y=serie(tr[i]); a=serie(te[i]); (isempty(y)||isempty(a)) && continue
    write(MARK, string(i))                      # <-- marca ANTES de tentar
    df=DataFrame(y=y); ds=Sarimax.loadDataset(df)
    try
        w = @elapsed begin
            m = BRACO=="stable" ?
                Sarimax.auto(ds; seasonality=S, objectiveFunction="stable", cvarLevel=0.5,
                    initialization=:innovations, seasonalForm=:multiplicative,
                    stationary=true, stationarityMargin=1e-6, invertible=false,
                    invertibilityMargin=1e-6, assertStationarity=true, assertInvertibility=true,
                    searchMethod="stepwise", informationCriteria="aicc",
                    integrationTest="kpssShort", seasonalIntegrationTest="seas",
                    optimizer=OPT, multistart=false, warmStartFromBox=false,
                    maxTimeSeconds=nothing, outlierDetection=false, rootMargin=1e-2,
                    requireTermsWhenOverDifferenced=true, requireMAWhenDoublyDifferenced=false,
                    maxp=5, maxq=5, maxP=2, maxQ=2, maxOrder=5, maxd=2, maxD=1,
                    exogDynamics=:armax, parallel=false, showLogs=false) :
                Sarimax.auto(ds; seasonality=S, objectiveFunction="mse",
                    initialization=:innovations, seasonalForm=:multiplicative,
                    stationary=true, stationarityMargin=1e-6, invertible=false,
                    invertibilityMargin=1e-6, assertStationarity=true, assertInvertibility=true,
                    searchMethod="stepwise", informationCriteria="aicc",
                    integrationTest="kpssShort", seasonalIntegrationTest="seas",
                    optimizer=OPT, multistart=false, warmStartFromBox=false,
                    maxTimeSeconds=nothing, outlierDetection=false, rootMargin=1e-2,
                    requireTermsWhenOverDifferenced=true, requireMAWhenDoublyDifferenced=false,
                    maxp=5, maxq=5, maxP=2, maxQ=2, maxOrder=5, maxd=2, maxD=1,
                    exogDynamics=:armax, parallel=false, showLogs=false)
            Sarimax.predict!(m; stepsAhead=length(a))
        end
        fc=Float64.(TimeSeries.values(m.forecast))
        open(OUT,"a") do io
            @printf(io,"%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.3f,%s\n",COMMIT,ENVID,FREQ,BRACO,
                i,length(y),m.p,m.d,m.q,m.P,m.D,m.Q,smape(a,fc),mase(a,fc,y,S),w,
                get(m.metadata,"solverStatus","?"))
        end
    catch e
        open(OUT,"a") do io
            @printf(io,"%s,%s,%s,%s,%d,%d,-1,-1,-1,-1,-1,-1,NaN,NaN,NaN,ERRO\n",COMMIT,ENVID,FREQ,BRACO,i,length(y))
        end
    end
    isfile(MARK) && rm(MARK; force=true)         # <-- desmarca so depois da linha gravada
  end
end
trabalha(tr,te,feitos,censurados)
println("$FREQ/$BRACO [$A..$B] COMPLETO")
