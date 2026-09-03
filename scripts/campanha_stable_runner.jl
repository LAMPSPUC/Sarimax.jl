# Campanha `stable` — quarterly, hourly, daily. Desenho declarado ao Luiz em 28/08
# (msg 101227) ANTES do disparo. Config dele, deliberadamente:
#   :innovations, stationary=true, invertible=false, asserts, stepwise, serie INTEIRA,
#   SEM teto de tempo, e mumps_mem_percent=1000 nos DOIS bracos (medido inerte).
# Retomada por sid: morte de worker custa a serie corrente, nao a rodada.
using Sarimax, Dates, TimeSeries, Printf, Statistics, DataFrames, JuMP, Ipopt, LinearAlgebra
BLAS.set_num_threads(1)
const D   = "/Users/davi/dev/ForecastTester/datasets"
const OUT = ENV["ARM_OUT"]; const A = parse(Int,ENV["ARM_A"]); const B = parse(Int,ENV["ARM_B"])
const FREQ = ENV["ARM_FREQ"]; const BRACO = ENV["ARM_BRACO"]
const S = parse(Int, ENV["ARM_S"])
const COMMIT = "dev@fc2c482+PR26"
const ENVID  = "MOI1.53.0|Ipopt1.11.0|JuMP1.31.2"
num(x)=(s=strip(x,['"',' ']); isempty(s) ? nothing : tryparse(Float64,s))
serie(l)=(v=Float64[]; for x in split(l,","); y=num(x); y===nothing || push!(v,y); end; v)
cap(f)=uppercase(f[1:1])*f[2:end]
tr=readlines("$D/$(cap(FREQ))-train.csv")[2:end]; te=readlines("$D/$(cap(FREQ))-test.csv")[2:end]
smape(a,f)=200/length(a)*sum(abs.(a.-f)./(abs.(a).+abs.(f)))
# DEFEITO CORRIGIDO 28/08: o denominador do MASE da M4 e o naive SAZONAL (lag m), nao o
# lag 1. Com m=1 (weekly, daily) os dois coincidem — foi por isso que o weekly bateu
# digito a digito e o defeito passou. Em quarterly (m=4) e hourly (m=24) o MASE saia
# inflado ~2x. Os CSV ja gravados foram corrigidos a posteriori pelo fator
# mean|D1 y| / mean|Dm y|, que sai so da serie de treino; daqui em diante nasce certo.
function mase(a,f,y,m)
    den = length(y) > m ? mean(abs.(y[m+1:end] .- y[1:end-m])) : mean(abs.(diff(y)))
    den <= 0 && (den = 1e-9)
    return mean(abs.(a .- f)) / den
end
isfile(OUT) || open(OUT,"w") do io
    println(io,"commit,env,freq,braco,sid,T,p,d,q,P,D,Q,smape,mase,wall,status")
end
feitos=Set{Int}()
for l in readlines(OUT)[2:end]; p=split(l,","); length(p)>5 && push!(feitos, parse(Int,p[5])); end
OPT = optimizer_with_attributes(Ipopt.Optimizer, "mumps_mem_percent"=>1000)
comum=(; seasonality=S, searchMethod="stepwise", initialization=:innovations,
        stationary=true, invertible=false, assertStationarity=true,
        assertInvertibility=true, optimizer=OPT)
for i in A:min(B,length(tr))
    i in feitos && continue
    y=serie(tr[i]); a=serie(te[i]); (isempty(y)||isempty(a)) && continue
    df=DataFrame(y=y); ds=Sarimax.loadDataset(df)
    try
        w = @elapsed begin
            m = BRACO=="stable" ?
                Sarimax.auto(ds; comum..., objectiveFunction="stable", cvarLevel=0.5) :
                Sarimax.auto(ds; comum..., objectiveFunction="mse")
            Sarimax.predict!(m; stepsAhead=length(a))
        end
        fc=Float64.(TimeSeries.values(m.forecast))
        open(OUT,"a") do io
            @printf(io,"%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.3f,%s\n",
                COMMIT,ENVID,FREQ,BRACO,i,length(y),m.p,m.d,m.q,m.P,m.D,m.Q,
                smape(a,fc),mase(a,fc,y,S),w,get(m.metadata,"solverStatus","?"))
        end
    catch e
        open(OUT,"a") do io
            @printf(io,"%s,%s,%s,%s,%d,%d,-1,-1,-1,-1,-1,-1,NaN,NaN,NaN,ERRO\n",
                COMMIT,ENVID,FREQ,BRACO,i,length(y))
        end
    end
end
println("$FREQ/$BRACO [$A..$B] COMPLETO")
