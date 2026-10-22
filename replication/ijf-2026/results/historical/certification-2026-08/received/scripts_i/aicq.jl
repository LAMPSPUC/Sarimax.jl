# [RECUPERADO DO TRANSCRIPT DA SESSAO EM 21/10/2026 — escrito 16/08 03:59]
using Sarimax, TimeSeries, Dates, CSV, DataFrames
function main()
    tr = CSV.read("/Users/davi/dev/ForecastTester/datasets/Monthly-train.csv", DataFrame)
    rv(row) = Float64[x for x in row[2:end] if !ismissing(x)]
    ids = parse.(Int, readlines("/tmp/m4s/rand400.txt"))[1:100]
    specs = [(p,q,P,Q) for (p,q) in ((1,1),(0,1)) for P in 0:2 for Q in 0:1]
    open("/tmp/m4s/jq_aicc.csv","w") do io
        println(io, "sid,p,q,P,Q,npar,aicc,exata")
        for i in ids
            y = rv(tr[i,:]); length(y) < 60 && continue
            ta = TimeArray(collect(Date(2000,1,1):Month(1):Date(2000,1,1)+Month(length(y)-1)), y)
            for sp in specs
                try
                    m = SARIMA(ta, sp[1], 1, sp[2]; P=sp[3], D=0, Q=sp[4],
                               seasonality=12, allowMean=false, allowDrift=false)
                    fit!(m; objectiveFunction="mse", initialization=:penalized,
                         stationary=true, invertible=false)
                    _,_,ex = Sarimax.criterionLoglikeAndN(m)
                    println(io, join((i, sp..., Sarimax.get_hyperparameters_number(m),
                                      round(aicc(m);digits=6), ex), ","))
                catch e; end
            end
            flush(io)
        end
    end
end
main()
