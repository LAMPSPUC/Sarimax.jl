# ARNES EXTERNO — o pacote NAO e tocado.
# Desenho do Fable, 21/08: celulas A/B/C/D, gap-no-timeout como PRODUTO,
# celula adversarial contra o vies de selecao. Piloto 12/40 morto e descartado.
#
#   julia --project=/Users/davi/dev/Sarimax.jl scip_prog.jl <i1> <i2> <saida.csv>
using JuMP, SCIP, Ipopt, Random, Printf, LinearAlgebra, CSV, DataFrames, Statistics
BLAS.set_num_threads(1)
const s  = 12
const TL = 600.0                 # NUNCA estendido
const FT = "/Users/davi/dev/ForecastTester"
const B  = 1 - 1e-6              # caixa = regiao admissivel (ordem 1 por bloco)

# ---------------------------------------------------------------- casos
struct Caso
    id::Int; cel::String; P::Int; Q::Int; T::Int; rep::Int
    modo::Symbol; rotulo::String
    par::NTuple{4,Float64}       # phi, theta, Phi, Theta do DGP (NaN se serie real)
    sid::Int                     # >0 se serie real da M4
end

"Sorteios de DGP semeados na caixa admissivel. Semente fixa, declarada."
function sorteio(cel::String, rep::Int, P::Int, Q::Int)
    rng = MersenneTwister(hash((cel, rep, P, Q)) % 100000)
    dr() = round(rand(rng) * 1.6 - 0.8; digits = 3)     # (-0.8, 0.8)
    ϕ, θ = dr(), dr()
    Φ = P > 0 ? dr() : 0.0
    Θ = Q > 0 ? dr() : 0.0
    # celula B, sorteio 3: DELIBERADAMENTE perto da fronteira sazonal
    if cel == "B" && rep == 3 && P > 0
        Φ = 0.88
    end
    (ϕ, θ, Φ, Θ)
end

function casos()
    cs = Caso[]; id = 0
    # ---- A: fronteira nao-sazonal, T ate 240 testa (nao assume) o efeito do bound OBBT
    for T in (30, 60, 120, 240), r in 1:3
        id += 1; push!(cs, Caso(id, "A", 0, 0, T, r, :zeroed, "fronteira-T",
                                sorteio("A", r, 0, 0), 0))
    end
    # ---- B: fronteira sazonal; sorteio 3 encosta na fronteira (|Phi| = 0.88)
    for (P, Q) in ((0, 1), (1, 1)), T in (36, 60), r in 1:3
        id += 1; push!(cs, Caso(id, "B", P, Q, T, r, :zeroed,
                                r == 3 ? "sazonal-fronteira" : "sazonal",
                                sorteio("B", r, P, Q), 0))
    end
    # ---- C: adversarial — onde otimos locais sao PLAUSIVEIS
    # C1 quase-cancelamento regular: raiz AR e MA quase coincidem (phi = -theta)
    id += 1; push!(cs, Caso(id, "C", 0, 0, 60, 1, :zeroed, "quase-cancelamento-regular",
                            (0.700, -0.699, 0.0, 0.0), 0))
    # C2 quase-cancelamento sazonal
    id += 1; push!(cs, Caso(id, "C", 1, 1, 60, 2, :zeroed, "quase-cancelamento-sazonal",
                            (0.500, 0.300, 0.700, -0.699), 0))
    # C3 raiz MA quase no circulo unitario — onde Szego converge devagar e o CSS e pior
    id += 1; push!(cs, Caso(id, "C", 0, 0, 60, 3, :zeroed, "MA-quase-nao-inversivel",
                            (0.500, 0.970, 0.0, 0.0), 0))
    # C4-C6 SERIES REAIS da M4, ordem fixa (1,0,1)(0,0,1)[12].
    # Regra de selecao declarada: menores sids com T<=72, censo escolhendo d=D=0 e
    # termo sazonal ARMA. Nao ha escolha por resultado.
    for (k, sid) in enumerate((2772, 7389, 7510))
        id += 1; push!(cs, Caso(id, "C", 0, 1, 0, k, :zeroed, "M4-real", (NaN,NaN,NaN,NaN), sid))
    end
    # ---- D: modo, 2 casos :innovations nao-sazonais em T=60 (mesmo DGP de A/rep1-2)
    for r in 1:2
        id += 1; push!(cs, Caso(id, "D", 0, 0, 60, r, :innovations, "modo-inovacoes",
                                sorteio("A", r, 0, 0), 0))
    end
    cs
end

# ---------------------------------------------------------------- dados
function gera(c::Caso)
    ϕ, θ, Φ, Θ = c.par
    rng = MersenneTwister(1000 + c.id); n = c.T + 400
    e = randn(rng, n); y = zeros(n)
    for t = 15:n
        y[t] = ϕ*y[t-1] + e[t] + θ*e[t-1]
        c.P > 0 && (y[t] += Φ*y[t-s] - ϕ*Φ*y[t-s-1])
        c.Q > 0 && (y[t] += Θ*e[t-s] + θ*Θ*e[t-s-1])
    end
    y[end-c.T+1:end]
end

const TRAIN = Ref{Union{Nothing,DataFrame}}(nothing)
function serie_real(sid::Int)
    TRAIN[] === nothing && (TRAIN[] = CSV.read(joinpath(FT, "datasets/Monthly-train.csv"), DataFrame))
    v = Float64[x for x in TRAIN[][sid, 2:end] if !ismissing(x)]
    # PADRONIZADA: a assercao do piso de ruido e a caixa pressupoem inovacao ~ unitaria.
    (v .- mean(v)) ./ std(v)
end

dados(c::Caso) = c.sid > 0 ? serie_real(c.sid) : gera(c)

# ---------------------------------------------------------------- modelo
"CSS em forma BILINEAR (produtos de coeficientes viram variaveis) => tudo quadratico."
function monta(y, c::Caso, opt; ub = nothing, warm = nothing)
    T = length(y); P, Q = c.P, c.Q
    lb = 1 + max(1, (P > 0 || Q > 0) ? s + 1 : 0)
    m = Model(opt)
    @variable(m, -B <= ϕ <= B); @variable(m, -B <= θ <= B)
    @variable(m, -B <= Φ <= B); @variable(m, -B <= Θ <= B)
    P == 0 && fix(Φ, 0.0; force = true)
    Q == 0 && fix(Θ, 0.0; force = true)
    # BOUND DERIVADO (OBBT): sum(eps^2) <= UB  =>  |eps_t| <= sqrt(UB).
    if ub === nothing || !isfinite(ub)
        @variable(m, ε[1:T])
    else
        E = sqrt(ub) * (1 + 1e-9)
        @variable(m, -E <= ε[1:T] <= E)
    end
    if c.modo === :zeroed
        for t = 1:lb-1; fix(ε[t], 0.0; force = true); end
        ly = t -> t < 1 ? 0.0 : y[t]
        le = t -> t < 1 ? zero(AffExpr) : one(Float64) * ε[t]
        objterms = lb:T
        y0expr = nothing
    else
        # :innovations, nao-sazonal, p=q=1 => o bloco pre-amostral e so t=0.
        # z_0 e VARIAVEL amarrada pela recursao: z_0 = eps_0 (termos anteriores fora do
        # alcance). Objetivo inclui eps_0^2 — e a parametrizacao por inovacoes.
        @variable(m, -B*10 <= z0 <= B*10)
        @variable(m, e0)
        if ub !== nothing && isfinite(ub)
            E = sqrt(ub) * (1 + 1e-9); set_lower_bound(e0, -E); set_upper_bound(e0, E)
        end
        @constraint(m, z0 == e0)
        ly = t -> t < 0 ? 0.0 : (t == 0 ? z0 : y[t])
        le = t -> t < 0 ? zero(AffExpr) : (t == 0 ? one(Float64)*e0 : one(Float64)*ε[t])
        objterms = 1:T
        y0expr = e0
    end
    if P > 0; @variable(m, -B*B <= uϕΦ <= B*B); @constraint(m, uϕΦ == ϕ*Φ); end
    if Q > 0; @variable(m, -B*B <= uθΘ <= B*B); @constraint(m, uθΘ == θ*Θ); end
    t1 = c.modo === :zeroed ? lb : 1
    for t = t1:T
        ex = @expression(m, ϕ*ly(t-1) + θ*le(t-1))
        P > 0 && (ex = @expression(m, ex + Φ*ly(t-s) - uϕΦ*ly(t-s-1)))
        Q > 0 && (ex = @expression(m, ex + Θ*le(t-s) + uθΘ*le(t-s-1)))
        @constraint(m, y[t] == ex + ε[t])
    end
    obj = @expression(m, sum(ε[t]^2 for t = objterms))
    y0expr !== nothing && (obj = @expression(m, obj + y0expr^2))
    @objective(m, Min, obj)
    # warm start COMPLETO: o SCIP descarta solucao parcial ("unknown rate too large"),
    # entao entregamos tambem o vetor eps e os produtos. E' informacao que ja temos; a
    # obrigacao de PROVA continua inteira com o dual bound dele.
    if warm !== nothing
        co, ev, e0v = warm
        set_start_value(ϕ, co[1]); set_start_value(θ, co[2])
        P > 0 && set_start_value(Φ, co[3]); Q > 0 && set_start_value(Θ, co[4])
        P > 0 && set_start_value(m[:uϕΦ], co[1]*co[3])
        Q > 0 && set_start_value(m[:uθΘ], co[2]*co[4])
        if length(ev) == T
            for t = 1:T; is_fixed(ε[t]) || set_start_value(ε[t], ev[t]); end
        end
        if c.modo === :innovations && isfinite(e0v)
            set_start_value(m[:e0], e0v); set_start_value(m[:z0], e0v)
        end
    end
    m
end

function roda(y, c::Caso, opt; ub = nothing, warm = nothing)
    m = monta(y, c, opt; ub = ub, warm = warm); t0 = time()
    optimize!(m); st = string(termination_status(m))
    co = has_values(m) ? [value(m[:ϕ]), value(m[:θ]),
                          c.P > 0 ? value(m[:Φ]) : 0.0, c.Q > 0 ? value(m[:Θ]) : 0.0] : fill(NaN, 4)
    ob = try objective_bound(m) catch; NaN end
    rg = try MOI.get(m, MOI.RelativeGap()) catch; NaN end
    ev = has_values(m) ? [value(m[:ε][t]) for t = 1:length(y)] : Float64[]
    e0v = (has_values(m) && c.modo === :innovations) ? value(m[:e0]) : NaN
    (has_values(m) ? objective_value(m) : NaN, st, time() - t0, co, ob, rg, ev, e0v)
end
IP()    = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
SC(tl)  = optimizer_with_attributes(SCIP.Optimizer, "limits/time" => tl, "display/verblevel" => 0)

# ---------------------------------------------------------------- main
function main()
    i1, i2, out = parse(Int, ARGS[1]), parse(Int, ARGS[2]), ARGS[3]
    cs = casos(); cs = cs[max(1,i1):min(length(cs), i2)]
    @printf("PROGRAMA SCIP — %d casos nesta fatia (de %d), limite %ds, SCIP %s\n",
            length(cs), length(casos()), Int(TL), string(SCIP.SCIP_versionnumber()))
    open(out, "w") do io
        println(io, "id,cel,rotulo,P,Q,T,rep,modo,sid,obj_ipopt,obj_scip,dual_scip,status,t_scip,gap_rel,gap_certificado,piso_ok")
        for c in cs
            y = dados(c)
            cc = c.sid > 0 ? Caso(c.id,c.cel,c.P,c.Q,length(y),c.rep,c.modo,c.rotulo,c.par,c.sid) : c
            oi = try roda(y, cc, IP()) catch e; (NaN, "erro", NaN, fill(NaN,4), NaN, NaN, Float64[], NaN) end
            os = try roda(y, cc, SC(TL); ub = oi[1], warm = (oi[4], oi[7], oi[8]))
                 catch e; (NaN, first(string(e), 16), NaN, fill(NaN,4), NaN, NaN, Float64[], NaN) end
            piso = 0.25 * (length(y) - 1)          # assercao: abaixo disso a caixa esta errada
            piso_ok = isnan(os[1]) || os[1] >= piso
            !piso_ok && @printf("  *** ALERTA piso: obj %.4f < %.4f\n", os[1], piso)
            g  = (isnan(oi[1])||isnan(os[1])) ? NaN : (oi[1]-os[1])/max(abs(os[1]),1e-12)
            # GAP CERTIFICADO: o dual bound em TIME_LIMIT e' um certificado parcial —
            # "nada mais que X% melhor existe na caixa". E' produto, nao falha.
            gc = (isnan(os[1])||isnan(os[5])) ? NaN : (os[1]-os[5])/max(abs(os[1]),1e-12)
            @printf("%3d %-2s %-26s P%dQ%d T%-4d %-12s | ip %10.4f  scip %10.4f  %-14s %6.1fs  gapcert %+.2e\n",
                    cc.id, cc.cel, cc.rotulo, cc.P, cc.Q, length(y), string(cc.modo),
                    oi[1], os[1], os[2], os[3], gc)
            println(io, join((cc.id,cc.cel,cc.rotulo,cc.P,cc.Q,length(y),cc.rep,cc.modo,cc.sid,
                              oi[1],os[1],os[5],os[2],os[3],g,gc,piso_ok), ","))
            flush(io); flush(stdout)
        end
    end
    println("fatia pronta: ", out)
end
main()
