"""
Verossimilhanca gaussiana EXATA de um SARIMA, avaliada em coeficientes DADOS.

Motivacao. O pacote tem tres objetos tipo-verossimilhanca que nao coincidem: a funcao objetivo
minimizada, a ML exata, e o `loglike()` que alimenta os criterios de informacao (CSS plug-in
sobre os residuos). Enquanto isso durar, comparar AICc ou trajetoria de busca com o
`forecast::auto.arima` e comparar objetos diferentes.

Este arquivo resolve o terceiro: dada uma serie e coeficientes, devolve a log-verossimilhanca
gaussiana EXATA — a mesma quantidade que o `stats::arima(method = "ML")` reporta.

Por que isso substitui o `nPresampleFree`. Aquele termo cobrava os valores pre-amostrais
livres como parametros no criterio, porque o CSS plug-in ignora a incerteza pre-amostral e
deixava candidatos sazonais de ordem alta absorverem `s*P` graus de liberdade de graca. A
verossimilhanca exata contabiliza essa incerteza POR CONSTRUCAO, entao ela e a substituicao
principiada do ajuste caseiro — e traz junto o termo de determinante cuja ausencia enviesava a
selecao a favor de `q` alto.

METODO: Durbin-Levinson sobre a autocovariancia teorica. Nao e Kalman e nao e Ansley.

    1. expande os polinomios multiplicativos:  phi(B)Phi(B^s)  e  theta(B)Theta(B^s)
    2. pesos psi da representacao MA(inf) e dai a autocovariancia gamma(k)
    3. Durbin-Levinson devolve os erros de previsao um-passo v_t e os residuos e_t
    4. concentrando sigma^2:

           -2l = n*log(sigma2) + sum_t log(v_t) + n,     sigma2 = (1/n) sum_t e_t^2 / v_t

E exato para qualquer ARMA estacionario, inclusive sazonal multiplicativo e misto — os casos
em que a forma fechada por perfilamento dos pre-amostrais NAO fecha (medido: espalhamento ~35
unidades de -2logLik no ARMA misto, contra 1e-13 no AR e no MA puros).

CUSTO: O(n^2). A n = 300 sao ~90 mil operacoes por avaliacao — irrelevante perto de um solve
do Ipopt. E como so precisa AVALIAR (nao otimizar), roda em Julia puro: nao precisa ser
expressavel em JuMP, nem diferenciavel, nem interagir com o solver. O otimizador segue livre
para usar qualquer objetivo — que e a propriedade que o pacote vende.
"""

"""
    expandMultiplicativePolynomial(nonSeasonal, seasonal, s; negate) -> Vector

Expande `(1 - sum a_i B^i)(1 - sum b_k B^{s k})` na forma plana `1 - sum c_j B^j`, devolvendo
`c`. Com `negate = false` expande a forma `(1 + sum a_i B^i)(1 + sum b_k B^{s k})`, que e a
convencao do lado MA deste pacote (`y_t = ... + theta_j eps_{t-j} + eps_t`).
"""
function expandMultiplicativePolynomial(
    nonSeasonal::Vector{Fl},
    seasonal::Vector{Fl},
    s::Int;
    negate::Bool = true,
) where {Fl<:AbstractFloat}
    na, nb = length(nonSeasonal), length(seasonal)
    (na == 0 && nb == 0) && return Fl[]
    sgn = negate ? -one(Fl) : one(Fl)
    c = zeros(Fl, na + s * nb)
    for i = 1:na
        c[i] += nonSeasonal[i]
    end
    for k = 1:nb
        c[s*k] += seasonal[k]
        for i = 1:na
            # o produto cruzado troca de sinal na convencao AR (1 - aB)(1 - bB^s)
            c[s*k+i] += sgn * nonSeasonal[i] * seasonal[k]
        end
    end
    return c
end

"""
    psiWeightsFromZero(ar, ma, n) -> Vector

Pesos `psi` da representacao MA(infinito) INCLUINDO `psi_0 = 1` na primeira posicao, de modo
que `psi[k+1] == psi_k`. Delega o calculo a [`psiWeights`](@ref), que ja existe no pacote e
devolve `psi_1..psi_n` (com `psi_0` implicito).

Existe como funcao separada por uma razao pratica: a primeira versao deste arquivo definia um
`psiWeights` proprio com indexacao base-zero e MESMA assinatura da funcao do pacote. Como
`exact_likelihood.jl` e incluido depois de `models/sarima.jl`, a definicao nova SOBRESCREVIA a
antiga, e `forecastErrors` — que espera base-um — passou a ler `psi_0` no lugar de `psi_1`.
Efeito: variancia de previsao de um ruido branco saindo `[1, 2, 2, 2]*sigma^2` em vez de
constante. Nao duplicar funcao que ja existe; quando a convencao difere, adaptar na borda.
"""
psiWeightsFromZero(ar::Vector{Fl}, ma::Vector{Fl}, n::Int) where {Fl<:AbstractFloat} =
    vcat(one(Fl), psiWeights(ar, ma, n))

"""
    theoreticalACF(ar, ma, n; psiLength) -> Vector

Autocovariancias `gamma(0..n-1)` do ARMA estacionario com variancia de inovacao unitaria,
obtidas da representacao MA(infinito): `gamma(k) = sum_j psi_j * psi_{j+k}`.

A truncagem em `psiLength` e legitima porque os `psi` de um processo ESTACIONARIO decaem
geometricamente. Se a truncagem morder (processo perto da fronteira), a funcao devolve
`nothing` em vez de um numero silenciosamente errado — a checagem e a cauda dos `psi`.
"""
function theoreticalACF(
    ar::Vector{Fl},
    ma::Vector{Fl},
    n::Int;
    psiLength::Int = max(2 * n, 1000),
) where {Fl<:AbstractFloat}
    ψ = psiWeightsFromZero(ar, ma, psiLength)
    all(isfinite, ψ) || return nothing
    # cauda ainda relevante => truncagem morde => nao devolver numero errado
    tailMax = maximum(abs, @view ψ[max(1, psiLength - 9):end])
    tailMax > 1e-6 && return nothing
    γ = zeros(Fl, n)
    for k = 0:(n-1)
        acc = zero(Fl)
        @inbounds for j = 1:(psiLength+1-k)
            acc += ψ[j] * ψ[j+k]
        end
        γ[k+1] = acc
    end
    (γ[1] > 0 && all(isfinite, γ)) || return nothing
    return γ
end

"""
    exactGaussianLogLikelihood(z, ar, ma) -> Union{Float64,Nothing}

Log-verossimilhanca gaussiana exata da serie `z` (ja diferenciada e ja limpa dos termos
deterministicos) sob o ARMA de coeficientes `ar`/`ma`, com `sigma^2` concentrado.

Devolve `nothing` quando o ponto e inadmissivel ou numericamente inviavel (nao estacionario,
autocovariancia nao positiva definida) — nunca um numero silenciosamente errado.

Durbin-Levinson: `v_t` e a variancia do erro de previsao um-passo em unidades de `sigma^2`, e
os coeficientes de previsao vem da recursao. `sum log v_t` e o termo de determinante que o CSS
plug-in nao tem — o mesmo que, ausente, enviesa a selecao a favor de ordem alta.
"""
function exactGaussianLogLikelihood(
    z::Vector{Fl},
    ar::Vector{Fl},
    ma::Vector{Fl},
) where {Fl<:AbstractFloat}
    n = length(z)
    n == 0 && return nothing
    γ = theoreticalACF(ar, ma, n)
    isnothing(γ) && return nothing

    v = zeros(Fl, n)          # variancia do erro de previsao (unidades de sigma^2)
    e = zeros(Fl, n)          # erro de previsao um-passo
    ϕprev = zeros(Fl, n)
    ϕcur = zeros(Fl, n)

    v[1] = γ[1]
    e[1] = z[1]
    for t = 1:(n-1)
        # coeficiente de reflexao
        acc = γ[t+1]
        for j = 1:(t-1)
            acc -= ϕprev[j] * γ[t-j+1]
        end
        v[t] <= 0 && return nothing
        κ = acc / v[t]
        abs(κ) >= 1 && return nothing            # perdeu positividade definida
        ϕcur[t] = κ
        for j = 1:(t-1)
            ϕcur[j] = ϕprev[j] - κ * ϕprev[t-j]
        end
        v[t+1] = v[t] * (1 - κ^2)
        pred = zero(Fl)
        for j = 1:t
            pred += ϕcur[j] * z[t-j+1]
        end
        e[t+1] = z[t+1] - pred
        ϕprev, ϕcur = ϕcur, ϕprev
    end
    (all(x -> x > 0 && isfinite(x), v) && all(isfinite, e)) || return nothing

    σ² = sum(e[t]^2 / v[t] for t = 1:n) / n
    (σ² > 0 && isfinite(σ²)) || return nothing
    sumLogV = sum(log, v)
    # -2l = n log(2 pi sigma2) + sum log v + n   =>   l = -(n/2)(log(2 pi sigma2) + 1) - (1/2) sum log v
    return -(n / 2) * (log(2π * σ²) + 1) - sumLogV / 2
end

"""
    exactLoglike(model::SARIMAModel) -> Union{Float64,Nothing}

Log-verossimilhanca gaussiana exata do modelo AJUSTADO, avaliada nos coeficientes estimados.

Diferenciacao e termos deterministicos sao removidos antes: a verossimilhanca e a da serie
diferenciada menos a parte deterministica, que e a convencao do `stats::arima`.

NOTA sobre avaliar fora do EMV: os coeficientes vem do objetivo escolhido pelo usuario (que
pode ser `mae`, `ridge`, `huber`, ...), nao do maximo desta funcao. Como estatistica de
COMPARACAO entre candidatos isso e legitimo desde que a mesma funcao pontue todos, mas nao e a
verossimilhanca maximizada que a teoria do AIC supoe. Para uso mais fiel, refinar os
finalistas (um passo de Newton nesta funcao a partir do ponto ajustado ja e assintoticamente
equivalente ao EMV) antes da decisao final — que e o que o `auto.arima` faz com
`approximation = TRUE`.
"""
function exactLoglike(model::SARIMAModel)
    isFitted(model) || return nothing
    s = model.seasonality
    ϕ = isnothing(model.ϕ) ? Float64[] : Float64.([model.ϕ...])
    θ = isnothing(model.θ) ? Float64[] : Float64.([model.θ...])
    Φ = isnothing(model.Φ) ? Float64[] : Float64.([model.Φ...])
    Θ = isnothing(model.Θ) ? Float64[] : Float64.([model.Θ...])
    ar = expandMultiplicativePolynomial(ϕ, Φ, s; negate = true)
    ma = expandMultiplicativePolynomial(θ, Θ, s; negate = false)

    diffY = differentiate(model.y, model.d, model.D, s)
    z = Float64.(values(diffY))
    isempty(z) && return nothing

    # remove a parte deterministica na mesma escala da serie diferenciada.
    #
    # DUAS CONVERSOES QUE FALTAVAM, ambas verificadas contra o `stats::arima(method="ML")`:
    #
    # 1. `model.c` e a CONSTANTE da regressao, nao a media. O nivel a remover e
    #    mu = c / (1 - sum(ar)). Medido na serie M4 44895: subtraindo `c` da -2305.891,
    #    subtraindo `mu` da -2304.253, e o R da -2304.253 exatamente. Erro de 1.64 de
    #    log-verossimilhanca = 3.3 unidades de AICc, o suficiente para virar selecao.
    #
    # 2. `model.trend` MULTIPLICA o regressor de tempo diferenciado (`trend * driftValues[t]`,
    #    ver `sarima.jl`), e esse regressor nao vale 1 em geral: com d=1,D=0 vale 1, mas com
    #    d=0,D=1 vale `s` (12 no mensal). Subtrair o escalar erra por um fator 12 nessa classe.
    #
    # Na M4 monthly as duas juntas atingem 15,7% das 48k series (10,7% + 5,0%).
    if !isnothing(model.c) && model.allowMean
        arSum = isempty(ar) ? zero(eltype(z)) : sum(ar)
        denom = 1 - arSum
        # `denom = phi(1)*Phi(1)`, ja com o polinomio expandido. Quando ele vai a zero o
        # processo tem raiz unitaria e a MEDIA NAO EXISTE — nenhum valor aqui esta certo.
        # Cai-se em `model.c` por ser finito, nao por ser correto; e o candidato deveria ter
        # sido rejeitado pela admissibilidade antes de chegar aqui.
        if abs(denom) <= 1e-8
            @warn "exactGaussianLogLikelihood: phi(1)*Phi(1) = $(denom) ~ 0 (raiz unitaria); " *
                  "a media do processo nao existe e o nivel removido e apenas finito, nao correto."
        end
        level = abs(denom) > 1e-8 ? model.c / denom : model.c
        z = z .- level
    end
    if !isnothing(model.trend) && model.allowDrift
        # O fallback `fill(1.0, ...)` E EXATAMENTE O BUG que este commit corrige: subtrair o
        # escalar em vez do regressor diferenciado. Se ele disparar em silencio, o defeito
        # volta numa configuracao que ninguem testou e sem deixar rastro. Avisa.
        driftReg = try
            r = values(differentiate(
                    TimeArray(timestamp(model.y), collect(1.0:length(values(model.y)))),
                    model.d, model.D, s))
            if length(r) == length(z)
                Float64.(r)
            else
                @warn "exactGaussianLogLikelihood: regressor de drift diferenciado tem " *
                      "comprimento $(length(r)), esperado $(length(z)); usando 1.0. " *
                      "Com d=$(model.d), D=$(model.D), s=$s o termo deterministico fica errado."
                fill(1.0, length(z))
            end
        catch e
            @warn "exactGaussianLogLikelihood: falha ao diferenciar o regressor de drift " *
                  "($(typeof(e))); usando 1.0, o que reintroduz o erro de escala."
            fill(1.0, length(z))
        end
        z = z .- model.trend .* driftReg
    end
    if !isnothing(model.exog) && !isnothing(model.exogCoefficients)
        try
            X = Float64.(values(differentiate(model.exog, model.d, model.D, s)))
            β = Float64.([model.exogCoefficients...])
            size(X, 1) == length(z) && length(β) == size(X, 2) && (z = z .- X * β)
        catch
            return nothing
        end
    end
    return exactGaussianLogLikelihood(z, ar, ma)
end
