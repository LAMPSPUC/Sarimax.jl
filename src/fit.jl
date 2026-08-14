"""
    has_fit_methods(modelType::Type{<:SarimaxModel}) -> Bool

Check if a given `SarimaxModel` type has the `fit!` method implemented.

# Arguments
- `modelType::Type{<:SarimaxModel}`: Type of the Sarimax model to check.

# Returns
A boolean indicating whether the `fit!` method is implemented for the specified model type.

"""
function has_fit_methods(modelType::Type{<:SarimaxModel})
    tupleModelType = Tuple{modelType}
    return hasmethod(fit!, tupleModelType)
end

"""
    has_hyperparameters_methods(modelType::Type{<:SarimaxModel}) -> Bool

Checks if a given `SarimaxModel` type has methods related to hyperparameters.

# Arguments
- `modelType::Type{<:SarimaxModel}`: Type of the Sarimax model to check.

# Returns
A boolean indicating whether the hyperparameter-related methods are implemented for the specified model type.

"""
function has_hyperparameters_methods(modelType::Type{<:SarimaxModel})
    tupleModelType = Tuple{modelType}
    return hasmethod(get_hyperparameters_number, tupleModelType)
end

"""
    aic(K::Int, loglikeVal::Fl) where Fl<:AbstractFloat -> Fl

Calculate the Akaike Information Criterion (AIC) for a given number of parameters and log-likelihood value.

# Arguments
- `K::Int`: Number of parameters in the model.
- `loglikeVal::Fl`: Log-likelihood value of the model.

# Returns
The AIC value calculated using the formula: AIC = 2*K - 2*loglikeVal.

"""
function aic(K::Int, loglikeVal::Fl) where {Fl<:AbstractFloat}
    return 2 * K - 2 * loglikeVal
end

"""
    aicc(T::Int, K::Int, loglikeVal::Fl) where Fl<:AbstractFloat -> Fl

Calculate the corrected Akaike Information Criterion (AICc) for a given number of observations, number of parameters, and log-likelihood value.

# Arguments
- `T::Int`: Number of observations in the data.
- `K::Int`: Number of parameters in the model.
- `loglikeVal::Fl`: Log-likelihood value of the model.

# Returns
The AICc value calculated using the formula: AICc = AIC(K, loglikeVal) + ((2*K*K + 2*K) / (T - K - 1)).

"""
function aicc(T::Int, K::Int, loglikeVal::Fl) where {Fl<:AbstractFloat}
    return aic(K, loglikeVal) + ((2 * K * K + 2 * K) / (T - K - 1))
end

"""
    bic(T::Int, K::Int, loglikeVal::Fl) -> Fl

Calculate the Bayesian Information Criterion (BIC) for a given number of observations, number of parameters, and log-likelihood value.

# Arguments
- `T::Int`: Number of observations in the data.
- `K::Int`: Number of parameters in the model.
- `loglikeVal::Fl`: Log-likelihood value of the model.

# Returns
The BIC value calculated using the formula: BIC = log(T) * K - 2 * loglikeVal.

"""
function bic(T::Int, K::Int, loglikeVal::Fl) where {Fl<:AbstractFloat}
    return log(T) * K - 2 * loglikeVal
end

"""
    criterionLoglike(model) -> Fl

Log-verossimilhanca usada pelos CRITERIOS DE INFORMACAO: a gaussiana EXATA
([`exactLoglike`](@ref)) quando ela e computavel, com recuo para a CSS plug-in
([`loglike`](@ref)) quando nao e.

Por que a exata e nao a CSS. O `forecast::auto.arima` seleciona por AICc calculado sobre a
verossimilhanca exata; enquanto o nosso criterio usava CSS plug-in, comparar AICc ou
trajetoria de busca com a dele era comparar objetos diferentes. Alem disso o CSS ignora a
incerteza pre-amostral, e era essa lacuna que o `nPresampleFree` tentava tapar cobrando os
valores pre-amostrais como parametros — remendo no criterio que agora sai, porque a exata
contabiliza essa incerteza por construcao.

O recuo existe porque `exactLoglike` devolve `nothing` de proposito em vez de um numero
errado (ponto nao estacionario, autocovariancia sem positividade definida, truncagem dos psi
mordendo). Nesses casos o criterio volta a ser o de antes — comportamento degradado, nunca
silenciosamente incorreto.

Sem `try/catch`: o contrato de `exactLoglike` ja e nothing-em-recusa, entao qualquer excecao
aqui e bug e deve subir — um `catch` largo mascararia exatamente a classe de erro
(`MethodError`, `UndefVarError`) que ja mordeu este arquivo, e tornaria a taxa de recuo
impossivel de interpretar. Tipos de modelo sem `exactLoglike` recuam via `applicable`.

QUASI-AIC, NAO AIC. Esta verossimilhanca e avaliada nos coeficientes que o OBJETIVO DO USUARIO
produziu (`mae`, `huber`, `CVaR`, ridge, ...), nao no maximo da gaussiana. Como estatistica de
comparacao entre candidatos e legitima — a mesma funcao pontua todos —, mas nao e a
verossimilhanca maximizada que a teoria do AIC supoe. Medido em ARMA(1,1) simulado, o deficit
`2*(l(EMV) - l(ponto ajustado))` tem mediana 0,01 sob `mse` e sobe com o raio da raiz MA:
mediana 0,98 e p90 7,03 com raiz em 0,98, isto e, ja na escala em que o AICc decide. Sob
`ridge` a mediana vai a 1,79, porque o encolhimento afasta o ponto do maximo de proposito.
Refinar por um passo de Newton NAO resolve: no regime que importa o otimo esta na fronteira de
invertibilidade (theta_EMV empilha em 1), onde a equivalencia assintotica de Le Cam nao vale —
medido, um passo fecha 6% do deficit e piora o valor em 40% dos casos.
"""
function criterionLoglike(model::SarimaxModel)
    return first(criterionLoglikeAndN(model))
end

"""
    criterionLoglikeAndN(model) -> (loglike, n, usedExact)

Nucleo de [`criterionLoglike`](@ref): devolve tambem o TAMANHO DA AMOSTRA sobre o qual a
log-verossimilhanca foi avaliada, e qual caminho foi usado.

O `n` importa porque as duas verossimilhancas nao vivem na mesma amostra: a exata e avaliada
sobre a serie diferenciada inteira (`T`), enquanto a CSS vive nos residuos condicionados
(`length(observedResiduals) = T - lb + 1`, com `lb` ate 30 nos defaults mensais de `auto`).
Correcoes de amostra finita (AICc) e o fator `log(n)` do BIC devem usar o `n` da
verossimilhanca efetivamente usada — caso contrario o criterio aplica uma penalidade de
tamanho extra, crescente em K, que o `forecast::Arima` nao tem.

O recuo e registrado em `model.metadata["criterionFallback"]` para ser mensuravel.
"""
function criterionLoglikeAndN(model::SarimaxModel)
    exact = applicable(exactLoglike, model) ? exactLoglike(model) : nothing
    usedExact = !isnothing(exact)
    if hasproperty(model, :metadata) && isa(model.metadata, AbstractDict)
        model.metadata["criterionFallback"] = !usedExact
    end
    usedExact && return exact, criterionSampleSize(model), true
    return loglike(model), length(observedResiduals(model)), false
end

"""
    criterionSampleSize(model) -> Int

Numero de observacoes da verossimilhanca EXATA: o comprimento da serie diferenciada, que e o
`n` que o `stats::arima` usa. Nao confundir com `length(observedResiduals)`, que desconta o
conditioning da CSS.

Calculado por aritmetica (`n - d - D*s`) e nao por `length(values(differentiate(...)))`: a
versao com `differentiate` alocava a serie diferenciada inteira so para medir o comprimento,
custando ~7.5us por avaliacao de criterio (medido: 164.6us vs 157.1us por `aicc`, mediana de
7 baterias de 3000 chamadas). A equivalencia das duas formas esta travada em
`test/exact_likelihood.jl`.

Sem anotacao de tipo porque `fit.jl` e incluido antes de `models/sarima.jl` definir
`SARIMAModel`; so e chamada no caminho exato, que exige `applicable(exactLoglike, model)`.
"""
criterionSampleSize(model) =
    length(values(model.y)) - model.d - model.D * model.seasonality

"""
Penalidade somada ao criterio de um candidato de BUSCA cujo criterio veio do recuo CSS.

Sem ela, o recuo premia exatamente os candidatos errados: a CSS e avaliada sobre menos
observacoes que a exata (`T - lb + 1` vs `T`), logo e menos negativa, logo AICc menor — e o
que dispara o recuo e raiz perto da fronteira. Um candidato quase nao estacionario ganharia
dezenas de unidades de AICc de vantagem por nao ter verossimilhanca exata computavel.

A penalidade impoe uma ordem em dois niveis, analoga a do `myarima` (que devolve `Inf` quando
a verossimilhanca nao e finita): candidato com verossimilhanca exata sempre vence candidato
sem; entre candidatos sem, a comparacao CSS continua valida porque todos condicionam na mesma
amostra (`searchLb`). Aditiva em vez de `Inf` para a busca nunca ficar sem selecao quando a
exata falha em todos os candidatos (series curtas ou patologicas).

So se aplica a SELECAO ([`searchCriterionFunction`](@ref)); os acessores publicos `aic`,
`aicc` e `bic` continuam devolvendo o valor com recuo documentado.
"""
const FALLBACK_CRITERION_PENALTY = 1e10

"""
    searchCriterionFunction(baseCriterion) -> Function

Envolve `aic`/`aicc`/`bic` com a semantica de SELECAO: soma
[`FALLBACK_CRITERION_PENALTY`](@ref) quando o criterio do candidato veio do recuo CSS
(lido de `model.metadata["criterionFallback"]`, gravado por [`criterionLoglikeAndN`](@ref)
na propria avaliacao do criterio).
"""
function searchCriterionFunction(baseCriterion::Function)
    # Transparente a argumentos: alem da forma de modelo, `aic`/`aicc`/`bic` tem formas
    # escalares ((K, ll), (T, K, ll)) que continuam acessiveis pela funcao retornada; a
    # penalidade so se aplica quando o primeiro argumento e um modelo.
    return function (args...; kwargs...)
        value = baseCriterion(args...; kwargs...)
        model = first(args)
        fallback =
            model isa SarimaxModel &&
            hasproperty(model, :metadata) &&
            isa(model.metadata, AbstractDict) &&
            get(model.metadata, "criterionFallback", false)
        return fallback ? value + FALLBACK_CRITERION_PENALTY : value
    end
end

"""
    aic(model::SarimaxModel; offset::Fl) -> Fl where Fl<:AbstractFloat

Calculate the Akaike Information Criterion (AIC) for a Sarimax model:
`AIC = 2K - 2ℓ`, where `ℓ` is [`criterionLoglike`](@ref) — the exact Gaussian
log-likelihood when computable, falling back to the conditional (CSS) one — and
`K = ncoef + 1` the number of estimated parameters, matching `forecast::Arima`.

# Arguments
- `model::SarimaxModel`: The Sarimax model for which AIC is calculated.
- `offset::Fl`: Optional value added to the AIC (kept for call compatibility).
- `K::Int`: Optional parameter-count override.

# Returns
The AIC value calculated using the number of parameters and log-likelihood value of the model.

# Errors
- Throws a `MissingMethodImplementation` if the `get_hyperparameters_number` method is not implemented for the given model type.

"""
function aic(model::SarimaxModel; offset::Union{AbstractFloat,Nothing} = nothing, K::Union{Int,Nothing} = nothing)
    !has_hyperparameters_methods(typeof(model)) &&
        throw(MissingMethodImplementation("get_hyperparameters_number"))
    K = isnothing(K) ? get_hyperparameters_number(model) : K
    offsetValue = isnothing(offset) ? 0.0 : offset
    return 2 * K - 2 * criterionLoglike(model) + offsetValue
end

"""
    aicc(model::SarimaxModel; offset::Fl) -> Fl where Fl<:AbstractFloat

Calculate the Corrected Akaike Information Criterion (AICc) for a Sarimax model.

# Arguments
- `model::SarimaxModel`: The Sarimax model for which AICc is calculated.
- `offset::Fl=0.0`: Offset value to be added to the AICc value.

# Returns
The AICc value calculated using the number of parameters, sample size, and log-likelihood value of the model.

# Errors
- Throws a `MissingMethodImplementation` if the `get_hyperparameters_number` method is not implemented for the given model type.

"""
function aicc(model::SarimaxModel; offset::Union{AbstractFloat,Nothing} = nothing, K::Union{Int,Nothing} = nothing)
    !has_hyperparameters_methods(typeof(model)) &&
        throw(MissingMethodImplementation("get_hyperparameters_number"))
    K = isnothing(K) ? get_hyperparameters_number(model) : K
    offsetValue = isnothing(offset) ? 0.0 : offset
    # O `n` da correcao e o da amostra da verossimilhanca efetivamente usada: `T` (serie
    # diferenciada inteira) no caminho exato, `length(observedResiduals)` no recuo CSS.
    # Misturar — corrigir uma verossimilhanca sobre T com um n condicionado menor — cobra
    # uma penalidade extra crescente em K que o `forecast::Arima` nao tem.
    ll, n, _ = criterionLoglikeAndN(model)
    return 2 * K - 2 * ll + offsetValue + ((2 * K * K + 2 * K) / (n - K - 1))
end

"""
    bic(model::SarimaxModel; offset::Fl) -> Fl where Fl<:AbstractFloat

Calculate the Bayesian Information Criterion (BIC) for a Sarimax model.

# Arguments
- `model::SarimaxModel`: The Sarimax model for which BIC is calculated.
- `offset::Fl=0.0`: Offset value to be added to the BIC value.

# Returns
The BIC value calculated using the number of parameters, sample size, and log-likelihood value of the model.

# Errors
- Throws a `MissingMethodImplementation` if the `get_hyperparameters_number` method is not implemented for the given model type.

"""
function bic(model::SarimaxModel; offset::Union{AbstractFloat,Nothing} = nothing, K::Union{Int,Nothing} = nothing)
    !has_hyperparameters_methods(typeof(model)) &&
        throw(MissingMethodImplementation("get_hyperparameters_number"))
    K = isnothing(K) ? get_hyperparameters_number(model) : K
    offsetValue = isnothing(offset) ? 0.0 : offset
    # Mesmo `n` da verossimilhanca usada — ver `aicc`.
    ll, n, _ = criterionLoglikeAndN(model)
    return K * log(n) - 2 * ll + offsetValue
end


