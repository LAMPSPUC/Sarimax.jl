"""
Coefficient-specific regularization weights.

This file is the SINGLE SOURCE OF TRUTH for two things the penalized objectives used to
answer separately, each with its own inline `reduce(vcat, ...)`:

  1. WHICH coefficients a penalty reaches, and in WHICH ORDER;
  2. WHAT WEIGHT each of them carries.

Keeping the two together is what makes a heterogeneous `lambda` safe: the weight vector and
the JuMP variable vector are built by the same walk over the same block list, so a weight can
never silently land on the wrong coefficient.

The penalty is on MODEL COEFFICIENTS. Innovations, pre-sample innovations, fitted values and
the auxiliary variables introduced to linearize `|.|` are never reachable from here -- the
walk only visits the five coefficient blocks named in [`PENALTY_BLOCKS`](@ref).
"""

"""
    PENALTY_BLOCKS

Coefficient blocks a penalty can reach, in the CANONICAL ORDER. A flat weight vector is read
in exactly this order, restricted to the blocks that exist in the model and that
`penaltyTarget` admits:

    [ar; ma; sar; sma; exog]

The intercept (`c`) and the drift (`trend`) are deliberately absent: penalizing the level has
no shrinkage interpretation here, and no `penaltyTarget` or weight can reintroduce them.

Use [`penaltyCoefficientNames`](@ref) to print the ordering for a given model rather than
reconstructing it by hand.
"""
const PENALTY_BLOCKS = (:ar, :ma, :sar, :sma, :exog)

"""
JuMP variable container behind each block. `:exog` maps to the exogenous coefficients, which
only exist when the model carries regressors.
"""
const PENALTY_BLOCK_VARIABLE = Dict{Symbol,Symbol}(
    :ar => :ϕ,
    :ma => :θ,
    :sar => :Φ,
    :sma => :Θ,
    :exog => :β,
)

"""
Names a caller may use for a block. The canonical spelling is the ASCII one
(`:ar`, `:ma`, `:sar`, `:sma`, `:exog`); the Greek letters are accepted because that is how
the coefficients are named on the fitted model (`model.ϕ`, `model.θ`, ...), and asking a
caller to remember which of the two spellings a keyword wants is exactly the kind of
ambiguity this package has paid for before.
"""
const PENALTY_BLOCK_ALIASES = Dict{Symbol,Symbol}(
    :ar => :ar, :AR => :ar, :ϕ => :ar, :phi => :ar,
    :ma => :ma, :MA => :ma, :θ => :ma, :theta => :ma,
    :sar => :sar, :SAR => :sar, :Φ => :sar, :Phi => :sar, :seasonal_ar => :sar,
    :sma => :sma, :SMA => :sma, :Θ => :sma, :Theta => :sma, :seasonal_ma => :sma,
    :exog => :exog, :β => :exog, :beta => :exog, :exogenous => :exog,
)

"""
    PenaltyLambda

The shapes `lambda` accepts:

  - `Real` -- one strength for every penalized coefficient. This is the historical API and
    remains the meaning of `lambda = L`: `lambda_j = L` for every coefficient that would
    previously have been penalized.
  - `AbstractVector{<:Real}` -- one weight per penalized coefficient, read in the order of
    [`penaltyCoefficientNames`](@ref).
  - `NamedTuple` / `AbstractDict` -- keyed by coefficient block (see
    [`PENALTY_BLOCK_ALIASES`](@ref)); each value is either a scalar (applied to the whole
    block) or a vector with one weight per coefficient of that block.

Zero is a legal weight and means "do not penalize this coefficient". Negative, `NaN` and
`Inf` weights are rejected.

The vector case is typed `AbstractVector` rather than `AbstractVector{<:Real}` on purpose:
a `Vector{Any}` — which is what an element-wise construction can easily produce — would
otherwise fail with a `MethodError` naming a type union instead of the `ArgumentError` that
says which entry is wrong. The element check happens in [`checkPenaltyWeight`](@ref).
"""
const PenaltyLambda = Union{Real,AbstractVector,NamedTuple,AbstractDict}

"""
    penaltyBlockLength(model, block) -> Int

Number of coefficients the block holds in this model; `0` when the block is absent.
"""
function penaltyBlockLength(model, block::Symbol)
    block === :ar && return model.p
    block === :ma && return model.q
    block === :sar && return model.P
    block === :sma && return model.Q
    block === :exog && return isnothing(model.exog) ? 0 : length(colnames(model.exog))
    throw(ArgumentError("Unknown coefficient block :$(block)."))
end

"""
    penaltyTargetBlocks(penaltyTarget) -> Tuple{Vararg{Symbol}}

Blocks the `penaltyTarget` selector admits, before intersecting with the model.
"""
function penaltyTargetBlocks(penaltyTarget::Symbol)
    penaltyTarget === :all && return PENALTY_BLOCKS
    penaltyTarget === :dynamics && return (:ar, :ma, :sar, :sma)
    penaltyTarget === :exogenous && return (:exog,)
    throw(
        ArgumentError(
            "penaltyTarget must be :all, :dynamics or :exogenous, got :$(penaltyTarget)",
        ),
    )
end

"""
    penaltyBlocks(model, penaltyTarget) -> Vector{Symbol}

Blocks actually penalized for this model: admitted by `penaltyTarget` AND non-empty in the
model, in canonical order.
"""
penaltyBlocks(model, penaltyTarget::Symbol) =
    Symbol[b for b in penaltyTargetBlocks(penaltyTarget) if penaltyBlockLength(model, b) > 0]

"""
    penaltyBlockCoefficientNames(model, block) -> Vector{String}

Names of the block's coefficients. Exogenous coefficients are named after their own column,
which is the only identification a caller can act on when the regressors are permuted.
"""
function penaltyBlockCoefficientNames(model, block::Symbol)
    n = penaltyBlockLength(model, block)
    block === :exog && return [string("exog:", colnames(model.exog)[i]) for i = 1:n]
    return [string(block, i) for i = 1:n]
end

"""
    penaltyCoefficientNames(model; penaltyTarget = :all) -> Vector{String}

The penalized coefficients of `model`, in the order a FLAT `lambda` vector is read.

This is the documented mapping between a weight vector and the coefficients it lands on, and
the intended way to build one -- including for adaptive-Lasso-style weights, where the vector
comes from a first-stage fit:

```julia
first = SARIMA(y, 2, 0, 1); fit!(first)
penaltyCoefficientNames(first; penaltyTarget = :dynamics)   # ["ar1", "ar2", "ma1"]
b = [first.ϕ...; first.θ...]
w = 1.0 ./ abs.(b) .^ 1.0
fit!(model; objectiveFunction = "elastic_net", alpha = 1.0,
     penaltyTarget = :dynamics, lambda = 10.0 .* w)
```
"""
function penaltyCoefficientNames(model; penaltyTarget::Symbol = :all)
    names = String[]
    for b in penaltyBlocks(model, penaltyTarget)
        append!(names, penaltyBlockCoefficientNames(model, b))
    end
    return names
end

"""
    checkPenaltyWeight(value, label) -> Float64

A weight must be a non-negative finite real. Zero is admitted -- it is how a caller excludes
one coefficient from the penalty -- but `NaN` and `Inf` are not: `NaN` would propagate into
the objective and `Inf` would make it unbounded, and neither failure is visible in the
fitted coefficients.
"""
function checkPenaltyWeight(value, label::AbstractString)
    value isa Real || throw(
        ArgumentError("Penalty weight for $(label) must be a real number, got $(value)."),
    )
    isfinite(value) || throw(
        ArgumentError(
            "Penalty weight for $(label) must be finite, got $(value). Use 0.0 to leave a " *
            "coefficient unpenalized.",
        ),
    )
    value < 0 && throw(
        ArgumentError("Penalty weight for $(label) must be non-negative, got $(value)."),
    )
    return Float64(value)
end

"""
    canonicalPenaltyBlock(key) -> Symbol

Resolves a user-supplied block key through [`PENALTY_BLOCK_ALIASES`](@ref).
"""
function canonicalPenaltyBlock(key)
    sym = key isa Symbol ? key : Symbol(key)
    haskey(PENALTY_BLOCK_ALIASES, sym) || throw(
        ArgumentError(
            "Unknown penalty block :$(sym). Valid blocks are " *
            join([":" * string(b) for b in PENALTY_BLOCKS], ", ") *
            " (the Greek coefficient names are accepted as aliases).",
        ),
    )
    return PENALTY_BLOCK_ALIASES[sym]
end

"""
    validatePenaltyLambdaValues(lambda)

Value-level check, usable where the model's blocks are not yet known (the `SARIMA`
constructors): every weight must be a non-negative finite real. The STRUCTURAL check -- that
the weights cover exactly the penalized blocks, with the right lengths -- needs
`penaltyTarget` and therefore happens at fit time, through
[`resolvePenaltyWeights`](@ref).
"""
function validatePenaltyLambdaValues(lambda)
    isnothing(lambda) && return nothing
    if lambda isa Real
        checkPenaltyWeight(lambda, "lambda")
    elseif lambda isa AbstractVector
        for (i, v) in enumerate(lambda)
            checkPenaltyWeight(v, "lambda[$(i)]")
        end
    elseif lambda isa NamedTuple || lambda isa AbstractDict
        for k in keys(lambda)
            block = canonicalPenaltyBlock(k)
            v = lambda[k]
            if v isa AbstractVector
                for (i, vi) in enumerate(v)
                    checkPenaltyWeight(vi, "lambda[:$(block)][$(i)]")
                end
            else
                checkPenaltyWeight(v, "lambda[:$(block)]")
            end
        end
    else
        throw(
            ArgumentError(
                "lambda must be a real number, a vector of weights, or a NamedTuple/Dict " *
                "keyed by coefficient block; got $(typeof(lambda)).",
            ),
        )
    end
    return nothing
end

"""
    resolvePenaltyWeights(model, lambda, penaltyTarget, defaultLambda) -> Vector{Float64}

Expands whatever the caller passed as `lambda` into ONE WEIGHT PER PENALIZED COEFFICIENT,
aligned position by position with [`penaltyCoefficientNames`](@ref)`(model; penaltyTarget)`.

`lambda === nothing` reproduces the historical default, `defaultLambda` on every penalized
coefficient; a scalar reproduces the historical scalar API, `lambda_j = lambda` on every one
of them. Both are the uniform special case of the general form, not a separate code path.

The structured forms must name EXACTLY the penalized blocks -- no more, no fewer. Filling an
unnamed block with the default instead would make `lambda = (ar = 0.0,)` silently leave the
moving-average block at `defaultLambda`, which reads as "penalize nothing" and is not.
"""
function resolvePenaltyWeights(model, lambda, penaltyTarget::Symbol, defaultLambda::Real)
    blocks = penaltyBlocks(model, penaltyTarget)
    lengths = Int[penaltyBlockLength(model, b) for b in blocks]
    total = sum(lengths; init = 0)
    total == 0 && return Float64[]

    isnothing(lambda) && return fill(Float64(defaultLambda), total)

    if lambda isa Real
        return fill(checkPenaltyWeight(lambda, "lambda"), total)
    end

    names = penaltyCoefficientNames(model; penaltyTarget = penaltyTarget)

    if lambda isa AbstractVector
        length(lambda) == total || throw(
            ArgumentError(
                "lambda has $(length(lambda)) weights but this model has $(total) " *
                "penalized coefficients under penaltyTarget = :$(penaltyTarget). The " *
                "expected order is [" * join(names, ", ") * "]. Call " *
                "penaltyCoefficientNames(model; penaltyTarget = :$(penaltyTarget)) to " *
                "obtain it.",
            ),
        )
        return Float64[checkPenaltyWeight(v, names[i]) for (i, v) in enumerate(lambda)]
    end

    if lambda isa NamedTuple || lambda isa AbstractDict
        given = Dict{Symbol,Any}()
        for k in keys(lambda)
            block = canonicalPenaltyBlock(k)
            haskey(given, block) && throw(
                ArgumentError(
                    "Penalty block :$(block) is named more than once in lambda (two " *
                    "aliases of the same block).",
                ),
            )
            given[block] = lambda[k]
        end
        missingBlocks = [b for b in blocks if !haskey(given, b)]
        isempty(missingBlocks) || throw(
            ArgumentError(
                "lambda does not name the penalized block(s) " *
                join([":" * string(b) for b in missingBlocks], ", ") *
                ". A structured lambda must name every penalized block of the model; pass " *
                "0.0 to leave one unpenalized. Penalized blocks under penaltyTarget = " *
                ":$(penaltyTarget): " *
                join([":" * string(b) for b in blocks], ", ") * ".",
            ),
        )
        extraBlocks = sort([b for b in keys(given) if !(b in blocks)])
        isempty(extraBlocks) || throw(
            ArgumentError(
                "lambda names the block(s) " *
                join([":" * string(b) for b in extraBlocks], ", ") *
                ", which are not penalized here -- either absent from the model or " *
                "excluded by penaltyTarget = :$(penaltyTarget). Penalized blocks: " *
                join([":" * string(b) for b in blocks], ", ") * ".",
            ),
        )

        weights = Float64[]
        for (b, n) in zip(blocks, lengths)
            v = given[b]
            blockNames = penaltyBlockCoefficientNames(model, b)
            if v isa AbstractVector
                length(v) == n || throw(
                    ArgumentError(
                        "lambda[:$(b)] has $(length(v)) weights but the block has $(n) " *
                        "coefficients (" * join(blockNames, ", ") * ").",
                    ),
                )
                for (i, vi) in enumerate(v)
                    push!(weights, checkPenaltyWeight(vi, blockNames[i]))
                end
            else
                w = checkPenaltyWeight(v, "lambda[:$(b)]")
                append!(weights, fill(w, n))
            end
        end
        return weights
    end

    throw(
        ArgumentError(
            "lambda must be a real number, a vector of weights, or a NamedTuple/Dict keyed " *
            "by coefficient block; got $(typeof(lambda)).",
        ),
    )
end

"""
    PenaltySpec

The penalized coefficients of one JuMP model: their names, their variables and their
weights, all in the canonical order and all of the same length. Built once per fit by
[`penaltySpec`](@ref) and consumed by the penalized objectives.
"""
struct PenaltySpec
    names::Vector{String}
    vars::Vector{VariableRef}
    weights::Vector{Float64}
end

"""
    penaltySpec(jumpModel, model, penaltyTarget, lambda, defaultLambda) -> PenaltySpec

Walks the penalized blocks ONCE, collecting the JuMP variables and the weights in the same
pass, so the two vectors cannot drift apart.
"""
function penaltySpec(
    jumpModel::Model,
    model,
    penaltyTarget::Symbol,
    lambda,
    defaultLambda::Real,
)
    blocks = penaltyBlocks(model, penaltyTarget)
    names = String[]
    vars = VariableRef[]
    for b in blocks
        append!(vars, Vector{VariableRef}([jumpModel[PENALTY_BLOCK_VARIABLE[b]]...]))
        append!(names, penaltyBlockCoefficientNames(model, b))
    end
    weights = resolvePenaltyWeights(model, lambda, penaltyTarget, defaultLambda)
    # Internal invariant, not a caller error: the JuMP container and the model order must
    # agree in length, otherwise a weight would land on the wrong coefficient.
    @assert length(vars) == length(weights) == length(names) "penalty specification is inconsistent: $(length(vars)) variables, $(length(weights)) weights, $(length(names)) names"
    return PenaltySpec(names, vars, weights)
end

"""
    penaltyIsUniform(spec) -> Bool

Whether every penalized coefficient carries the same weight. The uniform case emits the
HISTORICAL scalar expression verbatim (`lambda * sum(...)`) rather than the weighted sum, so
a scalar `lambda` -- and a heterogeneous one whose weights happen to be equal -- builds
exactly the objective the package built before this feature existed.
"""
penaltyIsUniform(spec::PenaltySpec) =
    isempty(spec.weights) || all(==(spec.weights[1]), spec.weights)

"""
    penaltyAbsoluteVariables!(jumpModel, spec) -> Vector{VariableRef}

Auxiliary variables linearizing the absolute value of each penalized coefficient, with the
usual pair of constraints. They are NOT model coefficients and are never themselves
penalized.
"""
function penaltyAbsoluteVariables!(jumpModel::Model, spec::PenaltySpec)
    n = length(spec.vars)
    @variable(jumpModel, absShrunk[i = 1:n] >= 0)
    @constraints(
        jumpModel,
        begin
            [i = 1:n], absShrunk[i] >= spec.vars[i]
            [i = 1:n], absShrunk[i] >= -spec.vars[i]
        end
    )
    return absShrunk
end
