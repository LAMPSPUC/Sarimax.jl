# Port of the STL implementation that R's `stats::stl` calls (Cleveland, Cleveland,
# McRae & Terpenning 1990), following `src/library/stats/src/stl.f` subroutine by
# subroutine.
#
# WHY A PORT. The seasonal differencing decision of `auto.arima` is
# `seas.heuristic(x) > 0.64`, and `seas.heuristic` reads the components of `mstl(x)`,
# which for a plain `ts` reduces to `stl(x, s.window = 11)`. Any difference in the
# decomposition moves the seasonal strength and therefore `D`. Measured against
# `SeasonalTrendLoess.jl`: the components agree only to 0.3-4% of sd(y) (correlations
# 0.92-0.99), which is the same algorithm but not the same implementation — the jump /
# interpolation scheme in particular is not exposed there. That residual is small in the
# components and large in the strength, because `Fs = 1 - var(R)/var(R+S)` is a ratio of
# variances: when the remainder is small relative to the seasonal figure, a small absolute
# change in it moves the ratio a lot. On 36.6k M4 monthly series that showed up as `D`
# disagreeing with `nsdiffs` on 7.4%.
#
# The loess here deliberately keeps R's approximation: the local regression is evaluated
# every `njump` points and linearly interpolated in between, rather than at every point.
# Evaluating everywhere would be *more* accurate and would not reproduce R.

"""
    nextodd(x)

Round to the nearest odd integer, as R's `stl` does for its window widths.
"""
nextodd(x::Real) = (n = round(Int, x); isodd(n) ? n : n + 1)

"""
    stlEst(y, n, len, ideg, xs, nleft, nright, w, userw, rw)

Local regression at the single position `xs` using the observations in `nleft:nright`
(`stlest` in stl.f). Returns `(ys, ok)`; `ok = false` when the weights vanish, which the
callers treat as "keep the raw value".
"""
function stlEst(
    y::Vector{Float64},
    n::Int,
    len::Int,
    ideg::Int,
    xs::Float64,
    nleft::Int,
    nright::Int,
    w::Vector{Float64},
    userw::Bool,
    rw::Vector{Float64},
)
    range = float(n) - 1.0
    h = max(xs - float(nleft), float(nright) - xs)
    len > n && (h += float((len - n) ÷ 2))
    h9 = 0.999h
    h1 = 0.001h

    a = 0.0
    for j = nleft:nright
        w[j] = 0.0
        r = abs(float(j) - xs)
        if r <= h9
            if r <= h1
                w[j] = 1.0
            else
                w[j] = (1.0 - (r / h)^3)^3
            end
            userw && (w[j] *= rw[j])
            a += w[j]
        end
    end

    a <= 0.0 && return (0.0, false)

    for j = nleft:nright
        w[j] /= a
    end
    if h > 0.0 && ideg > 0
        # linear (or higher) fit: tilt the kernel weights so the weighted fit is the
        # local regression evaluated at xs
        a = 0.0
        for j = nleft:nright
            a += w[j] * float(j)
        end
        b = xs - a
        c = 0.0
        for j = nleft:nright
            c += w[j] * (float(j) - a)^2
        end
        if sqrt(c) > 0.001 * range
            b /= c
            for j = nleft:nright
                w[j] *= (b * (float(j) - a) + 1.0)
            end
        end
    end
    ys = 0.0
    for j = nleft:nright
        ys += w[j] * y[j]
    end
    return (ys, true)
end

"""
    stlEss!(ys, y, n, len, ideg, njump, userw, rw, res)

Loess smoothing of `y[1:n]` into `ys` (`stless` in stl.f). Evaluates every `njump`
positions and interpolates linearly between them, exactly as R does.
"""
function stlEss!(
    ys::Vector{Float64},
    y::Vector{Float64},
    n::Int,
    len::Int,
    ideg::Int,
    njump::Int,
    userw::Bool,
    rw::Vector{Float64},
    res::Vector{Float64},
)
    if n < 2
        ys[1] = y[1]
        return ys
    end
    newnj = min(njump, n - 1)
    nleft = 1
    nright = n

    if len >= n
        nleft = 1
        nright = n
        for i = 1:newnj:n
            v, ok = stlEst(y, n, len, ideg, float(i), nleft, nright, res, userw, rw)
            ys[i] = ok ? v : y[i]
        end
    elseif newnj == 1
        nsh = (len + 1) ÷ 2
        nleft = 1
        nright = len
        for i = 1:n
            if i > nsh && nright != n
                nleft += 1
                nright += 1
            end
            v, ok = stlEst(y, n, len, ideg, float(i), nleft, nright, res, userw, rw)
            ys[i] = ok ? v : y[i]
        end
    else
        nsh = (len + 1) ÷ 2
        for i = 1:newnj:n
            if i < nsh
                nleft = 1
                nright = len
            elseif i >= (n - nsh + 1)
                nleft = n - len + 1
                nright = n
            else
                nleft = i - nsh + 1
                nright = len + i - nsh
            end
            v, ok = stlEst(y, n, len, ideg, float(i), nleft, nright, res, userw, rw)
            ys[i] = ok ? v : y[i]
        end
    end

    if newnj != 1
        i = 1
        while i <= n - newnj
            delta = (ys[i+newnj] - ys[i]) / float(newnj)
            for j = (i+1):(i+newnj-1)
                ys[j] = ys[i] + delta * float(j - i)
            end
            i += newnj
        end
        k = ((n - 1) ÷ newnj) * newnj + 1
        if k != n
            v, ok = stlEst(y, n, len, ideg, float(n), nleft, nright, res, userw, rw)
            ys[n] = ok ? v : y[n]
            if k != n - 1
                delta = (ys[n] - ys[k]) / float(n - k)
                for j = (k+1):(n-1)
                    ys[j] = ys[k] + delta * float(j - k)
                end
            end
        end
    end
    return ys
end

"""
    stlMa!(ave, x, n, len)

Moving average of length `len` (`stlma` in stl.f); writes `n - len + 1` values.
"""
function stlMa!(ave::Vector{Float64}, x::Vector{Float64}, n::Int, len::Int)
    newn = n - len + 1
    flen = float(len)
    v = 0.0
    for i = 1:len
        v += x[i]
    end
    ave[1] = v / flen
    if newn > 1
        k = len
        m = 0
        for j = 2:newn
            k += 1
            m += 1
            v = v - x[m] + x[k]
            ave[j] = v / flen
        end
    end
    return ave
end

"""
    stlFts!(trend, x, n, np, work)

Low-pass filter: moving averages of length `np`, `np` and 3 (`stlfts` in stl.f).
"""
function stlFts!(
    trend::Vector{Float64},
    x::Vector{Float64},
    n::Int,
    np::Int,
    work::Vector{Float64},
)
    stlMa!(trend, x, n, np)
    stlMa!(work, trend, n - np + 1, np)
    stlMa!(trend, work, n - 2np + 2, 3)
    return trend
end

"""
    stlSs!(season, y, n, np, ns, isdeg, nsjump, userw, rw, w1, w2, w3, w4)

Smooth each cycle-subseries and extend it one period at each end (`stlss` in stl.f).
`season` has length `n + 2np`.
"""
function stlSs!(
    season::Vector{Float64},
    y::Vector{Float64},
    n::Int,
    np::Int,
    ns::Int,
    isdeg::Int,
    nsjump::Int,
    userw::Bool,
    rw::Vector{Float64},
    w1::Vector{Float64},
    w2::Vector{Float64},
    w3::Vector{Float64},
    w4::Vector{Float64},
)
    for j = 1:np
        k = (n - j) ÷ np + 1
        for i = 1:k
            w1[i] = y[(i-1)*np+j]
        end
        if userw
            for i = 1:k
                w3[i] = rw[(i-1)*np+j]
            end
        end
        # w2[2 .. k+1] receives the smoothed subseries; w2[1] and w2[k+2] are the
        # one-period extensions at each end
        sub = view(w2, 2:(k+1))
        tmp = Vector{Float64}(undef, k)
        stlEss!(tmp, w1, k, ns, isdeg, nsjump, userw, w3, w4)
        sub .= tmp

        v, ok = stlEst(w1, k, ns, isdeg, 0.0, 1, min(ns, k), w4, userw, w3)
        w2[1] = ok ? v : w2[2]

        v, ok = stlEst(w1, k, ns, isdeg, float(k + 1), max(1, k - ns + 1), k, w4, userw, w3)
        w2[k+2] = ok ? v : w2[k+1]

        for m = 1:(k+2)
            season[(m-1)*np+j] = w2[m]
        end
    end
    return season
end

"""
    stlRwt!(rw, y, fit, n)

Bisquare robustness weights on the absolute residuals (`stlrwt` in stl.f), with the
`3 * median` scale R uses.
"""
function stlRwt!(rw::Vector{Float64}, y::Vector{Float64}, fit::Vector{Float64}, n::Int)
    r = [abs(y[i] - fit[i]) for i = 1:n]
    sorted = sort(r)
    # R takes the average of the two central order statistics via psort at n/2+1 and n-n/2
    mid1 = n ÷ 2 + 1
    mid2 = n - n ÷ 2
    cmad = 3.0 * (sorted[mid1] + sorted[mid2])
    c9 = 0.999cmad
    c1 = 0.001cmad
    for i = 1:n
        ri = r[i]
        if ri <= c1
            rw[i] = 1.0
        elseif ri <= c9
            rw[i] = (1.0 - (ri / cmad)^2)^2
        else
            rw[i] = 0.0
        end
    end
    return rw
end

"""
    stlStp!(season, trend, y, n, np, ns, nt, nl, isdeg, itdeg, ildeg,
            nsjump, ntjump, nljump, ni, userw, rw)

The inner loop (`stlstp` in stl.f): `ni` passes of seasonal smoothing, low-pass removal
and trend smoothing.
"""
function stlStp!(
    season::Vector{Float64},
    trend::Vector{Float64},
    y::Vector{Float64},
    n::Int,
    np::Int,
    ns::Int,
    nt::Int,
    nl::Int,
    isdeg::Int,
    itdeg::Int,
    ildeg::Int,
    nsjump::Int,
    ntjump::Int,
    nljump::Int,
    ni::Int,
    userw::Bool,
    rw::Vector{Float64},
)
    m = n + 2np
    w1 = zeros(Float64, m)
    w2 = zeros(Float64, m)
    w3 = zeros(Float64, m)
    w4 = zeros(Float64, m)
    w5 = zeros(Float64, m)

    for _ = 1:ni
        for i = 1:n
            w1[i] = y[i] - trend[i]
        end
        stlSs!(w2, w1, n, np, ns, isdeg, nsjump, userw, rw, w3, w4, w5, season)
        stlFts!(w3, w2, m, np, w1)
        stlEss!(w1, w3, n, nl, ildeg, nljump, false, w4, w5)
        for i = 1:n
            season[i] = w2[np+i] - w1[i]
        end
        for i = 1:n
            w1[i] = y[i] - season[i]
        end
        stlEss!(trend, w1, n, nt, itdeg, ntjump, userw, rw, w3)
    end
    return (season, trend)
end

"""
    stlR(y, np; s_window, s_degree, t_window, t_degree, l_window, l_degree,
         s_jump, t_jump, l_jump, robust, inner, outer)

Seasonal-trend decomposition by loess, reproducing `stats::stl`. Returns a named tuple
`(seasonal, trend, remainder)`.

`s_window` accepts the symbol `:periodic`, which R maps to a window of `10n + 1` with
degree 0 followed by replacing the seasonal component with its per-cycle means.

The defaults mirror R's: `t_window = nextodd(ceil(1.5np / (1 - 1.5/s_window)))`,
`l_window = nextodd(np)`, the jumps are `ceil(window/10)`, and `robust = false` implies
`inner = 2, outer = 0`.
"""
function stlR(
    y::Vector{Float64},
    np::Int;
    s_window::Union{Int,Symbol},
    s_degree::Int = 0,
    t_window::Union{Int,Nothing} = nothing,
    t_degree::Int = 1,
    l_window::Int = nextodd(np),
    l_degree::Int = t_degree,
    s_jump::Union{Int,Nothing} = nothing,
    t_jump::Union{Int,Nothing} = nothing,
    l_jump::Union{Int,Nothing} = nothing,
    robust::Bool = false,
    inner::Int = robust ? 1 : 2,
    outer::Int = robust ? 15 : 0,
)
    n = length(y)
    np < 2 && throw(ArgumentError("the seasonal period must be at least 2"))
    n <= 2np && throw(ArgumentError("the series must span more than two periods"))

    periodic = false
    sw = 0
    if s_window === :periodic
        periodic = true
        sw = 10n + 1
        s_degree = 0
    else
        sw = s_window::Int
    end

    tw = isnothing(t_window) ? nextodd(ceil(1.5np / (1 - 1.5 / sw))) : t_window
    sj = isnothing(s_jump) ? ceil(Int, sw / 10) : s_jump
    tj = isnothing(t_jump) ? ceil(Int, tw / 10) : t_jump
    lj = isnothing(l_jump) ? ceil(Int, l_window / 10) : l_jump

    # stl.f forces the windows odd and at least 3, and the period at least 2
    newns = max(3, sw); iseven(newns) && (newns += 1)
    newnt = max(3, tw); iseven(newnt) && (newnt += 1)
    newnl = max(3, l_window); iseven(newnl) && (newnl += 1)
    newnp = max(2, np)

    season = zeros(Float64, n + 2newnp)
    trend = zeros(Float64, n)
    rw = ones(Float64, n)

    userw = false
    k = 0
    while true
        stlStp!(season, trend, y, n, newnp, newns, newnt, newnl,
                s_degree, t_degree, l_degree, sj, tj, lj, inner, userw, rw)
        k += 1
        k > outer && break
        fit = [trend[i] + season[i] for i = 1:n]
        stlRwt!(rw, y, fit, n)
        userw = true
    end
    outer <= 0 && fill!(rw, 1.0)

    seasonal = season[1:n]
    if periodic
        # R replaces the seasonal component by its per-cycle means
        cyc = [((i - 1) % np) + 1 for i = 1:n]
        medias = zeros(Float64, np)
        contas = zeros(Int, np)
        for i = 1:n
            medias[cyc[i]] += seasonal[i]
            contas[cyc[i]] += 1
        end
        for c = 1:np
            contas[c] > 0 && (medias[c] /= contas[c])
        end
        seasonal = [medias[cyc[i]] for i = 1:n]
    end
    remainder = [y[i] - seasonal[i] - trend[i] for i = 1:n]
    return (seasonal = seasonal, trend = copy(trend), remainder = remainder)
end
