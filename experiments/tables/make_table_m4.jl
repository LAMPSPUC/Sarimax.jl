# Headline accuracy table: this package against `forecast::auto.arima`, by frequency AND by
# horizon block. Consumes results/raw/, fits nothing.
#
# This is why the runs stored the forecast, actual and MASE-denominator VECTORS rather than
# only the aggregates: any metric at any horizon stays recomputable without re-running.
#
# LOCKED CONVENTIONS (each of these has cost time at least once):
#   - the reference CSVs carry a `series,MASE,sMAPE,...` header: columns are read BY NAME;
#   - OWA is a RATIO OF MEANS, the published M4 convention. A mean of ratios gives ~0.99
#     and is not comparable to any published figure;
#   - the horizon blocks are the SAME ones the reference used to produce its
#     short/medium/long files. If they differed, the per-block comparison would be false;
#   - the MASE denominator is the one of the WHOLE series, identical across blocks.
#
#   julia make_table_m4.jl <resultCsv...>
using Statistics, Printf

const BLOCKS = Dict(
  "monthly"   => ("short"=>1:6,  "medium"=>7:12, "long"=>13:18, "total"=>1:18),
  "daily"     => ("short"=>1:4,  "medium"=>5:9,  "long"=>10:14, "total"=>1:14),
  "weekly"    => ("short"=>1:4,  "medium"=>5:9,  "long"=>10:13, "total"=>1:13),
  "quarterly" => ("short"=>1:2,  "medium"=>3:5,  "long"=>6:8,   "total"=>1:8),
  "hourly"    => ("short"=>1:16, "medium"=>17:32,"long"=>33:48, "total"=>1:48),
  "yearly"    => ("short"=>1:2,  "medium"=>3:4,  "long"=>5:6,   "total"=>1:6))

# Reference forecasts (auto.arima and Naive2) live outside this repository; see
# REPRODUCE.md for how to obtain them.
const REFDIR = get(ENV, "M4_REFERENCE_DIR", "reference")

function readOurs(path)
    rows = Dict{Int,NamedTuple}(); hdr = nothing; freq = ""
    for line in eachline(path)
        startswith(line, "#") && continue
        v = split(strip(line), ',')
        if hdr === nothing; hdr = v; continue; end
        ix = Dict(hdr[i] => i for i in eachindex(hdr))
        startswith(strip(v[ix["status"]], '"'), "OK") || continue
        sid = tryparse(Int, v[ix["series"]]); isnothing(sid) && continue
        fv = [parse(Float64, x) for x in split(strip(v[ix["forecast"]], '"'), ';') if !isempty(x)]
        av = [parse(Float64, x) for x in split(strip(v[ix["actual"]], '"'), ';') if !isempty(x)]
        (isempty(fv) || isempty(av)) && continue
        den = parse(Float64, v[ix["mase_den"]])
        freq = strip(v[ix["freq"]], '"')
        rows[sid] = (f = fv, a = av, den = den)
    end
    (rows, freq)
end

function readReference(freq, model, block)
    path = joinpath(REFDIR, freq, model, string(block, ".csv"))
    out = Dict{Int,Tuple{Float64,Float64}}(); isfile(path) || return out; hdr = nothing
    for line in eachline(path)
        v = split(strip(line), ',')
        if hdr === nothing; hdr = v; continue; end
        ix = Dict(hdr[i] => i for i in eachindex(hdr))
        sid = tryparse(Int, v[ix["series"]]); isnothing(sid) && continue
        a = tryparse(Float64, v[ix["sMAPE"]]); b = tryparse(Float64, v[ix["MASE"]])
        (isnothing(a) || isnothing(b) || !isfinite(a) || !isfinite(b)) && continue
        out[sid] = (a, b)
    end
    out
end

smape(f, a) = mean(2 .* abs.(f .- a) ./ (abs.(f) .+ abs.(a) .+ 1e-12)) * 100
mase(f, a, den) = mean(abs.(f .- a)) / max(den, 1e-12)

for path in ARGS
    (ours, freq) = readOurs(path)
    isempty(ours) && (println("$path: empty"); continue)
    println("="^86)
    @printf("%s   (%s)\n", uppercase(freq), basename(path))
    println("="^86)
    @printf("  %-8s %6s %19s %19s %19s\n", "block", "n", "sMAPE (ours/ref)",
            "MASE (ours/ref)", "OWA (ours/ref/gap)")
    println("  " * "-"^82)
    for (name, rng) in BLOCKS[freq]
        R  = readReference(freq, "autoarima_R", name)
        N2 = readReference(freq, "Naive", name)
        (isempty(R) || isempty(N2)) && (@printf("  %-8s  no reference\n", name); continue)
        common = [s for s in keys(ours) if haskey(R, s) && haskey(N2, s) &&
                  length(ours[s].f) >= last(rng) && length(ours[s].a) >= last(rng)]
        isempty(common) && continue
        sn = [smape(ours[s].f[rng], ours[s].a[rng]) for s in common]
        mn = [mase(ours[s].f[rng], ours[s].a[rng], ours[s].den) for s in common]
        sr = [R[s][1] for s in common];  mr = [R[s][2] for s in common]
        s2 = [N2[s][1] for s in common]; m2 = [N2[s][2] for s in common]
        owa(a, b) = 0.5 * (mean(a) / mean(s2)) + 0.5 * (mean(b) / mean(m2))
        on, orr = owa(sn, mn), owa(sr, mr)
        @printf("  %-8s %6d   %8.4f %8.4f   %8.4f %8.4f   %7.4f %7.4f %+7.4f\n",
                name, length(common), mean(sn), mean(sr), mean(mn), mean(mr), on, orr, on - orr)
    end
    println()
end
