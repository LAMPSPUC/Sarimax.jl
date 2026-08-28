# Paired arm comparison: for a result file with several arms measured on the same series,
# report level and paired win rate per arm, plus the validity columns.
#
# Serves campaigns B, C, D, E and F, whose files share the shape "one row per (series,
# arm)". Campaign A has its own generator because it is reported by horizon block against
# an external reference.
#
# PAIRING IS ENFORCED, NOT ASSUMED. Only series for which EVERY arm produced an OK row
# enter the comparison. Comparing arms over different series subsets is how a difference in
# coverage gets read as a difference in method — the arms that fail are usually the hard
# series, so dropping them silently favours whichever arm failed more.
#
# LEVELS ARE MEANS, DELTAS ARE PAIRED. The mean of a heavy-tailed per-series metric is
# dominated by its tail; the paired median and win rate say what happens to the typical
# series. Both are printed because they routinely disagree in sign, and a table that shows
# only one of them is choosing an answer.
#
#   julia make_table_arms.jl <resultCsv> [armColumn]
using Statistics, Printf

const PATH = length(ARGS) >= 1 ? ARGS[1] : error("usage: make_table_arms.jl <resultCsv> [armColumn]")
const ARMCOL = length(ARGS) >= 2 ? ARGS[2] : ""

function read(path, armcol)
    rows = Dict{Tuple{String,String,Int},NamedTuple}(); hdr = nothing
    for line in eachline(path)
        startswith(line, "#") && continue
        v = split(strip(line), ',')
        if hdr === nothing; hdr = v; continue; end
        length(v) < length(hdr) && continue
        ix = Dict(hdr[i] => i for i in eachindex(hdr))
        col = isempty(armcol) ? (haskey(ix, "arm") ? "arm" : "cell") : armcol
        haskey(ix, col) || error("no arm column found; pass one explicitly")
        sid = tryparse(Int, v[ix["series"]]); isnothing(sid) && continue
        arm = strip(v[ix[col]], '"')
        freq = haskey(ix, "freq") ? strip(v[ix["freq"]], '"') : "-"
        status = strip(v[ix["status"]], '"')
        sm = tryparse(Float64, v[ix["smape"]])
        ma = haskey(ix, "mase") ? tryparse(Float64, v[ix["mase"]]) : nothing
        sec = haskey(ix, "seconds") ? tryparse(Float64, strip(v[ix["seconds"]], '"')) : nothing
        solver = haskey(ix, "solver_status") ? strip(v[ix["solver_status"]], '"') : "-"
        rows[(arm, freq, sid)] = (
            ok = startswith(status, "OK"),
            smape = isnothing(sm) ? NaN : sm,
            mase = isnothing(ma) ? NaN : ma,
            seconds = isnothing(sec) ? NaN : sec,
            solver = solver)
    end
    rows
end

rows = read(PATH, ARMCOL)
isempty(rows) && (println("$PATH: empty"); exit(0))

arms = sort(unique([a for (a, _, _) in keys(rows)]))
freqs = sort(unique([f for (_, f, _) in keys(rows)]))

println("="^96)
println(basename(PATH))
println("="^96)

for freq in freqs
    ids = sort(unique([s for (a, f, s) in keys(rows) if f == freq]))
    # paired subset: every arm produced an OK row for this series
    paired = [s for s in ids if all(haskey(rows, (a, freq, s)) && rows[(a, freq, s)].ok for a in arms)]
    total = length(ids)
    println()
    @printf("%s   %d series, %d complete across all %d arms (%.2f%% dropped as incomplete)\n",
            uppercase(freq), total, length(paired), length(arms),
            total == 0 ? 0.0 : 100 * (total - length(paired)) / total)
    isempty(paired) && continue

    ref = arms[1]
    refS = [rows[(ref, freq, s)].smape for s in paired]

    @printf("  %-14s %9s %9s %11s %10s %9s %9s %9s\n",
            "arm", "sMAPE", "MASE", "vs " * first(ref, 8), "wins", "errors",
            "non-LS", ">=120s")
    println("  " * "-"^90)
    for a in arms
        cells = [rows[(a, freq, s)] for s in paired]
        allCells = [rows[(a, freq, s)] for s in ids if haskey(rows, (a, freq, s))]
        S = [c.smape for c in cells]; M = [c.mase for c in cells]
        d = S .- refS
        wins = count(x -> x < 0, d)
        nErr = count(c -> !c.ok, allCells)
        nNonLS = count(c -> c.ok && c.solver != "-" && c.solver != "LOCALLY_SOLVED", allCells)
        nCap = count(c -> !isnan(c.seconds) && c.seconds >= 119.5, allCells)
        @printf("  %-14s %9.4f %9.4f %11s %10s %9d %9s %9d\n",
                a, mean(S), mean(filter(!isnan, M)),
                a == ref ? "-" : @sprintf("%+.4f", mean(d)),
                a == ref ? "-" : @sprintf("%.1f%%", 100 * wins / length(paired)),
                nErr,
                nNonLS == 0 && all(c -> c.solver == "-", allCells) ? "n/r" : string(nNonLS),
                nCap)
    end
    println()
    println("  paired deltas against $ref (negative = better than $ref):")
    for a in arms
        a == ref && continue
        d = [rows[(a, freq, s)].smape - rows[(ref, freq, s)].smape for s in paired]
        @printf("    %-14s mean %+8.4f   median %+8.4f   p05 %+8.3f   p95 %+8.3f\n",
                a, mean(d), median(d), quantile(d, 0.05), quantile(d, 0.95))
    end
end
println()
println("non-LS = returned candidate not LOCALLY_SOLVED; `n/r` = solver status not recorded")
println("in this file. NOTE: the status describes the RETURNED candidate only. Candidates")
println("that failed inside the search are absorbed by `auto` and are not counted anywhere;")
println("see REPRODUCE.md, finding 5.")
println(">=120s counts fits at or beyond the 120 s cap where one was set, and is a")
println("CENSORING RATE for those campaigns. Where no cap was set it is a cost tail only.")
