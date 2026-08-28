# Back-fill provenance COLUMNS onto the archived raw results.
#
# The runners in this directory emit provenance per row. The archived results predate that,
# and carry provenance only as a header comment — or, for three campaigns, not at all. This
# script copies each archived file into results/raw/ with the provenance columns appended,
# so that a row separated from its file still carries its own origin.
#
# WHAT IT WILL AND WILL NOT WRITE.
#
# It writes `not_recorded` for every field that was not captured at run time, and it does
# NOT substitute a reconstruction. The solver stack in particular was never recorded by any
# campaign on this machine: every attribution of a MathOptInterface or Ipopt version is a
# reconstruction from manifest timestamps and file naming, and reconstructions belong in
# REPRODUCE.md where their evidence can be read, not in a data column where they would be
# indistinguishable from a measurement.
#
# `platform` is an exception and is filled in: the archived files were produced on the
# machine that holds them, and that is a fact about the file, not an inference about a
# version.
#
#   julia backfill_provenance.jl <harnessDir> <outDir>
using Printf

const HARNESS = length(ARGS) >= 1 ? ARGS[1] : ".."
const OUTDIR  = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "results", "raw")

# The platform of the machine that produced every file listed below.
const PLATFORM = "Windows-x86_64"

# source file => (campaign id, destination name)
#
# `sarimax_commit` and `julia` are lifted from the file's own `# provenance:` header when
# it has one, and set to `not_recorded` when it does not. Nothing else is inferred here.
const FILES = [
    ("m4_innov_yearly.csv",          "A", "m4_innovations_yearly.csv"),
    ("m4_innov_quarterly.csv",       "A", "m4_innovations_quarterly.csv"),
    ("m4_innov_weekly_moi148.csv",   "A", "m4_innovations_weekly.csv"),
    ("m4_innov_hourly_moi148.csv",   "A", "m4_innovations_hourly.csv"),
    ("m4_innov_daily_moi148.csv",    "A", "m4_innovations_daily.csv"),
    ("m4_innov_monthly.csv",         "A", "m4_innovations_monthly.csv"),
    ("objetivos_monthly.csv",        "B", "objectives_monthly.csv"),
    ("objetivos_penal.csv",          "B", "objectives_monthly_penalized.csv"),
    ("multistart_aleatorio.csv",     "C", "multistart_random.csv"),
    ("isolamento.csv",               "D", "isolation.csv"),
    ("eixos_weekly.csv",             "E", "axes_weekly.csv"),
    ("eixos_6e8e0bd.csv",            "E", "axes_weekly_earlier_commit.csv"),
    ("stable_m4.csv",                "F", "stable_weekly_yearly.csv"),
]

const NEWCOLS = ["campaign", "sarimax_commit", "sarimax_tree", "harness_commit",
                 "harness_tree", "julia", "moi", "ipopt", "jump", "platform",
                 "source_file"]

# Header-only rename, so the archive reads in the same vocabulary as the runners in this
# directory. NO VALUE IS TOUCHED: this renames columns, it does not transform data. The
# original header of each file is preserved in the comment block written above it.
const RENAME = Dict(
    "prev_neg"   => "forecast_negative",
    "tempo"      => "seconds",
    "solver"     => "solver_status",
    "braco"      => "arm",
    "celula"     => "cell",
    "venceu_css" => "multistart_beat_zero",
)

renameHeader(h) = join([get(RENAME, c, c) for c in split(h, ',')], ',')

"""
    parseStamp(line) -> Dict{String,String}

Pull the key=value pairs out of a `# provenance: ...` header line.
"""
function parseStamp(line)
    out = Dict{String,String}()
    for tok in split(replace(line, "# provenance:" => "", "# proveniencia:" => ""))
        parts = split(tok, '='; limit = 2)
        length(parts) == 2 && (out[parts[1]] = parts[2])
    end
    out
end

mkpath(OUTDIR)
summary = Tuple{String,String,String,Int}[]

for (src, campaign, dst) in FILES
    path = joinpath(HARNESS, src)
    if !isfile(path)
        @warn "missing, skipped" src
        continue
    end
    lines = readlines(path)
    isempty(lines) && continue

    stamp = startswith(lines[1], "#") ? parseStamp(lines[1]) : Dict{String,String}()
    commit = get(stamp, "commit", "not_recorded")
    tree   = get(stamp, "arvore", get(stamp, "tree", "not_recorded"))
    tree   = tree == "limpa" ? "clean" : tree
    julia  = get(stamp, "julia", "not_recorded")

    # The harness repository state was never captured by the original stamp — which is the
    # defect that let an uncommitted production wrapper run unnoticed. It cannot be
    # recovered after the fact.
    values = [campaign, commit, tree, "not_recorded", "not_recorded", julia,
              "not_recorded", "not_recorded", "not_recorded", PLATFORM, src]

    body = startswith(lines[1], "#") ? lines[2:end] : lines
    isempty(body) && continue
    header = body[1]
    rows = body[2:end]

    open(joinpath(OUTDIR, dst), "w") do io
        println(io, "# provenance: campaign=", campaign, " sarimax_commit=", commit,
                " sarimax_tree=", tree, " julia=", julia, " platform=", PLATFORM,
                " source_file=", src)
        println(io, "# Solver versions and harness repository state were not captured at ",
                "run time; see REPRODUCE.md for the reconstruction and its evidence.")
        println(io, "# original header: ", header)
        println(io, join(vcat([renameHeader(header)], NEWCOLS), ','))
        suffix = string(',', join(map(v -> string('"', v, '"'), values), ','))
        n = 0
        for r in rows
            isempty(strip(r)) && continue
            println(io, r, suffix)
            n += 1
        end
        push!(summary, (dst, campaign, commit, n))
    end
end

println()
@printf("%-42s %-4s %-14s %8s\n", "file", "camp", "commit", "rows")
println("-"^72)
for (f, c, k, n) in summary
    @printf("%-42s %-4s %-14s %8d\n", f, c, k, n)
end
println("\nBACKFILL_DONE -> ", OUTDIR)
