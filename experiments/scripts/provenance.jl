# Provenance stamp for the experiment runners.
#
# A measurement whose binary is not identifiable is not a measurement. Every output CSV
# therefore opens with a comment line naming the package commit, the working-tree state,
# the Julia version, the solver stack, and the platform. Analysers skip lines starting
# with `#`.
#
# SCOPE. Two trees can change the numbers, and both are checked:
#
#   1. the Sarimax.jl package itself;
#   2. the harness repository that holds the wrappers and the runners.
#
# Checking only (1) is not enough. A production wrapper living in the harness repository
# can change the configuration that is handed to `auto` — the short-series threshold, for
# instance — while the package tree stays pristine and the stamp still reads "clean".
#
# Call `stamp(io)` as the FIRST line of every output CSV, and `requireCleanTrees()` before
# any run whose numbers are meant to be cited.
module Provenance

using Dates: now
using Pkg

export stamp, requireCleanTrees, provenanceLine, provenanceFields

# `experiments/scripts` -> package root
const PKG_REPO = normpath(joinpath(@__DIR__, "..", ".."))
# The harness repository is passed in explicitly: it is not a fixed relative path, because
# the runners can be invoked from a checkout laid out differently than ours.
const HARNESS_REPO = get(ENV, "REPLICATION_HARNESS_REPO", pwd())

function _git(repo, args...)
    try
        strip(read(`git -C $repo $(collect(args))`, String))
    catch
        "unknown"
    end
end

# MODIFIED TRACKED FILES ONLY. Untracked files are deliberately not counted as dirty: a
# research harness always has scratch output beside the runners, and a check that trips on
# those would be switched off within a day — which is how the defect this module exists to
# prevent got in. What changes a result is an edit to a file that is part of the run, and
# that is what `--untracked-files=no` reports.
#
# The untracked count is still recorded, because it is occasionally the explanation for a
# result that cannot be reproduced from a clean checkout.
_dirty(repo) = !isempty(_git(repo, "status", "--porcelain", "--untracked-files=no"))

_untrackedCount(repo) =
    (s = _git(repo, "ls-files", "--others", "--exclude-standard");
     s == "unknown" ? -1 : (isempty(s) ? 0 : count(==('\n'), s) + 1))

"""
    solverVersions() -> Dict{String,String}

Versions of the dependencies that can move a numeric result: the JuMP/MathOptInterface
layer and the solvers. Read from the active manifest, not hard-coded, so the stamp cannot
drift away from the environment that is actually loaded.
"""
function solverVersions()
    out = Dict{String,String}()
    deps = Pkg.dependencies()
    for (_, info) in deps
        if info.name in ("MathOptInterface", "Ipopt", "JuMP", "SCIP", "HiGHS")
            out[info.name] = isnothing(info.version) ? "unknown" : string(info.version)
        end
    end
    out
end

# `Sys.KERNEL` reports `:NT` on Windows, which is accurate but reads oddly next to the
# archived rows. Normalised to the familiar name so the two agree.
platform() = string(Sys.iswindows() ? "Windows" :
                    Sys.isapple() ? "macOS" :
                    Sys.islinux() ? "Linux" : Sys.KERNEL, "-", Sys.ARCH)

"""
    provenanceFields() -> Vector{Pair{String,String}}

The provenance as ordered key/value pairs. The runners append these as COLUMNS on every
result row, so that a row separated from its file still carries its own origin.
"""
function provenanceFields()
    sv = solverVersions()
    [
        "sarimax_commit" => _git(PKG_REPO, "rev-parse", "--short", "HEAD"),
        "sarimax_tree" => _dirty(PKG_REPO) ? "DIRTY" : "clean",
        "sarimax_untracked" => string(_untrackedCount(PKG_REPO)),
        "harness_commit" => _git(HARNESS_REPO, "rev-parse", "--short", "HEAD"),
        "harness_tree" => _dirty(HARNESS_REPO) ? "DIRTY" : "clean",
        "harness_untracked" => string(_untrackedCount(HARNESS_REPO)),
        "julia" => string(VERSION),
        "moi" => get(sv, "MathOptInterface", "unknown"),
        "ipopt" => get(sv, "Ipopt", "unknown"),
        "jump" => get(sv, "JuMP", "unknown"),
        "scip" => get(sv, "SCIP", "absent"),
        "highs" => get(sv, "HiGHS", "absent"),
        "platform" => platform(),
    ]
end

provenanceLine() =
    string("# provenance: ",
           join((string(k, "=", v) for (k, v) in provenanceFields()), " "),
           " when=", string(now()))

stamp(io::IO) = println(io, provenanceLine())

"""
    requireCleanTrees(; allowDirty = false)

Abort if either working tree carries modifications to tracked files. The cost of
discovering afterwards that a grid ran against an uncommitted patch, and having to re-run
it, is far higher than the cost of stopping here.

Set `REPLICATION_ALLOW_DIRTY=1` for an exploratory run. That does not silence the problem:
the tree state is still recorded as `DIRTY` in the stamp and in every result row, so the
output stays self-describing and can never be mistaken for a citable measurement.
"""
function requireCleanTrees(; allowDirty::Bool = get(ENV, "REPLICATION_ALLOW_DIRTY", "") == "1")
    dirty = String[]
    _dirty(PKG_REPO) && push!(dirty, string("Sarimax.jl (", PKG_REPO, ")"))
    _dirty(HARNESS_REPO) && push!(dirty, string("harness (", HARNESS_REPO, ")"))
    if !isempty(dirty)
        msg = string("tracked files MODIFIED in: ", join(dirty, ", "), "\n",
                     _git(PKG_REPO, "status", "--short", "--untracked-files=no"), "\n",
                     _git(HARNESS_REPO, "status", "--short", "--untracked-files=no"))
        allowDirty ||
            error(msg, "\nThe measurement would not be identifiable.\n",
                  "Commit, stash, or set REPLICATION_ALLOW_DIRTY=1 for an exploratory run.")
        @warn string("RUNNING AGAINST A MODIFIED TREE — output is marked DIRTY and is not ",
                     "citable.\n", msg)
    end
    println(provenanceLine())
    flush(stdout)
end

end # module
