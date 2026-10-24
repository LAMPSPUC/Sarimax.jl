# Verifies that the active Julia environment is exactly the pinned replication environment:
# the Julia version and every package version recorded in Manifest.toml, the Sarimax.jl
# source location, and that the solvers load and report their versions.
#
# Prints one JSON object per line ({"check", "status", "expected", "generated", "note"});
# exits non-zero if anything fails.

using Pkg
using TOML

const RESULTS = Dict{String,Any}[]

function record(check, status; expected = "", generated = "", note = "")
    push!(RESULTS, Dict("check" => check, "status" => status, "expected" => string(expected),
                        "generated" => string(generated), "note" => string(note)))
end

jsonstr(s) = "\"" * escape_string(string(s)) * "\""
function emit(d)
    println("{" * join(("$(jsonstr(k)): $(jsonstr(v))" for (k, v) in d), ", ") * "}")
end

project = Base.active_project()
manifest_path = joinpath(dirname(project), "Manifest.toml")
manifest = TOML.parsefile(manifest_path)

expected_julia = get(manifest, "julia_version", "unknown")
record("julia version vs Manifest.toml", string(VERSION) == expected_julia ? "PASS" : "FAIL";
       expected = expected_julia, generated = VERSION)

installed = Pkg.dependencies()
mismatches = String[]
counted = 0
for (name, entries) in manifest["deps"], entry in entries
    haskey(entry, "version") || continue          # standard libraries carry no version
    global counted += 1
    uuid = Base.UUID(entry["uuid"])
    info = get(installed, uuid, nothing)
    if info === nothing
        push!(mismatches, "$name not installed")
    elseif string(info.version) != entry["version"]
        push!(mismatches, "$name $(info.version) (locked $(entry["version"]))")
    end
end
record("julia packages vs Manifest.toml", isempty(mismatches) ? "PASS" : "FAIL";
       expected = "$counted pinned packages", generated = "$(counted - length(mismatches)) match",
       note = join(mismatches, "; "))

# The package under study must be this checkout's source, not a registry copy.
sarimax = [e for (n, es) in manifest["deps"] for e in es if n == "Sarimax"]
if length(sarimax) == 1 && haskey(sarimax[1], "path")
    src = normpath(joinpath(dirname(manifest_path), sarimax[1]["path"]))
    record("Sarimax.jl source", isfile(joinpath(src, "src", "Sarimax.jl")) ? "PASS" : "FAIL";
           expected = "developed from this checkout", generated = src)
else
    record("Sarimax.jl source", "FAIL"; expected = "developed from this checkout (path entry)",
           generated = "registry or missing")
end

# The image does not ship SCIP's binaries (scip_artifacts.jl), and Sarimax.jl loads SCIP: until
# `./reproduce.sh fetch scip` has installed them nothing below can load, and `check` reports
# them as a missing input.
scip_record = get(ENV, "IJF_SCIP_RECORD", "/opt/ijf-env/scip-artifacts.toml")
scip_absent = isfile(scip_record) &&
              any(a -> !Pkg.Artifacts.artifact_exists(Base.SHA1(a["git-tree-sha1"])),
                  TOML.parsefile(scip_record)["artifact"])
if scip_absent
    record("Sarimax.jl, JuMP, Ipopt, SCIP load", "SKIPPED";
           note = "needs the SCIP binaries, not shipped in the image (ZIB Academic License): " *
                  "./reproduce.sh fetch scip installs them")
else
    try
        @eval using Sarimax, JuMP, Ipopt, SCIP
        MOI = JuMP.MOI
        record("Ipopt binary", "PASS"; generated = MOI.get(Ipopt.Optimizer(), MOI.SolverVersion()))
        record("SCIP binary", "PASS"; generated = MOI.get(SCIP.Optimizer(), MOI.SolverVersion()))
        record("Sarimax.jl loads", "PASS"; generated = string(pkgversion(Sarimax)))
    catch err
        record("Sarimax.jl, JuMP, Ipopt, SCIP load", "FAIL"; note = sprint(showerror, err))
    end
end

foreach(emit, RESULTS)
exit(any(r -> r["status"] == "FAIL", RESULTS) ? 1 : 0)
