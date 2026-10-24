# The SCIP binaries of the replication environment are not shipped in the image.
#
# SCIP_jll and SCIP_PaPILO_jll carry SCIP 8.0.0 and SoPlex 6.0, under the ZIB Academic License,
# whose terms for redistributing them are not met by a public container image. The image build
# therefore removes their artifacts in the same layer that installs them, and
# `./reproduce.sh fetch scip` installs them on the user's machine through Julia's package
# manager, which verifies each artifact against the content hash (git-tree-sha1) recorded in
# the package's Artifacts.toml.
#
#   julia --project=<replication env> scip_artifacts.jl record FILE   write their identity to FILE
#   julia --project=<replication env> scip_artifacts.jl strip         delete them from every depot
#   julia --project=<replication env> scip_artifacts.jl fetch         install the missing ones
#                                                                      into DEPOT_PATH[1]
#   julia --project=<replication env> scip_artifacts.jl status        exit 0 if all are installed
using Pkg, TOML
using Artifacts: artifact_exists, artifact_meta
using Base: SHA1

const PACKAGES = ("SCIP_jll" => "SCIP", "SCIP_PaPILO_jll" => "SCIP_PaPILO")
const LICENCE = "ZIB Academic License (SCIP 8.0.0, SoPlex 6.0); https://scipopt.org/academic.txt"

function entries()
    deps = Pkg.dependencies()
    return map(PACKAGES) do (pkg, name)
        info = only(i for i in values(deps) if i.name == pkg)
        toml = joinpath(info.source, "Artifacts.toml")
        meta = artifact_meta(name, toml)
        meta === nothing && error("$pkg has no artifact $name for this platform")
        (package = pkg, version = string(info.version), artifact = name, toml = toml,
         hash = SHA1(meta["git-tree-sha1"]))
    end
end

function main(args)
    mode = isempty(args) ? "status" : args[1]
    es = entries()
    if mode == "record"
        doc = Dict("licence" => LICENCE,
                   "artifact" => [Dict("package" => e.package, "version" => e.version, "name" => e.artifact,
                                       "git-tree-sha1" => string(e.hash)) for e in es])
        open(io -> TOML.print(io, doc; sorted = true), args[2], "w")
        println("recorded ", join(("$(e.package) $(e.hash)" for e in es), ", "), " in ", args[2])
    elseif mode == "strip"
        for e in es, depot in DEPOT_PATH
            rm(joinpath(depot, "artifacts", string(e.hash)); recursive = true, force = true)
        end
        present = [e.package for e in es if artifact_exists(e.hash)]
        isempty(present) || error("still present after removal: $(join(present, ", "))")
        println("removed ", join((e.package for e in es), ", "), " (", LICENCE, ")")
    elseif mode == "fetch"
        println("SCIP binaries: ", LICENCE)
        for e in es
            if artifact_exists(e.hash)
                println("  $(e.package) $(e.version): already installed ($(e.hash))")
                continue
            end
            path = Pkg.Artifacts.ensure_artifact_installed(e.artifact, e.toml)
            artifact_exists(e.hash) || error("$(e.package): not installed after download")
            println("  $(e.package) $(e.version): installed and verified ($(e.hash)) in $path")
        end
    elseif mode == "status"
        missing_ = [e.package for e in es if !artifact_exists(e.hash)]
        println(isempty(missing_) ? "installed" : "missing: $(join(missing_, ", "))")
        exit(isempty(missing_) ? 0 : 1)
    else
        error("unknown mode $mode (record FILE | strip | fetch | status)")
    end
end

main(ARGS)
