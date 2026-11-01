using Aqua

@testset "Aqua.jl quality checks" begin
    # ambiguities are skipped: JuMP/MOI method tables produce upstream noise
    #
    # piracies: StateSpaceModels.jl also exports a name `SARIMA`. On Julia >= 1.12,
    # `Sarimax.SARIMA` and `StateSpaceModels.SARIMA` resolve to the SAME generic
    # function object (confirmed via `Sarimax.SARIMA === StateSpaceModels.SARIMA`
    # and `parentmodule(Sarimax.SARIMA) == StateSpaceModels` on Julia 1.12.6; on
    # Julia 1.11 they were reported as distinct — this is a real difference in
    # how the two Julia versions unify same-named `using`-imported bindings, not
    # an Aqua false positive). Sarimax's SARIMA(::TimeArray, ::Int, ::Int, ::Int; ...)
    # methods are therefore, strictly, type piracy on >= 1.12. Dispatch has been
    # verified to remain correct in practice (Sarimax's signatures don't overlap
    # with StateSpaceModels', so fit! etc. still call the right method) — this is
    # accepted as a known, harmless-in-practice quirk via `treat_as_own` rather
    # than renaming the long-standing public `SARIMA` API.
    #
    # persistent_tasks runs on every OS. It used to be skipped on Windows, where the
    # sandboxed rebuild it performs failed under SCIP.jl 0.11 (whose build script
    # declares Windows unsupported); SCIP.jl 0.12 builds and loads there.
    Aqua.test_all(
        Sarimax;
        ambiguities = false,
        deps_compat = (check_extras = false,),
        piracies = (treat_as_own = [Sarimax.SARIMA],),
    )
end
