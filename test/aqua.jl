using Aqua

@testset "Aqua.jl quality checks" begin
    # ambiguities are skipped: JuMP/MOI method tables produce upstream noise
    Aqua.test_all(Sarimax; ambiguities = false, deps_compat = (check_extras = false,))
end
