# O termo de log-determinante do criterio concentrado, verificado contra a covariancia teorica
# construida diretamente. Sao dois fatos independentes, ambos exatos, e ambos ja erraram no
# pacote — por isso viram teste e nao comentario.
#
#   1. o EXPOENTE divide pelo numero de OBSERVACOES, nao pelo numero de termos quadrados;
#   2. o bloco MA tem determinante proprio, com a MESMA forma do bloco AR.
@testset "Determinant term of the concentrated criterion" begin
    "log|Omega| de um MA(q) puro, da ACF teorica"
    function maLogDet(θ::Vector{Float64}, n::Int)
        t = [1.0; θ]
        q = length(θ)
        γ = [k > q ? 0.0 : sum(t[j] * t[j+k] for j = 1:(length(t)-k)) for k = 0:(n-1)]
        logdet([γ[abs(i - j)+1] for i = 1:n, j = 1:n])
    end

    @testset "MA block: log|Omega| = -sum_j j*log(1 - kappa_j^2)" begin
        for θ in ([0.5], [0.7], [0.4, 0.2], [0.6, -0.3], [0.5, 0.3, 0.2])
            κ = Sarimax.maToReflection(θ)
            formula = -sum(j * log(1 - κ[j]^2) for j = 1:length(κ))
            @test isapprox(maLogDet(θ, 300), formula; atol = 1e-9)
        end
    end

    @testset "ARMA: the determinant does NOT factor into the two blocks" begin
        # registrado como limite conhecido: para ARMA misto a soma dos dois blocos NAO e o
        # determinante conjunto, e nao ha forma fechada polinomial nos coeficientes. Este teste
        # existe para que a suposicao contraria nao volte por engano.
        ϕ, θ = [0.6], [0.4]
        ψ = Sarimax.psiWeightsFromZero(ϕ, θ, 8000)
        n = 300
        γ = [sum(ψ[j] * ψ[j+k] for j = 1:(length(ψ)-k)) for k = 0:(n-1)]
        conjunto = logdet([γ[abs(i - j)+1] for i = 1:n, j = 1:n])
        soma =
            -sum(j * log(1 - Sarimax.arToReflection(ϕ)[j]^2) for j = 1:length(ϕ)) -
            sum(j * log(1 - Sarimax.maToReflection(θ)[j]^2) for j = 1:length(θ))
        @test abs(conjunto - soma) > 0.1
    end
end
