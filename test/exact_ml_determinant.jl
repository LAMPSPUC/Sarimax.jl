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

@testset "MA determinant does not require the invertible parameterisation" begin
    # A recursao inversa de Levinson-Durbin produz os `kappa` a partir dos proprios `theta`,
    # entao o termo vale com `theta` LIVRE. E ele proprio e a barreira: diverge em |kappa| -> 1.
    for θ in ([0.5], [0.4, 0.2], [0.6, -0.3])
        κ = Sarimax.maToReflection(θ)
        expr = Sarimax.maToReflectionExpr(θ, length(θ))
        @test isapprox(Float64.(expr), κ; atol = 1e-12)
    end
end
# O EXPOENTE em si, que o titulo do PR afirma e nenhum @test cobria. O caso nao-sazonal nao
# discrimina — com p<=2 e T=200 a razao T/nEf e 0,995 e os dois expoentes dao o mesmo argmin,
# que e por que a validacao original de 3e-5 passou com o expoente errado. Discrimina no
# SAZONAL, onde o bloco perfilado vale s*P e a razao cai a 0,88.
@testset "The determinant exponent divides by T, not by the term count" begin
    function omegaSAR(Φ::Vector{Float64}, s::Int, n::Int)
        ar = Sarimax.expandMultiplicativePolynomial(Float64[], Φ, s; negate = true)
        ψ = Sarimax.psiWeightsFromZero(ar, Float64[], 8000)
        γ = [sum(ψ[j] * ψ[j+k] for j = 1:(length(ψ)-k)) for k = 0:(n-1)]
        [γ[abs(i - j)+1] for i = 1:n, j = 1:n]
    end
    # -2logL exata concentrada, por algebra linear sobre a covariancia teorica
    function menos2logL(y::Vector{Float64}, Φ::Vector{Float64}, s::Int)
        n = length(y); Ω = omegaSAR(Φ, s, n)
        n * log((y' * (Ω \ y)) / n) + logdet(Ω)
    end
    criterio(y, Φ, s, expo) =
        (Ω = omegaSAR(Φ, s, length(y)); (y' * (Ω \ y)) * exp(logdet(Ω) / expo))

    s, n, P = 12, 180, 2
    Φtrue = [0.5, 0.25]
    nEf = n + s * P                       # o que `nEf` valeria: T mais o bloco perfilado
    Random.seed!(11)
    ar = Sarimax.expandMultiplicativePolynomial(Float64[], Φtrue, s; negate = true)
    m = length(ar)
    y = zeros(n + 600)
    for t = (m+1):(n+600)
        y[t] = sum(ar[i] * y[t-i] for i = 1:m) + randn()
    end
    y = y[601:end]

    grade = 0.05:0.05:0.95
    arg(f) = grade[argmin([f(φ) for φ in grade])]
    exato = arg(φ -> menos2logL(y, [φ, Φtrue[2]], s))
    comT = arg(φ -> criterio(y, [φ, Φtrue[2]], s, n))
    comNEf = arg(φ -> criterio(y, [φ, Φtrue[2]], s, nEf))

    # o expoente 1/T reproduz o argmin da verossimilhanca exata
    @test comT == exato
    # o expoente pelo numero de termos NAO reproduz, e erra para a borda do dominio
    @test comNEf != exato
    @test comNEf >= 0.9
end
