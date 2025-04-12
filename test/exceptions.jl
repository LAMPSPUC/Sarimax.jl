using Test

@testset "Sarimax Exceptions" begin
    @testset "ModelNotFitted" begin
        e = ModelNotFitted()
        @test e isa ModelNotFitted

        output = sprint(showerror, e)
        @test output == "The model has not been fitted yet. Please run fit!(model)"

        # Test error message
        buf = IOBuffer()
        showerror(buf, e)
        @test String(take!(buf)) == "The model has not been fitted yet. Please run fit!(model)"
        @test_throws ModelNotFitted throw(ModelNotFitted())
    end

    @testset "MissingMethodImplementation" begin
        method_name = "test_method"
        e = MissingMethodImplementation(method_name)
        @test e isa MissingMethodImplementation
        @test e.method == method_name

        # Test that the error message is correctly printed
        output = sprint(showerror, e)
        @test output == "The model does not implement the test_method method."

        # Test error message
        buf = IOBuffer()
        showerror(buf, e)
        @test String(take!(buf)) == "The model does not implement the test_method method."
        @test_throws MissingMethodImplementation throw(MissingMethodImplementation(method_name))

    end

    @testset "InconsistentDatePattern" begin
        e = InconsistentDatePattern()
        @test e isa InconsistentDatePattern

        output = sprint(showerror, e)
        @test output == "The timestamps do not follow a consistent pattern."

        # Test error message
        buf = IOBuffer()
        showerror(buf, e)
        @test String(take!(buf)) == "The timestamps do not follow a consistent pattern."
        @test_throws InconsistentDatePattern throw(InconsistentDatePattern())
    end

    @testset "MissingExogenousData" begin
        e = MissingExogenousData()
        @test e isa MissingExogenousData

        output = sprint(showerror, e)
        @test output == "There is no exogenous data to forecast the horizon requested"

        # Test error message
        buf = IOBuffer()
        showerror(buf, e)
        @test String(take!(buf)) == "There is no exogenous data to forecast the horizon requested"
        @test_throws MissingExogenousData throw(MissingExogenousData())
    end

    @testset "InvalidParametersCombination" begin
        msg = "Test error message"
        e = InvalidParametersCombination(msg)
        @test e isa InvalidParametersCombination
        @test e.msg == msg

        output = sprint(showerror, e)
        @test output == "The parameters provided are invalid for the model \n$msg"

        # Test error message
        buf = IOBuffer()
        showerror(buf, e)
        @test String(take!(buf)) == "The parameters provided are invalid for the model \n$msg"
        @test_throws InvalidParametersCombination throw(InvalidParametersCombination(msg))
    end
end
