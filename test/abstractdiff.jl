@testset "AbstractDiffFunction" begin
    @testset "Scalar-valued reverse-mode" begin
        global T = Nothing
        f = function (x)
            global T = eltype(x)
            return sum(x)
        end
        _f = ForwardDiffFunction(f)
        Zygote.gradient(_f, [1.0, 1.0])
        @test T <: ForwardDiff.Dual

        _f = AbstractDiffFunction(f, AD.ReverseDiffBackend())
        Zygote.gradient(_f, [1.0, 1.0])
        @test T <: ReverseDiff.TrackedReal

        _f = AbstractDiffFunction(f, AD.TrackerBackend())
        Zygote.gradient(_f, [1.0, 1.0])
        @test T <: Tracker.TrackedReal
    end
    @testset "Scalar-valued forward-mode" begin
        global T = Nothing
        f = function (x)
            global T = eltype(x)
            return sum(x)
        end
        _f = ForwardDiffFunction(f)
        ForwardDiff.gradient(_f, [1.0, 1.0])
        @test T <: ForwardDiff.Dual

        _f = AbstractDiffFunction(f, AD.ReverseDiffBackend())
        ForwardDiff.gradient(_f, [1.0, 1.0])
        @test T <: ReverseDiff.TrackedReal

        _f = AbstractDiffFunction(f, AD.TrackerBackend())
        ForwardDiff.gradient(_f, [1.0, 1.0])
        @test T <: Tracker.TrackedReal
    end

    @testset "Vector-valued reverse-mode" begin
        global T = Nothing
        f = function (x)
            global T = eltype(x)
            return 2x
        end
        _f = ForwardDiffFunction(f)
        Zygote.jacobian(_f, [1.0, 1.0])
        @test T <: ForwardDiff.Dual

        _f = AbstractDiffFunction(f, AD.ReverseDiffBackend())
        Zygote.jacobian(_f, [1.0, 1.0])
        @test T <: ReverseDiff.TrackedReal

        _f = AbstractDiffFunction(f, AD.TrackerBackend())
        Zygote.jacobian(_f, [1.0, 1.0])
        @test T <: Tracker.TrackedReal
    end
    @testset "Vector-valued forward-mode" begin
        global T = Nothing
        f = function (x)
            global T = eltype(x)
            return 2x
        end
        _f = ForwardDiffFunction(f)
        ForwardDiff.jacobian(_f, [1.0, 1.0])
        @test T <: ForwardDiff.Dual

        _f = AbstractDiffFunction(f, AD.ReverseDiffBackend())
        ForwardDiff.jacobian(_f, [1.0, 1.0])
        @test T <: ReverseDiff.TrackedReal

        _f = AbstractDiffFunction(f, AD.TrackerBackend())
        ForwardDiff.jacobian(_f, [1.0, 1.0])
        @test T <: Tracker.TrackedReal
    end
end
