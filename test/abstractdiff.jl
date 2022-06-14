@testset "abstractdiffy and forwarddiffy" begin
    @testset "Scalar-valued reverse-mode" begin
        global T = Nothing
        f = function (x)
            global T = eltype(x)
            return sum(x)
        end
        x = [1.0, 1.0]
        _f = forwarddiffy(f, x)
        Zygote.gradient(_f, x)
        @test T <: ForwardDiff.Dual

        _f = abstractdiffy(f, AD.ReverseDiffBackend(), x)
        Zygote.gradient(_f, x)
        @test T <: ReverseDiff.TrackedReal

        _f = abstractdiffy(f, AD.TrackerBackend(), x)
        Zygote.gradient(_f, x)
        @test T <: Tracker.TrackedReal
    end
    @testset "Scalar-valued forward-mode" begin
        global T = Nothing
        f = function (x)
            global T = eltype(x)
            return sum(x)
        end
        x = [1.0, 1.0]
        _f = forwarddiffy(f, x)
        ForwardDiff.gradient(_f, x)
        @test T <: ForwardDiff.Dual

        _f = abstractdiffy(f, AD.ReverseDiffBackend(), x)
        ForwardDiff.gradient(_f, x)
        @test T <: ReverseDiff.TrackedReal

        _f = abstractdiffy(f, AD.TrackerBackend(), x)
        ForwardDiff.gradient(_f, x)
        @test T <: Tracker.TrackedReal
    end

    @testset "Vector-valued reverse-mode" begin
        global T = Nothing
        f = function (x)
            global T = eltype(x)
            return 2x
        end
        x = [1.0, 1.0]
        _f = forwarddiffy(f, x)
        Zygote.jacobian(_f, x)
        @test T <: ForwardDiff.Dual

        _f = abstractdiffy(f, AD.ReverseDiffBackend(), x)
        Zygote.jacobian(_f, x)
        @test T <: ReverseDiff.TrackedReal

        _f = abstractdiffy(f, AD.TrackerBackend(), x)
        Zygote.jacobian(_f, x)
        @test T <: Tracker.TrackedReal
    end
    @testset "Vector-valued forward-mode" begin
        global T = Nothing
        f = function (x)
            global T = eltype(x)
            return 2x
        end
        x = [1.0, 1.0]
        _f = forwarddiffy(f, x)
        ForwardDiff.jacobian(_f, x)
        @test T <: ForwardDiff.Dual

        _f = abstractdiffy(f, AD.ReverseDiffBackend(), x)
        ForwardDiff.jacobian(_f, x)
        @test T <: ReverseDiff.TrackedReal

        _f = abstractdiffy(f, AD.TrackerBackend(), x)
        ForwardDiff.jacobian(_f, x)
        @test T <: Tracker.TrackedReal
    end

    @testset "Multiple inputs, multiple outputs" begin
        global T = Nothing
        __f = function (x::AbstractVector, y::Tuple)
            global T = eltype(x)
            return 2x[1] + x[2], y[1] * y[2]
        end
        x = ([1.0, 1.0], (2.0, 3.0))
        _f = forwarddiffy(__f, x...)
        f = x -> [_f(x[1:2], (x[3], x[4]))...]
        flatx = [1.0, 1.0, 2.0, 3.0]
        ForwardDiff.jacobian(f, flatx)
        @test T <: ForwardDiff.Dual

        _f = abstractdiffy(__f, AD.ReverseDiffBackend(), x...)
        f = x -> [_f(x[1:2], (x[3], x[4]))...]
        ForwardDiff.jacobian(f, flatx)
        @test T <: ReverseDiff.TrackedReal

        _f = abstractdiffy(__f, AD.TrackerBackend(), x...)
        f = x -> [_f(x[1:2], (x[3], x[4]))...]
        ForwardDiff.jacobian(f, flatx)
        @test T <: Tracker.TrackedReal
    end
end
