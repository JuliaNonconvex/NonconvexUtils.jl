@testset "diffiy" begin
    @testset "Scalar-valued reverse-mode" begin
        global T = Nothing
        f = function (x)
            global T = eltype(x)
            return sum(x)
        end
        x = [1.0, 1.0]
        _f = diffiy(f, ADTypes.AutoForwardDiff(), x)
        Zygote.gradient(_f, x)
        @test T <: ForwardDiff.Dual

        _f = diffiy(f, ADTypes.AutoReverseDiff(), x)
        Zygote.gradient(_f, x)
        @test T <: ReverseDiff.TrackedReal

        _f = diffiy(f, ADTypes.AutoTracker(), x)
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
        _f = diffiy(f, ADTypes.AutoForwardDiff(), x)
        ForwardDiff.gradient(_f, x)
        @test T <: ForwardDiff.Dual

        _f = diffiy(f, ADTypes.AutoReverseDiff(), x)
        ForwardDiff.gradient(_f, x)
        @test T <: ReverseDiff.TrackedReal

        _f = diffiy(f, ADTypes.AutoTracker(), x)
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
        _f = diffiy(f, ADTypes.AutoForwardDiff(), x)
        Zygote.jacobian(_f, x)
        @test T <: ForwardDiff.Dual

        _f = diffiy(f, ADTypes.AutoReverseDiff(), x)
        Zygote.jacobian(_f, x)
        @test T <: ReverseDiff.TrackedReal

        _f = diffiy(f, ADTypes.AutoTracker(), x)
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
        _f = diffiy(f, ADTypes.AutoForwardDiff(), x)
        ForwardDiff.jacobian(_f, x)
        @test T <: ForwardDiff.Dual

        _f = diffiy(f, ADTypes.AutoReverseDiff(), x)
        ForwardDiff.jacobian(_f, x)
        @test T <: ReverseDiff.TrackedReal

        _f = diffiy(f, ADTypes.AutoTracker(), x)
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
        _f = diffiy(__f, ADTypes.AutoForwardDiff(), x...)
        f = x -> [_f(x[1:2], (x[3], x[4]))...]
        flatx = [1.0, 1.0, 2.0, 3.0]
        ForwardDiff.jacobian(f, flatx)
        @test T <: ForwardDiff.Dual

        _f = diffiy(__f, ADTypes.AutoReverseDiff(), x...)
        f = x -> [_f(x[1:2], (x[3], x[4]))...]
        ForwardDiff.jacobian(f, flatx)
        @test T <: ReverseDiff.TrackedReal

        _f = diffiy(__f, ADTypes.AutoTracker(), x...)
        f = x -> [_f(x[1:2], (x[3], x[4]))...]
        ForwardDiff.jacobian(f, flatx)
        @test T <: Tracker.TrackedReal
    end

    @testset "Model - first order = $first_order" for first_order in (true, false)
        f = (x::AbstractVector) -> sqrt(x[2])
        g = (x::AbstractVector, a, b) -> (a * x[1] + b)^3 - x[2]
        options = IpoptOptions(first_order = first_order)
        m = Model(f)
        addvar!(m, [0.0, 0.0], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        alg = IpoptAlg()
        sp_model = diffiy(m, ADTypes.AutoForwardDiff())
        r = NonconvexIpopt.optimize(sp_model, alg, [1.234, 2.345], options = options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
    end
end

# Backward compatibility: the deprecated `abstractdiffy` / `forwarddiffy` aliases
# must still work, routing the old AbstractDifferentiation backends through to
# their ADTypes equivalents.
@testset "abstractdiffy and forwarddiffy (deprecated aliases)" begin
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
