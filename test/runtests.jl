using NonconvexUtils, ForwardDiff, ReverseDiff, Tracker, Zygote
using Test, LinearAlgebra, SparseArrays, NLsolve, IterativeSolvers
using StableRNGs, ChainRulesCore, NonconvexCore

@testset "AbstractDiffFunction" begin
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

@testset "TraceFunction" begin
    f = TraceFunction(sum, on_call = true)
    @test f.on_call == true
    @test f.on_grad == false
    f([2.0])
    @test f.trace == [(input = [2.0], output = 2.0)]
    Zygote.gradient(f, [3.0])
    @test f.trace == [(input = [2.0], output = 2.0)]

    f = TraceFunction(sum, on_grad = true)
    @test f.on_call == false
    @test f.on_grad == true
    f([2.0])
    @test f.trace == []
    Zygote.gradient(f, [3.0])
    @test f.trace == [(input = [3.0], output = 3.0, grad = [1.0])]

    f = TraceFunction(sum)
    @test f.on_call == true
    @test f.on_grad == true
    f = TraceFunction(sum, on_call = true, on_grad = true)
    @test f.on_call == true
    @test f.on_grad == true
    f([2.0])
    @test f.trace == [(input = [2.0], output = 2.0)]
    Zygote.gradient(f, [3.0])
    @test f.trace == [
        (input = [2.0], output = 2.0),
        (input = [3.0], output = 3.0, grad = [1.0]),
    ]

    f = TraceFunction(sum, on_call = false, on_grad = false)
    @test f.on_call == false
    @test f.on_grad == false
    f([2.0])
    @test f.trace == []
    Zygote.gradient(f, [3.0])
    @test f.trace == []
end

@testset "CustomGradFunction" begin
    fakeg = [2.0]
    f = CustomGradFunction(sum, x -> fakeg)
    @test Zygote.gradient(f, [1.0]) == (fakeg,)

    fakeJ = [1.0 2.0; 0.0 -1.0]
    f = CustomGradFunction(identity, x -> fakeJ)
    @test Zygote.jacobian(f, [1.0, 1.0]) == (fakeJ,)
end

@testset "CustomHessianFunction" begin
    fakeH = [3.0 -1.0; -1.0 2.0]
    f = CustomHessianFunction(sum, x -> fakeg, x -> fakeH)
    fakeg = [2.0, 2.0]
    @test Zygote.gradient(f, [1.0, 1.0]) == (fakeg,)
    @test Zygote.jacobian(x -> Zygote.gradient(f, x)[1], [1.0, 1.0]) == (fakeH,)

    hvp = (x, v) -> fakeH * v
    f = CustomHessianFunction(sum, x -> fakeg, hvp; hvp = true)
    H = Zygote.jacobian(x -> Zygote.gradient(f, x)[1], [1.0, 1.0])[1]
    @test norm(H - fakeH) < 1e-6
end

@testset "Implicit functions" begin
    @testset "Non-closure conditions" begin
        @testset "Vector input and output - Jacobian $jac - matrixfree $matrixfree" for jac in (false, true), matrixfree in (false, true)
            # Adapted from https://github.com/JuliaNLSolvers/NLsolve.jl/issues/205
            rng = StableRNG(123)
            nonlin = 0.1
            get_info_vec = N -> begin
                A = spdiagm(0 => fill(10.0, N), 1 => fill(-1.0, N-1), -1 => fill(-1.0, N-1))
                p0 = randn(rng, N)
                f = (p, x) -> A*x + nonlin*x.^2 - p
                solve_x = (p) -> begin
                    xstar = nlsolve(x -> f(p, x), zeros(N), method=:anderson, m=10).zero
                    return xstar, jac ? Zygote.jacobian(x -> f(p, x), xstar)[1] : nothing
                end
                g_analytic = gmres((A + Diagonal(2*nonlin*solve_x(p0)[1]))', ones(N))
                return solve_x, f, p0, g_analytic
            end

            solve_x, f, p0, g_analytic = get_info_vec(10)
            imf = ImplicitFunction(solve_x, f; matrixfree)
            obj = p -> sum(imf(p))
            g_auto = Zygote.gradient(obj, p0)[1]
            @test norm(g_analytic - g_auto) < 1e-6

            imf = ImplicitFunction(solve_x, f; matrixfree, tol = 0.0)
            obj = p -> sum(imf(p))
            g_auto = Zygote.gradient(obj, p0)[1]
            @test all(isnan.(g_auto))

            imf = ImplicitFunction(solve_x, f; matrixfree, tol = 0.0, error_on_tol_violation = true)
            obj = p -> sum(imf(p))
            @test_throws ArgumentError Zygote.gradient(obj, p0)[1]
        end

        @testset "Non-vector input and output" begin
            rng = StableRNG(123)
            nonlin = 0.1
            get_info_nonvec = N -> begin
                A = spdiagm(0 => fill(10.0, N), 1 => fill(-1.0, N-1), -1 => fill(-1.0, N-1))
                p0 = (a = randn(rng, N),)
                f = (p, x) -> A*x.a + nonlin*x.a.^2 - p.a
                solve_x = (p) -> begin
                    return (a = nlsolve(x -> f(p, (a = x,)), zeros(N), method=:anderson, m=10).zero,), nothing
                end
                g_analytic = (a = gmres((A + Diagonal(2*nonlin*solve_x(p0)[1].a))', ones(N)),)
                return solve_x, f, p0, g_analytic
            end

            for matrixfree in (false, true)
                solve_x, f, p0, g_analytic = get_info_nonvec(10)
                imf = ImplicitFunction(solve_x, f; matrixfree)
                obj = p -> sum(imf(p).a)
                g_auto = Zygote.gradient(obj, p0)[1]
                @test norm(g_analytic.a - g_auto.a) < 1e-6

                imf = ImplicitFunction(solve_x, f; matrixfree, tol = 0.0)
                obj = p -> sum(imf(p).a)
                g_auto = Zygote.gradient(obj, p0)[1]
                @test all(isnan.(g_auto.a))

                imf = ImplicitFunction(solve_x, f; matrixfree, tol = 0.0, error_on_tol_violation = true)
                obj = p -> sum(imf(p).a)
                @test_throws ArgumentError Zygote.gradient(obj, p0)[1]
            end
        end
    end

    @testset "Closure conditions" begin
        @testset "Vector input and output - Jacobian $jac - matrixfree $matrixfree" for jac in (false, true), matrixfree in (false, true)
            rng = StableRNG(123)
            nonlin = 0.1
            get_info_closure_vec = N -> begin
                A = spdiagm(0 => fill(10.0, N), 1 => fill(-1.0, N-1), -1 => fill(-1.0, N-1))
                p0 = randn(rng, N)
                f = (x) -> A*x + nonlin*x.^2 - p0
                solve_x = () -> begin
                    xstar = nlsolve(f, zeros(N), method=:anderson, m=10).zero
                    xstar, jac ? Zygote.jacobian(f, xstar)[1] : nothing
                end
                g_analytic = gmres((A + Diagonal(2*nonlin*solve_x()[1]))', ones(N))
                return p0, A, g_analytic
            end
    
            p0, A, g_analytic = get_info_closure_vec(10)
            obj = p -> begin
                N = 10
                # f closes over p
                f = (x) -> A*x + nonlin*x.^2 - p
                solve_x = () -> begin
                    return nlsolve(f, zeros(N), method=:anderson, m=10).zero, nothing
                end
                imf = ImplicitFunction(solve_x, f; matrixfree)
                return sum(imf())
            end
            g_auto = Zygote.gradient(obj, p0)[1]
            @test norm(g_analytic - g_auto) < 1e-6

            obj = p -> begin
                N = 10
                # f closes over p
                f = (x) -> A*x + nonlin*x.^2 - p
                solve_x = () -> begin
                    return nlsolve(f, zeros(N), method=:anderson, m=10).zero, nothing
                end
                imf = ImplicitFunction(solve_x, f; matrixfree, tol = 0.0)
                return sum(imf())
            end
            g_auto = Zygote.gradient(obj, p0)[1]
            @test all(isnan.(g_auto))

            obj = p -> begin
                N = 10
                # f closes over p
                f = (x) -> A*x + nonlin*x.^2 - p
                solve_x = () -> begin
                    return nlsolve(f, zeros(N), method=:anderson, m=10).zero, nothing
                end
                imf = ImplicitFunction(solve_x, f; matrixfree, tol = 0.0, error_on_tol_violation = true)
                return sum(imf())
            end
            @test_throws ArgumentError Zygote.gradient(obj, p0)[1]
        end

        @testset "Non-vector input and output" begin
            rng = StableRNG(123)
            nonlin = 0.1
            get_info_closure_nonvec = N -> begin
                N = 10
                A = spdiagm(0 => fill(10.0, N), 1 => fill(-1.0, N-1), -1 => fill(-1.0, N-1))
                p0 = (a = randn(rng, N),)
                _f = (x) -> A*x.a + nonlin*x.a.^2 - p0.a
                _solve_x = () -> begin
                    return (a = nlsolve(x -> _f((a = x,)), zeros(N), method=:anderson, m=10).zero,), nothing
                end
                return p0, A, (a = gmres((A + Diagonal(2*nonlin*_solve_x()[1].a))', ones(N)),)
            end
    
            for matrixfree in (false, true)
                p0, A, g_analytic = get_info_closure_nonvec(10)
                obj = p -> begin
                    N = 10
                    # f closes over p
                    f = (x) -> A*x.a + nonlin*x.a.^2 - p.a
                    solve_x = () -> begin
                        return (a = nlsolve(x -> f((a = x,)), zeros(N), method=:anderson, m=10).zero,), nothing
                    end
                    imf = ImplicitFunction(solve_x, f; matrixfree)
                    return sum(imf().a)
                end
                g_auto = Zygote.gradient(obj, p0)[1]
                @test norm(g_analytic.a - g_auto.a) < 1e-6

                obj = p -> begin
                    N = 10
                    # f closes over p
                    f = (x) -> A*x.a + nonlin*x.a.^2 - p.a
                    solve_x = () -> begin
                        return (a = nlsolve(x -> f((a = x,)), zeros(N), method=:anderson, m=10).zero,), nothing
                    end
                    imf = ImplicitFunction(solve_x, f; matrixfree, tol = 0.0)
                    return sum(imf().a)
                end
                g_auto = Zygote.gradient(obj, p0)[1]
                @test all(isnan.(g_auto.a))

                obj = p -> begin
                    N = 10
                    # f closes over p
                    f = (x) -> A*x.a + nonlin*x.a.^2 - p.a
                    solve_x = () -> begin
                        return (a = nlsolve(x -> f((a = x,)), zeros(N), method=:anderson, m=10).zero,), nothing
                    end
                    imf = ImplicitFunction(solve_x, f; matrixfree, tol = 0.0, error_on_tol_violation = true)
                    return sum(imf().a)
                end
                @test_throws ArgumentError Zygote.gradient(obj, p0)[1]
            end
        end
    end
end

# TODO tests:
## Chunk size of 1
## Multiple chunk sizes
## Multiple outputs
## Struct output
## Functor f - fix first
@testset "ForwardDiff frule" begin
    @eval begin
        f1(x, y) = x + y
        global frule_count = 0
        function ChainRulesCore.frule((_, Δx1, Δx2), ::typeof(f1), x1, x2)
            global frule_count += 1
            println("frule was called")
            return f1(x1, x2), Δx1 + Δx2
        end
        NonconvexUtils.@ForwardDiff_frule f1(x1::ForwardDiff.Dual, x2::ForwardDiff.Dual)
        NonconvexUtils.@ForwardDiff_frule f1(x1::AbstractVector{<:ForwardDiff.Dual}, x2::AbstractVector{<:ForwardDiff.Dual})
        NonconvexUtils.@ForwardDiff_frule f1(x1::AbstractMatrix{<:ForwardDiff.Dual}, x2::AbstractMatrix{<:ForwardDiff.Dual})

        f2(x::NamedTuple, y::NamedTuple) = (a = x.a + y.a, b = x.b + y.b)
        f2(x::AbstractVector, y::AbstractVector) = f2.(x, y)
        function ChainRulesCore.frule((_, Δx1, Δx2), ::typeof(f2), x1::NamedTuple, x2::NamedTuple)
            global frule_count += 1
            println("frule was called")
            return f2(x1, x2), (a = Δx1.a + Δx2.a, b = Δx1.b + Δx2.b)
        end
        NonconvexUtils.@ForwardDiff_frule f2(x1::NamedTuple{<:Any, <:Tuple{Vararg{<:ForwardDiff.Dual}}}, x2::NamedTuple{<:Any, <:Tuple{Vararg{<:ForwardDiff.Dual}}})

        struct MyStruct{T, T1, T2}
            a::T1
            b::T2
        end
        MyStruct(a, b) = MyStruct{typeof(a), typeof(a), typeof(b)}(a, b)

        # The @constructor macro takes the type (first) and constructor function (second)
        # The constructor function takes input the fields generated from ntfromstruct (as multiple positional arguments)
        # The ntfromstruct function can be overloaded for your type
        NonconvexCore.@constructor MyStruct MyStruct

        f2(x::MyStruct, y::MyStruct) = MyStruct(x.a + y.a, x.b + y.b)
        function ChainRulesCore.frule((_, Δx1, Δx2), ::typeof(f2), x1::MyStruct, x2::MyStruct)
            global frule_count += 1
            println("frule was called")
            return f2(x1, x2), MyStruct(Δx1.a + Δx2.a, Δx1.b + Δx2.b)
        end
        NonconvexUtils.@ForwardDiff_frule f2(x1::MyStruct{<:ForwardDiff.Dual}, x2::MyStruct{<:ForwardDiff.Dual})
        Base.sum(s::MyStruct) = s.a + s.b

        # I recommend creating your own type to avoid piracy
        _eigvals!(x) = eigvals!(x)
        function ChainRulesCore.frule((_, Δx), ::typeof(_eigvals!), x::Symmetric{<:Real})
            global frule_count += 1
            println("frule was called")
            return frule((NoTangent(), Δx), eigvals!, x)
        end

        # I recommend creating your own type to avoid piracy
        NonconvexCore.@constructor Symmetric Symmetric
        import NamedTupleTools: ntfromstruct, structfromnt
        ntfromstruct(a::Symmetric) = (data = a.data,)
        structfromnt(::Type{Symmetric}, x::NamedTuple) = Symmetric(x.data, :U)
        NonconvexUtils.@ForwardDiff_frule _eigvals!(A::Symmetric{<:ForwardDiff.Dual})
    end
    global frule_count = 0
    @testset "2 real inputs - 1 real output" begin
        _f = x -> f1(x[1], x[2])
        _f(rand(2))
        g1 = ForwardDiff.gradient(_f, rand(2))
        @test frule_count == 2
        g2 = Zygote.gradient(_f, rand(2))[1]
        @test g1 == g2
    end
    frule_count = 0
    @testset "2 vector inputs - 1 real output" begin
        g1 = ForwardDiff.gradient(x -> sum(f1(x[1:2], x[3:4])), rand(4))
        @test frule_count == 4
        g2 = Zygote.gradient(x -> sum(f1(x[1:2], x[3:4])), rand(4))[1]
        @test g1 == g2
    end
    frule_count = 0
    @testset "2 vector inputs - 1 vector output" begin
        j1 = ForwardDiff.jacobian(x -> f1(x[1:2], x[3:4]), rand(4))
        @test frule_count == 4
        j2 = Zygote.jacobian(x -> f1(x[1:2], x[3:4]), rand(4))[1]
        @test j1 == j2
    end
    frule_count = 0
    @testset "2 matrix inputs - 1 real output" begin
        g1 = ForwardDiff.gradient(x -> sum(f1(x[1:2,1:2], x[3:4,3:4])), rand(4, 4))
        @test frule_count == 16
        g2 = Zygote.gradient(x -> sum(f1(x[1:2,1:2], x[3:4,3:4])), rand(4, 4))[1]
        @test g1 == g2
    end
    frule_count = 0
    @testset "2 NamedTuple inputs - 1 real output" begin
        g1 = ForwardDiff.gradient(x -> sum(f2((a = x[1], b = x[2]), (a = x[3], b = x[4]))), rand(4))
        @test frule_count == 4
        g2 = Zygote.gradient(x -> sum(f2((a = x[1], b = x[2]), (a = x[3], b = x[4]))), rand(4))[1]
        @test g1 == g2
    end
    frule_count = 0
    @testset "2 struct inputs - 1 real output" begin
        g1 = ForwardDiff.gradient(x -> sum(f2(MyStruct(x[1], x[2]), MyStruct(x[3], x[4]))), rand(4))
        @test frule_count == 4
        g2 = Zygote.gradient(x -> sum(f2(MyStruct(x[1], x[2]), MyStruct(x[3], x[4]))), rand(4))[1]
        @test g1 == g2
    end
    frule_count = 0
    @testset "eigvals" begin
        # Gradient of trace
        g = ForwardDiff.gradient(x -> sum(_eigvals!(Symmetric(x))), rand(4, 4))
        @test frule_count == 16
        @test norm(g - I) < 1e-6
    end
end
