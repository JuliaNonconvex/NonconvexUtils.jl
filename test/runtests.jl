using NonconvexUtils, ForwardDiff, ReverseDiff, Tracker, Zygote
using Test, LinearAlgebra, SparseArrays, NLsolve, IterativeSolvers
using StableRNGs

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
    @testset "Vector - Jacobian $jac - matrixfree $matrixfree" for jac in (false, true), matrixfree in (false, true)
        # Adapted from https://github.com/JuliaNLSolvers/NLsolve.jl/issues/205
        rng = StableRNG(123)
        nonlin = 0.1
        get_info_vec = N -> begin
            N = 10
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
    end

    @testset "Non-vector input and output" begin
        rng = StableRNG(123)
        nonlin = 0.1
        get_info_nonvec = N -> begin
            N = 10
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
            imf = ImplicitFunction(solve_x, f, matrixfree = false)
            obj = p -> sum(imf(p).a)
            g_auto = Zygote.gradient(obj, p0)[1]
            @test norm(g_analytic.a - g_auto.a) < 1e-6
        end
    end
end
