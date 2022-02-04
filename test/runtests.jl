using NonconvexUtils, ForwardDiff, ReverseDiff, Tracker, Zygote
using Test

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
