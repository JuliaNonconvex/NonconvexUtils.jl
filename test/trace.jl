@testset "TraceFunction" begin
    @testset "Reverse-mode" begin
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

    @testset "Forward-mode" begin
        f = TraceFunction(sum, on_call = true)
        @test f.on_call == true
        @test f.on_grad == false
        f([2.0, 2.0])
        @test f.trace == [(input = [2.0, 2.0], output = 4.0)]
        ForwardDiff.gradient(f, [3.0, 3.0])
        @test f.trace == [(input = [2.0, 2.0], output = 4.0)]

        f = TraceFunction(sum, on_grad = true)
        @test f.on_call == false
        @test f.on_grad == true
        f([2.0, 2.0])
        @test f.trace == []
        ForwardDiff.gradient(f, [3.0, 3.0])
        @test f.trace == [(input = [3.0, 3.0], output = 6.0, grad = [1.0, 1.0])]

        f = TraceFunction(sum)
        @test f.on_call == true
        @test f.on_grad == true
        f = TraceFunction(sum, on_call = true, on_grad = true)
        @test f.on_call == true
        @test f.on_grad == true
        f([2.0, 2.0])
        @test f.trace == [(input = [2.0, 2.0], output = 4.0)]
        ForwardDiff.gradient(f, [3.0, 3.0])
        @test f.trace == [
            (input = [2.0, 2.0], output = 4.0),
            (input = [3.0, 3.0], output = 6.0, grad = [1.0, 1.0]),
        ]

        f = TraceFunction(sum, on_call = false, on_grad = false)
        @test f.on_call == false
        @test f.on_grad == false
        f([2.0, 2.0])
        @test f.trace == []
        ForwardDiff.gradient(f, [3.0, 3.0])
        @test f.trace == []
    end
end
