using ChainRulesCore

@testset "CustomGradFunction" begin
    fakeg = [2.0]
    f = CustomGradFunction(sum, x -> fakeg)
    @test Zygote.gradient(f, [1.0]) == (fakeg,)
    @test ForwardDiff.gradient(f, [1.0]) == fakeg

    fakeJ = [1.0 2.0; 0.0 -1.0]
    f = CustomGradFunction(identity, x -> fakeJ)
    @test Zygote.jacobian(f, [1.0, 1.0]) == (fakeJ,)
    @test ForwardDiff.jacobian(f, [1.0, 1.0]) == fakeJ
end

@testset "CustomHessianFunction" begin
    fakeH = [3.0 -1.0; -1.0 2.0]
    f = CustomHessianFunction(sum, x -> fakeg, x -> fakeH)
    fakeg = [2.0, 2.0]
    @test Zygote.gradient(f, [1.0, 1.0]) == (fakeg,)
    @test Zygote.jacobian(x -> Zygote.gradient(f, x)[1], [1.0, 1.0]) == (fakeH,)
    @test ForwardDiff.gradient(f, [1.0, 1.0]) == fakeg
    @test ForwardDiff.jacobian(x -> Zygote.gradient(f, x)[1], [1.0, 1.0]) == fakeH
    @test ForwardDiff.jacobian(x -> ForwardDiff.gradient(f, x), [1.0, 1.0]) == fakeH

    hvp = (x, v) -> fakeH * v
    f = CustomHessianFunction(sum, x -> fakeg, hvp; hvp=true)
    H = Zygote.jacobian(x -> Zygote.gradient(f, x)[1], [1.0, 1.0])[1]
    @test norm(H - fakeH) < 1e-6
    H = ForwardDiff.jacobian(x -> Zygote.gradient(f, x)[1], [1.0, 1.0])
    @test norm(H - fakeH) < 1e-6
    H = ForwardDiff.jacobian(x -> ForwardDiff.gradient(f, x), [1.0, 1.0])
    @test norm(H - fakeH) < 1e-6
end

@testset "CustomGradFunction rrule with InplaceableThunk" begin
    fakeJ = [1.0 2.0; 0.0 -1.0]
    f = CustomGradFunction(identity, x -> fakeJ)
    v, pullback = ChainRulesCore.rrule(f, [1.0, 1.0])
    Δ = ChainRulesCore.InplaceableThunk((d, x) -> d, ChainRulesCore.Thunk(() -> [1.0, 1.0]))
    result = pullback(Δ)
    @test result[2] == fakeJ' * [1.0, 1.0]
end

@testset "CustomHessianFunction rrule with InplaceableThunk" begin
    fakeg = [2.0, 2.0]
    fakeH = [3.0 -1.0; -1.0 2.0]
    f = CustomHessianFunction(sum, x -> fakeg, x -> fakeH)
    v, pullback = ChainRulesCore.rrule(f, [1.0, 1.0])
    Δ = ChainRulesCore.InplaceableThunk((d, x) -> d, ChainRulesCore.Thunk(() -> 1.0))
    result = pullback(Δ)
    @test result[2] == fakeg * 1.0
end

@testset "CustomGradFunction evaluates the custom gradient once per forward pass" begin
    # Regression test: the custom gradient used to be evaluated on every
    # pullback application (once per Jacobian row in value_jacobian).
    calls = Ref(0)
    g = x -> (calls[] += 1; [1.0 0.0; 0.0 1.0])
    f = CustomGradFunction(identity, g)
    calls[] = 0
    val, jac = NonconvexCore.value_jacobian(f, [1.0, 2.0])
    @test val == [1.0, 2.0]
    @test jac == [1.0 0.0; 0.0 1.0]
    @test calls[] == 1
end
