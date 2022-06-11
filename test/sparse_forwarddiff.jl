@testset "SparseForwardDiffFunction" begin
    f = SparseForwardDiffFunction(sum, rand(3); hessian = false)
    x = rand(3)
    @test Zygote.gradient(f, x)[1] ≈ ForwardDiff.gradient(f, rand(3))

    f = SparseForwardDiffFunction(x -> 2(x.^2) + x[1] * ones(3), rand(3); hessian = false)
    x = rand(3)
    @test Zygote.jacobian(f, x)[1] ≈ ForwardDiff.jacobian(f, x)

    f = SparseForwardDiffFunction(sum, rand(3); hessian = true)
    x = rand(3)
    @test Zygote.gradient(f, x)[1] ≈ ForwardDiff.gradient(f, rand(3))
    @test Zygote.hessian(f, x) ≈ ForwardDiff.hessian(f, rand(3))

    f = SparseForwardDiffFunction(x -> sum(x)^2 + x[1], rand(3); hessian = true)
    x = rand(3)
    @test Zygote.hessian(f, x) ≈ ForwardDiff.hessian(f, x)

    f = SparseForwardDiffFunction(x -> sum(x)^2 + x[1], rand(3); hessian = true)
    x = rand(3)
    @test Zygote.jacobian(x -> Zygote.gradient(f, x)[1], x)[1] ≈ ForwardDiff.hessian(f, x)
end
