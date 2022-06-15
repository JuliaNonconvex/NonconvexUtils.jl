@testset "sparsify" begin
    @testset "Functions" begin
        f = sparsify(sum, rand(3); hessian = false)
        x = rand(3)
        @test Zygote.gradient(f, x)[1] ≈ ForwardDiff.gradient(f, rand(3))

        f = sparsify(x -> 2(x.^2) + x[1] * ones(3), rand(3); hessian = false)
        x = rand(3)
        @test Zygote.jacobian(f, x)[1] ≈ ForwardDiff.jacobian(f, x)
        @test NonconvexCore.sparse_jacobian(f, x) ≈ Zygote.jacobian(f, x)[1]
        @test NonconvexCore.sparse_fd_jacobian(f, x) ≈ ForwardDiff.jacobian(f, x)
        @test NonconvexCore.sparse_jacobian(f, x) isa SparseMatrixCSC
        @test NonconvexCore.sparse_fd_jacobian(f, x) isa SparseMatrixCSC

        f = sparsify(sum, rand(3); hessian = true)
        x = rand(3)
        @test Zygote.gradient(f, x)[1] ≈ ForwardDiff.gradient(f, rand(3))
        @test Zygote.hessian(f, x) ≈ ForwardDiff.hessian(f, rand(3))
        @test NonconvexCore.sparse_hessian(f, x) ≈ Zygote.hessian(f, x)
        @test NonconvexCore.sparse_hessian(f, x) isa SparseMatrixCSC

        f = sparsify(x -> sum(x)^2 + x[1], rand(3); hessian = true)
        x = rand(3)
        @test Zygote.hessian(f, x) ≈ ForwardDiff.hessian(f, x)
        @test NonconvexCore.sparse_hessian(f, x) ≈ Zygote.hessian(f, x)
        @test NonconvexCore.sparse_hessian(f, x) isa SparseMatrixCSC

        f = sparsify(x -> sum(x)^2 + x[1], rand(3); hessian = true)
        x = rand(3)
        @test Zygote.jacobian(x -> Zygote.gradient(f, x)[1], x)[1] ≈ ForwardDiff.hessian(f, x)

        f = sparsify(x -> [sum(x)^2, x[1]], rand(3); hessian = true)
        x = rand(3)
        g = x -> sum(f(x))
        @test Zygote.gradient(g, x)[1] ≈ ForwardDiff.gradient(g, x)
        @test Zygote.hessian(g, x) ≈ ForwardDiff.hessian(g, x)

        f = sparsify(x -> [0.0, 0.0], rand(3); hessian = true)
        x = rand(3)
        g = x -> sum(f(x))
        @test Zygote.gradient(g, x)[1] ≈ ForwardDiff.gradient(g, x)
        @test Zygote.hessian(g, x) ≈ ForwardDiff.hessian(g, x)

        f = sparsify(x -> 0.0, rand(3); hessian = true)
        x = rand(3)
        @test Zygote.gradient(f, x)[1] ≈ ForwardDiff.gradient(f, x)
        @test Zygote.hessian(f, x) ≈ ForwardDiff.hessian(f, x)
    end
    @testset "Model - first order = $first_order" for first_order in (true, false)
        f = (x::AbstractVector) -> sqrt(x[2])
        g = (x::AbstractVector, a, b) -> (a*x[1] + b)^3 - x[2]
        options = IpoptOptions(first_order = first_order)
        m = Model(f)
        addvar!(m, [0.0, 0.0], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        alg = IpoptAlg()
        sp_model = sparsify(m)
        r = NonconvexIpopt.optimize(sp_model, alg, [1.234, 2.345], options = options)
        @test abs(r.minimum - sqrt(8/27)) < 1e-6
        @test norm(r.minimizer - [1/3, 8/27]) < 1e-6    
    end
end
