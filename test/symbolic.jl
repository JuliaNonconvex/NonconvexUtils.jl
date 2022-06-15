function _test_function_scoping()
    model = Model()
    addvar!(model, [0.0], [1.0])
    set_objective!(model, x -> x[1])
    add_eq_constraint!(model, x -> x[1])
    return NonconvexIpopt.optimize(
        symbolify(model),
        IpoptAlg(),
        [0.0];
        options = IpoptOptions(),
    )
end

@testset "symbolify" begin
    @testset "Functions - simplify = $simplify, sparse = $sparse" for simplify in (false, true), sparse in (false, true)
        f = symbolify(sum, rand(3); hessian = false, simplify, sparse)
        x = rand(3)
        @test Zygote.gradient(f, x)[1] ≈ ForwardDiff.gradient(f, rand(3))

        f = symbolify(x -> 2(x.^2) + x[1] * ones(3), rand(3); hessian = false, simplify, sparse)
        x = rand(3)
        @test Zygote.jacobian(f, x)[1] ≈ ForwardDiff.jacobian(f, x)
        if sparse
            @test NonconvexCore.sparse_jacobian(f, x) ≈ Zygote.jacobian(f, x)[1]
            @test NonconvexCore.sparse_fd_jacobian(f, x) ≈ ForwardDiff.jacobian(f, x)
            @test NonconvexCore.sparse_jacobian(f, x) isa SparseMatrixCSC
            @test NonconvexCore.sparse_fd_jacobian(f, x) isa SparseMatrixCSC
        end

        f = symbolify(sum, rand(3); hessian = true, simplify, sparse)
        x = rand(3)
        @test Zygote.gradient(f, x)[1] ≈ ForwardDiff.gradient(f, rand(3))
        @test Zygote.hessian(f, x) ≈ ForwardDiff.hessian(f, rand(3))
        if sparse
            @test NonconvexCore.sparse_hessian(f, x) ≈ Zygote.hessian(f, x)
            @test NonconvexCore.sparse_hessian(f, x) isa SparseMatrixCSC
        end

        f = symbolify(x -> norm(x) + x[1], rand(3); hessian = true, simplify, sparse)
        x = rand(3)
        @test Zygote.hessian(f, x) ≈ ForwardDiff.hessian(f, x)

        f = symbolify(x -> [norm(x), x[1]], rand(3); hessian = true, simplify, sparse)
        x = rand(3)
        g = x -> sum(f(x))
        @test Zygote.gradient(g, x)[1] ≈ ForwardDiff.gradient(g, x)
        @test Zygote.hessian(g, x) ≈ ForwardDiff.hessian(g, x)
    end
    @testset "Model - first order = $first_order - sparse $sparse" for first_order in (true, false), sparse in (true, false)
        f = (x::AbstractVector) -> sqrt(x[2])
        g = (x::AbstractVector, a, b) -> (a*x[1] + b)^3 - x[2]
        options = IpoptOptions(; first_order, sparse)
        m = Model(f)
        addvar!(m, [0.0, 0.0], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        alg = IpoptAlg()
        sym_model = symbolify(m)
        r = NonconvexIpopt.optimize(sym_model, alg, [1.234, 2.345], options = options)
        @test abs(r.minimum - sqrt(8/27)) < 1e-6
        @test norm(r.minimizer - [1/3, 8/27]) < 1e-6
    end
    @testset "function-scope" begin
        r = _test_function_scoping()
        @test abs(r.minimum) < 1e-6
    end
end
