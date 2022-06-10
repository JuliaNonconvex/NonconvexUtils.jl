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
