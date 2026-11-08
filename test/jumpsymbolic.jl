using NonconvexUtils, Test, Zygote, ForwardDiff, LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface

# x1^2 + sin(x2)
expr = MOI.ScalarNonlinearFunction(
    :+,
    Any[
        MOI.ScalarNonlinearFunction(:^, Any[MOI.VariableIndex(1), 2]),
        MOI.ScalarNonlinearFunction(:sin, Any[MOI.VariableIndex(2)]),
    ],
)
vars = MOI.VariableIndex.(1:2)
f = JuMPSymbolicFunction(expr, vars)
x = [0.7, 1.1]

@testset "JuMPSymbolicFunction" begin
    # value
    @test f(x) ≈ x[1]^2 + sin(x[2])

    # gradient via the evaluator
    g = jump_symbolic_gradient(f, x)
    @test g ≈ [2x[1], cos(x[2])]

    # hessian via the evaluator
    H = jump_symbolic_hessian(f, x)
    @test H ≈ [2.0 0.0; 0.0 -sin(x[2])]

    # reverse-mode differentiation uses the rrule
    gz = Zygote.gradient(v -> f(v), x)[1]
    @test gz ≈ [2x[1], cos(x[2])]
end
