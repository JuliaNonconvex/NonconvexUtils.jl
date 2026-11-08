# A function whose derivatives come from MathOptInterface's Nonlinear
# reverse-AD evaluator (the stack JuMP uses for its new nonlinear interface),
# wrapped for use inside NonconvexCore models. Reverse-mode backends
# differentiate through the ChainRules rrule, and solvers can call the
# gradient and Hessian functions directly.

"""
    JuMPSymbolicFunction(expr::MOI.ScalarNonlinearFunction, vars::Vector{MOI.VariableIndex})

Returns a callable that evaluates `expr` at a flat vector `x` (ordered like
`vars`) with value, gradient and Hessian computed by MathOptInterface's
`Nonlinear` sparse reverse-mode evaluator. Reverse-mode differentiation of
the callable uses the ChainRules `rrule` defined below; the gradient and
Hessian are also available as [`jump_symbolic_gradient`](@ref) and
[`jump_symbolic_hessian`](@ref).
"""
struct JuMPSymbolicFunction{E} <: Function
    evaluator::E
end

function JuMPSymbolicFunction(expr::MOI.ScalarNonlinearFunction, vars::Vector{MOI.VariableIndex})
    model = MOI.Nonlinear.Model()
    MOI.Nonlinear.set_objective(model, expr)
    evaluator = MOI.Nonlinear.Evaluator(model, MOI.Nonlinear.SparseReverseMode(), vars)
    MOI.initialize(evaluator, [:Grad, :Hess])
    return JuMPSymbolicFunction(evaluator)
end

function (f::JuMPSymbolicFunction)(x)
    return MOI.eval_objective(f.evaluator, x)
end

"""
    jump_symbolic_gradient(f::JuMPSymbolicFunction, x)

Returns the gradient of the wrapped function at `x`, computed by the
MathOptInterface Nonlinear evaluator.
"""
function jump_symbolic_gradient(f::JuMPSymbolicFunction, x)
    g = zeros(length(x))
    MOI.eval_objective_gradient(f.evaluator, g, x)
    return g
end

"""
    jump_symbolic_hessian(f::JuMPSymbolicFunction, x)

Returns the Hessian of the wrapped function at `x`, computed by the
MathOptInterface Nonlinear evaluator.
"""
function jump_symbolic_hessian(f::JuMPSymbolicFunction, x)
    H = zeros(length(x), length(x))
    for (i, j) in MOI.hessian_objective_structure(f.evaluator)
        H[i, j] = 1.0  # placeholder; values filled below via eval
    end
    vals = zeros(length(MOI.hessian_objective_structure(f.evaluator)))
    MOI.eval_hessian_objective(f.evaluator, vals, x)
    k = 0
    for (i, j) in MOI.hessian_objective_structure(f.evaluator)
        k += 1
        H[i, j] = vals[k]
        H[j, i] = vals[k]
    end
    return H
end

function ChainRulesCore.rrule(f::JuMPSymbolicFunction, x)
    y = f(x)
    return y, Δ -> (NoTangent(), jump_symbolic_gradient(f, x) * _unthunk(Δ))
end
