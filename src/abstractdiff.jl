struct AbstractDiffFunction{F, B} <: Function
    f::F
    backend::B
end
ForwardDiffFunction(f) = AbstractDiffFunction(f, AD.ForwardDiffBackend())
(f::AbstractDiffFunction)(x) = f.f(x)
function ChainRulesCore.rrule(
    f::AbstractDiffFunction, x::AbstractVector,
)
    v, (∇,) = AbstractDifferentiation.value_and_jacobian(f.backend, f.f, x)
    return v, Δ -> (NoTangent(), ∇' * Δ)
end
function ChainRulesCore.frule(
    (_, Δx), f::AbstractDiffFunction, x::AbstractVector,
)
    v, (∇,) = AbstractDifferentiation.value_and_jacobian(f.backend, f.f, x)
    return v, ∇ * Δx
end
@ForwardDiff_frule (f::AbstractDiffFunction)(x::AbstractVector{<:ForwardDiff.Dual})

# does not assume vector input and output
forwarddiffy(f_or_m, x...) = abstractdiffy(f_or_m, AD.ForwardDiffBackend(), x...)
function abstractdiffy(f, backend, x...)
    flat_f, _, unflatteny = tovecfunc(f, x...)
    ad_flat_f = AbstractDiffFunction(flat_f, backend)
    return (x...,) -> unflatteny(ad_flat_f(flatten(x)[1]))
end
function abstractdiffy(model::NonconvexCore.AbstractModel, backend; objective = true, ineq_constraints = true, eq_constraints = true, sd_constraints = true)
    x = getmin(model)
    if objective
        obj = NonconvexCore.Objective(abstractdiffy(model.objective, backend, x), flags = model.objective.flags)
    else
        obj = model.objective
    end
    if ineq_constraints
        ineq = length(model.ineq_constraints.fs) != 0 ? NonconvexCore.VectorOfFunctions(map(model.ineq_constraints.fs) do c
            return NonconvexCore.IneqConstraint(abstractdiffy(c, backend, x), c.rhs, c.dim, c.flags)
        end) : NonconvexCore.VectorOfFunctions(NonconvexCore.IneqConstraint[])
    else
        ineq = model.ineq_constraints
    end
    if eq_constraints
        eq = length(model.eq_constraints.fs) != 0 ? NonconvexCore.VectorOfFunctions(map(model.eq_constraints.fs) do c
            return NonconvexCore.EqConstraint(abstractdiffy(c, backend, x), c.rhs, c.dim, c.flags)
        end) : NonconvexCore.VectorOfFunctions(NonconvexCore.EqConstraint[])
    else
        eq = model.eq_constraints
    end
    if sd_constraints
        sd = length(model.sd_constraints.fs) != 0 ? NonconvexCore.VectorOfFunctions(map(model.sd_constraints.fs) do c
            return NonconvexCore.SDConstraint(abstractdiffy(c, backend, x), c.dim)
        end) : NonconvexCore.VectorOfFunctions(NonconvexCore.SDConstraint[])
    else
        sd = model.sd_constraints
    end
    if model isa NonconvexCore.Model
        ModelT = NonconvexCore.Model
    elseif model isa NonconvexCore.DictModel
        ModelT = NonconvexCore.DictModel
    else
        throw("Unsupported model type.")
    end
    return ModelT(obj, eq, ineq, sd, model.box_min, model.box_max, model.init, model.integer)
end
