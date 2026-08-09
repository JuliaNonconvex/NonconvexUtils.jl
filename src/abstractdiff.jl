"""
    diffiy(f, backend, x...)
    diffiy(f, x...)
    diffiy(model, backend; objective=true, ineq_constraints=true, eq_constraints=true, sd_constraints=true)

Wrap `f` (or every objective/constraint of `model`) so that it is differentiated
with the [`DifferentiationInterface`](https://github.com/JuliaDiff/DifferentiationInterface.jl)
backend `backend` (an ADTypes backend such as `AutoForwardDiff()`,
`AutoZygote()`, `AutoReverseDiff()`, `AutoTracker()`). The wrapper defines a
chain rule, so the result is differentiable from any outer AD that consumes
ChainRulesCore (Zygote, ForwardDiff, etc.).

When `backend` is omitted it defaults to `AutoForwardDiff()`.

`f` may take non-vector inputs and return non-vector outputs: they are flattened
via `NonconvexCore.tovecfunc` and unflattened back, so flattening always happens
before the function or input reaches DifferentiationInterface.
"""
diffiy(f_or_m, x...) = diffiy(f_or_m, ADTypes.AutoForwardDiff(), x...)
function diffiy(f, backend, x...)
    flat_f, _, unflatteny = tovecfunc(f, x...)
    di_flat_f = DifferentiationInterface.DifferentiateWith(flat_f, backend)
    return (x...,) -> unflatteny(di_flat_f(flatten(x)[1]))
end
function diffiy(
    model::NonconvexCore.AbstractModel,
    backend;
    objective = true,
    ineq_constraints = true,
    eq_constraints = true,
    sd_constraints = true,
)
    x = getmin(model)
    if objective
        obj = NonconvexCore.Objective(
            diffiy(model.objective, backend, x),
            flags = model.objective.flags,
        )
    else
        obj = model.objective
    end
    if ineq_constraints
        ineq =
            length(model.ineq_constraints.fs) != 0 ?
            NonconvexCore.VectorOfFunctions(
                map(model.ineq_constraints.fs) do c
                    return NonconvexCore.IneqConstraint(
                        diffiy(c, backend, x),
                        c.rhs,
                        c.dim,
                        c.flags,
                    )
                end,
            ) : NonconvexCore.VectorOfFunctions(NonconvexCore.IneqConstraint[])
    else
        ineq = model.ineq_constraints
    end
    if eq_constraints
        eq =
            length(model.eq_constraints.fs) != 0 ?
            NonconvexCore.VectorOfFunctions(
                map(model.eq_constraints.fs) do c
                    return NonconvexCore.EqConstraint(
                        diffiy(c, backend, x),
                        c.rhs,
                        c.dim,
                        c.flags,
                    )
                end,
            ) : NonconvexCore.VectorOfFunctions(NonconvexCore.EqConstraint[])
    else
        eq = model.eq_constraints
    end
    if sd_constraints
        sd =
            length(model.sd_constraints.fs) != 0 ?
            NonconvexCore.VectorOfFunctions(
                map(model.sd_constraints.fs) do c
                    return NonconvexCore.SDConstraint(diffiy(c, backend, x), c.dim)
                end,
            ) : NonconvexCore.VectorOfFunctions(NonconvexCore.SDConstraint[])
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
    return ModelT(
        obj,
        eq,
        ineq,
        sd,
        model.box_min,
        model.box_max,
        model.init,
        model.integer,
    )
end
