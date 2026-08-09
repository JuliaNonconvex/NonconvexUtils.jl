# Backend mapping for the deprecated `abstractdiffy` API: map the old
# AbstractDifferentiation backend objects to their ADTypes equivalents so the
# existing call sites keep working while routing through DifferentiationInterface.
# AD's backend constructors return wrapped types (e.g. `ZygoteBackend()` is a
# `ReverseRuleConfigBackend`), so dispatch on the abstract supertype and map by
# runtime type.
_ad_to_di(::AbstractDifferentiation.ForwardDiffBackend) = ADTypes.AutoForwardDiff()
_ad_to_di(::AbstractDifferentiation.ReverseDiffBackend) = ADTypes.AutoReverseDiff()
_ad_to_di(::AbstractDifferentiation.TrackerBackend) = ADTypes.AutoTracker()
function _ad_to_di(b::AbstractDifferentiation.AbstractBackend)
    # Fallback: any AD backend whose runtime type wraps a Zygote rule config
    # is treated as Zygote. This covers `AD.ZygoteBackend()` (a
    # `ReverseRuleConfigBackend{ZygoteRuleConfig}`) without depending on the
    # exact internal wrapper type.
    if b isa AbstractDifferentiation.ReverseRuleConfigBackend
        return ADTypes.AutoZygote()
    else
        throw(
            ArgumentError(
                "Unsupported AbstractDifferentiation backend `$b`. " *
                "Use an ADTypes backend (e.g. `AutoZygote()`) with `diffiy` directly.",
            ),
        )
    end
end
# Passthrough: anything that is already an ADTypes backend is used directly.
_ad_to_di(b::ADTypes.AbstractADType) = b

struct AbstractDiffFunction{F,B} <: Function
    f::F
    backend::B
end
ForwardDiffFunction(f) = AbstractDiffFunction(f, AD.ForwardDiffBackend())
(f::AbstractDiffFunction)(x) = f.f(x)
function ChainRulesCore.rrule(f::AbstractDiffFunction, x::AbstractVector)
    v, (∇,) = DifferentiationInterface.value_and_jacobian(f.f, _ad_to_di(f.backend), x)
    return v, Δ -> (NoTangent(), ∇' * Δ)
end
function ChainRulesCore.frule((_, Δx), f::AbstractDiffFunction, x::AbstractVector)
    v, (∇,) = DifferentiationInterface.value_and_jacobian(f.f, _ad_to_di(f.backend), x)
    return v, ∇ * Δx
end
@ForwardDiff_frule (f::AbstractDiffFunction)(x::AbstractVector{<:ForwardDiff.Dual})

# does not assume vector input and output
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
via `NonconvexCore.tovecfunc` and unflattened back.
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

# Deprecated: prefer `diffiy` with an ADTypes backend. Maps the old
# AbstractDifferentiation backends to their ADTypes equivalents and delegates.
function abstractdiffy(f, backend, x...)
    Base.depwarn(
        "`abstractdiffy` is deprecated; use `diffiy` with an ADTypes backend " *
        "(e.g. `diffiy(f, AutoZygote(), x...)`) from DifferentiationInterface instead.",
        :abstractdiffy,
    )
    return diffiy(f, _ad_to_di(backend), x...)
end
function abstractdiffy(
    model::NonconvexCore.AbstractModel,
    backend;
    objective = true,
    ineq_constraints = true,
    eq_constraints = true,
    sd_constraints = true,
)
    Base.depwarn(
        "`abstractdiffy` is deprecated; use `diffiy` with an ADTypes backend " *
        "(e.g. `diffiy(model, AutoZygote())`) from DifferentiationInterface instead.",
        :abstractdiffy,
    )
    return diffiy(
        model,
        _ad_to_di(backend);
        objective = objective,
        ineq_constraints = ineq_constraints,
        eq_constraints = eq_constraints,
        sd_constraints = sd_constraints,
    )
end

# Deprecated: prefer `diffiy(f, AutoForwardDiff(), x...)`.
function forwarddiffy(f_or_m, x...)
    Base.depwarn(
        "`forwarddiffy` is deprecated; use `diffiy(f, AutoForwardDiff(), x...)` " *
        "from DifferentiationInterface instead.",
        :forwarddiffy,
    )
    return diffiy(f_or_m, ADTypes.AutoForwardDiff(), x...)
end
