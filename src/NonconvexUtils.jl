module NonconvexUtils

export  ForwardDiffFunction,
        AbstractDiffFunction,
        AD,
        TraceFunction,
        CustomGradFunction,
        LazyJacobian,
        CustomHessianFunction,
        ImplicitFunction

using ChainRulesCore, AbstractDifferentiation, ForwardDiff, LinearAlgebra

struct AbstractDiffFunction{F, B} <: Function
    f::F
    backend::B
end
ForwardDiffFunction(f) = AbstractDiffFunction(f, AD.ForwardDiffBackend())
(f::AbstractDiffFunction)(x) = f.f(x)
function ChainRulesCore.rrule(
    f::AbstractDiffFunction, x::AbstractVector,
)
    v, (∇,) = AbstractDifferentiation.value_and_gradient(f.backend, f.f, x)
    return v, Δ -> (NoTangent(), Δ * ∇)
end

struct TraceFunction{F, V} <: Function
    f::F
    trace::V
    on_call::Bool
    on_grad::Bool
end
function TraceFunction(f; on_call::Union{Bool, Nothing} = nothing, on_grad::Union{Bool, Nothing} = nothing)
    if on_call === on_grad === nothing
        _on_call = true
        _on_grad = true
    elseif on_call === nothing
        _on_call = !on_grad
        _on_grad = on_grad
    elseif on_grad === nothing
        _on_call = on_call
        _on_grad = !on_call
    else
        _on_call = on_call
        _on_grad = on_grad
    end
    return TraceFunction(f, Any[], _on_call, _on_grad)
end
function (tf::TraceFunction)(x)
    v = tf.f(x)
    if tf.on_call
        push!(tf.trace, (input = copy(x), output = copy(v)))
    end
    return v
end
function ChainRulesCore.rrule(rc::RuleConfig, tf::TraceFunction, x)
    v, pb = ChainRulesCore.rrule_via_ad(rc, tf.f, x)
    return v, Δ -> begin
        Δin = pb(Δ)
        g = Δin[2].val.f()
        if tf.on_grad
            push!(tf.trace, (input = copy(x), output = copy(v), grad = copy(g)))
        end
        return (Δin[1], g)
    end
end

struct LazyJacobian{symmetric, J1, J2}
    jvp::J1
    jtvp::J2
end
function LazyJacobian(; jvp=nothing, jtvp=nothing, symmetric=false)
    return LazyJacobian{symmetric}(jvp, jtvp)
end
function LazyJacobian{symmetric}(jvp = nothing, jtvp = nothing) where {symmetric}
    if jvp === jtvp === nothing
        throw(ArgumentError("Both the jvp and jtvp operators cannot be nothing."))
    end
    if symmetric 
        if jvp !== nothing
            _jtvp = _jvp = jvp
        else
            _jvp = _jtvp = jtvp
        end
    else
        _jvp = jvp
        _jtvp = jtvp
    end
    return LazyJacobian{symmetric, typeof(_jvp), typeof(_jtvp)}(_jvp, _jtvp)
end

struct LazyJacobianTransposed{J}
    j::J
end

LinearAlgebra.adjoint(j::LazyJacobian{false}) = LazyJacobianTransposed(j)
LinearAlgebra.transpose(j::LazyJacobian{false}) = LazyJacobianTransposed(j)
LinearAlgebra.adjoint(j::LazyJacobian{true}) = j
LinearAlgebra.transpose(j::LazyJacobian{true}) = j
LinearAlgebra.adjoint(j::LazyJacobianTransposed) = j.j
LinearAlgebra.transpose(j::LazyJacobianTransposed) = j.j

LinearAlgebra.:*(j::LazyJacobian, v::AbstractVecOrMat) = j.jvp(v)
LinearAlgebra.:*(v::AbstractVecOrMat, j::LazyJacobian) = j.jtvp(v')'
LinearAlgebra.:*(j::LazyJacobianTransposed, v::AbstractVecOrMat) = (v' * j')'
LinearAlgebra.:*(v::AbstractVecOrMat, j::LazyJacobianTransposed) = (j' * v')'

struct CustomGradFunction{F, G} <: Function
    f::F
    g::G
end
(f::CustomGradFunction)(x) = f.f(x)
function ChainRulesCore.rrule(f::CustomGradFunction, x)
    return f.f(x), Δ -> begin
        G = f.g(x)
        if G isa AbstractVector
            return (NoTangent(), G * Δ)
        else
            return (NoTangent(), G' * Δ)
        end
    end
end

struct CustomHessianFunction{F, G, H} <: Function
    f::F
    g::G
    h::H
    function CustomHessianFunction(
        f::F, g::G, h::H; hvp = false,
    ) where {F, G, H}
        _h = hvp ? x -> LazyJacobian{true}(v -> h(x, v)) : h
        return new{F, G, typeof(_h)}(f, g, _h)
    end
end
(to::CustomHessianFunction)(x) = to.f(x)
function ChainRulesCore.rrule(f::CustomHessianFunction, x)
    g = CustomGradFunction(f.g, f.h)
    return f.f(x), Δ -> (NoTangent(), g(x) * Δ)
end

# Parameters x, variables y, residuals f
struct ImplicitFunction{F, C, L, T} <: Function
    # A function taking x as input and returning ystar as output such that f(x, ystar) = 0
    forward::F
    # The conditions function f(x, y) which must be 0
    conditions::C
    # A linear system solver
    linear_solver::L
    # The acceptable tolerance for f(x, y) to use the implicit function theorem
    tol::T
    # A booolean to error if the tolerance is violated, i.e. norm(f(x, ystar)) > tol
    error_on_tol_violation::Bool
end
function ImplicitFunction(
    forward, conditions; tol = 1e-5, error_on_tol_violation = false, linear_solver = (A, b) -> A \ b,
)
    return ImplicitFunction(
        forward, conditions, linear_solver, tol, error_on_tol_violation,
    )
end

(f::ImplicitFunction)(x) = f.forward(x)
function ChainRulesCore.rrule(f::ImplicitFunction, x)
    ystar = f(x)
    residual = f.conditions(x, ystar)
    dfdy = Zygote.jacobian(y -> f.conditions(x, y), ystar)[1]
    residual, pbx = Zygote.pullback(x -> f.conditions(x, ystar), x)
    return ystar, ∇ -> begin
        if norm(residual) <= tol
            return (NoTangent(), -pbx(f.linearsolver(dfdy', ∇))[1])
        elseif f.error_on_tol_violation
            throw(ArgumentError("The acceptable tolerance for the implicit function theorem is not satisfied for the current problem. Please double check your function definition, increase the tolerance, or set `error_on_tol_violation` to false to ignore the violation and return `NaN`s for the gradient."))
        end
        return (NoTangent(), (similar(x) .= NaN))
    end
end

end
