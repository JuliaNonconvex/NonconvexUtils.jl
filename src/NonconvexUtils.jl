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
using Zygote, LinearMaps, IterativeSolvers, NonconvexCore
using NonconvexCore: flatten
using MacroTools

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
struct ImplicitFunction{matrixfree, F, C, L, T} <: Function
    # A function which takes x as input and returns a tuple (ystar, df/dy) such that f(x, ystar) = 0. df/dy is optional and can be replaced by nothing to compute it via automatic differentiation. Jacobian should only be returned if it's more cheaply available than using AD, e.g. when using BFGS approximation of the Hessian in IPOPT.
    forward::F
    # The conditions function f(x, y) which must be 0 at ystar. Note that variables which don't show up in x and are closed over instead will be assumed to have no effect on the optimal solution. So it's the user's responsibility to ensure x includes all the interesting variables to be differentiated with respect to.
    conditions::C
    # A linear system solver to solve df/dy' \ v
    linear_solver::L
    # The acceptable tolerance for f(x, ystar) to use the implicit function theorem at x
    tol::T
    # A booolean to decide whether or not to error if the tolerance is violated, i.e. norm(f(x, ystar)) > tol. If false, we return a gradient of NaNs.
    error_on_tol_violation::Bool
end
function ImplicitFunction(
    forward::F, conditions::C; tol::T = 1e-5, error_on_tol_violation = false, matrixfree = false, linear_solver::L = _default_solver(matrixfree),
) where {F, C, L, T}
    return ImplicitFunction{matrixfree, F, C, L, T}(
        forward, conditions, linear_solver, tol, error_on_tol_violation,
    )
end

function _default_solver(matrixfree)
    if matrixfree
        return (A, b) -> begin
            L = LinearMap(A, length(b))
            return gmres(L, b)
        end
    else
        return (A, b) -> A \ b
    end
end

(f::ImplicitFunction)(x) = f.forward(x)[1]
(f::ImplicitFunction)() = f.forward()[1]

function ChainRulesCore.rrule(
    rc::RuleConfig, f::ImplicitFunction{matrixfree}, x,
) where {matrixfree}
    ystar, _dfdy = f.forward(x)
    flat_ystar, unflatten_y = flatten(ystar)
    forward_returns_jacobian = _dfdy !== nothing
    if forward_returns_jacobian
        dfdy = _dfdy
        if matrixfree
            # y assumed flat if dfdy is passed in
            pby = v -> dfdy' * v
        else
            pby = nothing
        end
    else
        _conditions_y = flat_y -> begin
            return flatten(f.conditions(x, unflatten_y(flat_y)))[1]
        end
        if matrixfree
            dfdy = nothing
            _, _pby = rrule_via_ad(rc, _conditions_y, flat_ystar)
            pby = v -> _pby(v)[2]
        else
            # Change this to AbstractDifferentiation
            dfdy = Zygote.jacobian(_conditions_y, flat_ystar)[1]
            pby = nothing
        end
    end
    _conditions_x = (conditions, x) -> begin
        return flatten(conditions(x, ystar))[1]
    end
    residual, pbx = rrule_via_ad(rc, _conditions_x, f.conditions, x)
    return ystar, ∇ -> begin
        if norm(residual) > f.tol && f.error_on_tol_violation
            throw(ArgumentError("The acceptable tolerance for the implicit function theorem is not satisfied for the current problem. Please double check your function definition, increase the tolerance, or set `error_on_tol_violation` to false to ignore the violation and return `NaN`s for the gradient."))
        end
        if matrixfree
            ∇f, ∇x = Base.tail(pbx(f.linear_solver(pby, -flatten(∇)[1])))
        else
            ∇f, ∇x = Base.tail(pbx(f.linear_solver(dfdy', -flatten(∇)[1])))
        end
        ∇imf = Tangent{typeof(f)}(
            conditions = Tangent{typeof(f.conditions)}(;
                ChainRulesCore.backing(∇f)...,
            ),
        )
        if norm(residual) <= f.tol
            return (∇imf, ∇x)
        else
            return (nanlike(∇imf), nanlike(∇x))
        end
    end
end

function ChainRulesCore.rrule(
    rc::RuleConfig, f::ImplicitFunction{matrixfree},
) where {matrixfree}
    ystar, _dfdy = f.forward()
    flat_ystar, unflatten_y = flatten(ystar)
    forward_returns_jacobian = _dfdy !== nothing
    if forward_returns_jacobian
        dfdy = _dfdy
        if matrixfree
            # y assumed flat if dfdy is passed in
            pby = v -> dfdy' * v
        else
            pby = nothing
        end
    else
        _conditions_y = flat_y -> begin
            return flatten(f.conditions(unflatten_y(flat_y)))[1]
        end
        if matrixfree
            dfdy = nothing
            _, _pby = rrule_via_ad(rc, _conditions_y, flat_ystar)
            pby = v -> _pby(v)[2]
        else
            # Change this to AbstractDifferentiation
            dfdy = Zygote.jacobian(_conditions_y, flat_ystar)[1]
            pby = nothing
        end
    end
    _conditions = (conditions) -> begin
        return flatten(conditions(ystar))[1]
    end
    residual, pbf = rrule_via_ad(rc, _conditions, f.conditions)
    return ystar, ∇ -> begin
        if norm(residual) > f.tol && f.error_on_tol_violation
            throw(ArgumentError("The acceptable tolerance for the implicit function theorem is not satisfied for the current problem. Please double check your function definition, increase the tolerance, or set `error_on_tol_violation` to false to ignore the violation and return `NaN`s for the gradient."))
        end
        if matrixfree
            ∇f = pbf(f.linear_solver(pby, -flatten(∇)[1]))[2]
        else
            ∇f = pbf(f.linear_solver(dfdy', -flatten(∇)[1]))[2]
        end
        ∇imf = Tangent{typeof(f)}(
            conditions = Tangent{typeof(f.conditions)}(;
                ChainRulesCore.backing(∇f)...,
            ),
        )
        if norm(residual) <= f.tol
            return (∇imf,)
        else
            return (nanlike(∇imf),)
        end
    end
end

function nanlike(x)
    flat, un = flatten(x)
    return un(similar(flat) .= NaN)
end

macro ForwardDiff_frule(sig)
    _fd_frule(sig)
end
function _fd_frule(sig)
    MacroTools.@capture(sig, f_(x__))
    return quote
        function $(esc(f))($(esc.(x)...))
            f = $(esc(f))
            x = ($(esc.(x)...),)
            flatx, unflattenx = flatten(x)
            CS = length(ForwardDiff.partials(first(flatx)))
            flat_xprimals = ForwardDiff.value.(flatx)
            flat_xpartials = reduce(vcat, transpose.(ForwardDiff.partials.(flatx)))

            xprimals = unflattenx(flat_xprimals)
            xpartials1 = unflattenx(flat_xpartials[:,1])

            yprimals, ypartials1 = ChainRulesCore.frule(
                (NoTangent(), xpartials1...), f, xprimals...,
            )
            flat_yprimals, unflatteny = flatten(yprimals)
            flat_ypartials1, _ = flatten(ypartials1)
            flat_ypartials = hcat(reshape(flat_ypartials1, :, 1), ntuple(Val(CS - 1)) do i
                xpartialsi = unflattenx(flat_xpartials[:, i+1])
                _, ypartialsi = ChainRulesCore.frule((NoTangent(), xpartialsi...), f, xprimals...)
                return flatten(ypartialsi)[1]
            end...)

            T = ForwardDiff.tagtype(eltype(flatx))
            flaty = ForwardDiff.Dual{T}.(
                flat_yprimals, ForwardDiff.Partials.(NTuple{CS}.(eachrow(flat_ypartials))),
            )
            return unflatteny(flaty)
        end
    end
end

end
