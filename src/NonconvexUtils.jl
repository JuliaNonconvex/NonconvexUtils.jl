module NonconvexUtils

export  ForwardDiffFunction,
        AbstractDiffFunction,
        AD,
        TraceFunction,
        CustomGradFunction,
        LazyJacobian,
        CustomHessianFunction

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

struct TraceFunction{F1, F2} <: Function
    f::F1
    ∇f::F2
    xtrace::Vector{Any}
    ftrace::Vector{Any}
    gtrace::Vector{Any}
    on_call::Bool
    on_grad::Bool
end
function TraceFunction(f, ∇f; on_call = false, on_grad = true)
    return TraceFunction(f, ∇f, Any[], Any[], Any[], on_call, on_grad)
end
function (to::TraceFunction)(x)
    v = to.f(x)
    if to.on_call
        push!(f.xtrace, copy(x))
        push!(f.ftrace, copy(v))
    end
    return v
end
function ChainRulesCore.rrule(f::TraceFunction, x)
    v, g = f.f(x), f.∇f(x)
    if to.on_grad
        push!(f.xtrace, copy(x))
        push!(f.ftrace, copy(v))
        push!(f.gtrace, copy(g))
    end
    return v, Δ -> (NoTangent(), Δ * g)
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
        return CustomHessianFunction{F, G, H}(f, g, _h)
    end
end
(to::CustomHessianFunction)(x) = to.f(x)
function ChainRulesCore.rrule(f::CustomHessianFunction, x)
    g = CustomGradFunction(f.g, f.h)
    return f.f(x), Δ -> (NoTangent(), g(x) * Δ)
end

end
