struct CustomGradFunction{F, G} <: Function
    f::F
    g::G
end
(f::CustomGradFunction)(x::AbstractVector) = f.f(x)
function ChainRulesCore.rrule(f::CustomGradFunction, x::AbstractVector)
    return f.f(x), Δ -> begin
        G = f.g(x)
        if G isa AbstractVector
            return (NoTangent(), G * Δ)
        else
            return (NoTangent(), G' * Δ)
        end
    end
end
function ChainRulesCore.frule(
    (_, Δx), f::CustomGradFunction, x::AbstractVector,
)
    v = f.f(x)
    if f.g === nothing
        if v isa Real
            ∇ = zeros(eltype(v), length(x))'
        else
            ∇ = zeros(eltype(v), length(v), length(x))
        end
    else
        ∇ = f.g(x)
    end
    if ∇ isa AbstractVector && Δx isa AbstractVector
        if !(∇ isa LazyJacobian) && issparse(∇) && nnz(∇) == 0
            return v, zero(eltype(Δx))
        else
            return v, ∇' * Δx
        end
    else
        if !(∇ isa LazyJacobian) && issparse(∇) && nnz(∇) == 0
            return v, zeros(eltype(Δx), size(∇, 1))
        else
            return v, ∇ * Δx
        end
    end
end
@ForwardDiff_frule (f::CustomGradFunction)(x::AbstractVector{<:ForwardDiff.Dual})

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
    return f(x), Δ -> (NoTangent(), g(x) * Δ)
end
function ChainRulesCore.frule(
    (_, Δx), f::CustomHessianFunction, x::AbstractVector,
)
    g = CustomGradFunction(f.g, f.h)
    v, ∇ = f(x), g(x)
    if ∇ isa AbstractVector && Δx isa AbstractVector
        return v, ∇' * Δx
    else
        return v, ∇ * Δx
    end
end
@ForwardDiff_frule (f::CustomHessianFunction)(x::AbstractVector{<:ForwardDiff.Dual})
