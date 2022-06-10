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
