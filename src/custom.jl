struct CustomGradFunction{F, G} <: Function
    f::F
    g::G
end
(f::CustomGradFunction)(x::AbstractVector) = f.f(x)
function ChainRulesCore.rrule(f::CustomGradFunction, x::AbstractVector)
    v = f.f(x)
    return v, Δ -> begin
        if f.g === nothing
            if v isa Real
                G = spzeros(eltype(v), length(x))
            else
                G = spzeros(eltype(v), length(v), length(x))
            end
        else
            G = f.g(x)
        end
        if G isa AbstractVector
            return (NoTangent(), G * Δ)
        elseif G isa LazyJacobian
            return (NoTangent(), G' * Δ)
        else
            spΔ = dropzeros!(sparse(copy(Δ)))
            if length(spΔ.nzval) == 1
                return (NoTangent(), G[spΔ.nzind[1], :] * spΔ.nzval[1])
            else
                return (NoTangent(), G' * Δ)
            end
        end
    end
end
function ChainRulesCore.frule(
    (_, Δx), f::CustomGradFunction, x::AbstractVector,
)
    v = f.f(x)
    if f.g === nothing
        if v isa Real
            ∇ = spzeros(eltype(v), 1, length(x))
        else
            ∇ = spzeros(eltype(v), length(v), length(x))
        end
    else
        ∇ = f.g(x)
    end
    project_to = ProjectTo(v)
    if ∇ isa AbstractVector && Δx isa AbstractVector
        if !(∇ isa LazyJacobian) && issparse(∇) && nnz(∇) == 0
            return v, project_to(zero(eltype(Δx)))
        else
            return v, project_to(∇' * Δx)
        end
    else
        if !(∇ isa LazyJacobian) && issparse(∇) && nnz(∇) == 0
            return v, project_to(spzeros(eltype(Δx), size(∇, 1)))
        else
            return v, project_to(_sparse_reshape(∇ * Δx, size(v)...))
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
    G = g(x)
    return f(x), Δ -> (NoTangent(), G * Δ)
end
function ChainRulesCore.frule(
    (_, Δx), f::CustomHessianFunction, x::AbstractVector,
)
    g = CustomGradFunction(f.g, f.h)
    v, ∇ = f(x), g(x)
    project_to = ProjectTo(v)
    if ∇ isa AbstractVector && Δx isa AbstractVector
        return v, project_to(∇' * Δx)
    else
        return v, project_to(∇ * Δx)
    end
end
@ForwardDiff_frule (f::CustomHessianFunction)(x::AbstractVector{<:ForwardDiff.Dual})
