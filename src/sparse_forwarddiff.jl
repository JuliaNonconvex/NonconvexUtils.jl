struct SparseForwardDiffFunction{F, F!, Y, J, JP, JC, JJ, JJ!, G, H, HP, HC} <: Function
    f::F
    f!::F!
    y::Y
    jac::J
    jac_pattern::JP
    jac_colors::JC
    vecJ::JJ
    vecJ!::JJ!
    G::G
    hess::H
    hess_pattern::HP
    hess_colors::HC
end

function SparseForwardDiffFunction(f, _x::AbstractVector; hessian = false, jac_pattern = nothing, hess_pattern = nothing)
    N = length(_x)
    val = f(_x)
    _f = val isa Real ? x -> [f(x)] : f
    f! = (y, x) -> begin
        v = f(x)
        y .= v
        return y
    end
    y = val isa Real ? [val] : copy(val)

    Symbolics.@variables x[1:N]
    if val isa Real
        vsyms = [f(x)]
    else
        vsyms = f(x)
    end
    jac_pattern = jac_pattern === nothing ? Symbolics.jacobian_sparsity(f!, y, _x) : jac_pattern
    if nnz(jac_pattern) > 0
        jac = float.(jac_pattern)
        jac_colors = SparseDiffTools.matrix_colors(jac)
    else
        T = eltype(G)
        jac = sparse(Int[], Int[], T[])
        jac_colors = Int[]
    end
    vecJ! = (G, x) -> begin
        _jac = SparseDiffTools.forwarddiff_color_jacobian(_f, x, colorvec = jac_colors)
        G .= vec(Array(_jac))
        return G
    end
    G = vec(Array(jac))
    vecJ = x -> copy(vecJ!(G, x))
    if hessian
        hess_pattern = hess_pattern === nothing ? Symbolics.jacobian_sparsity(vecJ!, G, _x) : hess_pattern
        if nnz(hess_pattern) > 0
            hess = float.(hess_pattern)
            hess_colors = SparseDiffTools.matrix_colors(hess)
        else
            T = eltype(G)
            hess = sparse(Int[], Int[], T[])
            hess_colors = Int[]
        end
    else
        hess = nothing
        hess_colors = nothing
    end
    return SparseForwardDiffFunction(f, f!, y, jac, jac_pattern, jac_colors, vecJ, vecJ!, G, hess, hess_pattern, hess_colors)
end
(f::SparseForwardDiffFunction)(x) = f.f(x)
function ChainRulesCore.rrule(f::SparseForwardDiffFunction, x::AbstractVector)
    if f.vecJ === nothing
        return f(x), _ -> (NoTangent(), NoTangent())
    else
        vecjac = SparseForwardDiffFunction(f.vecJ, f.vecJ!, f.G, f.hess, f.hess_pattern, f.hess_colors, nothing, nothing, nothing, nothing, nothing, nothing)
        val = f(x)
        if val isa Real
            jac = reshape(vecjac(x), 1, length(x))
        else
            jac = reshape(vecjac(x), length(val), length(x))
        end
        return val, Δ -> begin
            (NoTangent(), jac' * Δ)
        end
    end
end
function ChainRulesCore.frule((_, Δx), f::SparseForwardDiffFunction, x::AbstractVector)
    if f.vecJ === nothing
        val = f(x)
        return val, zero(val)
    else
        vecjac = SparseForwardDiffFunction(f.vecJ, f.vecJ!, f.G, f.hess, f.hess_pattern, f.hess_colors, nothing, nothing, nothing, nothing, nothing, nothing)
        val = f(x)
        if val isa Real
            jac = reshape(vecjac(x), 1, length(x))
        else
            jac = reshape(vecjac(x), length(val), length(x))
        end
        return val, jac * Δx
    end
end
@ForwardDiff_frule (f::SparseForwardDiffFunction)(x::AbstractVector{<:ForwardDiff.Dual})
