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

function SparseForwardDiffFunction(f, x::AbstractVector; hessian = false, jac_pattern = nothing, hess_pattern = nothing)
    val = f(x)
    _f = val isa Real ? x -> [f(x)] : f
    f! = (y, x) -> begin
        v = f(x)
        y .= v
        return y
    end
    y = val isa Real ? [val] : copy(val)
    jac_pattern = jac_pattern === nothing ? Symbolics.jacobian_sparsity(f!, y, x) : jac_pattern
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
        hess_pattern = hess_pattern === nothing ? Symbolics.jacobian_sparsity(vecJ!, G, x) : hess_pattern
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
            (NoTangent(), jac' * (Δ isa Real ? Δ : vec(Δ)))
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
        Δy = jac * (Δx isa Real ? Δx : vec(Δx))
        return val, (val isa Real) ? only(Δy) : reshape(Δy, size(val))
    end
end
@ForwardDiff_frule (f::SparseForwardDiffFunction)(x::AbstractVector{<:ForwardDiff.Dual})

function sparsify(model::NonconvexCore.AbstractModel; objective = true, ineq_constraints = true, eq_constraints = true, kwargs...)
    vmodel, v, _ = NonconvexCore.tovecmodel(model)
    if objective
        # Objective
        sparse_flat_obj = SparseForwardDiffFunction(vmodel.objective, v; kwargs...)
        obj = NonconvexCore.Objective(x -> sparse_flat_obj(flatten(x)[1]), flags = model.objective.flags)
    else
        obj = model.objective
    end
    if ineq_constraints
        ineq = length(vmodel.ineq_constraints.fs) != 0 ? NonconvexCore.VectorOfFunctions(map(vmodel.ineq_constraints.fs) do c
            sparse_flat_ineq = SparseForwardDiffFunction(c, v; kwargs...)
            NonconvexCore.IneqConstraint(x -> sparse_flat_ineq(flatten(x)[1]), c.rhs, c.dim, c.flags)
        end) : NonconvexCore.VectorOfFunctions(NonconvexCore.IneqConstraint[])
    else
        ineq = model.ineq_constraints
    end
    if eq_constraints
        eq = length(vmodel.eq_constraints.fs) != 0 ? NonconvexCore.VectorOfFunctions(map(vmodel.eq_constraints.fs) do c
            sparse_flat_eq = SparseForwardDiffFunction(c, v; kwargs...)
            NonconvexCore.EqConstraint(x -> sparse_flat_eq(flatten(x)[1]), c.rhs, c.dim, c.flags)
        end) : NonconvexCore.VectorOfFunctions(NonconvexCore.EqConstraint[])
    else
        eq = model.eq_constraints
    end
    if model isa NonconvexCore.Model
        ModelT = NonconvexCore.Model
    elseif model isa NonconvexCore.DictModel
        ModelT = NonconvexCore.DictModel
    else
        throw("Unsupported model type.")
    end
    return ModelT(obj, eq, ineq, model.sd_constraints, model.box_min, model.box_max, model.init, model.integer)
end
