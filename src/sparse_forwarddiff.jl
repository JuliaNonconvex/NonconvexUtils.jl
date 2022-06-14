struct SparseForwardDiffFunction{F, F!, Y, J, JP, JC, JJ, JJ!, G, HH1, HP, HC, HH2} <: Function
    f::F
    f!::F!
    y::Y
    jac::J
    jac_pattern::JP
    jac_colors::JC
    J::JJ
    vecJ!::JJ!
    vecJ::G
    hess::HH1
    hess_pattern::HP
    hess_colors::HC
    H::HH2
end

function SparseForwardDiffFunction(f, x::AbstractVector; hessian = false, jac_pattern = nothing, hess_pattern = nothing)
    val = f(x)
    _f = x -> _sparsevec(f(x))
    f! = (y, x) -> begin
        v = f(x)
        y .= v
        return y
    end
    y = _sparsevec(val)
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
        G .= _sparsevec(_jac)
        return G
    end
    G = vec(Array(jac))
    J = x -> begin
        xT = eltype(x)
        _jac = SparseDiffTools.forwarddiff_color_jacobian(_f, x, colorvec = jac_colors, sparsity = jac, jac_prototype = xT.(jac))
        return copy(_jac)
    end
    if hessian
        hess_pattern = hess_pattern === nothing ? Symbolics.jacobian_sparsity(vecJ!, G, x) : hess_pattern
        if nnz(hess_pattern) > 0
            hess = float.(hess_pattern)
            hess_colors = SparseDiffTools.matrix_colors(hess)
            _J = x -> _sparsevec(J(x))
            H = x -> begin
                _hess = SparseDiffTools.forwarddiff_color_jacobian(_J, x, colorvec = hess_colors, sparsity = hess_pattern, jac_prototype = hess)
                return copy(_hess)
            end
        else
            T = eltype(G)
            hess = sparse(Int[], Int[], T[], length(x), length(x))
            hess_colors = Int[]
            H = x -> hess
        end
    else
        hess = nothing
        hess_colors = nothing
        H = nothing
    end
    return SparseForwardDiffFunction(f, f!, y, jac, jac_pattern, jac_colors, J, vecJ!, G, hess, hess_pattern, hess_colors, H)
end

_sparsevec(x::Real) = [x]
_sparsevec(x::Vector) = copy(vec(x))
_sparsevec(x::Matrix) = copy(vec(x))
function _sparsevec(x::SparseMatrixCSC)
    m, n = size(x)
    linear_inds = zeros(Int, length(x.nzval))
    count = 1
    for colind in 1:length(x.colptr)-1
        for ind in x.colptr[colind]:x.colptr[colind+1]-1
            rowind = x.rowval[ind]
            val = x.nzval[ind]
            linear_inds[count] = rowind + (colind - 1) * m
            count += 1
        end
    end
    return sparsevec(linear_inds, copy(x.nzval))
end

(f::SparseForwardDiffFunction)(x) = f.f(x)
function ChainRulesCore.rrule(f::SparseForwardDiffFunction, x::AbstractVector)
    if f.H === nothing
        J = f.J
    else
        J = SparseForwardDiffFunction(f.J, f.vecJ!, f.vecJ, f.hess, f.hess_pattern, f.hess_colors, f.H, nothing, nothing, nothing, nothing, nothing, nothing)
    end
    val = f(x)
    jac = J(x)
    return val, Δ -> begin
        if val isa Real
            (NoTangent(), sparse(vec(jac' * Δ)))
        else
            (NoTangent(), jac' * sparse(vec(Δ)))
        end
    end
end
function ChainRulesCore.frule((_, Δx), f::SparseForwardDiffFunction, x::AbstractVector)
    if f.H === nothing
        J = f.J
    else
        J = SparseForwardDiffFunction(f.J, f.vecJ!, f.vecJ, f.hess, f.hess_pattern, f.hess_colors, f.H, nothing, nothing, nothing, nothing, nothing, nothing)
    end
    val = f(x)
    jac = J(x)
    if val isa Real
        Δy = only(jac * Δx)
    elseif val isa AbstractVector
        Δy = jac * sparse(vec(Δx))
    else
        Δy = reshape(jac * sparse(vec(Δx)), size(val)...)
    end
    return val, Δy
end
@ForwardDiff_frule (f::SparseForwardDiffFunction)(x::AbstractVector{<:ForwardDiff.Dual})

function sparsify(f, x...; kwargs...)
    # defined in the abstractdiff.jl file
    flat_f, vx, unflatteny = tovecfunc(f, x...)
    sp_flat_f = SparseForwardDiffFunction(flat_f, vx; kwargs...)
    return x -> unflatteny(sp_flat_f(flatten(x)[1]))
end

function sparsify(model::NonconvexCore.AbstractModel; objective = true, ineq_constraints = true, eq_constraints = true, sd_constraints = true, kwargs...)
    x = getmin(model)
    if objective
        obj = NonconvexCore.Objective(sparsify(model.objective, x; kwargs...), flags = model.objective.flags)
    else
        obj = model.objective
    end
    if ineq_constraints
        ineq = length(model.ineq_constraints.fs) != 0 ? NonconvexCore.VectorOfFunctions(map(model.ineq_constraints.fs) do c
            return NonconvexCore.IneqConstraint(sparsify(c, x; kwargs...), c.rhs, c.dim, c.flags)
        end) : NonconvexCore.VectorOfFunctions(NonconvexCore.IneqConstraint[])
    else
        ineq = model.ineq_constraints
    end
    if eq_constraints
        eq = length(model.eq_constraints.fs) != 0 ? NonconvexCore.VectorOfFunctions(map(model.eq_constraints.fs) do c
            return NonconvexCore.EqConstraint(sparsify(c, x; kwargs...), c.rhs, c.dim, c.flags)
        end) : NonconvexCore.VectorOfFunctions(NonconvexCore.EqConstraint[])
    else
        eq = model.eq_constraints
    end
    if sd_constraints
        sd = length(model.sd_constraints.fs) != 0 ? NonconvexCore.VectorOfFunctions(map(model.sd_constraints.fs) do c
            return NonconvexCore.SDConstraint(sparsify(c, x; kwargs...), c.dim)
        end) : NonconvexCore.VectorOfFunctions(NonconvexCore.SDConstraint[])
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
    return ModelT(obj, eq, ineq, sd, model.box_min, model.box_max, model.init, model.integer)
end
