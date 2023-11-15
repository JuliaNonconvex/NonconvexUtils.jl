struct SparseForwardDiffFunction{F,F!,Y,J,JP,JC,JJ,JJ!,G,HH1,HP,HC,HH2} <: Function
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

function SparseForwardDiffFunction(
    f,
    x::AbstractVector;
    hessian = false,
    jac_pattern = nothing,
    hess_pattern = nothing,
)
    val = f(x)
    _f = x -> _sparsevec(f(x))
    f! = (y, x) -> begin
        v = f(x)
        y .= v
        return y
    end
    y = _sparsevec(val)
    jac_pattern =
        jac_pattern === nothing ? Symbolics.jacobian_sparsity(f!, y, x) : jac_pattern
    if nnz(jac_pattern) > 0
        jac = float.(jac_pattern)
        jac_colors = SparseDiffTools.matrix_colors(jac)
    else
        T = eltype(y)
        jac = sparse(Int[], Int[], T[], length(y), length(x))
        jac_colors = Int[]
    end
    vecJ! =
        (G, x) -> begin
            _jac = SparseDiffTools.forwarddiff_color_jacobian(_f, x, colorvec = jac_colors)
            G .= _sparsevec(_jac)
            return G
        end
    G = vec(Array(jac))
    J =
        x -> begin
            xT = eltype(x)
            if length(jac.nzval) > 0
                _jac = SparseDiffTools.forwarddiff_color_jacobian(
                    _f,
                    x,
                    colorvec = jac_colors,
                    sparsity = jac,
                    jac_prototype = xT.(jac),
                )
                project_to = ChainRulesCore.ProjectTo(jac)
                return project_to(copy(_jac))
            else
                return sparse(Int[], Int[], xT[], size(jac)...)
            end
        end
    if hessian
        hess_pattern =
            hess_pattern === nothing ? Symbolics.jacobian_sparsity(vecJ!, G, x) :
            hess_pattern
        if nnz(hess_pattern) > 0
            hess = float.(hess_pattern)
            hess_colors = SparseDiffTools.matrix_colors(hess)
            _J = x -> _sparsevec(J(x))
            H =
                x -> begin
                    _hess = SparseDiffTools.forwarddiff_color_jacobian(
                        _J,
                        x,
                        colorvec = hess_colors,
                        sparsity = hess_pattern,
                        jac_prototype = hess,
                    )
                    project_to = ChainRulesCore.ProjectTo(hess)
                    return project_to(copy(_hess))
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
    return SparseForwardDiffFunction(
        f,
        f!,
        y,
        jac,
        jac_pattern,
        jac_colors,
        J,
        vecJ!,
        G,
        hess,
        hess_pattern,
        hess_colors,
        H,
    )
end

(f::SparseForwardDiffFunction)(x) = f.f(x)
function ChainRulesCore.rrule(f::SparseForwardDiffFunction, x::AbstractVector)
    if f.H === nothing
        J = f.J
    else
        J = SparseForwardDiffFunction(
            f.J,
            f.vecJ!,
            f.vecJ,
            f.hess,
            f.hess_pattern,
            f.hess_colors,
            f.H,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
        )
    end
    val = f(x)
    jac = J(x)
    if eltype(f.jac) === eltype(jac)
        nograd_cache!(f.jac, jac)
    end
    return val, Δ -> begin
        if val isa Real
            (NoTangent(), jac' * Δ)
        else
            spΔ = dropzeros!(sparse(_sparsevec(copy(Δ))))
            if length(spΔ.nzval) == 1
                (NoTangent(), jac[spΔ.nzind[1], :] * spΔ.nzval[1])
            else
                (NoTangent(), jac' * spΔ)
            end
        end
    end
end
function ChainRulesCore.frule((_, Δx), f::SparseForwardDiffFunction, x::AbstractVector)
    if f.H === nothing
        J = f.J
    else
        J = SparseForwardDiffFunction(
            f.J,
            f.vecJ!,
            f.vecJ,
            f.hess,
            f.hess_pattern,
            f.hess_colors,
            f.H,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
        )
    end
    val = f(x)
    jac = J(x)
    if eltype(f.jac) === eltype(jac)
        nograd_cache!(f.jac, jac)
    end
    if val isa Real
        Δy = only(jac * Δx)
    elseif val isa AbstractVector
        spΔx = dropzeros!(sparse(_sparsevec(copy(Δx))))
        if length(spΔx.nzval) == 1
            Δy = jac[:, spΔx.nzind[1]] * spΔx.nzval[1]
        else
            Δy = jac * spΔx
        end
    else
        spΔx = dropzeros!(sparse(_sparsevec(copy(Δx))))
        Δy = _sparse_reshape(jac * spΔx, size(val)...)
    end
    project_to = ChainRulesCore.ProjectTo(val)
    return val, project_to(Δy)
end
@ForwardDiff_frule (f::SparseForwardDiffFunction)(x::AbstractVector{<:ForwardDiff.Dual})

function nograd_cache!(A, B)
    A .= B
    return A
end
function ChainRulesCore.frule(_, ::typeof(nograd_cache!), A, B)
    nograd_cache!(A, B), NoTangent()
end
function ChainRulesCore.rrule(::typeof(nograd_cache!), A, B)
    nograd_cache!(A, B), _ -> (NoTangent(), NoTangent(), NoTangent())
end

struct UnflattennedFunction{F1,F2,V,U} <: Function
    f::F1
    flat_f::F2
    v::V
    unflatten::U
    flatteny::Bool
end
(f::UnflattennedFunction)(x...) = f.f(x...)
function NonconvexCore.tovecfunc(f::UnflattennedFunction, x...; flatteny = true)
    @assert flatteny == f.flatteny
    return f.flat_f, f.v, f.unflatten
end

function sparsify(f, x...; flatteny = true, kwargs...)
    flat_f, vx, unflatteny = tovecfunc(f, x...; flatteny)
    if length(x) == 1 && x[1] isa AbstractVector
        flat_f = f
        sp_flat_f = SparseForwardDiffFunction(flat_f, vx; kwargs...)
        return UnflattennedFunction(
            x -> unflatteny(sp_flat_f(x)),
            sp_flat_f,
            vx,
            unflatteny,
            flatteny,
        )
    else
        sp_flat_f = SparseForwardDiffFunction(flat_f, vx; kwargs...)
        return UnflattennedFunction(
            x -> unflatteny(sp_flat_f(flatten(x)[1])),
            sp_flat_f,
            vx,
            unflatteny,
            flatteny,
        )
    end
end

function sparsify(
    model::NonconvexCore.AbstractModel;
    objective = true,
    ineq_constraints = true,
    eq_constraints = true,
    sd_constraints = true,
    kwargs...,
)
    x = getmin(model)
    if objective
        obj = NonconvexCore.Objective(
            sparsify(model.objective.f, x; kwargs...),
            model.objective.multiple,
            model.objective.flags,
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
                        sparsify(c.f, x; kwargs...),
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
                        sparsify(c.f, x; kwargs...),
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
                    return NonconvexCore.SDConstraint(
                        sparsify(c.f, x; flatteny = false, kwargs...),
                        c.dim,
                    )
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
