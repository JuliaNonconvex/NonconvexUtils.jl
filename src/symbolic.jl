struct SymbolicFunction{F,G,H,X} <: Function
    f::F
    g::G
    h::H
    x::X
end

function SymbolicFunction(
    f,
    _x::AbstractVector;
    hessian = false,
    sparse = false,
    simplify = false,
)
    N = length(_x)
    val = f(_x)
    _T = eltype(val)
    T = x -> begin
        if (eltype(x) <: Symbolics.Num) || eltype(x) === _T
            return x
        elseif x isa SparseMatrixCSC && !(x isa Real) && nnz(x) == 0
            return SparseMatrixCSC(x.m, x.n, x.colptr, x.rowval, _T[])
        else
            return _T.(x)
        end
    end
    Symbolics.@variables tmpx[1:N]
    x = [tmpx[i] for i = 1:N]
    if val isa Real
        if sparse
            sgrad = Symbolics.sparsejacobian([f(x)], x; simplify)
            _g, _ = Symbolics.build_function(sgrad, x; expression = Val{false})
            g = x -> _sparsevec(T(_g(x)))
        else
            sgrad = Symbolics.jacobian([f(x)], x; simplify)
            _g, _ = Symbolics.build_function(sgrad, x; expression = Val{false})
            g = x -> vec(T(_g(x)))
        end
        if hessian
            if sparse
                shess = Symbolics.sparsejacobian(Vector(g(x)), x; simplify)
            else
                shess = Symbolics.jacobian(g(x), x; simplify)
            end
            _h, _ = Symbolics.build_function(shess, x; expression = Val{false})
            h = x -> T(_h(x))
        else
            h = nothing
        end
    else
        if sparse
            sjac = Symbolics.sparsejacobian(f(x), x; simplify)
            _g, _ = Symbolics.build_function(sjac, x; expression = Val{false})
            g = x -> T(_g(x))
        else
            sjac = Symbolics.jacobian(f(x), x; simplify)
            _g, _ = Symbolics.build_function(sjac, x; expression = Val{false})
            g = x -> T(_g(x))
        end
        if hessian
            if sparse
                shess = Symbolics.sparsejacobian(vec(Matrix(g(x))), x; simplify)
            else
                shess = Symbolics.jacobian(vec(g(x)), x; simplify)
            end
            _h, _ = Symbolics.build_function(shess, x; expression = Val{false})
            h = x -> T(_h(x))
        else
            h = nothing
        end
    end
    return SymbolicFunction(f, g, h, _x)
end
(f::SymbolicFunction)(x) = f.f(x)
function ChainRulesCore.rrule(f::SymbolicFunction, x)
    val = f.f(x)
    g = SymbolicFunction(f.g, f.h, nothing, f.x)
    if val isa Real
        G = g(x)
        return val, Δ -> (NoTangent(), G * Δ)
    else
        G = g(x)
        return val, Δ -> begin
            spΔ = dropzeros!(sparse(copy(Δ)))
            if length(spΔ.nzval) == 1
                return (NoTangent(), G[spΔ.nzind[1], :] * spΔ.nzval[1])
            else
                return (NoTangent(), G' * Δ)
            end
        end
    end
end
function ChainRulesCore.frule((_, Δx), f::SymbolicFunction, x::AbstractVector)
    val = f.f(x)
    g = SymbolicFunction(f.g, f.h, nothing, f.x)
    ∇ = g(x)
    if ∇ isa AbstractVector && Δx isa AbstractVector
        Δy = ∇' * _sparsevec(Δx)
    else
        Δy = reshape(∇ * _sparsevec(Δx), size(val)...)
    end
    project_to = ChainRulesCore.ProjectTo(val)
    return val, project_to(Δy)
end
@ForwardDiff_frule (f::SymbolicFunction)(x::AbstractVector{<:ForwardDiff.Dual})

function symbolify(f, x...; flatteny = true, kwargs...)
    flat_f, vx, unflatteny = tovecfunc(f, x...; flatteny)
    if length(x) == 1 && x[1] isa AbstractVector
        flat_f = f
        sym_flat_f = SymbolicFunction(flat_f, vx; kwargs...)
        return UnflattennedFunction(
            x -> unflatteny(sym_flat_f(x)),
            sym_flat_f,
            vx,
            unflatteny,
            flatteny,
        )
    else
        sym_flat_f = SymbolicFunction(flat_f, vx; kwargs...)
        return UnflattennedFunction(
            x -> unflatteny(sym_flat_f(flatten(x)[1])),
            sym_flat_f,
            vx,
            unflatteny,
            flatteny,
        )
    end
end

function symbolify(
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
            symbolify(model.objective.f, x; kwargs...),
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
                        symbolify(c.f, x; kwargs...),
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
                        symbolify(c.f, x; kwargs...),
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
                        symbolify(c.f, x; flatteny = false, kwargs...),
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
