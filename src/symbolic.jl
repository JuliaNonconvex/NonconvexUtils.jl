struct SymbolicFunction{F, G, H, X} <: Function
    f::F
    g::G
    h::H
    x::X
end

function SymbolicFunction(f, _x::AbstractVector; hessian = false, sparse = false, simplify = false)
    N = length(_x)
    val = f(_x)
    _T = eltype(val)
    T = x -> begin
        if (eltype(x) <: Symbolics.Num) || eltype(x) === _T || !(x isa Real) && (issparse(x) && nnz(x) == 0 || isempty(x))
            return x
        else
            return _T.(x)
        end
    end
    Symbolics.@variables x[1:N]
    if val isa Real
        if sparse
            sgrad = Symbolics.sparsejacobian([f(x)], x; simplify)
            f_expr = Symbolics.build_function(sgrad, x)[1]
            _g = eval(f_expr)
            g = x -> vec(Matrix(T(_g(x))))
        else
            sgrad = Symbolics.jacobian([f(x)], x; simplify)
            f_expr = Symbolics.build_function(sgrad, x)[1]
            _g = eval(f_expr)
            g = x -> vec(T(_g(x)))
        end
        if hessian
            if sparse
                shess = Symbolics.sparsejacobian(Base.invokelatest(g, x), x; simplify)
            else
                shess = Symbolics.jacobian(Base.invokelatest(g, x), x; simplify)
            end
            h_expr = Symbolics.build_function(shess, x)[1]
            _h = eval(h_expr)
            h = x -> T(_h(x))
        else
            h = nothing
        end
    else
        if sparse
            sjac = Symbolics.sparsejacobian(f(x), x; simplify)
            f_expr = Symbolics.build_function(sjac, x)[1]
            _g = eval(f_expr)
            g = x -> Matrix(T(_g(x)))
        else
            sjac = Symbolics.jacobian(f(x), x; simplify)
            f_expr = Symbolics.build_function(sjac, x)[1]
            _g = eval(f_expr)
            g = x -> T(_g(x))
        end
        if hessian
            if sparse
                shess = Symbolics.sparsejacobian(vec(Base.invokelatest(g, x)), x; simplify)
            else
                shess = Symbolics.jacobian(vec(Base.invokelatest(g, x)), x; simplify)
            end
            h_expr = Symbolics.build_function(shess, x)[1]
            _h = eval(h_expr)
            h = x -> reshape(T(_h(x)), length(val), N, N)
        else
            h = nothing
        end
    end
    return SymbolicFunction(f, g, h, _x)
end
(f::SymbolicFunction)(x) = f.f(x)
function ChainRulesCore.rrule(f::SymbolicFunction, x)
    val = f.f(x)
    if val isa Real
        g = CustomGradFunction(f.g, f.h)
        return val, Δ -> (NoTangent(), g(x) * Δ)
    else
        hvp = (x, v) -> reshape(f.h, length(val), length(x))' * vec(v)
        _h = x -> LazyJacobian{true}(v -> hvp(x, v))
        g = CustomGradFunction(f.g, _h)
        return val, Δ -> (NoTangent(), g(x)' * Δ)
    end
end
function ChainRulesCore.frule(
    (_, Δx), f::SymbolicFunction, x::AbstractVector,
)
    val = f.f(x)
    if val isa Real
        g = CustomGradFunction(f.g, f.h)
    else
        hvp = (x, v) -> reshape(f.h, length(val), length(x))' * vec(v)
        _h = x -> LazyJacobian{true}(v -> hvp(x, v))
        g = CustomGradFunction(f.g, _h)
    end
    ∇ = g(x)
    if ∇ isa AbstractVector && Δx isa AbstractVector
        return val, ∇' * Δx
    else
        return val, ∇ * Δx
    end
end
@ForwardDiff_frule (f::SymbolicFunction)(x::AbstractVector{<:ForwardDiff.Dual})
