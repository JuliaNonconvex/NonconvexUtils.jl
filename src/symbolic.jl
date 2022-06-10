struct SymbolicFunction{F, G, H, X} <: Function
    f::F
    g::G
    h::H
    x::X
end
function SymbolicFunction(f, _x::AbstractVector; hessian = false)
    N = length(_x)
    val = f(_x)
    Symbolics.@variables x[1:N]
    if val isa Real
        sgrad = Symbolics.jacobian([f(x)], x)
        f_expr = Symbolics.build_function(sgrad, x)[1]
        _g = eval(f_expr)
        g = x -> vec(_g(x))
        if hessian
            shess = Symbolics.jacobian(Base.invokelatest(g, x), x)
            h_expr = Symbolics.build_function(shess, x)[1]
            h = eval(h_expr)
        else
            h = nothing
        end
    else
        sjac = Symbolics.jacobian(f(x), x)
        f_expr = Symbolics.build_function(sjac, x)[1]
        g = eval(f_expr)
        if hessian
            shess = Symbolics.jacobian(vec(Base.invokelatest(g, x)), x)
            h_expr = Symbolics.build_function(shess, x)[1]
            _h = eval(h_expr)
            h = x -> reshape(_h(x), length(val), N, N)
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
