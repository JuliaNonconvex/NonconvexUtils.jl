module NonconvexUtils

struct ForwardDiffFunc{F} <: Function
    f::F
end
(f::FDFunc)(x) = f.f(x)
function ChainRulesCore.rrule(f::FDFunc, x::AbstractVector)
    ∇ = ForwardDiff.gradient(f, x)
    return f(x), Δ -> (NoTangent(), Δ * ∇)
end

struct TraceObjective{F1, F2} <: Function
    f::F1
    ∇f::F2
    xtrace::Vector{Any}
    ftrace::Vector{Any}
    gtrace::Vector{Any}
end
TraceObjective(f, ∇f) = TraceObjective(f, ∇f, Any[], Any[], Any[])

(to::TraceObjective)(x) = to.f(x)
function ChainRulesCore.rrule(f::TraceObjective, x)
    v, g = f.f(x), f.∇f(x)
    push!(f.xtrace, copy(x))
    push!(f.ftrace, copy(v))
    push!(f.gtrace, copy(g))
    return v, Δ -> (NoTangent(), Δ * g)
end

end
