struct TraceFunction{F, V} <: Function
    f::F
    trace::V
    on_call::Bool
    on_grad::Bool
end
function TraceFunction(f; on_call::Union{Bool, Nothing} = nothing, on_grad::Union{Bool, Nothing} = nothing)
    if on_call === on_grad === nothing
        _on_call = true
        _on_grad = true
    elseif on_call === nothing
        _on_call = !on_grad
        _on_grad = on_grad
    elseif on_grad === nothing
        _on_call = on_call
        _on_grad = !on_call
    else
        _on_call = on_call
        _on_grad = on_grad
    end
    return TraceFunction(f, Any[], _on_call, _on_grad)
end
function (tf::TraceFunction)(x)
    v = tf.f(x)
    if tf.on_call
        push!(tf.trace, (input = copy(x), output = copy(v)))
    end
    return v
end
function ChainRulesCore.rrule(rc::RuleConfig, tf::TraceFunction, x)
    v, pb = ChainRulesCore.rrule_via_ad(rc, tf.f, x)
    return v, Δ -> begin
        Δin = pb(Δ)
        g = Δin[2] isa Array ? Δin[2] : Δin[2].val.f()
        if tf.on_grad
            push!(tf.trace, (input = copy(x), output = copy(v), grad = copy(g)))
        end
        return (Δin[1], g)
    end
end
function ChainRulesCore.frule(
    rc::RuleConfig, (_, Δx), tf::TraceFunction, x,
)
    v, g = ChainRulesCore.frule(rc, (NoTangent(), Δx), tf.f, x)
    if tf.on_grad
        if !isempty(tf.trace) && x == tf.trace[end].input
            push!(tf.trace[end].grad, g)
        else
            push!(tf.trace, (input = copy(x), output = copy(v), grad = [copy(g)]))
        end
    end
    return v, g
end
function ChainRulesCore.frule(
    (_, Δx), tf::TraceFunction, x,
)
    v, g = ChainRulesCore.frule((NoTangent(), Δx), tf.f, x)
    if tf.on_grad
        if !isempty(tf.trace) && x == tf.trace[end].input
            push!(tf.trace[end].grad, g)
        else
            push!(tf.trace, (input = copy(x), output = copy(v), grad = [copy(g)]))
        end
    end
    return v, g
end

@ForwardDiff_frule (f::TraceFunction)(x::AbstractVector{<:ForwardDiff.Dual})
@ForwardDiff_frule (f::TraceFunction)(x::ForwardDiff.Dual)
