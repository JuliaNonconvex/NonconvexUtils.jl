macro ForwardDiff_frule(sig)
    _fd_frule(sig)
end
function _fd_frule(sig)
    MacroTools.@capture(sig, f_(x__))
    return quote
        function $(esc(f))($(esc.(x)...))
            f = $(esc(f))
            x = ($(esc.(x)...),)
            flatx, unflattenx = NonconvexCore.flatten(x)
            CS = length(ForwardDiff.partials(first(flatx)))
            flat_xprimals = ForwardDiff.value.(flatx)
            flat_xpartials = reduce(vcat, transpose.(ForwardDiff.partials.(flatx)))

            xprimals = unflattenx(flat_xprimals)
            xpartials1 = unflattenx(flat_xpartials[:,1])

            yprimals, ypartials1 = ChainRulesCore.frule(
                (NoTangent(), xpartials1...), f, xprimals...,
            )
            flat_yprimals, unflatteny = NonconvexCore.flatten(yprimals)
            flat_ypartials1, _ = NonconvexCore.flatten(ypartials1)
            flat_ypartials = hcat(reshape(flat_ypartials1, :, 1), ntuple(Val(CS - 1)) do i
                xpartialsi = unflattenx(flat_xpartials[:, i+1])
                _, ypartialsi = ChainRulesCore.frule((NoTangent(), xpartialsi...), f, xprimals...)
                return NonconvexCore.flatten(ypartialsi)[1]
            end...)

            T = ForwardDiff.tagtype(eltype(flatx))
            flaty = ForwardDiff.Dual{T}.(
                flat_yprimals, ForwardDiff.Partials.(NTuple{CS}.(eachrow(flat_ypartials))),
            )
            return unflatteny(flaty)
        end
    end
end
