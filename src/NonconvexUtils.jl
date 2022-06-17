module NonconvexUtils

export  forwarddiffy,
        abstractdiffy,
        AD,
        TraceFunction,
        CustomGradFunction,
        LazyJacobian,
        CustomHessianFunction,
        ImplicitFunction,
        sparsify,
        symbolify

using ChainRulesCore, AbstractDifferentiation, ForwardDiff, LinearAlgebra
using Zygote, LinearMaps, IterativeSolvers, NonconvexCore, SparseArrays
using NonconvexCore: flatten, tovecfunc, _sparsevec, _sparse_reshape
using MacroTools
using Symbolics: Symbolics
using SparseDiffTools: SparseDiffTools

include("forwarddiff_frule.jl")
include("abstractdiff.jl")
include("lazy.jl")
include("trace.jl")
include("custom.jl")
include("implicit.jl")
include("symbolic.jl")
include("sparse_forwarddiff.jl")

end
