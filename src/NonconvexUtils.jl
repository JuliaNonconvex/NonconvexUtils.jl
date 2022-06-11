module NonconvexUtils

export  ForwardDiffFunction,
        AbstractDiffFunction,
        AD,
        TraceFunction,
        CustomGradFunction,
        LazyJacobian,
        CustomHessianFunction,
        ImplicitFunction,
        SymbolicFunction,
        SparseForwardDiffFunction

using ChainRulesCore, AbstractDifferentiation, ForwardDiff, LinearAlgebra
using Zygote, LinearMaps, IterativeSolvers, NonconvexCore, SparseArrays
using NonconvexCore: flatten
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
