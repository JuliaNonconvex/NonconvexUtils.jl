module NonconvexUtils

export diffiy,
    DI,
    TraceFunction,
    CustomGradFunction,
    LazyJacobian,
    CustomHessianFunction,
    ImplicitFunction,
    symbolify,
    JuMPSymbolicFunction,
    jump_symbolic_gradient,
    jump_symbolic_hessian

using ChainRulesCore, ForwardDiff, LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface
using DifferentiationInterface
const DI = DifferentiationInterface
using ADTypes
using Zygote, LinearMaps, IterativeSolvers, NonconvexCore, SparseArrays
using NonconvexCore: flatten, tovecfunc, _sparsevec, _sparse_reshape
using MacroTools
using Symbolics: Symbolics

include("forwarddiff_frule.jl")
include("abstractdiff.jl")
include("lazy.jl")
include("trace.jl")
include("custom.jl")
include("implicit.jl")
include("symbolic.jl")
include("jumpsymbolic.jl")

end
