using NonconvexUtils, ForwardDiff, ReverseDiff, Tracker, Zygote
using ADTypes, DifferentiationInterface
using Test, LinearAlgebra, SparseArrays, NLsolve, IterativeSolvers
using StableRNGs, ChainRulesCore, NonconvexCore, NonconvexIpopt
using DifferentiableFlatten

include("forwarddiff_frule.jl")
include("abstractdiff.jl")
include("trace.jl")
include("custom.jl")
include("implicit.jl")
include("symbolic.jl")
# include("sparse_forwarddiff.jl")


@testset "Aqua" begin
    using Aqua
    Aqua.test_all(NonconvexUtils)
end

@testset "JET" begin
    using JET
    JET.test_package(NonconvexUtils; target_modules=(NonconvexUtils,), analyze_from_definitions=true)
end
