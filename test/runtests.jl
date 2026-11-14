using NonconvexUtils, NonconvexCore, Test

@testset "NonconvexUtils shim" begin
    # The package re-exports NonconvexCore's API unchanged.
    @test NonconvexUtils.DictModel === NonconvexCore.DictModel
    @test NonconvexUtils.Model === NonconvexCore.Model
    @test NonconvexUtils.optimize === NonconvexCore.optimize
    @test NonconvexUtils.getobjective === NonconvexCore.getobjective
    @test NonconvexUtils.addvar! === NonconvexCore.addvar!
    @test NonconvexUtils.symbolify === NonconvexCore.symbolify

    # A model built through the re-exported names evaluates correctly.
    m = DictModel()
    addvar!(m, :x, -1.0, 1.0, init=0.5)
    set_objective!(m, x -> (x[:x] - 0.25)^2)
    @test getobjective(m)(NonconvexCore.getinit(m)) == (0.5 - 0.25)^2
end

@testset "Aqua" begin
    include("aqua.jl")
end
