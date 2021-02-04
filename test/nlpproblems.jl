using LinearAlgebra, Logging, NLPModels, PDENLPModels

using Gridap
include("../../PDENLPModels/PDEOptimizationProblems/Burger1d.jl")

nlp = Burger1d(n=50)

using DCI, FletcherPenaltyNLPSolver

@show "Solve with DCI"
stats = dci(nlp, nlp.meta.x0)
@show "Solve with FPS"
stats2 = Fletcher_penalty_solver(nlp) 
