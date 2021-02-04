using CUTEst, NLPModels, NLPModelsIpopt, NLPModelsKnitro, Plots, SolverBenchmark, SolverTools
#This package
using DCI
gr()

#I. A nice example where our heuristic to avoid infeasible stationary points works:
nlp = CUTEstModel("POWERSUMNE")
dci(nlp)
finalize(nlp)

#II. Larger list of problems
pnames = CUTEst.select(max_var=300, min_con=1, max_con=300, only_free_var=true, only_equ_con=true, objtype=1:2)

cutest_problems = (CUTEstModel(p) for p in pnames)

solvers = Dict(:DCI => nlp -> dci(nlp, nlp.meta.x0), #atol=rtol=ctol=1e-5 by default
                 :knitro =>(nlp; kwargs...) -> knitro(nlp, out_hints = 0, outlev = 0,
                                                       feastol = 1e-5,
                                                       feastol_abs = 1e-5,
                                                       opttol = 1e-5,
                                                       opttol_abs = 1e-5,
                                                       x0 = nlp.meta.x0,
                                                       kwargs...))
stats = bmark_solvers(solvers, cutest_problems)

#=
We probably need to be more agressive in the restoration step.
Is there some problems with very small steps?
=#
