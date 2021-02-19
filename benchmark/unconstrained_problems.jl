# stdlib
using Plots, Test
#JSO
using NLPModels, NLPModelsIpopt, NLPModelsKnitro, NLPModelsJuMP, OptimizationProblems, SolverBenchmark, SolverTools
#This package
using DCI

# Test that every problem can be instantiated.
global pnames = setdiff(names(OptimizationProblems), [:OptimizationProblems])
for prob in setdiff(names(OptimizationProblems), [:OptimizationProblems])
  prob_fn = eval(prob)
  nlp = MathOptNLPModel(prob_fn())
  if !(equality_constrained(nlp) || unconstrained(nlp))
    setdiff!(pnames, [prob])
  end
end

op_problems = (MathOptNLPModel(eval(p)(), name = string(p)) for p in pnames)

solvers = Dict(:DCI_LDL => nlp -> dci(nlp, nlp.meta.x0, linear_solver = :ldlfact, max_time = 60., max_iter = 5000, max_eval = typemax(Int64)))

stats = bmark_solvers(solvers, op_problems)

#=
T.M., February 19th 21': three problems looks difficult:
[ Info:           hs112      10       3         max_time   6.0e+01  -2.1e+01   7.2e+00   1.0e+00
[ Info:            hs41       4       1         max_iter   8.1e-02  -2.7e+32   2.4e+21   8.8e+00
[ Info:            hs62       3       1         max_time   6.0e+01  -7.7e+03   2.8e+05   1.0e+00
=#
