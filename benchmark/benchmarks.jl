using Pkg
Pkg.add(url="https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl#main")
using BenchmarkTools, DataFrames, Dates, DelimitedFiles, JLD2, Logging
#JSO packages
using OptimizationProblems, NLPModels, NLPModelsKnitro, NLPModelsIpopt, SolverBenchmark, SolverCore
import ADNLPModels
#This package
using DCISolver

function runcutest(ad_problems, solvers; today::String = string(today()))
  list = ""
  for solver in keys(solvers)
    list = string(list, "_$(solver)")
  end
  stats = bmark_solvers(solvers, ad_problems)

  @save "$(today)_$(list)_$(string(length(ad_problems))).jld2" stats

  return stats
end

ad_problems = [
  eval(Meta.parse("ADNLPProblems.$(prob)()"))
  for prob in setdiff(names(ADNLPProblems), [:ADNLPProblems, :clplatea, :clplateb, :clplatec, :fminsrf2])[1:10]
]

#Same time limit for all the solvers
max_time = 60.0 #20 minutes
solvers = Dict(
  :ipopt =>
    nlp -> ipopt(
      nlp,
      print_level = 0,
      dual_inf_tol = Inf,
      constr_viol_tol = Inf,
      compl_inf_tol = Inf,
      acceptable_iter = 0,
      max_cpu_time = max_time,
      x0 = nlp.meta.x0,
    ),
  :DCILDL =>
    nlp -> dci(
      nlp,
      nlp.meta.x0,
      linear_solver = :ldlfact,
      max_time = max_time,
      max_iter = typemax(Int64),
      max_eval = typemax(Int64),
    ),
)

with_logger(NullLogger()) do
  runcutest(ad_problems, solvers) # for precompilation
end

const SUITE = BenchmarkGroup()
SUITE[:cutest_dcildl_ipopt_benchmark] = @benchmarkable with_logger(NullLogger()) do
   runcutest(ad_problems, solvers)
  end #samples = 5 seconds = 300
