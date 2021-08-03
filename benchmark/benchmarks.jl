using DelimitedFiles, LinearAlgebra, Printf, SparseArrays
using BenchmarkTools, DataFrames, Dates, JLD2, Plots
#JSO packages
using CUTEst,
  NLPModels, NLPModelsKnitro, NLPModelsIpopt, BenchmarkProfiles, SolverBenchmark, SolverCore
#This package
using DCISolver

function runcutest(cutest_problems, solvers; today::String = string(today()))
  list = ""
  for solver in keys(solvers)
    list = string(list, "_$(solver)")
  end
  stats = bmark_solvers(solvers, cutest_problems)

  @save "$(today)_$(list)_$(string(length(pnames))).jld2" stats

  return stats
end

nmax = 300
_pnames = CUTEst.select(
  max_var = nmax,
  min_con = 1,
  max_con = nmax,
  only_free_var = true,
  only_equ_con = true,
  objtype = 3:6,
)

#Remove all the problems ending by NE as Ipopt cannot handle them.
pnamesNE = _pnames[findall(x -> occursin(r"NE\b", x), _pnames)]
pnames = setdiff(_pnames, pnamesNE)
cutest_problems = (CUTEstModel(p) for p in pnames)

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

const SUITE = BenchmarkGroup()
SUITE[:cutest_dcildl_ipopt_benchmark] =
  @benchmarkable runcutest(cutest_problems, solvers) samples = 5
