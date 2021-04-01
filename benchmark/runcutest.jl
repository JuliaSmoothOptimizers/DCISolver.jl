using Pkg; Pkg.activate("bench")
using CUTEst, NLPModels, NLPModelsKnitro, NLPModelsIpopt, SolverBenchmark, SolverCore
#This package
using DCI
#
using Dates, JLD2

function runcutest(; today :: String = string(today()))

  #pnames = readlines("paper-problems.list")
  _pnames = CUTEst.select(max_var=300, min_con=1, max_con=300, only_free_var=true, only_equ_con=true, objtype=3:6)

  #Remove all the problems ending by NE as Ipopt cannot handle them.
  pnamesNE = _pnames[findall(x->occursin(r"NE\b", x), _pnames)]
  pnames = setdiff(_pnames, pnamesNE)
  cutest_problems = (CUTEstModel(p) for p in pnames)

  #Same time limit for all the solvers
  max_time = 60.

  solvers = Dict(:ipopt => nlp -> ipopt(nlp, print_level = 0,
                                             dual_inf_tol = Inf,
                                             constr_viol_tol = Inf,
                                             compl_inf_tol = Inf,
                                             acceptable_iter = 0,
                                             max_cpu_time = max_time,
                                             x0 = nlp.meta.x0),
                :knitro =>nlp -> knitro(nlp, out_hints = 0, outlev = 0,
                                             feastol = 1e-5,
                                             feastol_abs = 1e-5,
                                             opttol = 1e-5,
                                             opttol_abs = 1e-5,
                                             maxtime_cpu = max_time,
                                             x0 = nlp.meta.x0),
                :DCILDL => nlp -> dci(nlp, nlp.meta.x0, linear_solver = :ldlfact,
                                                          max_time = max_time,
                                                          max_iter = typemax(Int64),
                                                          max_eval = typemax(Int64)),
                :DCIMA57 => nlp -> dci(nlp, nlp.meta.x0, linear_solver = :ma57,
                                                          max_time = max_time,
                                                          max_iter = typemax(Int64),
                                                          max_eval = typemax(Int64)))

  list=""; for solver in keys(solvers) list=string(list,"_$(solver)") end

  stats = bmark_solvers(solvers, cutest_problems)

  @save "$(today)_$(list)_$(string(length(pnames))).jld2" stats

  return stats
end

stats = runcutest()
