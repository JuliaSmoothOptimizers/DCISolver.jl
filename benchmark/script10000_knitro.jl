using Pkg
Pkg.activate("")
using CUTEst, NLPModels, NLPModelsIpopt, SolverBenchmark, SolverCore
using NLPModelsKnitro
#This package
using DCISolver

nmax = 10000
_pnames = CUTEst.select(
  max_var = nmax, 
  min_con = 1, 
  max_con = nmax, 
  only_free_var = true, 
  only_equ_con = true, 
  objtype = 3:6
)

#Remove all the problems ending by NE as Ipopt cannot handle them.
pnamesNE = _pnames[findall(x->occursin(r"NE\b", x), _pnames)]
pnames = setdiff(_pnames, pnamesNE)
cutest_problems = (CUTEstModel(p) for p in pnames)

#Same time limit for all the solvers
max_time = 1200. #20 minutes
tol = 1e-5

solvers = Dict(
  :ipopt => nlp -> ipopt(
    nlp,
    print_level = 0,
    dual_inf_tol = Inf,
    constr_viol_tol = Inf,
    compl_inf_tol = Inf,
    acceptable_iter = 0,
    max_cpu_time = max_time,
    x0 = nlp.meta.x0,
    tol = tol,
  ),
  :knitro => nlp -> knitro(
    nlp,
    x0 = nlp.meta.x0,
    feastol = tol,
    feastol_abs = tol,
    opttol = tol,
    opttol_abs = tol,
    maxfevals = typemax(Int32),
    maxit = 0,
    maxtime_real = max_time,
    outlev = 0,
  ),
  :dcildl => nlp -> dci(
    nlp,
    nlp.meta.x0,
    linear_solver = :ldlfact,
    max_time = max_time,
    max_iter = typemax(Int64),
    max_eval = typemax(Int64),
    atol = tol,
    ctol = tol,
    rtol = tol,
  ),
)
stats = bmark_solvers(solvers, cutest_problems)

using JLD2
@save "ipopt_knitro_dcildl_$(string(length(pnames))).jld2" stats
