using CUTEst, DCI, LinearAlgebra, Test

using NLPModels, NLPModelsIpopt, NLPModelsKnitro

#=
This file list problems from benchmarks that we didn't solve.
=#

problems = ["MSS1", "S308NE", "COATINGNE"] 
#=
S308NE and COATINGNE could really be infeasible ?

Actually, we just struggle to achieve a high precision with MSS1, Knitro does 1e-2 while DCI 1e-3.
=#

i=1
nlp = CUTEstModel(problems[i])

#stats_ipopt = ipopt(nlp, x0 = nlp.meta.x0, dual_inf_tol=Inf, constr_viol_tol=Inf, compl_inf_tol=Inf, acceptable_iter=0)
#x = stats_ipopt.solution
#norm(cons(nlp, x))
#norm(grad(nlp, x) + jtprod(nlp, x, stats_ipopt.multipliers))
#=
tol=1e-5 is the parameter killing Ipopt.
ipopt(nlp, dual_inf_tol=Inf, constr_viol_tol=Inf, compl_inf_tol=Inf, acceptable_iter=0, tol = 1e-5, x0 = nlp.meta.x0)
=#
#  reset!(nlp)
#  stats_knitro = knitro(nlp)
#=
knitro(nlp, out_hints = 0, outlev = 0,
                                                       feastol = 1e-5,
                                                       feastol_abs = 1e-5,
                                                       opttol = 1e-5,
                                                       opttol_abs = 1e-5,
                                                       x0 = nlp.meta.x0,
                                                       kwargs...))
=#
  reset!(nlp)
  #thanks to the restoration step we avoid the infeasible stationary point
  #it is impressive how many iterations we need for λ !
  #Probably, we need some scaling or pre-conditioning ?
  #stats = dci(nlp, nlp.meta.x0, linear_solver = :ldlfact, max_time = 160., max_iter = 1000)
  stats = dci(nlp, nlp.meta.x0, linear_solver = :ma57, max_time = 160., max_iter = 1000)

finalize(nlp)
