using CUTEst, DCI, LinearAlgebra, Test

using NLPModels, NLPModelsIpopt, NLPModelsKnitro

#=
This file list problems from benchmarks that we didn't solve.
=#

#status is infeasible for these three problems:
problems = ["MSS1", "S308NE", "COATINGNE"] 
#=
MSS1 is unknown for Knitro, and the other two are infeasible.

The three are unknowns for Ipopt
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
  stats_dci = dci(nlp, nlp.meta.x0, max_eval = 1000)

finalize(nlp)
