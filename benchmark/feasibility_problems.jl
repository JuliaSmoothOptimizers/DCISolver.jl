# stdlib
using Plots, Test
#JSO
using CUTEst, NLPModels, NLPModelsIpopt, NLPModelsKnitro, SolverBenchmark, SolverCore
#This package
using DCI
gr()

#=
#I. A nice example where our heuristic to avoid infeasible stationary points works:
nlp = CUTEstModel("POWERSUMNE")
dci(nlp)
finalize(nlp)
=#

#II. Larger list of problems
#Probably, we should also run from different starting points
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
T.M., February 19th 21':
hcat(stats[:DCI][:name], stats[:DCI][:nvar], stats[:DCI][:ncon],
     stats[:DCI][:primal_feas], stats[::DCI][:status],
     stats[:knitro][:primal_feas], stats[:knitro][:status])

# The main differences are:
"GULFNE"        3   99     2.03504e-6   :first_order     0.99         :infeasible
"HATFLDF"       3    3      4.08161e-6   :first_order      0.00633344   :infeasible
"METHANB8"     31   31      1.78801e-5   :infeasible       1.02621e-7   :first_order
"POWELLBS"      2    2      0.000776101  :infeasible       7.84282e-7   :first_order
"HYDCAR20"     99   99      0.024057     :unknown          5.86464e-7   :first_order
"LUKSAN11"    100  198   2.06639e-6   :first_order  10.0          :infeasible
"LANCZOS2"      6   24   5.01073e-5   :infeasible    5.02291e-6   :first_order
"LANCZOS1"      6   24   5.20818e-5   :infeasible    8.21039e-7   :first_order
"FREURONE"      2    2   6.99888      :infeasible    9.72022e-12  :first_order

=#
