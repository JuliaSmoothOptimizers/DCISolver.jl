#activate hsl

using CUTEst, NLPModels, NLPModelsKnitro, Plots, SolverBenchmark, SolverTools
#This package
using DCI
gr()

function runcutest()
  #pnames = readlines("paper-problems.list")
  #pnames = pnames[1:3]
  #_pnames = CUTEst.select(max_var=100, min_con=1, max_con=100, only_free_var=true, only_equ_con=true)
  pnames = CUTEst.select(max_var=10000, min_con=1, max_con=10000, only_free_var=true, only_equ_con=true, objtype=3:6)
  #remove MSS1 and S308NE that are very hard?
  #pnames = setdiff(pnames, ["MSS1", "S308NE"])

  #Remove all the problems ending by NE as Ipopt cannot handle them.
  #pnamesNE = _pnames[findall(x->occursin(r"NE\b", x), _pnames)]
  #pnames = setdiff(_pnames, pnamesNE)
  cutest_problems = (CUTEstModel(p) for p in pnames)
#=
  solvers = Dict(:DCI_MA57 => nlp -> dci(nlp, nlp.meta.x0),# linear_solver = :ma57),
                 :knitro =>(nlp; kwargs...) -> knitro(nlp, out_hints = 0, outlev = 0,
                                                       feastol = 1e-5,
                                                       feastol_abs = 1e-5,
                                                       opttol = 1e-5,
                                                       opttol_abs = 1e-5,
                                                       x0 = nlp.meta.x0,
                                                       kwargs...))
=#
 solvers = Dict(:DCI_LDL => nlp -> dci(nlp, nlp.meta.x0, linear_solver = :ldlfact, max_time = 600., max_iter = 5000), #atol=rtol=ctol=1e-5 by default
                :DCI_MA57 => nlp -> dci(nlp, nlp.meta.x0, linear_solver = :ma57, max_time = 600., max_iter = 5000))
# =#
  stats = bmark_solvers(solvers, cutest_problems)

  #join_df = join(stats, [:objective, :dual_feas, :primal_feas, :neval_obj, :status], invariant_cols=[:name])
  #SolverBenchmark.markdown_table(stdout, join_df)
#:neval_jprod, :neval_jtprod are not used by Knitro and none use :neval_hprod
  for col in [:neval_obj, :neval_cons, :neval_grad, :neval_jac, :neval_jprod, :neval_jtprod, :neval_hess, :elapsed_time]
    for df in values(stats)
      if all(df[col] .== 0)
        println("df[col] = $(df[col])")
      end
    end
    ϵ = minimum(minimum(filter(x -> x > 0, df[col])) for df in values(stats))
    cost(df) = (max.(df[col], ϵ) + (df.status .!= :first_order) * Inf)
    performance_profile(stats, cost)
    png("perf-$col")
  end

  open("data_dci_ldl.md", "w") do io
        SolverBenchmark.markdown_table(io, stats[:DCI_LDL])
      end
  open("data_dci_ma57.md", "w") do io
        SolverBenchmark.markdown_table(io, stats[:DCI_MA57])
      end

  return stats
end

stats = runcutest()

status
hcat(stats[:DCI_LDL][:name], stats[:DCI_LDL][:nvar], stats[:DCI_LDL][:ncon], stats[:DCI_LDL][:primal_feas], stats[:DCI_LDL][:dual_feas], stats[:DCI_MA57][:primal_feas], stats[:DCI_MA57][:dual_feas])
