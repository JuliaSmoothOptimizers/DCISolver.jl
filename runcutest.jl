using CUTEst, NLPModels, NLPModelsIpopt, NLPModelsKnitro, Plots, SolverBenchmark, SolverTools
#This package
using DCI
gr()

function runcutest()
  #pnames = readlines("paper-problems.list")
  #pnames = pnames[1:3]
  #_pnames = CUTEst.select(max_var=100, min_con=1, max_con=100, only_free_var=true, only_equ_con=true)
  pnames = CUTEst.select(max_var=100, min_con=1, max_con=100, only_free_var=true, only_equ_con=true, objtype=3:6)
  #remove MSS1 and S308NE that are very hard?
  #pnames = setdiff(pnames, ["MSS1", "S308NE"])

  #Remove all the problems ending by NE as Ipopt cannot handle them.
  #pnamesNE = _pnames[findall(x->occursin(r"NE\b", x), _pnames)]
  #pnames = setdiff(_pnames, pnamesNE)
  cutest_problems = (CUTEstModel(p) for p in pnames)

  #=
  In order for stop
ping conditions to be comparable in both solvers, we set the IPOPT parameters
dual_inf_tol=Inf, constr_viol_tol=Inf and compl_inf_tol=Inf
to disable additional stopping conditions related to those tolerances, accepta-
ble_iter=0 to disable the search for an acceptable point
  =#
  solvers = Dict(:DCI => nlp -> dci(nlp, nlp.meta.x0), #atol=rtol=ctol=1e-5 by default
                 :ipopt => (nlp; kwargs...) -> ipopt(nlp, print_level = 0, 
                                                          dual_inf_tol = Inf,
                                                          constr_viol_tol = Inf,
                                                          compl_inf_tol = Inf, 
                                                          acceptable_iter = 0,
                                                          x0 = nlp.meta.x0,
                                                          kwargs...), #tol = 1e-5, 
                 :knitro =>(nlp; kwargs...) -> knitro(nlp, out_hints = 0, outlev = 0,
                                                       feastol = 1e-5,
                                                       feastol_abs = 1e-5,
                                                       opttol = 1e-5,
                                                       opttol_abs = 1e-5,
                                                       x0 = nlp.meta.x0,
                                                       kwargs...))
  stats = bmark_solvers(solvers, cutest_problems)

  join_df = join(stats, [:objective, :dual_feas, :primal_feas, :neval_obj, :status], invariant_cols=[:name])
  SolverBenchmark.markdown_table(stdout, join_df)
  for col in [:neval_obj, :elapsed_time]
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

  return stats
end

stats = runcutest()
