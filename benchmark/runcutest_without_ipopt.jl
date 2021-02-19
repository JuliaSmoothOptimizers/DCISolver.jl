#activate hsl

using CUTEst, NLPModels, NLPModelsKnitro, NLPModelsIpopt, Plots, SolverBenchmark, SolverTools
#This package
using DCI
gr()

function runcutest()
  #pnames = readlines("paper-problems.list")
  _pnames = CUTEst.select(max_var=100, min_con=1, max_con=100, only_free_var=true, only_equ_con=true, objtype=3:6)

  #Remove all the problems ending by NE as Ipopt cannot handle them.
  pnamesNE = _pnames[findall(x->occursin(r"NE\b", x), _pnames)]
  pnames = setdiff(_pnames, pnamesNE)[1:3]
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
#=
 solvers = Dict(:DCI_LDL => nlp -> dci(nlp, nlp.meta.x0, linear_solver = :ldlfact, max_time = 60., max_iter = 5000), #atol=rtol=ctol=1e-5 by default
                :DCI_MA57 => nlp -> dci(nlp, nlp.meta.x0, linear_solver = :ma57, max_time = 60., max_iter = 5000))
=#

  solvers = Dict(:ipopt => (nlp; kwargs...) -> ipopt(nlp, print_level = 0, 
                                                          dual_inf_tol = Inf,
                                                          constr_viol_tol = Inf,
                                                          compl_inf_tol = Inf, 
                                                          acceptable_iter = 0,
                                                          max_cpu_time = 600., #max_cpu_time and its default value is 10+06.
                                                          x0 = nlp.meta.x0,
                                                          kwargs...),
                :DCI_MA57 => nlp -> dci(nlp, nlp.meta.x0, linear_solver = :ma57, 
                                                          max_time = 600., 
                                                          max_iter = typemax(Int64), 
                                                          max_eval = typemax(Int64)))
  list=""; for solver in keys(solvers) list=string(list,"_$(solver)") end
  today = "20210218"

  stats = bmark_solvers(solvers, cutest_problems)
  #join_df = join(stats, [:objective, :dual_feas, :primal_feas, :neval_obj, :status], invariant_cols=[:name])
  #SolverBenchmark.markdown_table(stdout, join_df)
  #:neval_jprod, :neval_jtprod are not used by Knitro or Ipopt and none use :neval_hprod
  for col in [:neval_obj, :neval_cons, :neval_grad, :neval_jac, :neval_jprod, :neval_jtprod, :neval_hess, :elapsed_time]
    empty = false
    for df in values(stats)
      if all(df[col] .== 0)
        empty = true
      end
    end

    if !empty
      ϵ = minimum(minimum(filter(x -> x > 0, df[col])) for df in values(stats))
      cost(df) = (max.(df[col], ϵ) + (df.status .!= :first_order) * Inf)
      performance_profile(stats, cost)
      png("$(today)_$(list)_perf-$col")
    end
  end

  for solver in keys(solvers)
    open("data_$(solver).md", "w") do io
          SolverBenchmark.markdown_table(io, stats[solver])
        end
  end

  return stats
end

stats = runcutest()

#=
status
hcat(stats[:ipopt][:name], stats[:ipopt][:nvar], stats[:ipopt][:ncon],
     stats[:ipopt][:primal_feas], stats[:ipopt][:dual_feas],
     stats[:DCI_MA57][:primal_feas], stats[:DCI_MA57][:dual_feas])
=#
