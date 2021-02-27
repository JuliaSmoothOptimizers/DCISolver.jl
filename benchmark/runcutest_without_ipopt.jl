using Pkg; Pkg.activate("bench")
using CUTEst, NLPModels, NLPModelsKnitro, NLPModelsIpopt, Plots, SolverBenchmark, SolverTools
#This package
using DCI
gr()

function runcutest()
  #pnames = readlines("paper-problems.list")
  _pnames = CUTEst.select(max_var=300, min_con=1, max_con=300, only_free_var=true, only_equ_con=true, objtype=3:6)

  #Remove all the problems ending by NE as Ipopt cannot handle them.
  pnamesNE = _pnames[findall(x->occursin(r"NE\b", x), _pnames)]
  pnames = setdiff(_pnames, pnamesNE)
  cutest_problems = (CUTEstModel(p) for p in pnames)

  dsolvers = Dict(:ipopt => nlp -> ipopt(nlp, print_level = 0, 
                                                          dual_inf_tol = Inf,
                                                          constr_viol_tol = Inf,
                                                          compl_inf_tol = Inf, 
                                                          acceptable_iter = 0,
                                                          max_cpu_time = 600., #max_cpu_time and its default value is 10+06.
                                                          x0 = nlp.meta.x0),
                :knitro =>nlp -> knitro(nlp, out_hints = 0, outlev = 0,
                                                       feastol = 1e-5,
                                                       feastol_abs = 1e-5,
                                                       opttol = 1e-5,
                                                       opttol_abs = 1e-5,
                                                       maxtime_cpu = 600.,
                                                       x0 = nlp.meta.x0),
                :DCI_LDL => nlp -> dci(nlp, nlp.meta.x0, linear_solver = :ldlfact, 
                                                          max_time = 600., 
                                                          max_iter = typemax(Int64), 
                                                          max_eval = typemax(Int64)),
                :DCI_MA57 => nlp -> dci(nlp, nlp.meta.x0, linear_solver = :ma57, 
                                                          max_time = 600., 
                                                          max_iter = typemax(Int64), 
                                                          max_eval = typemax(Int64)))

  list=""; for solver in keys(dsolvers) list=string(list,"_$(solver)") end
  today = "20210224"

  stats = bmark_solvers(dsolvers, cutest_problems)
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
      first_order(df) = df.status .== :first_order
      unbounded(df) = df.status .== :unbounded
      solved(df) = first_order(df) .| unbounded(df)
      cost(df) = (max.(df[col], ϵ) + .!solved(df) .* Inf)
      performance_profile(stats, cost)
      png("$(today)_$(list)_perf-$col")
      #profile_solvers(stats, [cost], ["$(col)"])
      costs = [cost]
      solvers = collect(keys(stats))
      nsolvers = length(solvers)
      npairs = div(nsolvers * (nsolvers - 1), 2)
      colors = get_color_palette(:auto, nsolvers)
      if nsolvers > 2
          ipairs = 0
          # combinations of solvers 2 by 2
          for i = 2 : nsolvers
            for j = 1 : i-1
              ipairs += 1
              pair = [solvers[i], solvers[j]]
              dfs = (stats[solver] for solver in pair)
              Ps = [hcat([cost(df) for df in dfs]...) for cost in costs]

              clrs = [colors[i], colors[j]]
              p = performance_profile(Ps[1], string.(pair), palette=clrs, legend=:bottomright)
              ipairs < npairs && xlabel!(p, "")
              png("$(today)_$(solvers[i])_$(solvers[j])_perf-$col")
            end
          end
      else
      end
    end
  end

  for solver in keys(dsolvers)
    open("data_$(solver).md", "w") do io
          SolverBenchmark.markdown_table(io, stats[solver])
        end
  end

  return stats
end

stats = runcutest()
