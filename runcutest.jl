using CUTEst, NLPModels, NLPModelsIpopt, Plots, SolverBenchmark, SolverTools
#This package
using Main.DCI
gr()

function runcutest()
  #pnames = readlines("paper-problems.list")
  #pnames = pnames[1:3]
  pnames = CUTEst.select(max_var=100, min_con=1, max_con=100, only_free_var=true, only_equ_con=true)
  cutest_problems = (CUTEstModel(p) for p in pnames)

  solvers = Dict(:DCI => dci, :ipopt => (nlp; kwargs...) -> ipopt(nlp, print_level=0, kwargs...))
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
end

runcutest()
