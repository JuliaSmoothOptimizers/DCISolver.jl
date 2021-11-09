using Dates, JLD2, SolverCore, SolverBenchmark
using Plots

@load "script/ipopt_knitro_dcildl_82.jld2" stats
solved(df) = (df.status .== :first_order)
costs = [df -> .!solved(df) * Inf + df.elapsed_time, df -> .!solved(df) * Inf + df.iter]
costnames = ["Time", "Iterations"]
p = profile_solvers(stats, costs, costnames)
png(p, "ipopt_knitro_dcildl_82")
