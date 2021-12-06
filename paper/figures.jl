using Pkg; Pkg.activate(".")
using JLD2, Plots, SolverBenchmark

@load "ipopt_dcildl_82.jld2" stats
solved(df) = (df.status .== :first_order)

# Number of problems solved by ipopt
@show length(stats[:ipopt][solved(stats[:ipopt]), [:name, :elapsed_time, :status]])
# Number of problems solved by dci
@show length(stats[:dcildl][solved(stats[:dcildl]), [:name, :elapsed_time, :status]])

tim_dci = stats[:dcildl][solved(stats[:dcildl]), :elapsed_time]
tim_ipopt = stats[:ipopt][solved(stats[:dcildl]), :elapsed_time]
# Number of problems where Ipopt is fastest
@show sum(tim_dci .> tim_ipopt) # 20
# Number of problems where DCI is fastest
@show sum(tim_dci .< tim_ipopt) # 51

obj_dci = stats[:dcildl][solved(stats[:dcildl]), :neval_obj]
obj_ipopt = stats[:ipopt][solved(stats[:dcildl]), :neval_obj]
con_dci = stats[:dcildl][solved(stats[:dcildl]), :neval_cons]
con_ipopt = stats[:ipopt][solved(stats[:dcildl]), :neval_cons]
# Number of problems where Ipopt use less evaluations
@show sum((obj_dci .+ con_dci) .> (obj_ipopt .+ con_ipopt)) # 50
# Number of problems where DCI use less evaluations
@show sum((obj_dci .+ con_dci) .< (obj_ipopt .+ con_ipopt)) # 17
# Number where it is a tie
@show sum((obj_dci .+ con_dci) .== (obj_ipopt .+ con_ipopt)) # 4

costs = [
  df -> .!solved(df) * Inf + df.elapsed_time,
  df -> .!solved(df) * Inf + df.neval_obj + df.neval_cons,
]
costnames = ["Time", "Evaluations of obj + cons"]
p = profile_solvers(stats, costs, costnames)
png(p, "ipopt_dcildl_82")
