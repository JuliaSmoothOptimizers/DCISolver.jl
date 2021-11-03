We are following here the tutorial in [SolverBenchmark.jl](https://juliasmoothoptimizers.github.io/SolverBenchmark.jl/v0.3/tutorial/) to run benchmarks on JSO-compliant solvers.
We compare here the Ipopt via the [NLPModelsIpopt.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl) thin wrapper with DCISolver on a subset of CUTEst problems.

``` @example ex1
using CUTEst, NLPModels, NLPModelsIpopt, SolverBenchmark, SolverCore
#This package
using DCISolver

nmax = 100
_pnames = CUTEst.select(
  max_var = nmax, 
  min_con = 1, 
  max_con = nmax, 
  only_free_var = true, 
  only_equ_con = true, 
  objtype = 3:6
)

#Remove all the problems ending by NE as Ipopt cannot handle them.
pnamesNE = _pnames[findall(x->occursin(r"NE\b", x), _pnames)]
pnames = setdiff(_pnames, pnamesNE)
cutest_problems = (CUTEstModel(p) for p in pnames)

#Same time limit for all the solvers
max_time = 1200. #20 minutes

solvers = Dict(
  :ipopt => nlp -> ipopt(
    nlp,
    print_level = 0,
    dual_inf_tol = Inf,
    constr_viol_tol = Inf,
    compl_inf_tol = Inf,
    acceptable_iter = 0,
    max_cpu_time = max_time,
    x0 = nlp.meta.x0,
  ),
  :dcildl => nlp -> dci(
    nlp,
    nlp.meta.x0,
    linear_solver = :ldlfact,
    max_time = max_time,
    max_iter = typemax(Int64),
    max_eval = typemax(Int64),
  ),
)

stats = bmark_solvers(solvers, cutest_problems)

using JLD2

@save "ipopt_dcildl_$(string(length(pnames))).jld2" stats
```

``` @example ex1
pretty_stats(stats[:dcildl])
```

``` @example ex1
using Plots
gr()

legend = Dict(
  :neval_obj => "number of f evals", 
  :neval_cons => "number of c evals", 
  :neval_grad => "number of ∇f evals", 
  :neval_jac => "number of ∇c evals", 
  :neval_jprod => "number of ∇c*v evals", 
  :neval_jtprod  => "number of ∇cᵀ*v evals", 
  :neval_hess  => "number of ∇²f evals", 
  :elapsed_time => "elapsed time"
)
perf_title(col) = "Performance profile on CUTEst w.r.t. $(string(legend[col]))"

styles = [:solid,:dash,:dot,:dashdot] #[:auto, :solid, :dash, :dot, :dashdot, :dashdotdot]

function print_pp_column(col::Symbol, stats)
  
  ϵ = minimum(minimum(filter(x -> x > 0, df[!, col])) for df in values(stats))
  first_order(df) = df.status .== :first_order
  unbounded(df) = df.status .== :unbounded
  solved(df) = first_order(df) .| unbounded(df)
  cost(df) = (max.(df[!, col], ϵ) + .!solved(df) .* Inf)

  p = performance_profile(
    stats, 
    cost, 
    title=perf_title(col), 
    legend=:bottomright, 
    linestyles=styles
  )
end

print_pp_column(:elapsed_time, stats)
```