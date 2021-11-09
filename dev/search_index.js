var documenterSearchIndex = {"docs":
[{"location":"fine-tuneDCI/#Advanced-usage-of-DCI","page":"Fine-tune DCI","title":"Advanced-usage of DCI","text":"","category":"section"},{"location":"fine-tuneDCI/#Contents","page":"Fine-tune DCI","title":"Contents","text":"","category":"section"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"Pages = [\"fine-tuneDCI.md\"]","category":"page"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"The main function exported by this package is the function dci whose basic usage has been illustrated previously. It is also possible to fine-tune the parameters used in the implementation in two different ways.","category":"page"},{"location":"fine-tuneDCI/#Examples","page":"Fine-tune DCI","title":"Examples","text":"","category":"section"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"DCISolver.jl exports the function dci:","category":"page"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"   dci(nlp :: AbstractNLPModel)\n   dci(nlp :: AbstractNLPModel, x :: AbstractVector)\n   dci(nlp :: AbstractNLPModel, meta :: MetaDCI, x :: AbstractVector)\n   dci(nlp :: AbstractNLPModel, meta :: MetaDCI, workspace :: DCIWorkspace)","category":"page"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"where MetaDCI is a structure handling all the parameters used in the algorithm, and DCIWorkspace pre-allocates all the memory used during the iterative process.","category":"page"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"It is therefore possible to either call dci(nlp, x, kwargs...) and the keywords arguments are passed to the MetaDCI constructor or build an instance of MetaDCI directly.","category":"page"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"using ADNLPModels, DCISolver\n\nnlp = ADNLPModel(\n  x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, \n  [-1.2; 1.0],\n  x->[x[1] * x[2] - 1], \n  [0.0], [0.0],\n  name = \"Rosenbrock with x₁x₂=1\"\n)\n\n#The alternative would be:\nstats = dci(\n  nlp, nlp.meta.x0, \n  max_time = 600., \n  linear_solver = :ldlfact, \n  TR_compute_step = :TR_lsmr\n)","category":"page"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"The alternative would be:","category":"page"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"meta = DCISolver.MetaDCI(\n  nlp.meta.x0, nlp.meta.y0, \n  max_time = 600., \n  linear_solver = :ldlfact, \n  TR_compute_step = :TR_lsmr\n)\nstats = dci(nlp, meta, nlp.meta.x0)","category":"page"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"The DCIWorkspace allows to reuse the same memory if one would re-solve a problem of the same dimension.","category":"page"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"workspace = DCISolver.DCIWorkspace(nlp, meta, nlp.meta.x0)\nstats = dci(nlp, meta, workspace)\nworspace.x0 .= ones(2) # change the initial guess, and resolve\nstats = dci(nlp, meta, workspace)","category":"page"},{"location":"fine-tuneDCI/#List-of-possible-options","page":"Fine-tune DCI","title":"List of possible options","text":"","category":"section"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"Find below a list of the main options of dci.","category":"page"},{"location":"fine-tuneDCI/#Tolerances-on-the-problem","page":"Fine-tune DCI","title":"Tolerances on the problem","text":"","category":"section"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"We use ϵ = atol + rtol * dualnorm.","category":"page"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"| Parameters           | Type          | Default      | Description                                    |\n| -------------------- | ------------- | ------------ | ---------------------------------------------- |\n| atol                 | AbstractFloat | 1e-5         | absolute tolerance.                            |\n| rtol                 | AbstractFloat | 1e-5         | relative tolerance.                            |\n| ctol                 | AbstractFloat | 1e-5         | feasibility tolerance.                         |\n| unbounded_threshold  | AbstractFloat | -1e5         | below this threshold the problem is unbounded. |\n| max_eval             | Integer       | 50000        | maximum number of cons + obj evaluations.      |\n| max_time             | AbstractFloat | 120.         | maximum number of seconds.                     |\n| max_iter             | Integer       | 500          | maximum number of iterations.                  |\n| max_iter_normal_step | Integer       | typemax(Int) | maximum number of iterations in normal step.   |","category":"page"},{"location":"fine-tuneDCI/#Compute-Lagrange-multipliers","page":"Fine-tune DCI","title":"Compute Lagrange multipliers","text":"","category":"section"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"| Parameters  | Type        | Default                                         | Description                                           |\n| ----------- | ----------- | ----------------------------------------------- | ----------------------------------------------------- |\n| comp_λ      | Symbol      | :cgls                                           | eval(comp_λ) is used to compute Lagrange multipliers. |\n| λ_struct    | comp_λ_cgls | comp_λ_cgls(length(x0), length(y0), typeof(x0)) | companion structure of `comp_λ`.                      |","category":"page"},{"location":"fine-tuneDCI/#Tangent-step","page":"Fine-tune DCI","title":"Tangent step","text":"","category":"section"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"| Parameters    | Type          | Default  | Description                                                                                               |\n| ------------- | ------------- | -------- | --------------------------------------------------------------------------------------------------------- |\n| linear_solver | Symbol        | :ldlfact | Solver for the factorization. options: :ma57.                                                             | \n| decrease_γ    | AbstractFloat | 0.1      | Regularization for the factorization: reduce γ if possible, > √eps(T), between tangent steps.             |\n| increase_γ    | AbstractFloat | 100.0    | Regularization for the factorization: up γ if possible, < 1/√eps(T), during the factorization.            |\n| δmin          | AbstractFloat | √eps(T)  | Regularization for the factorization: smallest value of δ used for the regularization.                    |\n| tan_Δ         | AbstractFloat | 1.0      | Tangent step trust-region parameters: initial trust-region radius.                                        |\n| tan_η₁        | AbstractFloat | 1e-2     | Tangent step trust-region parameters: decrease the trust-region radius when Ared/Pred < η₁.               |\n| tan_η₂        | AbstractFloat | 0.75     | Tangent step trust-region parameters: increase the trust-region radius when Ared/Pred > η₂.               |\n| tan_σ₁        | AbstractFloat | 0.25     | Tangent step trust-region parameters: decrease coefficient of the trust-region radius.                    |\n| tan_σ₂        | AbstractFloat | 2.0      | Tangent step trust-region parameters: increase coefficient of the trust-region radius.                    |\n| tan_small_d   | AbstractFloat | eps(T)   | Tangent step trust-region parameters: ||d|| is too small.                                                 |\n| increase_Δtg  | AbstractFloat | 10.0     | Tangent step trust-region parameters: increase if possible, < 1 / √eps(T), the Δtg between tangent steps. |","category":"page"},{"location":"fine-tuneDCI/#Normal-step","page":"Fine-tune DCI","title":"Normal step","text":"","category":"section"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"| Parameters             | Type                                    | Default                                            | Description                                                                                               |\n| ---------------------- | --------------------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |\n| feas_step              | Symbol                                  | :feasibility_step                                  | Normal step                                                                                               |\n| feas_η₁                | AbstractFloat                           | 1e-3                                               | Feasibility step: decrease the trust-region radius when Ared/Pred < η₁.                                   |\n| feas_η₂                | AbstractFloat                           | 0.66                                               | Feasibility step: increase the trust-region radius when Ared/Pred > η₂.                                   |\n| feas_σ₁                | AbstractFloat                           | 0.25                                               | Feasibility step: decrease coefficient of the trust-region radius.                                        |\n| feas_σ₂                | AbstractFloat                           | 2.0                                                | Feasibility step: increase coefficient of the trust-region radius.                                        |\n| feas_Δ₀                | AbstractFloat                           | 1.0                                                | Feasibility step: initial radius.                                                                         |\n| feas_expected_decrease | AbstractFloat                           | 0.95                                               | Feasibility step: bad steps are when ‖c(z)‖ / ‖c(x)‖ >feas_expected_decrease.                             |\n| bad_steps_lim          | Integer                                 | 3                                                  | Feasibility step: consecutive bad steps before using a second order step.                                 |\n| TR_compute_step        | Symbol                                  | :TR_lsmr                                           | Compute the direction in feasibility step: options: :TR_dogleg.                                           |\n| TR_compute_step_struct | Union{TR_lsmr_struct, TR_dogleg_struct} | TR_lsmr_struct(length(x0), length(y0), typeof(x0)) | Compute the direction in feasibility step: options: TR_dogleg_struct(length(x0), length(y0), typeof(x0)). |","category":"page"},{"location":"fine-tuneDCI/#Parameters-updating-ρ-(or-redefine-the-function-compute_ρ)","page":"Fine-tune DCI","title":"Parameters updating ρ (or redefine the function compute_ρ)","text":"","category":"section"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"| Parameters  | Type          | Default | Description                                                     |\n| ----------- | ------------- | ------- | --------------------------------------------------------------- |\n| compρ_p1    | AbstractFloat | 0.75    | update ρ as `ρ = max(min(ngp, p1) * ρmax, ϵ)`.                  |\n| compρ_p2    | AbstractFloat | 0.90    | update ρ as `ρ = primalnorm * p2` if not sufficiently feasible. |\n| ρbar        | AbstractFloat | 2.0     | radius of the larger cylinder is `ρbar * ρ`.                    |","category":"page"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"The computation of ρ can also be modified by importing compute_ρ(dualnorm, primalnorm, norm∇fx, ρmax, ϵ, iter, meta::MetaDCI)","category":"page"},{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Contents","page":"Reference","title":"Contents","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [DCISolver]","category":"page"},{"location":"reference/#DCISolver.SymCOOSolver","page":"Reference","title":"DCISolver.SymCOOSolver","text":"An SymCOOSolver is an interface to allow simple usage of different solvers. Ideally, having rows, cols, vals and the dimension ndim of a symmetric matrix should allow the user to call     M = LinearSolver(ndim, rows, cols, vals)     factorize!(M)     solve!(x, M, b) # Also x = M \\ b Only the lower triangle of the matrix should be passed.\n\n\n\n\n\n","category":"type"},{"location":"reference/#DCISolver.comp_λ_cgls","page":"Reference","title":"DCISolver.comp_λ_cgls","text":"comp_λ_cgls{<: AbstractFloat} attributes correspond to input parameters of cgls used in the computation of Lagrange multipliers.\n\n\n\n\n\n","category":"type"},{"location":"reference/#Base.success","page":"Reference","title":"Base.success","text":"success(M :: SymCOOSolver)\n\nReturns whether factorize!(M) was successful.\n\n\n\n\n\n","category":"function"},{"location":"reference/#DCISolver.TR_dogleg-Union{Tuple{T}, Tuple{AbstractVector{T}, Any, AbstractFloat, T, AbstractFloat, AbstractVector{T}, DCISolver.TR_dogleg_struct}} where T","page":"Reference","title":"DCISolver.TR_dogleg","text":"feasibility_step(nls, x, cx, Jx)\n\nCompute a direction d such that min ‖cₖ + Jₖd‖ s.t. ||d|| ≤ Δ using a dogleg.\n\nAlso checks if problem is infeasible.\n\nReturns 4 entries: (d, Jd, solved, infeasible)\n\n\n\n\n\n","category":"method"},{"location":"reference/#DCISolver._compute_gradient_step!-Union{Tuple{T}, Tuple{NLPModels.AbstractNLPModel, T, AbstractVector{T}, T, AbstractVector{T}}} where T","page":"Reference","title":"DCISolver._compute_gradient_step!","text":"Compute a solution to minα q(-α g) s.t. ‖αg‖2 ≤ Δ\n\nreturn dcp_on_boundary true if ‖αg‖ = Δ, return dcp = - α g return dcpBdcp = α^2 gBg and α the solution.\n\n\n\n\n\n","category":"method"},{"location":"reference/#DCISolver._compute_newton_step!-Union{Tuple{T}, Tuple{NLPModels.AbstractNLPModel, DCISolver.SymCOOSolver, AbstractVector{T}, T, T, AbstractVector{T}, AbstractVector{T}, DCISolver.MetaDCI, Any}} where T","page":"Reference","title":"DCISolver._compute_newton_step!","text":"1st idea: we stop whenever the fonction looks convex-ish\n\n\n\n\n\n","category":"method"},{"location":"reference/#DCISolver._compute_step_length-Union{Tuple{T}, NTuple{4, T}} where T<:AbstractFloat","page":"Reference","title":"DCISolver._compute_step_length","text":"Given two directions dcp and dn, compute the largest 0 ≤ τ ≤ 1 such that ‖dn + τ (dcp -dn)‖ = Δ\n\n\n\n\n\n","category":"method"},{"location":"reference/#DCISolver.compute_descent_direction!-Union{Tuple{T}, Tuple{NLPModels.AbstractNLPModel, T, AbstractVector{T}, T, DCISolver.SymCOOSolver, T, T, AbstractVector{T}, AbstractVector{T}, DCISolver.MetaDCI, DCISolver.DCIWorkspace}} where T","page":"Reference","title":"DCISolver.compute_descent_direction!","text":"Compute a direction d with three possible outcomes:\n\n:cauchy_step\n:newton\n:dogleg\n:interior_cauchy_step when γ is too large.\n\nfor min_d q(d) s.t. ‖d‖ ≤ Δ.\n\n\n\n\n\n","category":"method"},{"location":"reference/#DCISolver.compute_gBg-Union{Tuple{T}, Tuple{NLPModels.AbstractNLPModel, AbstractVector{T} where T, AbstractVector{T} where T, AbstractVector{T}, AbstractVector{T}}} where T","page":"Reference","title":"DCISolver.compute_gBg","text":"compute_gBg   B is a symmetric sparse matrix   whose lower triangular given in COO: (rows, cols, vals)\n\nCompute ∇ℓxλ' * B * ∇ℓxλ\n\n\n\n\n\n","category":"method"},{"location":"reference/#DCISolver.compute_lx!-Union{Tuple{T}, Tuple{Any, AbstractVector{T}, AbstractVector{T}, DCISolver.MetaDCI}} where T<:AbstractFloat","page":"Reference","title":"DCISolver.compute_lx!","text":"Compute the solution of ‖Jx' λ - ∇fx‖\n\n\n\n\n\n","category":"method"},{"location":"reference/#DCISolver.dci-Union{Tuple{T}, Tuple{NLPModels.AbstractNLPModel, AbstractVector{T}}} where T","page":"Reference","title":"DCISolver.dci","text":"dci(nlp, x; kwargs...)\n\nThis method implements the Dynamic Control of Infeasibility for equality-constrained problems described in\n\nDynamic Control of Infeasibility in Equality Constrained Optimization\nRoberto H. Bielschowsky and Francisco A. M. Gomes\nSIAM J. Optim., 19(3), 1299–1325.\nhttps://doi.org/10.1137/070679557\n\n\n\n\n\n","category":"method"},{"location":"reference/#DCISolver.factorize!","page":"Reference","title":"DCISolver.factorize!","text":"factorize!(M :: SymCOOSolver)\n\nCalls the factorization of the symmetric solver given by M. Use success(M) to check whether the factorization was successful.\n\n\n\n\n\n","category":"function"},{"location":"reference/#DCISolver.feasibility_step-Union{Tuple{T}, Tuple{NLPModels.AbstractNLPModel, AbstractVector{T}, AbstractVector{T}, T, Any, T, AbstractFloat, DCISolver.MetaDCI, Any}} where T","page":"Reference","title":"DCISolver.feasibility_step","text":"feasibility_step(nls, x, cx, Jx)\n\nApproximately solves min ‖c(x)‖.\n\nGiven xₖ, finds min ‖cₖ + Jₖd‖\n\n\n\n\n\n","category":"method"},{"location":"reference/#DCISolver.num_neg_eig","page":"Reference","title":"DCISolver.num_neg_eig","text":"num_neg_eig(M :: SymCOOSolver)\n\nReturns the number of negative eigenvalues of M.\n\n\n\n\n\n","category":"function"},{"location":"reference/#DCISolver.regularized_coo_saddle_system!-Union{Tuple{T}, Tuple{S}, Tuple{NLPModels.AbstractNLPModel, AbstractVector{S}, AbstractVector{S}, AbstractVector{T}}} where {S<:Int64, T<:AbstractFloat}","page":"Reference","title":"DCISolver.regularized_coo_saddle_system!","text":"regularized_coo_saddle_system!(nlp, rows, cols, vals, γ = γ, δ = δ)   Compute the structure for the saddle system [H + γI  [Jᵀ]; J -δI]   in COO-format in the following order:   H J γ -δ\n\n\n\n\n\n","category":"method"},{"location":"reference/#DCISolver.solve!","page":"Reference","title":"DCISolver.solve!","text":"solve!(x, M :: SymCOOSolver, b)\n\nSolve the system M x = b. factorize!(M) should be called first.\n\n\n\n\n\n","category":"function"},{"location":"reference/#DCISolver.tangent_step!-Union{Tuple{T}, Tuple{NLPModels.AbstractNLPModel, AbstractVector{T}, AbstractVector{T}, AbstractVector{T}, T, T, DCISolver.SymCOOSolver, AbstractVector{T}, AbstractVector{T}, T, T, AbstractFloat, T, T, DCISolver.MetaDCI, DCISolver.DCIWorkspace}} where T","page":"Reference","title":"DCISolver.tangent_step!","text":"min q(d):=¹/₂dᵀBd + dᵀg s.t Ad = 0     ‖d‖ ≦ Δ where B is an approximation of hessian of the Lagrangian, A is the jacobian matrix and g is the projected gradient.\n\nReturn status with outcomes:\n\n:cauchy_step, :newton, :dogleg,\n:unknown if we didn't enter the loop.\n:smallhorizontalstep\n:tired if we stop due to maxeval or maxtime\n:success if we computed z such that ‖c(z)‖ ≤ meta.ρbar * ρ and Δℓ ≥ η₁ q(d)\n\nSee https://github.com/JuliaSmoothOptimizers/SolverTools.jl/blob/78f6793f161c3aac2234aee8a27aa07f1df3e8ee/src/trust-region/trust-region.jl#L37 for SolverTools.aredpred\n\n\n\n\n\n","category":"method"},{"location":"benchmark/#Benchmarks","page":"Benchmark","title":"Benchmarks","text":"","category":"section"},{"location":"benchmark/#CUTEst-benchmark","page":"Benchmark","title":"CUTEst benchmark","text":"","category":"section"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"With a JSO-compliant solver, such as DCI, we can run the solver on a set of problems, explore the results, and compare to other JSO-compliant solvers using specialized benchmark tools.  We are following here the tutorial in SolverBenchmark.jl to run benchmarks on JSO-compliant solvers.","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"using CUTEst","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"To test the implementation of DCI, we use the package CUTEst.jl, which implements CUTEstModel an instance of AbstractNLPModel. ","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"using SolverBenchmark","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"Let us select equality-constrained problems from CUTEst with a maximum of 100 variables or constraints. After removing problems with fixed variables, examples with a constant objective, and infeasibility residuals.","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"_pnames = CUTEst.select(\n  max_var = 100, \n  min_con = 1, \n  max_con = 100, \n  only_free_var = true, \n  only_equ_con = true, \n  objtype = 3:6\n)\n\n#Remove all the problems ending by NE as Ipopt cannot handle them.\npnamesNE = _pnames[findall(x->occursin(r\"NE\\b\", x), _pnames)]\npnames = setdiff(_pnames, pnamesNE)\ncutest_problems = (CUTEstModel(p) for p in pnames)\n\nlength(cutest_problems) # number of problems","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"We compare here DCISolver with Ipopt (Wächter, A., & Biegler, L. T. (2006). On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming. Mathematical programming, 106(1), 25-57.), via the NLPModelsIpopt.jl thin wrapper, with DCISolver on a subset of CUTEst problems.","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"using DCISolver, NLPModelsIpopt","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"To make stopping conditions comparable, we set Ipopt's parameters dual_inf_tol=Inf, constr_viol_tol=Inf and compl_inf_tol=Inf to disable additional stopping conditions related to those tolerances, acceptable_iter=0 to disable the search for an acceptable point.","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"#Same time limit for all the solvers\nmax_time = 1200. #20 minutes\ntol = 1e-5\n\nsolvers = Dict(\n  :ipopt => nlp -> ipopt(\n    nlp,\n    print_level = 0,\n    dual_inf_tol = Inf,\n    constr_viol_tol = Inf,\n    compl_inf_tol = Inf,\n    acceptable_iter = 0,\n    max_cpu_time = max_time,\n    x0 = nlp.meta.x0,\n    tol = tol,\n  ),\n  :dcildl => nlp -> dci(\n    nlp,\n    nlp.meta.x0,\n    linear_solver = :ldlfact,\n    max_time = max_time,\n    max_iter = typemax(Int64),\n    max_eval = typemax(Int64),\n    atol = tol,\n    ctol = tol,\n    rtol = tol,\n  ),\n)\n\nstats = bmark_solvers(solvers, cutest_problems)","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"The function bmark_solvers return a Dict of DataFrames with detailed information on the execution. This output can be saved in a data file.","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"using JLD2\n@save \"ipopt_dcildl_$(string(length(pnames))).jld2\" stats","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"The result of the benchmark can be explored via tables,","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"pretty_stats(stats[:dcildl])","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"or it can also be used to make performance profiles.","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"using Plots\ngr()\n\nlegend = Dict(\n  :neval_obj => \"number of f evals\", \n  :neval_cons => \"number of c evals\", \n  :neval_grad => \"number of ∇f evals\", \n  :neval_jac => \"number of ∇c evals\", \n  :neval_jprod => \"number of ∇c*v evals\", \n  :neval_jtprod  => \"number of ∇cᵀ*v evals\", \n  :neval_hess  => \"number of ∇²f evals\", \n  :elapsed_time => \"elapsed time\"\n)\nperf_title(col) = \"Performance profile on CUTEst w.r.t. $(string(legend[col]))\"\n\nstyles = [:solid,:dash,:dot,:dashdot] #[:auto, :solid, :dash, :dot, :dashdot, :dashdotdot]\n\nfunction print_pp_column(col::Symbol, stats)\n  \n  ϵ = minimum(minimum(filter(x -> x > 0, df[!, col])) for df in values(stats))\n  first_order(df) = df.status .== :first_order\n  unbounded(df) = df.status .== :unbounded\n  solved(df) = first_order(df) .| unbounded(df)\n  cost(df) = (max.(df[!, col], ϵ) + .!solved(df) .* Inf)\n\n  p = performance_profile(\n    stats, \n    cost, \n    title=perf_title(col), \n    legend=:bottomright, \n    linestyles=styles\n  )\nend\n\nprint_pp_column(:elapsed_time, stats) # with respect to time","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"print_pp_column(:neval_jac, stats) # with respect to number of jacobian evaluations","category":"page"},{"location":"benchmark/#CUTEst-benchmark-with-Knitro","page":"Benchmark","title":"CUTEst benchmark with Knitro","text":"","category":"section"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"In this second part, we present the result of a similar benchmark with a maximum of 10000 variables and constraints (82 problems), and including the solver KNITRO (Byrd, R. H., Nocedal, J., & Waltz, R. A. (2006). K nitro: An integrated package for nonlinear optimization. In Large-scale nonlinear optimization (pp. 35-59). Springer, Boston, MA.) via NLPModelsKnitro.jl. The script is included in /docs/script/script10000_knitro.jl). We report here a performance profile with respect to the elapsed time to solve the problems.","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"(Image: )","category":"page"},{"location":"#DCISolver-Dynamic-Control-of-Infeasibility-Solver","page":"Introduction","title":"DCISolver - Dynamic Control of Infeasibility Solver","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"DCI is a solver for equality-constrained nonlinear problems, i.e., optimization problems of the form","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"    min_x  f(x) quad textst quad  c(x) = 0","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"based on the paper","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Bielschowsky, R. H., & Gomes, F. A. Dynamic control of infeasibility in equality constrained optimization. SIAM Journal on Optimization, 19(3), 1299-1325 (2008). 10.1007/s10589-020-00201-2","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"DCISolver is a JuliaSmoothOptimizers-compliant solver. It takes an AbstractNLPModel as an input and returns a GenericExecutionStats.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"We refer to juliasmoothoptimizers.github.io for tutorials on the NLPModel API. This framework allows the usage of models from Ampl (using AmplNLReader.jl), CUTEst (using CUTEst.jl), JuMP (using NLPModelsJuMP.jl), PDE-constrained optimization problems (using PDENLPModels.jl) and models defined with automatic differentiation (using ADNLPModels.jl).","category":"page"},{"location":"#Installation","page":"Introduction","title":"Installation","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"DCISolver is a registered package. To install this package, open the Julia REPL (i.e., execute the julia binary), type ] to enter package mode, and install DCISolver as follows","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"add DCISolver","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"The DCI algorithm is an iterative method that has the flavor of a projected gradient algorithm and could be characterized as a relaxed feasible point method with dynamic control of infeasibility. It is a combination of two steps: a tangent step and a feasibility step. It uses LDLFactorizations.jl by default to compute the factorization in the tangent step. Follow HSL.jl's MA57 installation for an alternative. The feasibility steps are factorization-free and use iterative methods from Krylov.jl.","category":"page"},{"location":"#Example","page":"Introduction","title":"Example","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"We consider in this example the minization of the Rosenbrock function over an equality constraint.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"    min_x  100 * (x₂ - x₁²)² + (x₁ - 1)² quad textst quad  x₁x₂=1","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"The problem is modeled using ADNLPModels.jl with [-1.2; 1.0] as default initial point, and then solved using dci.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"using DCISolver, ADNLPModels, Logging\nnlp = ADNLPModel(\n  x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, \n  [-1.2; 1.0],\n  x -> [x[1] * x[2] - 1], \n  [0.0], [0.0],\n  name = \"Rosenbrock with x₁x₂=1\"\n)\nstats = with_logger(NullLogger()) do\n  dci(nlp)\nend\n\nprintln(stats)","category":"page"},{"location":"#Bug-reports-and-discussions","page":"Introduction","title":"Bug reports and discussions","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"If you think you found a bug, feel free to open an issue. Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"If you want to ask a question not suited for a bug report, feel free to start a discussion here. This forum is for general discussion about this repository and the JuliaSmoothOptimizers, so questions about any of our packages are welcome.","category":"page"}]
}
