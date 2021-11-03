var documenterSearchIndex = {"docs":
[{"location":"fine-tuneDCI/#Advanced-usage-of-DCI","page":"Fine-tune DCI","title":"Advanced-usage of DCI","text":"","category":"section"},{"location":"fine-tuneDCI/#Contents","page":"Fine-tune DCI","title":"Contents","text":"","category":"section"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"Pages = [\"fine-tuneDCI.md\"]","category":"page"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"The main function exported by this package is the function dci whose basic usage has been illustrated previously. It is also possible to fine-tune the parameters used in the implementation in two different ways.","category":"page"},{"location":"fine-tuneDCI/#Examples","page":"Fine-tune DCI","title":"Examples","text":"","category":"section"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"DCISolver.jl exports the function dci:","category":"page"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"dci(nlp :: AbstractNLPModel, x :: AbstractVector{T}, meta :: MetaDCI) where T","category":"page"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"where MetaDCI is a structure handling all the parameters used in the algorithm.","category":"page"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"It is therefore possible to either call dci(nlp, x, kwargs...) and the keywords arguments are passed to the MetaDCI constructor or build an instance of MetaDCI directly.","category":"page"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"using ADNLPModels, DCISolver\n\nnlp = ADNLPModel(\n  x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, \n  [-1.2; 1.0],\n  x->[x[1] * x[2] - 1], \n  [0.0], [0.0],\n  name = \"Rosenbrock with x₁x₂=1\"\n)\n\nm, n = nlp.meta.ncon, nlp.meta.nvar\nmeta = DCISolver.MetaDCI(\n  nlp.meta.x0, nlp.meta.y0, \n  max_time = 600., \n  linear_solver = :ldlfact, \n  TR_compute_step = :TR_lsmr\n)\nworkspace = DCISolver.DCIWorkspace(nlp, meta, nlp.meta.x0)\nstats = dci(nlp, meta, workspace)\n\n#The alternative would be:\nstats2 = dci(\n  nlp, nlp.meta.x0, \n  max_time = 600., \n  linear_solver = :ldlfact, \n  TR_compute_step = :TR_lsmr\n)","category":"page"},{"location":"fine-tuneDCI/#List-of-possible-options","page":"Fine-tune DCI","title":"List of possible options","text":"","category":"section"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"Find below a list of the main options of dci:","category":"page"},{"location":"fine-tuneDCI/","page":"Fine-tune DCI","title":"Fine-tune DCI","text":"# Tolerances on the problem: in general, we use `ϵ = atol + rtol * dualnorm`\natol :: AbstractFloat # default: 1e-5 ; absolute tolerance.\nrtol :: AbstractFloat # default: 1e-5 ; relative tolerance.\nctol :: AbstractFloat # default: 1e-5 ; feasibility tolerance.\n\nunbounded_threshold :: AbstractFloat # default: -1e5 ; below this threshold the problem is unbounded.\n\n# Evaluation limits\nmax_eval :: Integer # default: 50000 ; maximum number of cons + obj evaluations.\nmax_time :: AbstractFloat # default: 120 ; maximum number of seconds.\nmax_iter :: Integer # default: 500 ; maximum number of iterations.\nmax_iter_normal_step :: Integer # default: typemax(Int) ; maximum number of iterations in normal step.\n\n# Compute Lagrange multipliers\ncomp_λ :: Symbol # default: :cgls ; eval(comp_λ) is used to compute Lagrange multipliers.\nλ_struct :: comp_λ_cgls # default: comp_λ_cgls(length(x0), length(y0), typeof(x0)) ; companion structure of `comp_λ`.\n   \n# Tangent step\n## Solver for the factorization\nlinear_solver :: Symbol # default: :ldlfact, options: :ma57.\n## Regularization for the factorization\ndecrease_γ :: AbstractFloat # default: 0.1 ; reduce γ if possible, > √eps(T), between tangent steps.\nincrease_γ :: AbstractFloat # default: 100.0 ; up γ if possible, < 1/√eps(T), during the factorization.\nδmin :: AbstractFloat # default: √eps(T) ; smallest value of δ used for the regularization.\n## Tangent step trust-region parameters\ntan_Δ :: AbstractFloat # default: 1.0 ; initial trust-region radius.\ntan_η₁ :: AbstractFloat # default: 1e-2 ; decrease the trust-region radius when Ared/Pred < η₁.\ntan_η₂ :: AbstractFloat # default: 0.75 ; increase the trust-region radius when Ared/Pred > η₂.\ntan_σ₁ :: AbstractFloat # default: 0.25 ; decrease coefficient of the trust-region radius.\ntan_σ₂ :: AbstractFloat # default: 2.0 ; increase coefficient of the trust-region radius.\ntan_small_d :: AbstractFloat # default: eps(T) ; ||d|| is too small.\nincrease_Δtg :: AbstractFloat # default: 10.0 ; increase if possible, < 1 / √eps(T), the Δtg between tangent steps.\n\n# Normal step\nfeas_step :: Symbol # default: :feasibility_step\n## Feasibility step\nfeas_η₁ :: AbstractFloat # default: 1e-3 ; decrease the trust-region radius when Ared/Pred < η₁.\nfeas_η₂ :: AbstractFloat # default: 0.66 ; increase the trust-region radius when Ared/Pred > η₂.\nfeas_σ₁ :: AbstractFloat # default: 0.25 ; decrease coefficient of the trust-region radius.\nfeas_σ₂ :: AbstractFloat # default: 2.0 ; increase coefficient of the trust-region radius.\nfeas_Δ₀ :: AbstractFloat # default: 1.0 ; initial radius.\nfeas_expected_decrease :: AbstractFloat # default: 0.95 ; bad steps are when ‖c(z)‖ / ‖c(x)‖ >feas_expected_decrease.\nbad_steps_lim :: Integer # default: 3 ; consecutive bad steps before using a second order step.\n## Compute the direction in feasibility step\nTR_compute_step :: Symbol # default: :TR_lsmr, options: :TR_dogleg.\nTR_compute_step_struct :: Union{TR_lsmr_struct, TR_dogleg_struct} # default: TR_lsmr_struct(length(x0), length(y0), typeof(x0)), options: TR_dogleg_struct(length(x0), length(y0), typeof(x0)).\n\n# Parameters updating ρ (or redefine the function `compute_ρ`)\ncompρ_p1 :: AbstractFloat # default: 0.75 ; update ρ as `ρ = max(min(ngp, p1) * ρmax, ϵ)`.\ncompρ_p2 :: AbstractFloat # default: 0.90 ; update ρ as `ρ = primalnorm * p2` if not sufficiently feasible.\nρbar :: AbstractFloat # default: 2.0 ; radius of the larger cylinder is `ρbar * ρ`.\n#Computation of ρ can be modified by importing `compute_ρ(dualnorm, primalnorm, norm∇fx, ρmax, ϵ, iter, meta::MetaDCI)`","category":"page"},{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Contents","page":"Reference","title":"Contents","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [DCISolver]","category":"page"},{"location":"reference/#DCISolver.SymCOOSolver","page":"Reference","title":"DCISolver.SymCOOSolver","text":"An SymCOOSolver is an interface to allow simple usage of different solvers. Ideally, having rows, cols, vals and the dimension ndim of a symmetric matrix should allow the user to call     M = LinearSolver(ndim, rows, cols, vals)     factorize!(M)     solve!(x, M, b) # Also x = M \\ b Only the lower triangle of the matrix should be passed.\n\n\n\n\n\n","category":"type"},{"location":"reference/#DCISolver.comp_λ_cgls","page":"Reference","title":"DCISolver.comp_λ_cgls","text":"comp_λ_cgls{<: AbstractFloat} attributes correspond to input parameters of cgls used in the computation of Lagrange multipliers.\n\n\n\n\n\n","category":"type"},{"location":"reference/#Base.success","page":"Reference","title":"Base.success","text":"success(M :: SymCOOSolver)\n\nReturns whether factorize!(M) was successful.\n\n\n\n\n\n","category":"function"},{"location":"reference/#DCISolver.TR_dogleg-Union{Tuple{T}, Tuple{AbstractVector{T}, Any, AbstractFloat, T, AbstractFloat, AbstractVector{T}, DCISolver.TR_dogleg_struct}} where T","page":"Reference","title":"DCISolver.TR_dogleg","text":"feasibility_step(nls, x, cx, Jx)\n\nCompute a direction d such that min ‖cₖ + Jₖd‖ s.t. ||d|| ≤ Δ using a dogleg.\n\nAlso checks if problem is infeasible.\n\nReturns 4 entries: (d, Jd, solved, infeasible)\n\n\n\n\n\n","category":"method"},{"location":"reference/#DCISolver._compute_gradient_step!-Union{Tuple{T}, Tuple{NLPModels.AbstractNLPModel, T, AbstractVector{T}, T, AbstractVector{T}}} where T","page":"Reference","title":"DCISolver._compute_gradient_step!","text":"Compute a solution to minα q(-α g) s.t. ‖αg‖2 ≤ Δ\n\nreturn dcp_on_boundary true if ‖αg‖ = Δ, return dcp = - α g return dcpBdcp = α^2 gBg and α the solution.\n\n\n\n\n\n","category":"method"},{"location":"reference/#DCISolver._compute_newton_step!-Union{Tuple{T}, Tuple{NLPModels.AbstractNLPModel, DCISolver.SymCOOSolver, AbstractVector{T}, T, T, AbstractVector{T}, AbstractVector{T}, DCISolver.MetaDCI, Any}} where T","page":"Reference","title":"DCISolver._compute_newton_step!","text":"1st idea: we stop whenever the fonction looks convex-ish\n\n\n\n\n\n","category":"method"},{"location":"reference/#DCISolver._compute_step_length-Union{Tuple{T}, NTuple{4, T}} where T<:AbstractFloat","page":"Reference","title":"DCISolver._compute_step_length","text":"Given two directions dcp and dn, compute the largest 0 ≤ τ ≤ 1 such that ‖dn + τ (dcp -dn)‖ = Δ\n\n\n\n\n\n","category":"method"},{"location":"reference/#DCISolver.compute_descent_direction!-Union{Tuple{T}, Tuple{NLPModels.AbstractNLPModel, T, AbstractVector{T}, T, DCISolver.SymCOOSolver, T, T, AbstractVector{T}, AbstractVector{T}, DCISolver.MetaDCI, DCISolver.DCIWorkspace}} where T","page":"Reference","title":"DCISolver.compute_descent_direction!","text":"Compute a direction d with three possible outcomes:\n\n:cauchy_step\n:newton\n:dogleg\n:interior_cauchy_step when γ is too large.\n\nfor min_d q(d) s.t. ‖d‖ ≤ Δ.\n\n\n\n\n\n","category":"method"},{"location":"reference/#DCISolver.compute_gBg-Union{Tuple{T}, Tuple{NLPModels.AbstractNLPModel, AbstractVector{T} where T, AbstractVector{T} where T, AbstractVector{T}, AbstractVector{T}}} where T","page":"Reference","title":"DCISolver.compute_gBg","text":"compute_gBg   B is a symmetric sparse matrix   whose lower triangular given in COO: (rows, cols, vals)\n\nCompute ∇ℓxλ' * B * ∇ℓxλ\n\n\n\n\n\n","category":"method"},{"location":"reference/#DCISolver.compute_lx!-Union{Tuple{T}, Tuple{Any, AbstractVector{T}, AbstractVector{T}, DCISolver.MetaDCI}} where T<:AbstractFloat","page":"Reference","title":"DCISolver.compute_lx!","text":"Compute the solution of ‖Jx' λ - ∇fx‖\n\n\n\n\n\n","category":"method"},{"location":"reference/#DCISolver.dci-Union{Tuple{T}, Tuple{NLPModels.AbstractNLPModel, AbstractVector{T}}} where T","page":"Reference","title":"DCISolver.dci","text":"dci(nlp, x; kwargs...)\n\nThis method implements the Dynamic Control of Infeasibility for equality-constrained problems described in\n\nDynamic Control of Infeasibility in Equality Constrained Optimization\nRoberto H. Bielschowsky and Francisco A. M. Gomes\nSIAM J. Optim., 19(3), 1299–1325.\nhttps://doi.org/10.1137/070679557\n\n\n\n\n\n","category":"method"},{"location":"reference/#DCISolver.factorize!","page":"Reference","title":"DCISolver.factorize!","text":"factorize!(M :: SymCOOSolver)\n\nCalls the factorization of the symmetric solver given by M. Use success(M) to check whether the factorization was successful.\n\n\n\n\n\n","category":"function"},{"location":"reference/#DCISolver.feasibility_step-Union{Tuple{T}, Tuple{NLPModels.AbstractNLPModel, AbstractVector{T}, AbstractVector{T}, T, Any, T, AbstractFloat, DCISolver.MetaDCI, Any}} where T","page":"Reference","title":"DCISolver.feasibility_step","text":"feasibility_step(nls, x, cx, Jx)\n\nApproximately solves min ‖c(x)‖.\n\nGiven xₖ, finds min ‖cₖ + Jₖd‖\n\n\n\n\n\n","category":"method"},{"location":"reference/#DCISolver.num_neg_eig","page":"Reference","title":"DCISolver.num_neg_eig","text":"num_neg_eig(M :: SymCOOSolver)\n\nReturns the number of negative eigenvalues of M.\n\n\n\n\n\n","category":"function"},{"location":"reference/#DCISolver.regularized_coo_saddle_system!-Union{Tuple{T}, Tuple{S}, Tuple{NLPModels.AbstractNLPModel, AbstractVector{S}, AbstractVector{S}, AbstractVector{T}}} where {S<:Int64, T<:AbstractFloat}","page":"Reference","title":"DCISolver.regularized_coo_saddle_system!","text":"regularized_coo_saddle_system!(nlp, rows, cols, vals, γ = γ, δ = δ)   Compute the structure for the saddle system [H + γI  [Jᵀ]; J -δI]   in COO-format in the following order:   H J γ -δ\n\n\n\n\n\n","category":"method"},{"location":"reference/#DCISolver.solve!","page":"Reference","title":"DCISolver.solve!","text":"solve!(x, M :: SymCOOSolver, b)\n\nSolve the system M x = b. factorize!(M) should be called first.\n\n\n\n\n\n","category":"function"},{"location":"reference/#DCISolver.tangent_step!-Union{Tuple{T}, Tuple{NLPModels.AbstractNLPModel, AbstractVector{T}, AbstractVector{T}, AbstractVector{T}, T, T, DCISolver.SymCOOSolver, AbstractVector{T}, AbstractVector{T}, T, T, AbstractFloat, T, T, DCISolver.MetaDCI, DCISolver.DCIWorkspace}} where T","page":"Reference","title":"DCISolver.tangent_step!","text":"min q(d):=¹/₂dᵀBd + dᵀg s.t Ad = 0     ‖d‖ ≦ Δ where B is an approximation of hessian of the Lagrangian, A is the jacobian matrix and g is the projected gradient.\n\nReturn status with outcomes:\n\n:cauchy_step, :newton, :dogleg,\n:unknown if we didn't enter the loop.\n:smallhorizontalstep\n:tired if we stop due to maxeval or maxtime\n:success if we computed z such that ‖c(z)‖ ≤ meta.ρbar * ρ and Δℓ ≥ η₁ q(d)\n\nSee https://github.com/JuliaSmoothOptimizers/SolverTools.jl/blob/78f6793f161c3aac2234aee8a27aa07f1df3e8ee/src/trust-region/trust-region.jl#L37 for SolverTools.aredpred\n\n\n\n\n\n","category":"method"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"We are following here the tutorial in SolverBenchmark.jl to run benchmarks on JSO-compliant solvers. We compare here the Ipopt via the NLPModelsIpopt.jl thin wrapper with DCISolver on a subset of CUTEst problems.","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"using CUTEst, NLPModels, NLPModelsIpopt, SolverBenchmark, SolverCore\n#This package\nusing DCISolver\n\nnmax = 100\n_pnames = CUTEst.select(\n  max_var = nmax, \n  min_con = 1, \n  max_con = nmax, \n  only_free_var = true, \n  only_equ_con = true, \n  objtype = 3:6\n)\n\n#Remove all the problems ending by NE as Ipopt cannot handle them.\npnamesNE = _pnames[findall(x->occursin(r\"NE\\b\", x), _pnames)]\npnames = setdiff(_pnames, pnamesNE)\ncutest_problems = (CUTEstModel(p) for p in pnames)\n\n#Same time limit for all the solvers\nmax_time = 1200. #20 minutes\n\nsolvers = Dict(\n  :ipopt => nlp -> ipopt(\n    nlp,\n    print_level = 0,\n    dual_inf_tol = Inf,\n    constr_viol_tol = Inf,\n    compl_inf_tol = Inf,\n    acceptable_iter = 0,\n    max_cpu_time = max_time,\n    x0 = nlp.meta.x0,\n  ),\n  :dcildl => nlp -> dci(\n    nlp,\n    nlp.meta.x0,\n    linear_solver = :ldlfact,\n    max_time = max_time,\n    max_iter = typemax(Int64),\n    max_eval = typemax(Int64),\n  ),\n)\n\nstats = bmark_solvers(solvers, cutest_problems)\n\nusing JLD2\n\n@save \"ipopt_dcildl_$(string(length(pnames))).jld2\" stats","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"pretty_stats(stats[:dcildl])","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"using Plots\ngr()\n\nlegend = Dict(\n  :neval_obj => \"number of f evals\", \n  :neval_cons => \"number of c evals\", \n  :neval_grad => \"number of ∇f evals\", \n  :neval_jac => \"number of ∇c evals\", \n  :neval_jprod => \"number of ∇c*v evals\", \n  :neval_jtprod  => \"number of ∇cᵀ*v evals\", \n  :neval_hess  => \"number of ∇²f evals\", \n  :elapsed_time => \"elapsed time\"\n)\nperf_title(col) = \"Performance profile on CUTEst w.r.t. $(string(legend[col]))\"\n\nstyles = [:solid,:dash,:dot,:dashdot] #[:auto, :solid, :dash, :dot, :dashdot, :dashdotdot]\n\nfunction print_pp_column(col::Symbol, stats)\n  \n  ϵ = minimum(minimum(filter(x -> x > 0, df[!, col])) for df in values(stats))\n  first_order(df) = df.status .== :first_order\n  unbounded(df) = df.status .== :unbounded\n  solved(df) = first_order(df) .| unbounded(df)\n  cost(df) = (max.(df[!, col], ϵ) + .!solved(df) .* Inf)\n\n  p = performance_profile(\n    stats, \n    cost, \n    title=perf_title(col), \n    legend=:bottomright, \n    linestyles=styles\n  )\nend\n\nprint_pp_column(:elapsed_time, stats)","category":"page"},{"location":"#DCISolver-Dynamic-Control-of-Infeasibility-Solver","page":"Introduction","title":"DCISolver - Dynamic Control of Infeasibility Solver","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"DCI is a solver for equality-constrained nonlinear problems, i.e., optimization problems of the form","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"    min_x  f(x) quad textst quad  c(x) = 0","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"based on the paper","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Bielschowsky, R. H., & Gomes, F. A. Dynamic control of infeasibility in equality constrained optimization. SIAM Journal on Optimization, 19(3), 1299-1325 (2008). 10.1007/s10589-020-00201-2","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"DCISolver is a JuliaSmoothOptimizers-compliant solver. It takes an AbstractNLPModel as an input and returns a GenericExecutionStats.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"We refer to juliasmoothoptimizers.github.io for tutorials on the NLPModel API and its usage. This framework allows the usage of models from Ampl (using AmplNLReader.jl), CUTEst (using CUTEst.jl), JuMP (using NLPModelsJuMP.jl), PDE-constrained optimization problems (using PDENLPModels.jl) and models defined with automatic differentiation (using ADNLPModels.jl).","category":"page"},{"location":"#Installation","page":"Introduction","title":"Installation","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"DCISolver is a registered package. To install this package, open the Julia REPL (i.e., execute the julia binary), type ] to enter package mode, and install DCISolver as follows","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"add DCISolver","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"It uses LDLFactorizations.jl by default to compute the factorization in the tangent step. Follow HSL.jl's MA57 installation for an alternative.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"The feasibility steps are factorization-free and use iterative methods from Krylov.jl.","category":"page"},{"location":"#Example","page":"Introduction","title":"Example","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"using DCISolver, ADNLPModels, Logging\nnlp = ADNLPModel(\n  x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, \n  [-1.2; 1.0],\n  x -> [x[1] * x[2] - 1], \n  [0.0], [0.0],\n  name = \"Rosenbrock with x₁x₂=1\"\n)\nstats = with_logger(NullLogger()) do\n  dci(nlp, nlp.meta.x0)\nend\n\nprintln(stats)","category":"page"},{"location":"#Bug-reports-and-discussions","page":"Introduction","title":"Bug reports and discussions","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"If you think you found a bug, feel free to open an issue. Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"If you want to ask a question not suited for a bug report, feel free to start a discussion here. This forum is for general discussion about this repository and the JuliaSmoothOptimizers, so questions about any of our packages are welcome.","category":"page"}]
}
