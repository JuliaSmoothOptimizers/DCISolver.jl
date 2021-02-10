module DCI

  using LinearAlgebra, Logging

  using HSL, Krylov, LinearOperators, NLPModels, SolverTools, SparseArrays, SymCOOSolverInterface

  const solver_correspondence = if isdefined(HSL, :libhsl_ma57)
    Dict(:ma57 => MA57Struct, 
         :ldlfact => LDLFactorizationStruct)
  else
    Dict(:ldlfact => LDLFactorizationStruct)
  end

  export dci

  include("dci_feasibility.jl")
  #using CaNNOLeS
  #include("dci_cannoles_feasibility.jl")
  include("dci_normal.jl")
  include("dci_tangent.jl")
  include("main.jl")

  struct MetaDCI

    #Tolerances on the problem:
    atol # = 1e-5,
    rtol # = 1e-5, #ϵd = atol + rtol * dualnorm
    ctol # = 1e-5, #feasibility tolerance

    #Limits
    max_eval # = 50000,
    max_time # = 60.

    #Compute Lagrange multiplier min_λ ‖Jx' λ - ∇fx‖
    solver :: Symbol #cgls by default, ε = atol + rtol * ArNorm
    λatol  :: AbstractFloat #√eps(T)
    λrtol  :: AbstractFloat #√eps(T)

    #List of intermediary functions:
    feas_step :: Symbol #minimize ‖c(x)‖  #dogleg or TR_lsmr
    TR_computation_step :: Symbol #min_d ‖cₖ + Jₖd‖ s.t. ||d|| ≤ Δ #feasibility_step or cannoles_step

    #Parameters in tangent step
    Δ       :: AbstractFloat # = one(T), #trust-region radius
    η₁      :: AbstractFloat # = T(1e-2),
    η₂      :: AbstractFloat # = T(0.75),
    σ₁      :: AbstractFloat # = T(0.25), #decrease trust-region radius
    σ₂      :: AbstractFloat # = T(2.0), #increase trust-region radius after success
    small_d :: AbstractFloat # = eps(T)  #below this threshold on ||d|| the step is too small.
    #Parameters for the regularization of the factorization in tangent step
    linear_solver # = :ldlfact,#:ma57,
    δmin     :: AbstractFloat # = √eps(T),

  end

  """compute_gBg
    B is a symmetric sparse matrix 
    whose lower triangular given in COO: (rows, cols, vals)

    Compute ∇ℓxλ' * B * ∇ℓxλ
  """
  function compute_gBg(nlp :: AbstractNLPModel, 
                       rows :: AbstractVector, 
                       cols :: AbstractVector, 
                       vals :: AbstractVector{T}, 
                       ∇ℓzλ :: AbstractVector{T}) where T
    gBg = zero(T)
    for k = 1:nlp.meta.nnzh
      i, j, v = rows[k], cols[k], vals[k]
      gBg += v * ∇ℓzλ[i] * ∇ℓzλ[j] * (i == j ? 1 : 2) #v * ∇ℓxλ[i] * ∇ℓxλ[j] * (i == j ? 1 : 2)
    end
    return gBg
  end

  """
  `regularized_coo_saddle_system!(nlp, rows, cols, vals, γ = γ, δ = δ)`
    Compute the structure for the saddle system [H + γI  [Jᵀ]; J -δI] in COO-format in the following order:
    H J γ -δ
  """
  function regularized_coo_saddle_system!(nlp  :: AbstractNLPModel,
                                          rows :: AbstractVector{S},
                                          cols :: AbstractVector{S},
                                          vals :: AbstractVector{T};
                                          γ    :: T = zero(T),
                                          δ    :: T = zero(T),
                                         ) where {S <: Int, T <: AbstractFloat}
    #n = nlp.meta.nnzh + nlp.meta.nnzj + nlp.meta.nvar + nlp.meta.ncon
    #Test length rows, cols, vals
    #@lencheck n rows cols vals

    # H (1:nvar, 1:nvar)
    nnz_idx = 1:nlp.meta.nnzh
    @views hess_structure!(nlp, rows[nnz_idx], cols[nnz_idx])
    # J (nvar .+ 1:ncon, 1:nvar)
    nnz_idx = nlp.meta.nnzh .+ (1:nlp.meta.nnzj)
    @views jac_structure!(nlp, rows[nnz_idx], cols[nnz_idx])
    rows[nnz_idx] .+= nlp.meta.nvar
    # γI (1:nvar, 1:nvar)
    nnz_idx = nlp.meta.nnzh .+ nlp.meta.nnzj .+ (1:nlp.meta.nvar)
    rows[nnz_idx] .= 1:nlp.meta.nvar
    cols[nnz_idx] .= 1:nlp.meta.nvar
    vals[nnz_idx] .= γ
    # -δI (nvar .+ 1:ncon, nvar .+ 1:ncon)
    nnz_idx = nlp.meta.nnzh .+ nlp.meta.nnzj .+ nlp.meta.nvar .+ (1:nlp.meta.ncon)
    rows[nnz_idx] .= nlp.meta.nvar .+ (1:nlp.meta.ncon)
    cols[nnz_idx] .= nlp.meta.nvar .+ (1:nlp.meta.ncon)
    vals[nnz_idx] .= - δ

    return rows, cols, vals
  end

  """
  Compute the solution of ‖Jx' λ - ∇fx‖
  """
  function compute_lx(Jx :: LinearOperator{T}, ∇fx) where T
    m, n = size(Jx) 
    (λ, stats) = cgls(Jx', -∇fx)#, itmax = 10 * (m + n)) #atol, rtol
    if !stats.solved
      @warn "Fail cgls computation Lagrange multiplier: $(stats.status)"
    end
    return λ
  end

  function compute_lx(Jx, ∇fx)
    return Jx' \ ( - ∇fx)
  end

  function compute_lx!(Jx :: LinearOperator{T}, ∇fx, λ) where T
     
    m, n = size(Jx) 
    (l, stats) = cgls(Jx', -∇fx)#, itmax = 10 * (m + n)) #atol, rtol
    if !stats.solved
      @warn "Fail cgls computation Lagrange multiplier: $(stats.status)"
    end
    λ .= l
    return λ
  end

  function compute_lx!(Jx, ∇fx, λ)
    λ .= Jx' \ ( - ∇fx)
    return λ
  end

end # end of module
