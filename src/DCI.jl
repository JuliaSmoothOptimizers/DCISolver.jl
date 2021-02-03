module DCI

  using LinearAlgebra, Logging

  using Krylov, LinearOperators, NLPModels, SolverTools, SparseArrays, SymCOOSolverInterface

  const solver_correspondence = Dict(#:ma57 => MA57Struct, 
                                    :ldlfact => LDLFactorizationStruct)

  export dci

  include("dci_feasibility.jl")
  using CaNNOLeS
  include("dci_cannoles_feasibility.jl")
  include("dci_normal.jl")
  include("dci_tangent.jl")
  include("main.jl")

  struct MetaDCI
    #Tolerances:
    atol # = 1e-5,
    rtol # = 1e-5, #ϵd = atol + rtol * dualnorm
    ctol # = 1e-5, #feasibility tolerance
    linear_solver # = :ldlfact,#:ma57,

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
  Compute the solution of ‖Jx' λ - ∇fx‖
  """
  function compute_lx(Jx :: LinearOperator{T}, ∇fx) where T
    (λ, stats) = cgls(Jx', -∇fx) #atol, rtol
    if !stats.solved
      @warn "Fail cgls computation Lagrange multiplier: $(stats.status)"
    end
    return λ
  end

  function compute_lx(Jx, ∇fx)
    return Jx' \ ( - ∇fx)
  end

  function compute_lx!(Jx :: LinearOperator{T}, ∇fx, λ) where T
    (l, stats) = cgls(Jx', -∇fx) #atol, rtol
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
