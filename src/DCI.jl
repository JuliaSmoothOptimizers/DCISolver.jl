module DCI

  using LinearAlgebra, Logging

  using HSL, Krylov, LinearOperators, NLPModels, SolverTools, SparseArrays, SymCOOSolverInterface

  export dci

  include("param_struct.jl")
  include("dci_feasibility.jl")
  #using CaNNOLeS
  #include("dci_cannoles_feasibility.jl")
  include("dci_normal.jl")
  include("dci_tangent.jl")
  include("main.jl")

  """
      dci(nlp, x; kwargs...)

  This method implements the Dynamic Control of Infeasibility for equality-constrained
  problems described in

      Dynamic Control of Infeasibility in Equality Constrained Optimization
      Roberto H. Bielschowsky and Francisco A. M. Gomes
      SIAM J. Optim., 19(3), 1299–1325.
      https://doi.org/10.1137/070679557

  """
  #=
  function dci(nlp  :: AbstractNLPModel,
              x    :: AbstractVector{T};# = nlp.meta.x0,
              atol :: AbstractFloat = 1e-5,
              rtol :: AbstractFloat = 1e-5,
              ctol :: AbstractFloat = 1e-5,
              linear_solver :: Symbol = :ldlfact,  # :ma57,#
              max_eval :: Int = 50000,
              max_time :: Float64 = 10.,
              max_iter :: Int = 500,
              ) where T
  =#
  function dci(nlp  :: AbstractNLPModel,
              x    :: AbstractVector{T};
              kwargs...
              ) where T
    meta = MetaDCI(x, nlp.meta.y0; kwargs...)
    return dci(nlp, x, meta)
  end


  """compute_gBg
    B is a symmetric sparse matrix 
    whose lower triangular given in COO: (rows, cols, vals)

    Compute ∇ℓxλ' * B * ∇ℓxλ
  """
  function compute_gBg(nlp  :: AbstractNLPModel, 
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
    nnzh, nnzj = nlp.meta.nnzh, nlp.meta.nnzj
    nvar, ncon = nlp.meta.nvar, nlp.meta.ncon

    # H (1:nvar, 1:nvar)
    nnz_idx = 1:nnzh
    @views hess_structure!(nlp, rows[nnz_idx], cols[nnz_idx])
    # J (nvar .+ 1:ncon, 1:nvar)
    nnz_idx = nnzh .+ (1:nnzj)
    @views jac_structure!(nlp, rows[nnz_idx], cols[nnz_idx])
    rows[nnz_idx] .+= nvar
    # γI (1:nvar, 1:nvar)
    nnz_idx = nnzh .+ nnzj .+ (1:nvar)
    rows[nnz_idx] .= 1:nvar
    cols[nnz_idx] .= 1:nvar
    vals[nnz_idx] .= γ
    # -δI (nvar .+ 1:ncon, nvar .+ 1:ncon)
    nnz_idx = nnzh .+ nnzj .+ nvar .+ (1:ncon)
    rows[nnz_idx] .= nvar .+ (1:ncon)
    cols[nnz_idx] .= nvar .+ (1:ncon)
    vals[nnz_idx] .= - δ

    return rows, cols, vals
  end

  """
  Compute the solution of ‖Jx' λ - ∇fx‖
  """
  function compute_lx(Jx  :: LinearOperator{T}, 
                      ∇fx :: AbstractVector{T}) where T <: AbstractFloat
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

  function compute_lx!(Jx  :: LinearOperator{T}, 
                       ∇fx :: AbstractVector{T}, 
                       λ   :: AbstractVector{T};
                       linear_solver :: Function = cgls) where T <: AbstractFloat
     
    m, n = size(Jx) 
    #(l, stats) = cgls(Jx', -∇fx, itmax = 5 * (m + n), λ = 1e-5, atol = 1e-5, rtol = 1e-5) #atol, rtol
    (l, stats) = linear_solver(Jx', -∇fx, itmax = 5 * (m + n))
    if !stats.solved
      @warn "Fail cgls computation Lagrange multiplier: $(stats.status)"
      print(stats)
    end
    λ .= l #should we really update if !stats.solved
    return λ
  end

  function compute_lx!(Jx, ∇fx, λ)
    λ .= Jx' \ ( - ∇fx)
    return λ
  end

end # end of module
