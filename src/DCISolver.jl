module DCISolver

  #using SymCOOSolverInterface
  include("SymCOOSolverInterface/SymCOOSolverInterface.jl")

  using LinearAlgebra, SparseArrays
  #JSO packages
  using HSL, Krylov, LinearOperators, NLPModels, SolverCore, SolverTools

  export dci

  include("param_struct.jl")
  include("dci_feasibility.jl")
  include("dci_normal.jl")
  include("dci_tangent.jl")
  include("main.jl")

  """
      dci(nlp, x; kwargs...)

  This method implements the Dynamic Control of Infeasibility for
  equality-constrained problems described in

      Dynamic Control of Infeasibility in Equality Constrained Optimization
      Roberto H. Bielschowsky and Francisco A. M. Gomes
      SIAM J. Optim., 19(3), 1299–1325.
      https://doi.org/10.1137/070679557

  """
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
       #v * ∇ℓxλ[i] * ∇ℓxλ[j] * (i == j ? 1 : 2)
      gBg += v * ∇ℓzλ[i] * ∇ℓzλ[j] * (i == j ? 1 : 2)
    end
    return gBg
  end

  """
  `regularized_coo_saddle_system!(nlp, rows, cols, vals, γ = γ, δ = δ)`
    Compute the structure for the saddle system [H + γI  [Jᵀ]; J -δI]
    in COO-format in the following order:
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
  function compute_lx(Jx   :: LinearOperator{T},
                      ∇fx  :: AbstractVector{T},
                      meta :: MetaDCI) where T <: AbstractFloat
    m, n = size(Jx)
    λ = Array{T}(undef, m)
    compute_lx!(Jx, ∇fx, λ, meta)
    return λ
  end

  function compute_lx!(Jx   :: LinearOperator{T},
                       ∇fx  :: AbstractVector{T},
                       λ    :: AbstractVector{T},
                       meta :: MetaDCI) where T <: AbstractFloat

    (l, stats) = eval(meta.comp_λ)(Jx', -∇fx, M     = meta.λ_struct.M,
                                              λ     = meta.λ_struct.λ,
                                              atol  = meta.λ_struct.atol,
                                              rtol  = meta.λ_struct.rtol,
                                              itmax = meta.λ_struct.itmax)
    if !stats.solved
      @warn "Fail $(meta.comp_λ) computation Lagrange multiplier: $(stats.status)"
      #print(stats)
    end
    λ .= l #Should we really update if !stats.solved?
    return λ
  end

  function compute_lx(Jx, ∇fx, meta :: MetaDCI)
    return Jx' \ ( - ∇fx)
  end

  function compute_lx!(Jx, ∇fx, λ, meta :: MetaDCI)
    λ .= Jx' \ ( - ∇fx)
    return λ
  end

end # end of module
