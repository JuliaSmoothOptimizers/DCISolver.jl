module DCI

  using LinearAlgebra, Logging

  using Krylov, LinearOperators, NLPModels, SolverTools, SparseArrays, SymCOOSolverInterface

  const solver_correspondence = Dict(#:ma57 => MA57Struct, 
                                    :ldlfact => LDLFactorizationStruct)

  export dci

  include("dci_feasibility.jl")
  include("dci_normal.jl")
  include("dci_tangent.jl")
  include("main.jl")

  #T.M.: ∇ℓxλ' * B * ∇ℓxλ (recall the only the lower triangular is in vals)
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
  Compute the solution of || Jx' λ - ∇fx ||
  """
  function compute_lx(Jx :: LinearOperator, ∇fx)
      return cgls(Jx', -∇fx)[1]
  end

  function compute_lx(Jx, ∇fx)
      return Jx' \ ( - ∇fx)
  end

  function compute_lx!(Jx :: LinearOperator, ∇fx, λ)
      λ .= cgls(Jx', -∇fx)[1]
      return λ
  end

  function compute_lx!(Jx, ∇fx, λ)
      λ .= Jx' \ ( - ∇fx)
      return λ
  end

end # end of module
