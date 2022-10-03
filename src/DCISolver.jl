module DCISolver

using SolverCore
import SolverCore.solve!

#using SymCOOSolverInterface
include("SymCOOSolverInterface/SymCOOSolverInterface.jl")

using LinearAlgebra, SparseArrays
#JSO packages
using HSL, Krylov, NLPModels, SolverCore, SolverTools

function cons_norhs!(nlp, x, cx)
  cons!(nlp, x, cx)
  if nlp.meta.ncon > 0
    cx .-= get_lcon(nlp)
  end
  return cx
end

export dci, solve!

include("param_struct.jl")
include("workspace.jl")
include("dci_feasibility.jl")
include("dci_normal.jl")
include("dci_tangent.jl")
include("main.jl")

"""
    dci(nlp; kwargs...)
    dci(nlp, x; kwargs...)
    dci(nlp, meta, x)

Compute a local minimum of an equality-constrained optimization problem using DCI algorithm from Bielschowsky & Gomes (2008).

# Arguments
- `nlp::AbstractNLPModel`: the model solved, see `NLPModels.jl`.
- `x`: Initial guess. If `x` is not specified, then `nlp.meta.x0` is used.
- `meta`: The keyword arguments are used to initialize a [`MetaDCI`](@ref).

For advanced usage, the principal call to the solver uses a [`DCIWorkspace`](@ref).

    solve!(workspace, nlp)
    solve!(workspace, nlp, stats)

# Output
The returned value is a `GenericExecutionStats`, see `SolverCore.jl`.

# References
This method implements the Dynamic Control of Infeasibility for
equality-constrained problems described in

    Dynamic Control of Infeasibility in Equality Constrained Optimization
    Roberto H. Bielschowsky and Francisco A. M. Gomes
    SIAM J. Optim., 19(3), 2008, 1299–1325.
    https://doi.org/10.1137/070679557

# Examples
```jldoctest; output = false
using ADNLPModels, DCISolver
nlp = ADNLPModel(x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, [-1.2; 1.0])
stats = dci(nlp, verbose = 0)
stats

# output

"Execution stats: first-order stationary"
```
"""
function dci(nlp::AbstractNLPModel, x::AbstractVector{T}; kwargs...) where {T}
  meta = MetaDCI(x, nlp.meta.y0; kwargs...)
  return dci(nlp, meta, x)
end
dci(nlp::AbstractNLPModel; kwargs...) = dci(nlp, nlp.meta.x0; kwargs...)
function dci(
  nlp::AbstractNLPModel,
  meta::MetaDCI{T, In, COO},
  x::AbstractVector{T},
) where {T, In, COO}
  workspace = DCIWorkspace(nlp, meta, x)
  return solve!(workspace, nlp)
end

"""
    compute_gBg(nlp, rows, cols, vals, ∇ℓzλ)
  
Compute `gBg = ∇ℓxλ' * B * ∇ℓxλ`, where `B` is a symmetric sparse matrix whose lower triangular is given in COO-format.
"""
function compute_gBg(
  nlp::AbstractNLPModel,
  rows::AbstractVector,
  cols::AbstractVector,
  vals::AbstractVector{T},
  ∇ℓzλ::AbstractVector{T},
) where {T}
  gBg = zero(T)
  for k = 1:(nlp.meta.nnzh) # TODO: use |vals| and remove dependency in nlp
    i, j, v = rows[k], cols[k], vals[k]
    #v * ∇ℓxλ[i] * ∇ℓxλ[j] * (i == j ? 1 : 2)
    gBg += v * ∇ℓzλ[i] * ∇ℓzλ[j] * (i == j ? 1 : 2)
  end
  return gBg
end

"""
    regularized_coo_saddle_system!(nlp, rows, cols, vals, γ = γ, δ = δ)

Compute the structure for the saddle system `[H + γI  [Jᵀ]; J -δI]` in COO-format `(rows, cols, vals)` in the following order: `H, J, γ, -δ,`.
"""
function regularized_coo_saddle_system!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{S},
  cols::AbstractVector{S},
  vals::AbstractVector{T};
  γ::T = zero(T),
  δ::T = zero(T),
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
  vals[nnz_idx] .= -δ

  return rows, cols, vals
end

"""
    compute_lx!(Jx, ∇fx, λ, meta)

Compute the solution of `‖Jx' λ - ∇fx‖` using solvers from `Krylov.jl` as defined by `meta.λ_struct.comp_λ_solver`.
Return the solution `λ`.
"""
function compute_lx!(
  Jx,
  ∇fx::AbstractVector{T},
  λ::AbstractVector{T},
  meta::MetaDCI,
) where {T <: AbstractFloat}
  l = meta.λ_struct.comp_λ_solver.x
  stats = meta.λ_struct.comp_λ_solver.stats
  Krylov.solve!(
    meta.λ_struct.comp_λ_solver,
    Jx',
    ∇fx,
    M = meta.λ_struct.M,
    λ = meta.λ_struct.λ,
    atol = meta.λ_struct.atol,
    rtol = meta.λ_struct.rtol,
    itmax = meta.λ_struct.itmax,
  )
  if !stats.solved
    @warn "Fail computation of Lagrange multiplier: $(stats.status)"
    #print(stats)
  end
  @. λ = -l #Should we really update if !stats.solved?
  return λ
end

end # end of module
