"""
min q(d):=¹/₂dᵀBd + dᵀg
s.t Ad = 0
    ‖d‖ ≦ Δ
where B is an approximation of hessian of the Lagrangian, A is the jacobian
matrix and `g` is the projected gradient.

Return status with outcomes:
- :cauchy_step, :newton, :dogleg,
- :unknown if we didn't enter the loop.
- :small_horizontal_step
- :tired if we stop due to max_eval or max_time
- :success if we computed z such that ‖c(z)‖ ≤ meta.ρbar * ρ and Δℓ ≥ η₁ q(d)

See https://github.com/JuliaSmoothOptimizers/SolverTools.jl/blob/78f6793f161c3aac2234aee8a27aa07f1df3e8ee/src/trust-region/trust-region.jl#L37
for `SolverTools.aredpred`
"""
function tangent_step!(
  nlp::AbstractNLPModel,
  z::AbstractVector{T},
  λ::AbstractVector{T},
  cz::AbstractVector{T},
  normcz::T,
  fz::T,
  LDL::SymCOOSolver,
  vals::AbstractVector{T},
  g::AbstractVector{T},
  ℓzλ::T,
  gBg::T,
  ρ::AbstractFloat,
  γ::T,
  δ::T,
  meta::MetaDCI,
  workspace::DCIWorkspace;
  Δ::AbstractFloat = meta.tan_Δ,
  η₁::AbstractFloat = meta.tan_η₁,
  η₂::AbstractFloat = meta.tan_η₂,
  σ₁::AbstractFloat = meta.tan_σ₁,
  σ₂::AbstractFloat = meta.tan_σ₂,
  small_d::AbstractFloat = meta.tan_small_d,
  max_eval::Int = 1_000,
  max_time::AbstractFloat = 1_000.0,
) where {T}
  d, tr, xt = workspace.dtan, workspace.tr, workspace.xtan
  Δℓ = zero(T)
  r = -one(T)

  status = :unknown
  iter = 0
  start_time = time()
  el_time = 0.0

  tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || el_time > max_time

  while !((normcz ≤ meta.ρbar * ρ && r ≥ η₁) || tired)
    #Compute a descent direction d (no evals)
    d, dBd, status, γ, δ, vals = compute_descent_direction!(nlp, gBg, g, Δ, LDL, γ, δ, vals, d, meta, workspace)
    n2d = dot(d, d)
    if √n2d > Δ
      d .*= Δ / √n2d #Just in case.
    end
    if √n2d < small_d
      status = :small_horizontal_step
      break
    end

    @. xt = z + d
    cons!(nlp, xt, cz)
    normcz = norm(cz)

    if normcz ≤ meta.ρbar * ρ
      ft = obj(nlp, xt)
      ℓxtλ = ft + dot(λ, cz)
      qd = dBd / 2 + dot(g, d)

      Δℓ, pred = aredpred!(tr, nlp, ℓzλ, ℓxtλ, qd, xt, d, dot(g, d))

      r = Δℓ / qd
      if r < η₁ #we decrease further Δ so that ≤ ||d||
        m = max(ceil(log(√n2d / Δ) / log(σ₁)), 1)
        Δ *= σ₁^m
      else #success
        status = :success
        z .= xt
        fz = ft
        ℓzλ = ℓxtλ
        if r ≥ η₂ && √n2d ≥ 0.99 * Δ
          Δ *= σ₂
        end
      end
    else
      m = max(ceil(log(√n2d / Δ) / log(σ₁)), 1)
      Δ *= σ₁^m
    end

    @info log_row(
      Any[
        "Tr",
        iter,
        neval_obj(nlp) + neval_cons(nlp),
        fz,
        ℓzλ,
        Float64,
        normcz,
        Float64,
        Float64,
        status,
        √n2d,
        Δ,
      ],
    )
    iter += 1

    el_time = time() - start_time
    tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || el_time > max_time
  end

  if tired
    status = :tired
  end

  return z, cz, fz, status, Δ, Δℓ, γ, δ #ℓzλ
end

"""
Compute a direction `d` with three possible outcomes:
- `:cauchy_step`
- `:newton`
- `:dogleg`
- `:interior_cauchy_step` when γ is too large.
for `min_d q(d) s.t. ‖d‖ ≤ Δ`.
"""
function compute_descent_direction!(
  nlp::AbstractNLPModel,
  gBg::T,
  g::AbstractVector{T},
  Δ::T,
  LDL::SymCOOSolver,
  γ::T,
  δ::T,
  vals::AbstractVector{T},
  d::AbstractVector{T},
  meta::MetaDCI,
  workspace::DCIWorkspace,
) where {T}
  m, n = nlp.meta.ncon, nlp.meta.nvar
  dcp = workspace.dcp

  #first compute a gradient step
  dcp_on_boundary, dcp, dcpBdcp, α = _compute_gradient_step!(nlp, gBg, g, Δ, dcp)

  if dcp_on_boundary # When the Cauchy step is in the boundary, we use it
    status = :cauchy_step
    dBd = dcpBdcp
    d = dcp
  else
    dn, dnBdn, dcpBdn, γ_too_large, γ, δ, vals =
      _compute_newton_step!(nlp, LDL, g, γ, δ, dcp, vals, meta, workspace)
    norm2dn = dot(dn, dn)
    if γ_too_large || dnBdn ≤ 1e-10 #or same test as gBg in _compute_gradient_step ?
      #dn = 0 here.
      if norm(dcp) < Δ #just to be sure
        d = dcp
        dBd = dcpBdcp
        status = :interior_cauchy
        return d, dBd, status, γ, δ, vals
      end
    end

    if √norm2dn < Δ # Both Newton and Cauchy are inside the TR.
      status = :newton
      dBd = dnBdn
      d = dn
    else
      dotdndcp, norm2dcp = dot(dn, dcp), dot(dcp, dcp)
      τ = _compute_step_length(norm2dn, dotdndcp, norm2dcp, Δ)
      @. d = dn + τ * (dcp - dn)
      dBd = τ^2 * dcpBdcp + 2 * τ * (1 - τ) * dcpBdn + (1 - τ)^2 * dnBdn
      status = :dogleg
    end
  end

  return d, dBd, status, γ, δ, vals
end

"""
Compute a solution to
min_α q(-α g) s.t. ‖αg‖_2 ≤ Δ

return `dcp_on_boundary` true if ‖αg‖ = Δ,
return `dcp = - α g`
return `dcpBdcp = α^2 gBg`
and `α` the solution.
"""
function _compute_gradient_step!(nlp::AbstractNLPModel, gBg::T, g::AbstractVector{T}, Δ::T, dcp::AbstractVector{T}) where {T}
  dcp_on_boundary = false
  dgg = dot(g, g)
  if gBg ≤ 1e-12 * dgg #generalize this test
    α = Δ / √dgg #norm(g)
    dcp_on_boundary = true
  else
    α = dgg / gBg #dot(g, g) / gBg
    if α > Δ / √dgg #norm(g)
      α = Δ / √dgg #norm(g)
      dcp_on_boundary = true
    end
  end
  @. dcp = -α * g
  dcpBdcp = α^2 * gBg

  return dcp_on_boundary, dcp, dcpBdcp, α
end

"""
Given two directions dcp and dn, compute the largest 0 ≤ τ ≤ 1 such that
‖dn + τ (dcp -dn)‖ = Δ
"""
function _compute_step_length(norm2dn::T, dotdndcp::T, norm2dcp::T, Δ::T) where {T <: AbstractFloat}
  # d = τ dcp + (1 - τ) * dn = dn + τ * (dcp - dn)
  # ‖d‖² = Δ² => τ² ‖dcp - dn‖² + 2τ dnᵀ(dcp - dn) + ‖dn‖² - Δ² = 0
  # Δ = b² - 4ac
  scal = norm2dcp - 2 * dotdndcp + norm2dn
  q₀ = (norm2dn - Δ^2) / scal
  q₁ = 2 * (dotdndcp - norm2dn) / scal
  q₂ = one(T)
  # q₀, q₁, q₂ = [q₀, q₁, q₂] / q₂ #so the first coefficient is 1.
  roots = Krylov.roots_quadratic(q₂, q₁, q₀)
  τ = length(roots) == 0 ? one(T) : min(one(T), roots...)
  return τ
end

include("factorization.jl")
#=
dn, dnBdn, dcpBdn,
γ_too_large, γ, δ, vals = _compute_newton_step!(nlp, LDL, g, γ, δ, dcp, vals)
=#
