"""
    feasibility_step(nls, x, cx, normcx, Jx, ρ, ctol, meta, workspace; kwargs...)

Approximately solves `min ‖c(x)‖` using a trust-region Levenberg-Marquardt method, i.e. given `xₖ`, finds `min ‖cₖ + Jₖd‖`.

# Arguments
- `η₁::AbstractFloat = meta.feas_η₁`: decrease the trust-region radius when Ared/Pred < η₁.
- `η₂::AbstractFloat = meta.feas_η₂`: increase the trust-region radius when Ared/Pred > η₂.
- `σ₁::AbstractFloat = meta.feas_σ₁`: decrease coefficient of the trust-region radius.
- `σ₂::AbstractFloat = meta.feas_σ₂`:increase coefficient of the trust-region radius.
- `Δ₀::T = meta.feas_Δ₀`: initial trust-region radius.
- `bad_steps_lim::Integer = meta.bad_steps_lim`: consecutive bad steps before using a second order step.
- `expected_decrease::T = meta.feas_expected_decrease`: bad steps are when `‖c(z)‖ / ‖c(x)‖ >feas_expected_decrease`.
- `max_eval::Int = 1_000`: maximum evaluations.
- `max_time::AbstractFloat = 60.0`: maximum time.
- `max_feas_iter::Int = typemax(Int64)`: maximum number of iterations.

# Output
- `z`, `cz`, `normcz`, `Jz`: the new iterate, and updated evaluations.
- `status`: Computation status. Possible outcomes are: `:success`, `max_eval`, `max_time`, `max_iter`, `unknown_tired`, `:infeasible`, `:unknown`.
"""
function feasibility_step(
  nlp::AbstractNLPModel,
  x::AbstractVector{T},
  cx::AbstractVector{T},
  normcx::T,
  Jx,
  ρ::T,
  ctol::AbstractFloat,
  meta::MetaDCI,
  workspace,
  verbose::Bool;
  η₁::AbstractFloat = meta.feas_η₁,
  η₂::AbstractFloat = meta.feas_η₂,
  σ₁::AbstractFloat = meta.feas_σ₁,
  σ₂::AbstractFloat = meta.feas_σ₂,
  Δ₀::T = meta.feas_Δ₀,
  bad_steps_lim::Integer = meta.bad_steps_lim,
  expected_decrease::T = meta.feas_expected_decrease,
  max_eval::Int = 1_000,
  max_time::AbstractFloat = 60.0,
  max_feas_iter::Int = typemax(Int64),
) where {T}
  z = x
  cz = cx
  zp, czp, Jd = workspace.zp, workspace.czp, workspace.Jd
  Jz = Jx
  (m, n) = size(Jz)
  normcz = normcx # cons(nlp, x) = normcx = normcz for the first z
  Δ = Δ₀
  feas_iter = 0
  consecutive_bad_steps = 0
  failed_step_comp = false

  el_time = 0.0
  tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || el_time > max_time
  status = :unknown

  start_time = time()
  infeasible = false

  while !(normcz ≤ ρ || tired || infeasible)

    #Compute the a direction satisfying the trust-region constraint
    d, Jd, infeasible, solved =
      eval(meta.TR_compute_step)(cz, Jz, ctol, Δ, normcz, Jd, meta.TR_compute_step_struct)

    if infeasible #the direction is too small
      failed_step_comp = true #too small step
      status = :too_small
    else
      zp .= z .+ d
      cons_norhs!(nlp, zp, czp)
      normczp = norm(czp)

      Jd .+= cz
      Pred = T(0.5) * (normcz^2 - norm(Jd)^2) # T(0.5) * (normcz^2 - norm(Jd + cz)^2)
      Ared = T(0.5) * (normcz^2 - normczp^2)

      if Ared / Pred < η₁
        Δ = max(T(1e-8), Δ * σ₁)
        status = :reduce_Δ
      else #success
        z = zp
        Jz = jac_op!(nlp, z, workspace.Jv, workspace.Jtv)
        cz = czp
        if normczp / normcz > expected_decrease
          consecutive_bad_steps += 1
        else
          consecutive_bad_steps = 0
        end
        normcz = normczp
        status = :success
        if Ared / Pred > η₂ && norm(d) >= T(0.99) * Δ
          Δ *= σ₂
        end
      end
    end

    verbose && @info log_row(
      Any[
        "F",
        feas_iter,
        neval_obj(nlp) + neval_cons(nlp),
        Float64,
        Float64,
        Float64,
        normcz,
        Float64,
        Float64,
        status,
        norm(d),
        Δ,
        time() - start_time,
      ],
    )

    # Safeguard: agressive normal step
    if normcz > ρ && (consecutive_bad_steps ≥ bad_steps_lim || failed_step_comp)
      Hz = hess_op(nlp, z, cz, obj_weight = zero(T))
      d = meta.agressive_cgsolver.x
      stats = meta.agressive_cgsolver.stats
      Krylov.solve!(meta.agressive_cgsolver, Hz + Jz' * Jz, Jz' * cz)
      if !stats.solved
        @warn "Fail 2nd order correction in feasibility_step: $(stats.status)"
      end
      @. zp = z - d
      cons_norhs!(nlp, zp, czp)
      nczp = norm(czp)
      if nczp < normcz #even if d is small we keep going
        infeasible = false
        failed_step_comp = false
        status = :agressive
        z, cz = zp, czp
        normcz = nczp
        Jz = jac_op!(nlp, z, workspace.Jv, workspace.Jtv)
      elseif norm(d) < ctol * min(nczp, one(T))
        infeasible = true
        status = :agressive_fail
      else #unsuccessful,nczp > normcz,infeasible = true,status = :too_small
        cg_iter = length(stats.residuals)
        #@show cg_iter, stats.residuals[end], nczp, normcz, norm(Jz' * czp)
        #should we increase the iteration limit if we busted it?
        #Adding regularization might be more efficient
      end
      verbose && @info log_row(
        Any[
          "F-safe",
          feas_iter,
          neval_obj(nlp) + neval_cons(nlp),
          Float64,
          Float64,
          Float64,
          normcz,
          Float64,
          Float64,
          status,
          norm(d),
          Δ,
          time() - start_time,
        ],
      )
    end

    el_time = time() - start_time
    feas_iter += 1
    many_evals = neval_obj(nlp) + neval_cons(nlp) > max_eval
    iter_limit = feas_iter > max_feas_iter
    tired = many_evals || el_time > max_time || iter_limit
  end

  status = if normcz ≤ ρ
    :success
  elseif tired
    if neval_obj(nlp) + neval_cons(nlp) > max_eval
      :max_eval
    elseif el_time > max_time
      :max_time
    elseif feas_iter > max_feas_iter
      :max_iter
    else
      :unknown_tired
    end
  elseif infeasible
    :infeasible
  else
    :unknown
  end

  return z, cz, normcz, Jz, status
end

@doc raw"""
    TR_dogleg(cz, Jz, ctol, Δ, normcz, Jd, meta)

Compute a direction `d` such that
```math
\begin{aligned}
\min_{d} \quad & \|c + Jz' d \| \\
\text{s.t.} \quad & \|d\| \leq \Delta,
\end{aligned}
```
using a dogleg.

# Output
- `d`: solution
- `Jd`: product of the solution with `J`.
- `infeasible`: `true` if the problem is infeasible.
- `solved`: `true` if the problem has been successfully solved.
"""
function TR_dogleg(
  cz::AbstractVector{T},
  Jz,
  ctol::AbstractFloat,
  Δ::T,
  normcz::AbstractFloat,
  Jd::AbstractVector{T},
  meta::TR_dogleg_struct,
) where {T}
  infeasible, solved = false, true

  d = -Jz' * cz
  nd2 = dot(d, d)
  Jd .= Jz * d

  if √nd2 < ctol * min(normcz, one(T)) #norm(d) < ctol * normcz
    infeasible = true
    return d, Jd, solved, infeasible
  end

  t = nd2 / dot(Jd, Jd) #dot(d, d) / dot(Jd, Jd)
  dcp = t * d
  ndcp2 = t^2 * nd2 #dot(dcp, dcp)

  if √ndcp2 > Δ
    d = dcp * Δ / √ndcp2  #so ||d||=Δ
    Jd .= Jd * t * Δ / √ndcp2  #avoid recomputing Jd
  else
    dn = meta.lsmr_solver.x
    stats = meta.lsmr_solver.stats
    Krylov.solve!(meta.lsmr_solver, Jz, -cz)
    solved = stats.solved
    if !solved #stats.status ∈ ("maximum number of iterations exceeded")
      @warn "Fail lsmr in TR_dogleg: $(stats.status)"
    end

    if norm(dn) <= Δ
      d = dn
    else
      v = dn - dcp
      nv2 = dot(v, v)
      dcpv = dot(dcp, v)
      τ = (-dcpv + sqrt(dcpv^2 + 4 * nv2 * (Δ^2 - ndcp2))) / nv2
      d = dcp + τ * v
    end
    mul!(Jd, Jz, d) # d has been updated
  end

  return d, Jd, infeasible, solved
end

@doc raw"""
    TR_lsmr(cz, Jz, ctol, Δ, normcz, Jd, meta)

Compute a direction `d` such that
```math
\begin{aligned}
\min_{d} \quad & \|c + Jz' d \| \\
\text{s.t.} \quad & \|d\| \leq \Delta,
\end{aligned}
```
using `lsmr` method from `Krylov.jl`.

# Output
- `d`: solution
- `Jd`: product of the solution with `J`.
- `infeasible`: `true` if the problem is infeasible.
- `solved`: `true` if the problem has been successfully solved.
"""
function TR_lsmr(
  cz::AbstractVector{T},
  Jz,
  ctol::AbstractFloat,
  Δ::T,
  normcz::AbstractFloat,
  Jd::AbstractVector{T},
  meta::TR_lsmr_struct,
) where {T}
  solver = meta.lsmr_solver
  d = solver.x
  stats = solver.stats
  Krylov.solve!(
    solver,
    Jz,
    cz,
    radius = Δ,
    M = meta.M,
    λ = meta.λ,
    axtol = meta.axtol,
    btol = meta.btol,
    atol = meta.atol,
    rtol = meta.rtol,
    etol = meta.etol,
    itmax = meta.itmax,
  )

  infeasible = norm(d) < ctol * min(normcz, one(T))
  solved = stats.solved
  if !solved
    @warn "Fail lsmr in TR_lsmr: $(stats.status)"
  end

  @. d = -d
  mul!(Jd, Jz, d) #lsmr doesn't return this information

  return d, Jd, infeasible, solved
end
