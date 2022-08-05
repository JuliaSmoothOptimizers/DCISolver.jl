"""
    normal_step!(nlp, x, cx, Jx, fx, ∇fx, λ, ℓxλ, ∇ℓxλ, dualnorm, primalnorm, ρmax, ϵp, max_eval, max_time, max_iter, meta, workspace)

Normal step: find `z` such that `||h(z)|| ≤ ρ` where `` is the trust-cylinder radius.

# Output
- `z`, `cz`, `fz`, `ℓzλ`, `∇ℓzλ`:  the new iterate, and updated evaluations.
- `ρ`: updated trust-cylinder radius.
- `primalnorm`, `dualnorm`: updated primal and dual feasibility norms.
- `status`: Computation status. The possible outcomes are: `:init_success`, `:success`, `:max_eval`, `:max_time`, `:max_iter`, `:unknown_tired`, `:infeasible`, `:unknown`.
"""
function normal_step!(
  nlp::AbstractNLPModel,
  x::AbstractVector{T},
  cx::AbstractVector{T},
  Jx,
  fx::T,
  ∇fx::AbstractVector{T},
  λ::AbstractVector{T},
  ℓxλ::T,
  ∇ℓxλ::AbstractVector{T},
  dualnorm::T,
  primalnorm::T, #norm(cx)
  ρmax::T,
  ϵp::T,
  max_eval,
  max_time,
  max_iter,
  meta::MetaDCI,
  workspace::DCIWorkspace,
  verbose::Bool,
) where {T}

  #assign z variable:
  workspace.z .= x
  workspace.cz .= cx
  workspace.∇fz .= ∇fx
  Jz, fz, ℓzλ = Jx, fx, ℓxλ
  norm∇fz = norm(∇fx) #can be avoided if we use dualnorm
  workspace.∇ℓzλ .= ∇ℓxλ
  z, cz, ∇fz, ∇ℓzλ = workspace.z, workspace.cz, workspace.∇fz, workspace.∇ℓzλ

  infeasible = false
  restoration = false
  tired = false
  start_time = time()
  eltime = 0.0

  #Initialize ρ at x
  ρ = compute_ρ(dualnorm, primalnorm, norm∇fz, ρmax, ϵp, 0, meta)

  done_with_normal_step = primalnorm ≤ ρ
  iter_normal_step = 0

  while !done_with_normal_step

    #primalnorm = norm(cz)
    z, cz, primalnorm, Jz, normal_status = eval(meta.feas_step)(
      nlp,
      z,
      cz,
      primalnorm,
      Jz,
      ρ,
      ϵp,
      meta,
      workspace,
      max_eval = max_eval,
      max_time = max_time,
    )

    fz, ∇fz = objgrad!(nlp, z, ∇fz)
    norm∇fz = norm(∇fz) #can be avoided if we use dualnorm
    compute_lx!(Jz, ∇fz, λ, meta)
    ℓzλ = fz + dot(λ, cz)
    ∇ℓzλ .= ∇fz .+ Jz' * λ
    dualnorm = norm(∇ℓzλ)

    #update rho
    iter_normal_step += 1
    ρ = compute_ρ(dualnorm, primalnorm, norm∇fz, ρmax, ϵp, iter_normal_step, meta)

    verbose && @info log_row(
      Any[
        "N",
        iter_normal_step,
        neval_obj(nlp) + neval_cons(nlp),
        fz,
        ℓzλ,
        dualnorm,
        primalnorm,
        ρmax,
        ρ,
        normal_status,
        Float64,
        Float64,
      ],
    )

    eltime = time() - start_time
    many_evals = neval_obj(nlp) + neval_cons(nlp) > max_eval
    tired = many_evals || eltime > max_time || iter_normal_step > max_iter
    infeasible = normal_status == :infeasible

    if infeasible && !restoration && !(primalnorm ≤ ρ || tired)
      #Enter restoration phase to avoid infeasible stationary points.
      #Heuristic that forces a random move from z
      restoration, infeasible = true, false
      perturbation_length = min(primalnorm, √ϵp) / norm(z) #sqrt(ϵp)/norm(z)
      z .+= (2 * rand(T, nlp.meta.nvar) .- one(T)) * perturbation_length
      cons_norhs!(nlp, z, cz)
      Jz = jac_op!(nlp, z, workspace.Jv, workspace.Jtv) # workspace.Jx
      primalnorm = norm(cz)
      ρ = compute_ρ(dualnorm, primalnorm, norm∇fz, ρmax, ϵp, 0, meta)
    end

    done_with_normal_step = primalnorm ≤ ρ || tired || infeasible
  end

  status = if primalnorm ≤ ρ && iter_normal_step == 0
    :init_success
  elseif primalnorm ≤ ρ
    :success
  elseif tired
    if neval_obj(nlp) + neval_cons(nlp) > max_eval
      :max_eval
    elseif eltime > max_time
      :max_time
    elseif iter_normal_step > max_iter
      :max_iter
    else
      :unknown_tired
    end
  elseif infeasible
    :infeasible
  else
    :unknown
  end

  return z, cz, fz, ℓzλ, ∇ℓzλ, ρ, primalnorm, dualnorm, status
end

"""
    compute_ρ(dualnorm, primalnorm, norm∇fx, ρmax, ϵ, iter, meta::MetaDCI)
    compute_ρ(dualnorm, primalnorm, norm∇fx, ρmax, ϵ, iter, p1, p2)

Update and return the trust-cylinder radius `ρ`.

Theory asks for ngp ρmax 10^-4 < ρ <= ngp ρmax
There are no evaluations of functions here.

`ρ = O(‖g_p(z)‖)` and in the paper `ρ = ν n_p(z) ρ_max` with `n_p(z) = norm(g_p(z)) / (norm(g(z)) + 1)`.
"""
function compute_ρ(dualnorm, primalnorm, norm∇fx, ρmax, ϵ, iter, meta::MetaDCI)
  p1, p2 = meta.compρ_p1, meta.compρ_p2
  return compute_ρ(dualnorm, primalnorm, norm∇fx, ρmax, ϵ, iter, p1, p2)
end
# T.M., 2021 Feb. 5th: what if dualnorm is excessively small ?
#            Feb. 8th: don't let ρ decrease too crazy
function compute_ρ(
  dualnorm::T,
  primalnorm::T,
  norm∇fx::T,
  ρmax::T,
  ϵ::T, #ctol
  iter::Integer,
  p1::Real,
  p2::Real,
) where {T}
  if iter > 100
    return p1 * ρmax
  end
  ngp = dualnorm / (norm∇fx + 1)
  ρ = max(min(ngp, p1) * ρmax, ϵ) # max(min(ngp/ρmax, p1) * ρmax, ϵ)
  if ρ ≤ ϵ && primalnorm > 100 * ϵ
    ρ = primalnorm * p2 #/ 10
    #elseif ngp ≤ 5ϵ
    #  ρ = ϵ
  end

  return ρ
end
