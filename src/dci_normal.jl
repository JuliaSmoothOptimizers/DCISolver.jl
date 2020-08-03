"""    normal_step(nls, x, cx, Jx)

Approximately solves min ‖c(x)‖.

Given xₖ, finds min ‖cₖ + Jₖd‖
"""
function normal_step(nlp, ctol, x, cx, Jx, ρ;
                     η₁ = 1e-3, η₂ = 0.66, σ₁ = 0.25, σ₂ = 4.0,
                     max_eval = 1_000, max_time = 60,
                    )

  c(x) = cons(nlp, x)
  z = copy(x)
  cz = copy(cx)
  normcz = norm(cz)

  Δ = 1.0

  normal_iter = 0
  consecutive_bad_steps = 0 # Bad steps are when ‖c(z)‖ / ‖c(x)‖ > 0.95
  normcx = normcz           # c(x) = normcx = normcz for the first z

  start_time = time()
  el_time = 0.0
  tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || el_time > max_time
  infeasible = false
  while !(normcz ≤ ρ || tired || infeasible)
    d = -Jx'*cz
    if norm(d) < ctol * normcz
      infeasible = true
      break
    end
    Jd = Jx * d
    t = dot(d, d) / dot(Jd, Jd)
    dcp = t * d
    if norm(dcp) > Δ
      d = dcp * Δ / norm(dcp)
    else
      dn  = lsmr(Jx, -cz)[1]
      if norm(dn) <= Δ
        d = dn
      else
        v = dn - dcp
        τ = (-dot(dcp, v) + sqrt(dot(dcp, v)^2 + 4 * dot(v, v) * (Δ^2 - dot(dcp, dcp)))) / dot(v, v)
        d = dcp + τ * v
      end
    end
    Jd = Jx * d
    zp = z + d
    czp = c(zp)
    Pred = 0.5*(normcz^2 - norm(Jd + cz)^2)
    Ared = 0.5*(normcz^2 - norm(czp)^2)

    if Ared/Pred < η₁
      Δ = max(1e-8, Δ * σ₁)
    else
      z = zp
      Jx = jac_op(nlp, z)
      cz = czp
      normcz = norm(czp)
      if Ared/Pred > η₂ && norm(d) >= 0.99Δ
        Δ *= σ₂
      end
    end

    if normcz / normcx > 0.95
      consecutive_bad_steps += 1
    else
      consecutive_bad_steps = 0
    end

    # Safeguard AKA agressive normal step - Loses robustness, doesn't seem to fix any
    # if normcz > ρ && consecutive_bad_steps ≥ 3
    #   d = cg(hess_op(nlp, z, cz, obj_weight=0.0), Jx' * cz)[1]
    #   z -= d
    #   cz = c(z)
    #   normcz = norm(cz)
    # end

    el_time = time() - start_time
    tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || el_time > max_time
  end

  status = if normcz ≤ ρ
    :success
  elseif tired
    if neval_obj(nlp) + neval_cons(nlp) > max_eval
      :max_eval
    elseif el_time > max_time
      :max_time
    else
      :unknown_tired
    end
  elseif infeasible
    :infeasible
  else
    :unknown
  end

  return z, cz, status
end
