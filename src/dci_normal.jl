"""    normal_step(nls, x, cx, Jx)

Approximately solves min ‖c(x)‖.

Given xₖ, finds min ‖cₖ + Jₖd‖
"""
function normal_step(nlp, ctol, x, cx, Jx, ρmax, ngp;
                     η₁ = 1e-2, η₂ = 0.66, σ₁ = 0.25, σ₂ = 4.0,
                     max_eval = 1_000, max_time = 60,
                    )

  c(x) = cons(nlp, x)
  z = copy(x)
  cz = copy(cx)
  normcz = norm(cz)

  Δ = 1.0

  ρ = max(min(ρmax * ngp, 0.75ρmax), ctol)
  normal_iter = 0

  start_time = time()
  el_time = 0.0
  tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || el_time > max_time
  while !(normcz ≤ ρ || tired)
    d = -Jx'*cz
    Jd = Jx * d
    t = dot(d,d)/dot(Jd,Jd)
    dcp = t * d
    if norm(dcp) > Δ
      d = dcp * Δ / norm(dcp)
    else
      dn  = cgls(Jx, -cz)[1]
      if norm(dn) <= Δ
        d = dn
      else
        v = dn - dcp
        τ = (-dot(dcp, v) + sqrt(dot(dcp, v)^2 + 4*dot(v, v)*(Δ^2 - dot(dcp, dcp))))/dot(v, v)
        d = dcp + τ * v
      end
    end
    Jd = Jx * d
    zp = z + d
    czp = c(zp)
    Pred = 0.5*(normcz^2 - norm(Jd + cz)^2)
    Ared = 0.5*(normcz^2 - norm(czp)^2)

    if Ared/Pred < η₁
      Δ *= σ₁
    else
      z = zp
      Jx = jac_op(nlp, z)
      cz = czp
      normcz = norm(czp)
      if Ared/Pred > η₂ && norm(d) >= 0.99Δ
        Δ *= σ₂
      end
    end

    el_time = time() - start_time
    tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || el_time > max_time
  end

  return z, cz, ρ
end
