using NLPModels, LinearOperators, Krylov

function dci(nlp :: AbstractNLPModel,
             atol = 1e-8,
             rtol = 1e-6,
             ctol = 1e-6,
             max_f = 1000,
             max_time = 60
            )
  if !equality_constrained(nlp)
    error("DCI only works for equality constrained problems")
  end

  f(x) = obj(nlp, x)
  ∇f(x) = grad(nlp, x)
  H(x,y) = hess_op(nlp, x, y=y)
  c(x) = cons(nlp, x)
  J(x) = jac_op(nlp, x)

  x = nlp.meta.x0
  fx = f(x)
  ∇fx = ∇f(x)
  cx = c(x)
  Jx = J(x)
  # λ = argmin ‖∇f + Jᵀλ‖
  λ = cgls(Jx', -∇fx)[1]

  #ℓ(x,λ) = f(x) + λᵀc(x)
  ℓxλ = fx + dot(λ, cx)
  ∇ℓxλ = ∇fx + Jx'*λ
  Bx = hess_op(nlp, x, y=λ)

  ρmax = 1.0

  dualnorm = norm(∇ℓxλ)
  primalnorm = norm(cx)

  start_time = time()
  eltime = 0.0

  ϵ = 1e-4

  solved = primalnorm < ϵ && dualnorm < ϵ
  tired = sum_counters(nlp) > max_f || eltime > max_time

  iter = 0

  while !(solved || tired)
    println("$x $primalnorm $dualnorm")
    ngp = dualnorm/(norm(∇fx) + 1)
    z, cz, ρ = normal_step(nlp, ctol, x, cx, Jx, ρmax, ngp)
    ℓzλ = f(z) + dot(λ, cz)

    x = tangent_step(nlp, x, λ, Bx, ∇ℓxλ, Jx, ℓxλ, ρ)
    cx = c(x)
    ∇fx = ∇f(x)
    Jx = J(x)
    λ = cgls(Jx', -∇fx)[1]
    ℓxλ = fx + dot(λ, cx)
    ∇ℓxλ = ∇fx + Jx'*λ
    Bx = hess_op(nlp, x, y=λ)
    primalnorm = norm(cx)
    dualnorm = norm(∇ℓxλ)
    solved = primalnorm < ϵ && dualnorm < ϵ
    tired = sum_counters(nlp) > max_f || eltime > max_time
  end

  return x
end

"""    normal_step(nls, x, cx, Jx)

Approximately solves min ‖c(x)‖.

Given xₖ, finds min ‖cₖ + Jₖd‖
"""
function normal_step(nlp, ctol, x, cx, Jx, ρmax, ngp;
                     η₁ = 1e-2, η₂ = 0.66, σ₁ = 0.25, σ₂ = 4.0)

  c(x) = cons(nlp, x)
  z = copy(x)
  cz = copy(cx)
  normcz = norm(cz)

  Δ = 1.0

  ρ = max(min(ρmax * ngp, 0.75ρmax), ctol)
  normal_iter = 0

  while normcz > ρ
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

  end

  return z, cz, ρ
end

"""

min ¹/₂dᵀBd + dᵀg
s.t Ad = 0
    ‖d‖ ≦ Δ
"""
function tangent_step(nlp, x, λ, B, g, A, ℓzλ, ρ;
                      Δ=1.0,
                      η₁ = 0.25,
                      η₂ = 0.75,
                      σ₁ = 0.25,
                      σ₂ = 2.0
                     )
  m, n = size(A)

  Z = LinearOperator(nullspace(full(A)))
  normct = 1.0
  r = -1.0

  while !(normct <= 2ρ && r >= η₁)
    d = cg(Z' * B * Z, -Z' * g, radius=Δ)[1]
    d = Z * d
    println("d = $d")
    xt = x + d
    ct = cons(nlp, xt)
    normct = norm(ct)

    if normct > 2ρ
      ft = obj(nlp, xt)
      ℓxtλ = ft + dot(λ, ct)
      qd = dot(d, B * d)/2 + dot(g, d)

      r = (ℓzλ - ℓxtλ)/qd
      if r < η₁
        Δ *= σ₁
      else
        x = xt
        cx = ct
        ℓxλ = ℓxtλ
        if r > η₂ && norm(d) >= 0.99
          Δ *= σ₂
        end
      end
    else
      Δ *= σ₁
    end

  end

  return x
end
