"""

min ¹/₂dᵀBd + dᵀg
s.t Ad = 0
    ‖d‖ ≦ Δ
"""
function tangent_step(nlp, z, λ, B, g, A, ℓzλ, ρ;
                      Δ = 1.0,
                      η₁ = 0.25,
                      η₂ = 0.75,
                      σ₁ = 0.25,
                      σ₂ = 2.0,
                      max_eval = 1_000,
                      max_time = 1_000,
                     )
  m, n = size(A)

  status = :success

  Z = LinearOperator(nullspace(Matrix(A)))
  normct = 1.0
  r = -1.0

  iter = 0

  start_time = time()
  el_time = 0.0
  tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || el_time > max_time
  while !((normct <= 2ρ && r >= η₁) || tired)
    d = cg(Z' * B * Z, -Z' * g, radius=Δ)[1]
    d = Z * d
    if norm(d) > Δ
      d = d * (norm(d) / Δ)
    end
    if norm(d) < 1e-20
      status = :small_horizontal_step
      break
    end
    xt = z + d
    ct = cons(nlp, xt)
    normct = norm(ct)

    if normct <= 2ρ
      ft = obj(nlp, xt)
      ℓxtλ = ft + dot(λ, ct)
      qd = dot(d, B * d)/2 + dot(g, d)
      if qd >= 0
        @error("iter = $iter", "qd = $qd", "‖d‖ = $(norm(d))", "Δ = $Δ")
      end
      @assert qd < 0

      Δℓ = ℓxtλ - ℓzλ + max(1.0, abs(ℓzλ)) * 10 * eps()
      # roundoff error correction from Optimize.jl
      if abs(qd) < 1e4 * eps() || abs(Δℓ) < 1e4 * eps() * abs(ℓzλ)
        gt = grad(nlp, xt)
        Δℓ = (dot(g, d) + dot(gt, d)) / 2
      end
      r = Δℓ / qd
      if r < η₁
        Δ *= σ₁
      else
        z = xt
        cx = ct
        ℓxλ = ℓxtλ
        if r > η₂ && norm(d) >= 0.99
          Δ *= σ₂
        end
      end
    else
      Δ *= σ₁
    end
    iter += 1

    el_time = time() - start_time
    tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || el_time > max_time
  end

  return z, status
end
