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
                      σ₂ = 2.0,
                      verbose = false
                     )
  m, n = size(A)

  Z = LinearOperator(nullspace(full(A)))
  normct = 1.0
  r = -1.0

  iter = 0

  while !(normct <= 2ρ && r >= η₁)
    d = cg(Z' * B * Z, -Z' * g, radius=Δ)[1]
    d = Z * d
    xt = x + d
    ct = cons(nlp, xt)
    normct = norm(ct)
    print("  ‖c(xt)‖ = $normct")

    if normct <= 2ρ
      ft = obj(nlp, xt)
      ℓxtλ = ft + dot(λ, ct)
      qd = dot(d, B * d)/2 + dot(g, d)
      @assert qd < 0

      r = (ℓxtλ - ℓzλ)/qd
      println(", r = $r, ft = $ft")
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
      println("")
    end
    iter += 1

  end

  return x
end
