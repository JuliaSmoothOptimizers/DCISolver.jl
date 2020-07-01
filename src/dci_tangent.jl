"""

min ¹/₂dᵀBd + dᵀg
s.t Ad = 0
    ‖d‖ ≦ Δ
"""
function tangent_step(nlp, z, λ, rows, cols, vals, g, A, ℓzλ, ρ, γ;
                      Δ = 1.0,
                      η₁ = 0.25,
                      η₂ = 0.75,
                      σ₁ = 0.25,
                      σ₂ = 2.0,
                      max_eval = 1_000,
                      max_time = 1_000,
                     )
  m, n = size(A)
  status = :unknown

  nnzh = nlp.meta.nnzh

  B = Symmetric(sparse(rows[1:nnzh], cols[1:nnzh], vals[1:nnzh], n, n), :L) # TODO: !!!

  normct = 1.0
  r = -1.0

  iter = 0

  start_time = time()
  el_time = 0.0
  tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || el_time > max_time
  while !((normct <= 2ρ && r >= η₁) || tired)

    dcp_on_boundary = false
    gBg = dot(g, B * g)
    if gBg ≤ 1e-12 * dot(g,g)
      α = Δ / norm(g)
      dcp_on_boundary = true
    else
      α = dot(g, g) / gBg
      if α > Δ / norm(g)
        α = Δ / norm(g)
        dcp_on_boundary = true
      end
    end
    dcp = -α * g

    d = if dcp_on_boundary
      # When the Cauchy step is in the boundary, we use it
      status = :cauchy_step
      dcp
    else
      # When there is room for improvement, we try a dogleg step
      # A CG variant can be implemented, but it needs the nullspace matrix.
      descent = false
      local dn
      while !descent
        # TODO: Use MA57 and LDLFactorization to solve the system
        H = Symmetric(sparse(rows, cols, vals, m + n, m + n), :L)
        try
          dζ = H \ [-g; zeros(m)]
          dn = dζ[1:n]
        catch
          dn = zeros(n)
        end
        if dot(dn, g) ≥ -1e-4 * norm(g)^2
          γ = max(10γ, 1e-8)
          if γ > 1e8
            error("γ too large. TODO: Fix here")
          end
          nnz_idx = nlp.meta.nnzh .+ nlp.meta.nnzj .+ (1:nlp.meta.nvar)
          vals[nnz_idx] .= γ
          H = Symmetric(sparse(rows, cols, vals, m + n, m + n), :L)
          try
            dζ = H \ [-g; zeros(m)]
            dn = dζ[1:n]
          catch
            dn = zeros(n)
          end
        else
          descent = true
        end
      end

      if norm(dn) < Δ # Both Newton and Cauchy are inside the TR.
        status = :newton
        dn
      else
        # d = τ dcp + (1 - τ) * dn = dn + τ * (dcp - dn)
        # ‖d‖² = Δ² => τ² ‖dcp - dn‖² + 2τ dnᵀ(dcp - dn) + ‖dn‖² - Δ² = 0
        # Δ = b² - 4ac
        q₀ = dot(dn, dn) - Δ^2
        q₁ = 2 * (dot(dn, dcp) - dot(dn, dn))
        q₂ = dot(dcp, dcp) - 2 * dot(dcp, dn) + dot(dn, dn)
        q₀, q₁, q₂ = [q₀, q₁, q₂] / maximum(abs.([q₀, q₁, q₂]))
        roots = Krylov.roots_quadratic(q₂, q₁, q₀)
        τ = length(roots) == 0 ? 1.0 : min(1.0, roots...)

        d = dn + τ * (dcp - dn)
        status = :dogleg
        d
      end
    end
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
        @error "status = $status"
        @error("iter = $iter", "qd = $qd", "‖d‖ = $(norm(d))", "Δ = $Δ")
      end
      @assert qd < 0

      # Trust region update. TODO: Change to SolverTools.jl
      Δℓ = ℓxtλ - ℓzλ + max(1.0, abs(ℓzλ)) * 10 * eps()
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
        if r > η₂ && norm(d) >= 0.99 Δ
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

  return z, status, Δ, γ
end
