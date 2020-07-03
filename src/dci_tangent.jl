"""

min ¹/₂dᵀBd + dᵀg
s.t Ad = 0
    ‖d‖ ≦ Δ
"""
function tangent_step(nlp, z, λ, LDL, vals, g, ℓzλ, gBg, ρ, δ, γ;
                      Δ = 1.0,
                      η₁ = 0.25,
                      η₂ = 0.75,
                      σ₁ = 0.25,
                      σ₂ = 2.0,
                      max_eval = 1_000,
                      max_time = 1_000,
                     )
  m, n = nlp.meta.ncon, nlp.meta.nvar
  status = :unknown

  nnzh = nlp.meta.nnzh

  normct = 1.0
  r = -1.0
  dζ = zeros(m + n)
  rhs = [-g; zeros(m)]

  iter = 0

  δmin = 1e-8

  start_time = time()
  el_time = 0.0
  tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || el_time > max_time
  while !((normct <= 2ρ && r >= η₁) || tired)
    dcp_on_boundary = false
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
    dcpBdcp = α^2 * gBg
    dnBdn = dBd = dcpBdn = 0.0

    d = if dcp_on_boundary
      # When the Cauchy step is in the boundary, we use it
      status = :cauchy_step
      dBd = dcpBdcp
      dcp
    else
      # When there is room for improvement, we try a dogleg step
      # A CG variant can be implemented, but it needs the nullspace matrix.
      descent = false
      local dn
      while !descent
        factorize!(LDL)
        if success(LDL) && num_neg_eig(LDL) == m
          solve!(dζ, LDL, rhs)
          dn = dζ[1:n]
          dλ = view(dζ, n+1:n+m)
          dnBdn = -dot(g, dn) - γ * dot(dn, dn) - δ * dot(dλ, dλ)
          dcpBdn = -dot(g, dcp) - γ * dot(dcp, dn) # dcpᵀ Aᵀ dλ = (Adcp)ᵀ dλ = 0ᵀ dλ = 0
        end

        if !success(LDL) || dot(dn, g) ≥ -1e-4 * norm(g)^2 # Can I change to while !success(LDL)?
          γ = max(10γ, 1e-8)
          if γ > 1e8
            error("γ too large. TODO: Fix here")
          end
          nnz_idx = nlp.meta.nnzh .+ nlp.meta.nnzj .+ (1:nlp.meta.nvar)
          vals[nnz_idx] .= γ
          nnz_idx = nlp.meta.nnzh .+ nlp.meta.nnzj .+ nlp.meta.nvar .+ (1:nlp.meta.ncon)
          δ = δmin * ρ
          vals[nnz_idx] .= -δ
          factorize!(LDL)
          if success(LDL) && num_neg_eig(LDL) == m
            solve!(dζ, LDL, rhs)
            dn = dζ[1:n]
            dλ = view(dζ, n+1:n+m)
            dnBdn = -dot(g, dn) - γ * dot(dn, dn) - δ * dot(dλ, dλ)
            dcpBdn = -dot(g, dcp) - γ * dot(dcp, dn) # dcpᵀ Aᵀ dλ = (Adcp)ᵀ dλ = 0ᵀ dλ = 0
          end
        else
          descent = true
        end
      end

      if norm(dn) < Δ # Both Newton and Cauchy are inside the TR.
        status = :newton
        dBd = dnBdn
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
        dBd = τ^2 * dcpBdcp + 2 * τ * (1 - τ) * dcpBdn + (1 - τ)^2 * dnBdn
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
      qd = dBd / 2 + dot(g, d)
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

  return z, status, Δ, γ, δ
end
