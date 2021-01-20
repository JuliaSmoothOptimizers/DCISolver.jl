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
- :success if we computed z such that ||c(z)|| ≤ 2ρ and Δℓ ≥ η₁ q(d)
"""
function tangent_step(nlp, z, λ, LDL, vals, g, ℓzλ, gBg, ρ, δ, γ;
                      Δ = 1.0, #trust-region radius
                      η₁ = 1e-2,
                      η₂ = 0.75,
                      σ₁ = 0.25, #decrease trust-region radius
                      σ₂ = 2.0, #increase trust-region radius after success
                      small_d = 1e-20, #below this value ||d|| is too small
                      max_eval = 1_000, #max number of evaluation of obj + cons
                      max_time = 1_000., #max real time
                     )
  m, n = nlp.meta.ncon, nlp.meta.nvar
  
  d = Array{Float64,1}(undef, n)
  
  δmin = 1e-8
  Δℓ = 0.0

  status = :unknown
  iter = 0
  start_time = time()
  el_time = 0.0
  tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || el_time > max_time
  normct, r = 1.0, -1.0
  while !((normct ≤ 2ρ && r ≥ η₁) || tired)
    #Compute a descent direction d
    d, dBd, status, γ, δ, vals = compute_descent_direction(nlp, gBg, g, Δ, LDL, γ, δ, δmin, vals, d)
    n2d = dot(d,d)
    if √n2d > Δ
      d = d * (Δ / √n2d) #Just in case.
    end
    if √n2d < small_d
      status = :small_horizontal_step
      break
    end
    xt = z + d
    ct = cons(nlp, xt)
    normct = norm(ct)

    if normct ≤ 2ρ
      ft   = obj(nlp, xt)
      ℓxtλ = ft + dot(λ, ct)
      qd   = dBd / 2 + dot(g, d)

      #SolverTools.aredpred https://github.com/JuliaSmoothOptimizers/SolverTools.jl/blob/78f6793f161c3aac2234aee8a27aa07f1df3e8ee/src/trust-region/trust-region.jl#L37
      Δℓ, pred = aredpred(nlp, ℓzλ, ℓxtλ, qd, xt, d, dot(g, d))

      r = Δℓ / qd
      if r < η₁
        Δ *= σ₁
      else #success
        status = :success
        z = xt
        cx = ct
        ℓxλ = ℓxtλ
        if r ≥ η₂ && √n2d >= 0.99 Δ
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
  
  if tired
      status = :tired
  end

  return z, status, Δ, Δℓ, γ, δ #cx, ℓxλ
end

"""
Compute a direction `d` with three possible outcomes:
- `:cauchy_step`
- `:newton`
- `:dogleg`
- `:interior_cauchy_step` when γ is too large.
for `min_d q(d) s.t. ||d|| ≤ Δ`.
"""
function compute_descent_direction(nlp, gBg, g, Δ, LDL, γ, δ, δmin, vals, d)
    m, n = nlp.meta.ncon, nlp.meta.nvar
    
    #first compute a gradient step
    dcp_on_boundary, dcp, dcpBdcp, α = _compute_gradient_step(nlp, gBg, g, Δ)

    if dcp_on_boundary # When the Cauchy step is in the boundary, we use it
      status = :cauchy_step
      dBd = dcpBdcp
      d = dcp
    else
      dn, dnBdn, dcpBdn,  γ_too_large, γ, δ, vals = _compute_newton_step!(nlp, LDL, g, γ, δ, δmin, dcp, vals)
      if γ_too_large 
          #dn = 0 here.
          if norm(dcp) < Δ #just to be sure
              d = dcp
              dBd = dcpBdcp
              status = :interior_cauchy_step
          end
      end
      norm2dn = dot(dn, dn)
      
      if √norm2dn < Δ # Both Newton and Cauchy are inside the TR.
        status = :newton
        dBd = dnBdn
        d = dn
      else
        dotdndcp, norm2dcp = dot(dn, dcp), dot(dcp, dcp)
        τ = _compute_step_length(norm2dn, dotdndcp, norm2dcp, Δ)
        d = dn + τ * (dcp - dn)
        dBd = τ^2 * dcpBdcp + 2 * τ * (1 - τ) * dcpBdn + (1 - τ)^2 * dnBdn
        status = :dogleg
      end
    end #end of "d="
  return d, dBd, status, γ, δ, vals
end

"""
Compute a solution to
min_α q(-α g) s.t. ||αg||_2 ≤ Δ

return `dcp_on_boundary` true if ||αg|| = Δ,
return `dcp = - α g`
return `dcpBdcp = α^2 gBg`
and `α` the solution.
"""
function _compute_gradient_step(nlp, gBg, g, Δ)
    dcp_on_boundary = false
    dgg = dot(g,g)
    if gBg ≤ 1e-12 * dgg
      α = Δ / sqrt(dgg) #norm(g)
      dcp_on_boundary = true
    else
      α = dgg / gBg #dot(g, g) / gBg
      if α > Δ / sqrt(dgg) #norm(g)
        α = Δ / sqrt(dgg) #norm(g)
        dcp_on_boundary = true
      end
    end
    dcp = -α * g
    dcpBdcp = α^2 * gBg
    
    return dcp_on_boundary, dcp, dcpBdcp, α
end

"""
Given two directions dcp and dn, compute the largest 0 ≤ τ ≤ 1 such that
||dn + τ (dcp -dn)|| = Δ
"""
function _compute_step_length(norm2dn, dotdndcp, norm2dcp, Δ)
    # d = τ dcp + (1 - τ) * dn = dn + τ * (dcp - dn)
    # ‖d‖² = Δ² => τ² ‖dcp - dn‖² + 2τ dnᵀ(dcp - dn) + ‖dn‖² - Δ² = 0
    # Δ = b² - 4ac
    q₀ = norm2dn - Δ^2
    q₁ = 2 * (dotdndcp - norm2dn)
    q₂ = norm2dcp - 2 * dotdndcp + norm2dn
    #q₀, q₁, q₂ = [q₀, q₁, q₂] / maximum(abs.([q₀, q₁, q₂]))
    q₀, q₁, q₂ = [q₀, q₁, q₂] / q₂ #so the first coefficient is 1.
    roots = Krylov.roots_quadratic(q₂, q₁, q₀)
    τ = length(roots) == 0 ? 1.0 : min(1.0, roots...)
    
    return τ
end

"""
compute a step ****

return dn = 0. whenever γ > 1e16
"""
function _compute_newton_step!(nlp, LDL, g, γ, δ, δmin, dcp, vals)
    m, n, nnzh, nnzj = nlp.meta.ncon, nlp.meta.nvar, nlp.meta.nnzh, nlp.meta.nnzj
    dζ = Array{Float64}(undef, m + n)#zeros(m + n)
    dn = zeros(n) #Array{Float64}(undef, n)
    # When there is room for improvement, we try a dogleg step
    # A CG variant can be implemented, but it needs the nullspace matrix.
    rhs = [-g; zeros(m)]
    descent = false
    dnBdn = dcpBdn = 0.0
    γ_too_large = false
    while !descent
      factorize!(LDL)
      if success(LDL) && num_neg_eig(LDL) == m
        solve!(dζ, LDL, rhs)
        dn = dζ[1:n]
        dλ = view(dζ, n+1:n+m)
        dnBdn  = - dot(g, dn) - γ * dot(dn, dn) - δ * dot(dλ, dλ)
        dcpBdn = - dot(g, dcp) - γ * dot(dcp, dn) # dcpᵀ Aᵀ dλ = (Adcp)ᵀ dλ = 0ᵀ dλ = 0
      end

      while !(success(LDL) && num_neg_eig(LDL) == m)
        γ = max(10γ, 1e-8)
        if γ > 1e16
          γ_too_large = true
          dnBdn = 0.
          dcpBdn = 0.
          dn = zeros(n)
          break
        end
        nnz_idx = nnzh .+ nnzj .+ (1:n)
        vals[nnz_idx] .= γ
        nnz_idx = nnzh .+ nnzj .+ n .+ (1:m)
        δ = δmin
        vals[nnz_idx] .= -δ
        factorize!(LDL)
        if success(LDL) && num_neg_eig(LDL) == m
          solve!(dζ, LDL, rhs)
          dn = dζ[1:n]
          dλ = view(dζ, n+1:n+m)
          dnBdn = -dot(g, dn) - γ * dot(dn, dn) - δ * dot(dλ, dλ)
          dcpBdn = -dot(g, dcp) - γ * dot(dcp, dn) # dcpᵀ Aᵀ dλ = (Adcp)ᵀ dλ = 0ᵀ dλ = 0
        end
      end
      descent = true
    end
    
    return dn, dnBdn, dcpBdn, γ_too_large, γ, δ, vals
end
