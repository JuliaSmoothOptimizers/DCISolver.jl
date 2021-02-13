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
- :success if we computed z such that ‖c(z)‖ ≤ 2ρ and Δℓ ≥ η₁ q(d)

See https://github.com/JuliaSmoothOptimizers/SolverTools.jl/blob/78f6793f161c3aac2234aee8a27aa07f1df3e8ee/src/trust-region/trust-region.jl#L37
for `SolverTools.aredpred`
"""
function tangent_step(nlp      :: AbstractNLPModel, 
                      z        :: AbstractVector{T}, 
                      λ        :: AbstractVector{T}, 
                      cz       :: AbstractVector{T},
                      normcz   :: T,
                      fz       :: T,
                      LDL, 
                      vals     :: AbstractVector{T}, 
                      g        :: AbstractVector{T}, 
                      ℓzλ      :: T, 
                      gBg      :: T, 
                      ρ        :: AbstractFloat, 
                      γ        :: T, 
                      δ        :: T;
                      Δ        :: AbstractFloat= one(T), #trust-region radius
                      η₁       :: AbstractFloat= T(1e-2),
                      η₂       :: AbstractFloat= T(0.75),
                      σ₁       :: AbstractFloat= T(0.25), #decrease trust-region radius
                      σ₂       :: AbstractFloat= T(2.0), #increase trust-region radius after success
                      δmin     :: T = √eps(T),
                      small_d  :: AbstractFloat = eps(T), #below this value ||d|| is too small
                      max_eval :: Int = 1_000, #max number of evaluation of obj + cons
                      max_time :: AbstractFloat = 1_000., #max real time
                     ) where T

  m, n = nlp.meta.ncon, nlp.meta.nvar
  
  d = Array{T, 1}(undef, n)
  Δℓ = zero(T)

  status = :unknown
  iter = 0
  start_time = time()
  el_time = 0.0
  tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || el_time > max_time

  normct, r = normcz, -one(T)

  while !((normct ≤ 2ρ && r ≥ η₁) || tired)
    #Compute a descent direction d
    d, dBd, status, γ, δ, vals = compute_descent_direction(nlp, gBg, g, Δ, LDL, γ, δ, δmin, vals, d) #no evals
    n2d = dot(d,d)
    if √n2d > Δ
      d = d * (Δ / √n2d) #Just in case.
    end
    if √n2d < small_d
      status = :small_horizontal_step
      break
    end

    xt     = z + d
    ct     = cons(nlp, xt)
    normct = norm(ct)

    if normct ≤ 2ρ
      ft   = obj(nlp, xt)
      ℓxtλ = ft + dot(λ, ct)
      qd   = dBd / 2 + dot(g, d)

      Δℓ, pred = aredpred(nlp, ℓzλ, ℓxtλ, qd, xt, d, dot(g, d))

      r = Δℓ / qd
      if r < η₁ #we can decrease further Δ so that ≤ ||d||
        m = max(ceil(log(√n2d / Δ) / log(σ₁)), 1)
        Δ *= σ₁^m
      else #success
        status = :success
        z  = xt
        cz = ct
        fz = ft
        ℓzλ = ℓxtλ
        if r ≥ η₂ && √n2d ≥ 0.99Δ
          Δ *= σ₂
        end
      end
    else #we can decrease further Δ so that ≤ ||d||
      m = max(ceil(log(√n2d / Δ) / log(σ₁)), 1)
      Δ *= σ₁^m
    end

    @info log_row(Any["Tr", iter, neval_obj(nlp) + neval_cons(nlp), fz, ℓzλ,
                           NaN, norm(ct), NaN, NaN, status, √n2d, Δ])
    iter += 1

    el_time = time() - start_time
    tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || el_time > max_time
  end
  
  if tired
      status = :tired
  end

  return z, cz, fz, status, Δ, Δℓ, γ, δ #ℓzλ
end

"""
Compute a direction `d` with three possible outcomes:
- `:cauchy_step`
- `:newton`
- `:dogleg`
- `:interior_cauchy_step` when γ is too large.
for `min_d q(d) s.t. ‖d‖ ≤ Δ`.
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
      dn, dnBdn, dcpBdn, γ_too_large, γ, δ, vals = _compute_newton_step!(nlp, LDL, g, γ, δ, δmin, dcp, vals)
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
    end
@show gBg, dBd, dot(g, d), norm(dn), norm(dcp), dnBdn, dcpBdn, status
  return d, dBd, status, γ, δ, vals
end

"""
Compute a solution to
min_α q(-α g) s.t. ‖αg‖_2 ≤ Δ

return `dcp_on_boundary` true if ‖αg‖ = Δ,
return `dcp = - α g`
return `dcpBdcp = α^2 gBg`
and `α` the solution.
"""
function _compute_gradient_step(nlp, gBg, g, Δ)

    dcp_on_boundary = false
    dgg = dot(g,g)
    if gBg ≤ 1e-12 * dgg #generalize this test
      α = Δ / √dgg #norm(g)
      dcp_on_boundary = true
    else
      α = dgg / gBg #dot(g, g) / gBg
      if α > Δ / √dgg #norm(g)
        α = Δ / √dgg #norm(g)
        dcp_on_boundary = true
      end
    end
    dcp = -α * g
    dcpBdcp = α^2 * gBg
    
    return dcp_on_boundary, dcp, dcpBdcp, α
end

"""
Given two directions dcp and dn, compute the largest 0 ≤ τ ≤ 1 such that
‖dn + τ (dcp -dn)‖ = Δ
"""
function _compute_step_length(norm2dn, dotdndcp, norm2dcp, Δ :: T) where T <: AbstractFloat
    # d = τ dcp + (1 - τ) * dn = dn + τ * (dcp - dn)
    # ‖d‖² = Δ² => τ² ‖dcp - dn‖² + 2τ dnᵀ(dcp - dn) + ‖dn‖² - Δ² = 0
    # Δ = b² - 4ac
    q₀ = norm2dn - Δ^2
    q₁ = 2 * (dotdndcp - norm2dn)
    q₂ = norm2dcp - 2 * dotdndcp + norm2dn
    #q₀, q₁, q₂ = [q₀, q₁, q₂] / maximum(abs.([q₀, q₁, q₂]))
    q₀, q₁, q₂ = [q₀, q₁, q₂] / q₂ #so the first coefficient is 1.
    roots = Krylov.roots_quadratic(q₂, q₁, q₀) #Is this type stable?
    τ = length(roots) == 0 ? one(T) : min(one(T), roots...)
    
    return τ
end

"""
compute a step ****

return dn = 0. whenever γ > 1/eps(T)
"""
function _compute_newton_step!2(nlp, LDL, g, γ, δ, δmin, dcp, vals)

    m, n, nnzh, nnzj = nlp.meta.ncon, nlp.meta.nvar, nlp.meta.nnzh, nlp.meta.nnzj
    T = eltype(nlp.meta.x0)

    dζ = Array{T}(undef, m + n)
    dn = zeros(T, n) #Array{Float64}(undef, n)

    # When there is room for improvement, we try a dogleg step
    # A CG variant can be implemented, but it needs the nullspace matrix.
    rhs = [-g; zeros(T, m)]
    descent = false
    dnBdn = dcpBdn = zero(T)
    γ_too_large = false
    status = :unknown #:γ_too_large, :success_fact, :success_psd, :regularize

    @info log_header([:stage, :gamma, :gamma_max, :delta, :delta_min, :status],
                   [String, Float64, Float64, Float64, Float64, Symbol],
                   hdr_override=Dict(:gamma => "γ", :gamma_max => "γmax", :delta => "δ", :delta_min => "δmin")
                  )
    #@info log_row(Any["init", γ, 1/eps(T), δ, δmin])

    while !descent
      factorize!(LDL)
      status = if success(LDL)
        :success_fact
      elseif num_neg_eig(LDL) == m
        :success_psd
      else
        :regularize
      end
      if success(LDL) && num_neg_eig(LDL) == m
        solve!(dζ, LDL, rhs)
        dn = dζ[1:n]
        dλ = view(dζ, n+1:n+m)
        dnBdn  = - dot(g, dn) - γ * dot(dn, dn) - δ * dot(dλ, dλ)
        dcpBdn = - dot(g, dcp) - γ * dot(dcp, dn) # dcpᵀ Aᵀ dλ = (Adcp)ᵀ dλ = 0ᵀ dλ = 0
        status = :success
      end
      @info log_row(Any["Fact", γ, 1/eps(T), δ, δmin, status])

      while !(success(LDL) && num_neg_eig(LDL) == m)
        γ = max(100γ, √eps(T)) #max(10γ, √eps(T))
        if γ > 1/eps(T)
          γ_too_large = true
          dnBdn = zero(T)
          dcpBdn = zero(T)
          dn = zeros(n)
          break
        end
        nnz_idx = nnzh .+ nnzj .+ (1:n)
        vals[nnz_idx] .= γ
        nnz_idx = nnzh .+ nnzj .+ n .+ (1:m)
        δ = δmin
        vals[nnz_idx] .= -δ
        factorize!(LDL)
        status = if success(LDL)
                :success_fact
              elseif num_neg_eig(LDL) == m
                :success_psd
              else
                :regularize
              end
        if success(LDL) && num_neg_eig(LDL) == m
          solve!(dζ, LDL, rhs)
          dn = dζ[1:n]
          dλ = view(dζ, n+1:n+m)
          dnBdn = -dot(g, dn) - γ * dot(dn, dn) - δ * dot(dλ, dλ)
          dcpBdn = -dot(g, dcp) - γ * dot(dcp, dn) # dcpᵀ Aᵀ dλ = (Adcp)ᵀ dλ = 0ᵀ dλ = 0
          status = :success
        end
        @info log_row(Any["Fact", γ, 1/eps(T), δ, δmin, status])
      end
      descent = true
    end
    
    return dn, dnBdn, dcpBdn, γ_too_large, γ, δ, vals
end

include("factorization.jl")
