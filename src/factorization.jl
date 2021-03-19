"""
1st idea: we stop whenever the fonction looks convex-ish
"""
function _compute_newton_step!(nlp  :: AbstractNLPModel, 
                               LDL  :: SymCOOSolver, 
                               g    :: AbstractVector{T}, 
                               γ    :: T, 
                               δ    :: T, 
                               δmin :: T, 
                               dcp  :: AbstractVector{T}, 
                               vals :: AbstractVector{T},
                               meta :: MetaDCI) where T

  m, n = nlp.meta.ncon, nlp.meta.nvar
  nnzh, nnzj = nlp.meta.nnzh, nlp.meta.nnzj

  dζ = Array{T}(undef, m + n)
  dn = zeros(T, n) #Array{Float64}(undef, n)

  # When there is room for improvement, we try a dogleg step
  rhs = [-g; zeros(T, m)]
  dnBdn = dcpBdn = zero(T)
  gnorm = norm(g)
  slope = NaN
  γ_too_large = false
  γ0 = copy(γ)
  status = :unknown #:γ_too_large, :success_fact, :regularize

  @info log_header([:stage, :-, :-, :gamma, :delta, :delta_min, :-, :slope, :-, :-, :-, :-],
                  [String, Int, Int, Float64, Float64, Float64, Float64, Float64, Float64, Symbol],
                  hdr_override=Dict(:gamma => "γ",
                                    :delta => "δ", :delta_min => "δmin")
                  )

  descent = false
  while !descent
    factorize!(LDL)

    if success(LDL)
      solve!(dζ, LDL, rhs)
      dn = dζ[1:n]
      dλ = view(dζ, n+1:n+m)
      slope  = dot(g, dn)
      dnBdn  = - slope - γ * dot(dn, dn) - δ * dot(dλ, dλ)
      dcpBdn = - dot(g, dcp) - γ * dot(dcp, dn) # dcpᵀ Aᵀ dλ = 0
      if dnBdn > 0.0 #slope < -1.0e-5 #4 * norm(dn) * gnorm #dnBdn > 0.0 
        status = :success
        descent = true
      else
        status = :success_fact
      end
    else
      status = :regularize
    end
    @info log_row(Any["Fact", Int, Int, γ, δ, δmin, Float64, slope, Float64, status, norm(dn), Float64])

    if !descent
      if γ ≥ 1/√eps(T)
        γ_too_large = true
        γ = γ0 #regularization failed
        dnBdn = zero(T)
        dcpBdn = zero(T)
        dn = zeros(n)
        break
      end
      γ = min(max(100γ, √eps(T)), 1/√eps(T))
      nnz_idx = nnzh .+ nnzj .+ (1:n)
      vals[nnz_idx] .= γ
      nnz_idx = nnzh .+ nnzj .+ n .+ (1:m)
      δ = δmin
      vals[nnz_idx] .= -δ
    end
  end
    
  return dn, dnBdn, dcpBdn, γ_too_large, γ, δ, vals
end
