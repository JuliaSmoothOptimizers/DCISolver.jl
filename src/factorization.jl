"""
1st idea: we stop whenever the fonction looks convex-ish
"""
function _compute_newton_step!(nlp  :: AbstractNLPModel, 
                               LDL  :: SymCOOSolverInterface.SymCOOSolver, 
                               g    :: AbstractVector{T}, 
                               γ    :: T, 
                               δ    :: T, 
                               δmin :: T, 
                               dcp  :: AbstractVector{T}, 
                               vals :: AbstractVector{T}) where T

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
    status = :unknown #:γ_too_large, :success_fact, :regularize

    @info log_header([:stage, :gamma, :delta, :delta_min, :ndn, :slope, :status],
                   [String, Float64, Float64, Float64, Float64, Float64, Symbol],
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
        dcpBdn = - dot(g, dcp) - γ * dot(dcp, dn) # dcpᵀ Aᵀ dλ = (Adcp)ᵀ dλ = 0ᵀ dλ = 0
        if dnBdn > 0.0 #slope < -1.0e-5 #4 * norm(dn) * gnorm #dnBdn > 0.0 
          status = :success
          descent = true
        else
          status = :success_fact
        end
      else
        status = :regularize
      end
      @info log_row(Any["Fact", γ, δ, δmin, norm(dn), slope, status])

      if !descent
        if γ ≥ 1/√eps(T)
          γ_too_large = true
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

#=
"""
2nd idea: we take the absolute value of the diagonal
Only with LDLFactorizations, how do we get/set the diagonal with HSL?
"""
function _compute_newton_step!(nlp  :: AbstractNLPModel, 
                               LDL  :: SymCOOSolverInterface.LDLFactorizationStruct, 
                               g    :: AbstractVector{T}, 
                               γ    :: T, 
                               δ    :: T, 
                               δmin :: T, 
                               dcp  :: AbstractVector{T}, 
                               vals :: AbstractVector{T}) where T

    m, n = nlp.meta.ncon, nlp.meta.nvar
    nnzh, nnzj = nlp.meta.nnzh, nlp.meta.nnzj

    dζ = Array{T}(undef, m + n)
    dn = zeros(T, n) #Array{Float64}(undef, n)

    # When there is room for improvement, we try a dogleg step
    rhs = [-g; zeros(T, m)]
    dnBdn = dcpBdn = zero(T)
    γ_too_large = false
    status = :unknown #:γ_too_large, :success_fact, :regularize

    @info log_header([:stage, :gamma, :gamma_max, :delta, :delta_min, :status],
                   [String, Float64, Float64, Float64, Float64, Symbol],
                   hdr_override=Dict(:gamma => "γ", :gamma_max => "γmax", 
                                     :delta => "δ", :delta_min => "δmin")
                  )

    #Dynamic regularization:
    LDL.factor.n_d =  n
    LDL.factor.r1  =  √eps(T) #γ
    LDL.factor.r2  = -√eps(T) #δ
    LDL.factor.tol = √eps(T)

    descent = false
    while !descent
      factorize!(LDL)

      if success(LDL) #why would we fail?
        LDL.factor.d .= abs.(LDL.factor.d)
        solve!(dζ, LDL, rhs)
        dn = dζ[1:n]
        dλ = view(dζ, n+1:n+m)
        dnBdn  = - dot(g, dn) - γ * dot(dn, dn) - δ * dot(dλ, dλ)
        dcpBdn = - dot(g, dcp) - γ * dot(dcp, dn) # dcpᵀ Aᵀ dλ = (Adcp)ᵀ dλ = 0ᵀ dλ = 0
        if dnBdn > 0.0 #we have a descent direction
          status = :success
          descent = true
        else
          status = :success_fact
        end
      else
        @warn "Why would we fail?"
        status = :regularize
      end
      @info log_row(Any["Fact-|dyn|", γ, 1/√eps(T), δ, δmin, status])

      if !descent
        γ = min(max(100γ, √eps(T)), 1/√eps(T))
        if γ ≥ 1/√eps(T)
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
      end
    end
    
    return dn, dnBdn, dcpBdn, γ_too_large, γ, δ, vals
end
=#

#=
"""
2nd idea: we take the absolute value of the diagonal
Only with LDLFactorizations, how do we get/set the diagonal with HSL?
"""
function _compute_newton_step!(nlp  :: AbstractNLPModel, 
                               LDL  :: SymCOOSolverInterface.LDLFactorizationStruct, 
                               g    :: AbstractVector{T}, 
                               γ    :: T, 
                               δ    :: T, 
                               δmin :: T, 
                               dcp  :: AbstractVector{T}, 
                               vals :: AbstractVector{T}) where T

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
    status = :unknown #:γ_too_large, :success_fact, :regularize

    @info log_header([:stage, :gamma, :delta, :delta_min, :ndn, :slope, :status],
                   [String, Float64, Float64, Float64, Float64, Float64, Symbol],
                   hdr_override=Dict(:gamma => "γ", 
                                     :delta => "δ", :delta_min => "δmin")
                  )

    #Dynamic regularization:

    LDL.factor.n_d =  n
    LDL.factor.r1  = √eps(T) #1e-5#δmin # √eps(T) #γ
    LDL.factor.r2  = -√eps(T) #-√eps(T)# 1e-5#-δmin  # -√eps(T) #δ
    LDL.factor.tol = √eps(T) #1e-5#δmin

nb_fact = 0
    descent = false
    while !descent
      if !success(LDL)
        factorize!(LDL)
        nb_fact += 1
      end

      if success(LDL) #why would we fail?
        #=
        dd_indx = findall(x-> (1 ≤ x ≤ n), LDL.factor.P) #indices of 1st block
        dd2 = findall(x-> (n+1 ≤ x ≤ n+m), LDL.factor.P)
        λ = minimum(LDL.factor.d[dd_indx])
        γreg = -λ + γ + √eps(T)
        δreg   = √eps(T) #assuming we always regularize
        if λ < 0.0
          LDL.factor.d[dd_indx] .+= γreg
        end
@show λ, minimum(LDL.factor.d[dd_indx]), maximum(LDL.factor.d[dd_indx]), minimum(LDL.factor.d[dd2]), maximum(LDL.factor.d[dd2])
        =#
        solve!(dζ, LDL, rhs)
        dn = dζ[1:n]
        dλ = view(dζ, n+1:n+m)
        slope  = dot(g, dn)
        Bdζ = copy(dζ)
        LDLFactorizations.lmul!(LDL.factor, Bdζ)
        #dnBdn  = - slope #- γreg * dot(dn, dn) #- δreg * dot(dλ, dλ) #- slope - γ * dot(dn, dn) - δ * dot(dλ, dλ)
#@show dnBdn, dn' * Bdζ[1:n]
        dnBdn = dn' * Bdζ[1:n]
        #dcpBdn = - dot(g, dcp) #- γreg * dot(dcp, dn) #- dot(g, dcp) - γ * dot(dcp, dn) # dcpᵀ Aᵀ dλ = (Adcp)ᵀ dλ = 0ᵀ dλ = 0
#@show dcpBdn, dcp' * Bdζ[1:n]
        dcpBdn = dcp' * Bdζ[1:n]
        if slope < -1.0e-4 * gnorm#* norm(dn) * gnorm #dnBdn > 0.0  #slope < -1.0e-5 #4 * norm(dn) * gnorm #dnBdn > 0.0 
                       #we have a descent direction, we should be more strict dnBdn > 1e-5
                       #we could also check dot(dn, g)
          status = :success
          descent = true
        else
          status = :success_fact
        end
      else
        @warn "Why would we fail?"
        status = :regularize
      end
      @info log_row(Any["Fact-|dyn|", γ, δ, δmin, norm(dn), slope, status])

      if !descent
        if γ ≥ 1/√eps(T)
          γ_too_large = true
          dnBdn = zero(T)
          dcpBdn = zero(T)
          dn = zeros(n)
          break
        end
        γ = min(max(100γ, √eps(T)), 1/√eps(T))
        δ = δmin#δmin #done by the regularization
        #if success(LDL)
        dd_indx = findall(x-> (1 ≤ x ≤ n), LDL.factor.P) #indices of 1st block
        λ = min(minimum(LDL.factor.d[dd_indx]), zero(T))
        #  λ = min(minimum(LDL.factor.d[1:n]), zero(T))
        #LDL.factor.d[dd_indx] .+= -λ #max.(LDL.factor.d[dd_indx], 0.) .+ γ #abs(LDL.factor.d[dd_indx]) #-λ + γ
        LDL.factor.d[dd_indx] .+= -min.(LDL.factor.d[dd_indx] .- γ, zero(T))
        #LDL.factor.d .-= min.(LDL.factor.d .- γ, zero(T)) # -λ + γ
          #@show num_neg_eig(LDL), λ, minimum(LDL.factor.d[1:n]), γ, maximum(LDL.factor.d[n+1:n+m])
          #LDL.factor.d[n+1:n+m] .+= -δ #already handled by the regularization
        #else
        #  nnz_idx = nnzh .+ nnzj .+ (1:n)
        #  vals[nnz_idx] .= γ
        #  nnz_idx = nnzh .+ nnzj .+ n .+ (1:m)
        #  vals[nnz_idx] .= -δ
        #end
      end
    end
    if nb_fact > 1 @warn "Wait more that one factorization?" end
    return dn, dnBdn, dcpBdn, γ_too_large, γ, δ, vals
end
=#

"""
compute a step ****

return dn = 0. whenever γ > 1/eps(T)
"""
function _compute_newton_step!2(nlp  :: AbstractNLPModel, 
                                LDL  :: SymCOOSolverInterface.SymCOOSolver, 
                                g    :: AbstractVector{T}, 
                                γ    :: T, 
                                δ    :: T, 
                                δmin :: T, 
                                dcp  :: AbstractVector{T}, 
                                vals :: AbstractVector{T}) where T

    m, n, nnzh, nnzj = nlp.meta.ncon, nlp.meta.nvar, nlp.meta.nnzh, nlp.meta.nnzj

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
