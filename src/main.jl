"""
    dci(nlp; kwargs...)

This method implements the Dynamic Control of Infeasibility for equality-constrained
problems described in

    Dynamic Control of Infeasibility in Equality Constrained Optimization
    Roberto H. Bielschowsky and Francisco A. M. Gomes
    SIAM J. Optim., 19(3), 1299–1325.
    https://doi.org/10.1137/070679557

"""
function dci(nlp  :: AbstractNLPModel,
             x    :: AbstractVector{T};# = nlp.meta.x0,
             atol :: AbstractFloat = 1e-5,
             rtol :: AbstractFloat = 1e-5,
             ctol :: AbstractFloat = 1e-5,
             linear_solver :: Symbol = :ma57, #:ldlfact,
             max_eval :: Int = 50000,
             max_time :: Float64 = 60.
            ) where T

  if !equality_constrained(nlp)
    error("DCI only works for equality constrained problems")
  end
  if !(linear_solver ∈ keys(solver_correspondence))
    @warn "linear solver $linear_solver not found in $(collect(keys(solver_correspondence))). Using :ldlfact instead"
    linear_solver = :ldlfact
  end

  evals(nlp) = neval_obj(nlp) + neval_cons(nlp)

  ##########################
  #This is not really useful?
  f(x) = obj(nlp, x)
  ∇f(x) = grad(nlp, x)
  c(x) = cons(nlp, x)
  J(x) = jac_op(nlp, x)
  #########################

  z = copy(x)
  fz = fx = f(x)
  ∇fx = ∇f(x)
  cx = c(x)
  
  #T.M: we probably don't want to compute Jx and λ, if cx > ρ
  Jx = J(x)
  λ  = compute_lx(Jx, ∇fx)  # λ = argmin ‖∇f + Jᵀλ‖

  # Regularization
  γ = zero(T)
  δmin = T(1e-8) #√eps(T)
  δ = zero(T)

  # Allocate the sparse structure of K = [H + γI  [Jᵀ]; J -δI]
  nnz = nlp.meta.nnzh + nlp.meta.nnzj + nlp.meta.nvar + nlp.meta.ncon # H, J, γI, -δI
  rows = zeros(Int, nnz)
  cols = zeros(Int, nnz)
  vals = zeros(nnz)
  # H (1:nvar, 1:nvar)
  nnz_idx = 1:nlp.meta.nnzh
  @views hess_structure!(nlp, rows[nnz_idx], cols[nnz_idx])
  # J (nvar .+ 1:ncon, 1:nvar)
  nnz_idx = nlp.meta.nnzh .+ (1:nlp.meta.nnzj)
  @views jac_structure!(nlp, rows[nnz_idx], cols[nnz_idx])
  @views jac_coord!(nlp, x, vals[nnz_idx])
  rows[nnz_idx] .+= nlp.meta.nvar
  # γI (1:nvar, 1:nvar)
  nnz_idx = nlp.meta.nnzh .+ nlp.meta.nnzj .+ (1:nlp.meta.nvar)
  rows[nnz_idx] .= 1:nlp.meta.nvar
  cols[nnz_idx] .= 1:nlp.meta.nvar
  vals[nnz_idx] .= zero(T)
  # -δI (nvar .+ 1:ncon, nvar .+ 1:ncon)
  nnz_idx = nlp.meta.nnzh .+ nlp.meta.nnzj .+ nlp.meta.nvar .+ (1:nlp.meta.ncon)
  rows[nnz_idx] .= nlp.meta.nvar .+ (1:nlp.meta.ncon)
  cols[nnz_idx] .= nlp.meta.nvar .+ (1:nlp.meta.ncon)
  vals[nnz_idx] .= - δ

  LDL = solver_correspondence[linear_solver](nlp.meta.nvar + nlp.meta.ncon, rows, cols, vals)

  #ℓ(x,λ) = f(x) + λᵀc(x)
  ℓxλ = fx + dot(λ, cx)
  ∇ℓxλ = ∇fx + Jx'*λ

  Δℓₜ = T(Inf)
  Δℓₙ = zero(T)
  ℓᵣ = T(Inf)

  dualnorm = norm(∇ℓxλ)
  primalnorm = norm(cx)

  ρmax = max(ctol, 5primalnorm, 50dualnorm)
  ρ = NaN #not needed

  start_time = time()
  eltime = 0.0

  ϵd = atol + rtol * dualnorm
  ϵp = ctol

  Δtg = one(T)

  #stopping statuses
  solved     = primalnorm < ϵp && dualnorm < ϵd
  tired      = evals(nlp) > max_eval || eltime > max_time
  infeasible = false

  iter = 0

  @info log_header([:stage, :iter, :nf, :fx, :dual, :primal, :ρmax, :ρ, :status],
                   [String, Int, Int, Float64, Float64, Float64, Float64, Float64, String],
                   hdr_override=Dict(:nf => "#f+#c", :fx => "f(x)", :dual => "‖∇L‖", :primal => "‖c(x)‖")
                  )
  @info log_row(Any["init", iter, evals(nlp), fx, 
                            dualnorm, primalnorm, ρmax, ρ])

  while !(solved || tired || infeasible)
    # Trust-cylinder Normal step: find z such that ||h(z)|| ≤ ρ
    z, cz, fz, ℓzλ,  ∇ℓzλ, ρ, 
      primalnorm, dualnorm, 
      normal_status = normal_step(nlp, x, cx, Jx, fx, ∇fx, λ, ℓxλ, ∇ℓxλ, 
                                             dualnorm, primalnorm, ρmax, 
                                             ctol, ϵp, 
                                             max_eval, max_time, 
                                             eltime, start_time)
    # Convergence test
    solved = primalnorm < ϵp && dualnorm < ϵd
    infeasible = normal_status == :infeasible
    if solved || infeasible || (normal_status ∉ (:init_success, :success))
      break
    end

    # TODO Comment
    Δℓₙ = ℓzλ - ℓxλ
    if Δℓₙ ≥ (ℓᵣ - ℓxλ) / 2
      ρmax /= 2
    end
    if Δℓₙ > -Δℓₜ / 2
      ℓᵣ = ℓzλ
    end

    #Update matrix system if we moved
    #if normal_status != :init_success
    @views hess_coord!(nlp, z, λ, vals[1:nlp.meta.nnzh])
    @views jac_coord!(nlp, z, vals[nlp.meta.nnzh .+ (1:nlp.meta.nnzj)])
    # TODO: Update γ and δ here
    #end

    gBg = compute_gBg(nlp, rows, cols, vals, ∇ℓzλ)
    
    x, cx, fx, tg_status, Δtg, Δℓₜ, γ, δ = tangent_step(nlp, z, λ, cz, 
                                                        primalnorm, fz,
                                                        LDL, vals, 
                                                        ∇ℓzλ, ℓzλ, gBg, 
                                                        ρ, γ, δ,
                                                        Δ = Δtg, 
                                                        max_eval = max_eval, 
                                                        max_time = max_time - eltime)
    #γ
    if tg_status == :tired
      tired = true
      #now it depends whether we are feasibility or not.
      continue
    elseif tg_status == :small_horizontal_step
        #Try something ?
    else
        #success :)
    end
    
    γ = γ / 10
    Δtg *= 10

    if tg_status == :unknown #nothing happened in tangent_step
      # skip some computations z, cz, fz, ℓzλ,  ∇ℓzλ
      #@show "Pass here sometimes?"
    else
      #run computations
    end
    ∇fx = ∇f(x)
    Jx  = J(x)
    compute_lx!(Jx, ∇fx, λ)
    ℓxλ  = fx + dot(λ, cx) #differs from the tangent step as λ is different
    ∇ℓxλ = ∇fx + Jx'*λ
    
    primalnorm = norm(cx)
    dualnorm   = norm(∇ℓxλ)
    
    @info log_row(Any["T", iter, evals(nlp), fx, 
                           dualnorm, primalnorm, ρmax, ρ, tg_status])
    iter  += 1
    solved = primalnorm < ϵp && dualnorm < ϵd
    eltime = time() - start_time
    tired  = evals(nlp) > max_eval || eltime > max_time
  end

  status = if solved
    :first_order
  elseif tired
    if evals(nlp) > max_eval
      :max_eval
    elseif eltime > max_time
      :max_time
    else
      :exception
    end
  elseif infeasible
    :infeasible
  else
    :unknown
  end

  return GenericExecutionStats(status, nlp, solution     = z, 
                                            objective    = fz, 
                                            dual_feas    = dualnorm, 
                                            primal_feas  = primalnorm, 
                                            elapsed_time = eltime)
end
