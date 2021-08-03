function dci(nlp::AbstractNLPModel, x::AbstractVector{T}, meta::MetaDCI) where {T}
  if !(nlp.meta.minimize)
    error("DCI only works for minimization problem")
  end
  if !(equality_constrained(nlp) || unconstrained(nlp))
    error("DCI only works for equality constrained problems")
  end

  evals(nlp) = neval_obj(nlp) + neval_cons(nlp)
  tired_check(nlp, eltime, iter, meta) = begin
    (evals(nlp) > meta.max_eval || eltime > meta.max_time || iter > meta.max_iter)
  end

  z = copy(x)
  fz = fx = obj(nlp, x)
  ∇fx = grad(nlp, x)
  cx = cons(nlp, x)

  #T.M: we probably don't want to compute Jx and λ, if cx > ρ
  Jx = jac_op(nlp, x)
  λ = compute_lx(Jx, ∇fx, meta)  # λ = argmin ‖∇f + Jᵀλ‖
  ℓxλ = fx + dot(λ, cx)
  ∇ℓxλ = ∇fx + Jx' * λ

  dualnorm = norm(∇ℓxλ)
  primalnorm = norm(cx)

  # Regularization
  δmin = √eps(T)
  γ = zero(T)
  δ = zero(T)

  # Allocate the sparse structure of K = [H + γI  [Jᵀ]; J -δI]
  n, m = nlp.meta.nvar, nlp.meta.ncon
  nnz = nlp.meta.nnzh + nlp.meta.nnzj + n + m # H, J, γI, -δI
  rows = zeros(Int, nnz)
  cols = zeros(Int, nnz)
  vals = zeros(nnz)
  regularized_coo_saddle_system!(nlp, rows, cols, vals, γ = γ, δ = δ)

  LDL = solver_correspondence[meta.linear_solver](n + m, rows, cols, vals)

  Δℓₜ = T(Inf)
  Δℓₙ = zero(T)
  ℓᵣ = T(Inf)

  start_time = time()
  eltime = 0.0

  ϵd = meta.atol + meta.rtol * max(dualnorm, norm(∇fx))
  ϵp = meta.ctol

  ρmax = max(ϵp, 5primalnorm, 50dualnorm)
  ρ = NaN #not needed at iteration 0

  Δtg = one(T)

  #stopping statuses
  solved = primalnorm < ϵp && dualnorm < ϵd
  tired = tired_check(nlp, eltime, 0, meta)
  infeasible = false
  small_step_rescue = false
  stalled = false

  iter = 0

  @info log_header(
    [:stage, :iter, :nf, :fx, :lag, :dual, :primal, :ρmax, :ρ, :status, :nd, :Δ],
    [
      String,
      Int,
      Int,
      Float64,
      Float64,
      Float64,
      Float64,
      Float64,
      Float64,
      String,
      Float64,
      Float64,
    ],
    hdr_override = Dict(
      :nf => "#f+#c",
      :fx => "f(x)",
      :lag => "ℓ",
      :dual => "‖∇L‖",
      :primal => "‖c(x)‖",
      :nd => "‖d‖",
    ),
  )
  @info log_row(
    Any["init", iter, evals(nlp), fx, ℓxλ, dualnorm, primalnorm, ρmax, ρ, Symbol, Float64, Float64],
  )

  while !(solved || tired || infeasible || stalled)
    # Trust-cylinder Normal step: find z such that ||h(z)|| ≤ ρ
    z, cz, fz, ℓzλ, ∇ℓzλ, ρ, primalnorm, dualnorm, normal_status = normal_step(
      nlp,
      x,
      cx,
      Jx,
      fx,
      ∇fx,
      λ,
      ℓxλ,
      ∇ℓxλ,
      dualnorm,
      primalnorm,
      ρmax,
      ϵp,
      meta.max_eval,
      meta.max_time - eltime,
      meta.max_iter_normal_step,
      meta,
    )
    # Convergence test
    solved = primalnorm < ϵp && (dualnorm < ϵd || fz < meta.unbounded_threshold)
    infeasible = normal_status == :infeasible
    if solved || infeasible || (normal_status ∉ (:init_success, :success))
      eltime = time() - start_time
      tired = tired_check(nlp, eltime, iter, meta)
      x, fx = z, fz
      #stalled?
      break
    end

    Δℓₙ = ℓzλ - ℓxλ
    if Δℓₙ ≥ (ℓᵣ - ℓxλ) / 2
      ρmax = max(ϵp, ρmax / 2)
      ρ = min(ρ, ρmax)
    else #we don't let ρmax too far from the residuals
      ρmax = min(ρmax, max(ϵp, 5primalnorm, 50dualnorm))
      ρ = min(ρ, ρmax)
    end
    if Δℓₙ > -Δℓₜ / 2
      ℓᵣ = ℓzλ
    end

    #Update matrix system
    @views hess_coord!(nlp, z, λ, vals[1:(nlp.meta.nnzh)])
    @views jac_coord!(nlp, z, vals[nlp.meta.nnzh .+ (1:(nlp.meta.nnzj))])
    if γ != 0.0 && γ != √eps(T)
      γ = max(γ / 10, √eps(T))
      vals[nlp.meta.nnzh .+ nlp.meta.nnzj .+ (1:(nlp.meta.nvar))] .= γ
    end

    gBg = compute_gBg(nlp, rows, cols, vals, ∇ℓzλ)

    rmng_time = meta.max_time - (time() - start_time)
    x, cx, fx, tg_status, Δtg, Δℓₜ, γ, δ = tangent_step(
      nlp,
      z,
      λ,
      cz,
      primalnorm,
      fz,
      LDL,
      vals,
      ∇ℓzλ,
      ℓzλ,
      gBg,
      ρ,
      γ,
      δ,
      meta,
      Δ = Δtg,
      max_eval = meta.max_eval,
      max_time = rmng_time,
    )
    if tg_status == :tired
      tired = true
      eltime = time() - start_time
      #now it depends whether we are feasibility or not.
      continue
    elseif tg_status == :small_horizontal_step
      if !small_step_rescue
        #Try something ?
        ρ = primalnorm / 2 #force a feasibility step
        #maybe decrease ρmax too ?
        small_step_rescue = true
      else
        stalled = true
      end
      #Or stop!
    else
      #success :)
    end

    #increase the trust-region paramter
    Δtg = min(10Δtg, 1 / √eps(T))

    if tg_status == :unknown #nothing happened in tangent_step
      # skip some computations z, cz, fz, ℓzλ,  ∇ℓzλ
      #@warn "Pass here sometimes?"
    end

    ∇fx = grad(nlp, x)
    Jx = jac_op(nlp, x)
    compute_lx!(Jx, ∇fx, λ, meta)
    ℓxλ = fx + dot(λ, cx) #differs from the tangent step as λ is different
    ∇ℓxλ = ∇fx + Jx' * λ

    primalnorm = norm(cx)
    dualnorm = norm(∇ℓxλ)

    @info log_row(
      Any[
        "T",
        iter,
        evals(nlp),
        fx,
        ℓxλ,
        dualnorm,
        primalnorm,
        ρmax,
        ρ,
        tg_status,
        Float64,
        Float64,
      ],
    )
    iter += 1
    solved = primalnorm < ϵp && (dualnorm < ϵd || fx < meta.unbounded_threshold)
    eltime = time() - start_time
    tired = tired_check(nlp, eltime, iter, meta)
  end

  #check unboundedness
  status = if solved
    fx < meta.unbounded_threshold ? :unbounded : :first_order
  elseif tired
    if evals(nlp) > meta.max_eval
      :max_eval
    elseif eltime > meta.max_time
      :max_time
    elseif iter > meta.max_iter
      :max_iter
    else
      :exception
    end
  elseif stalled
    :small_step
  elseif infeasible
    :infeasible
  else
    :unknown
  end

  return GenericExecutionStats(
    status,
    nlp,
    solution = x,
    objective = fx,
    dual_feas = dualnorm,
    primal_feas = primalnorm,
    iter = iter,
    elapsed_time = eltime,
    solver_specific = Dict(:multiplier => λ, :lagrangian => ℓxλ),
  ) #add more?
end
