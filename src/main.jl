function SolverCore.solve!(
  workspace::DCIWorkspace{T, S, Si, Op, In, COO},
  nlp::AbstractNLPModel{T, S},
  stats::GenericExecutionStats{T, S, V, Tsp};
  callback = (args...) -> nothing,
) where {T, S, V, Tsp, Si, Op, In, COO}
  meta = get_meta(workspace)
  if !(nlp.meta.minimize)
    error("DCI only works for minimization problem")
  end
  if !(equality_constrained(nlp) || unconstrained(nlp))
    error("DCI only works for equality constrained problems")
  end
  reset!(stats)

  verbose = meta.verbose

  z, x = workspace.z, workspace.x
  ∇fx, cx = workspace.∇fx, workspace.cx
  ∇ℓzλ, cz = workspace.∇ℓzλ, workspace.cz

  x .= workspace.x0
  set_solution!(stats, x)
  fz = fx = obj(nlp, x)
  set_objective!(stats, fx)
  grad!(nlp, x, ∇fx)
  cons_norhs!(nlp, x, cx) # issue with the type of cx

  #T.M: we probably don't want to compute Jx and λ, if cx > ρ
  λ, ∇ℓxλ = workspace.λ, workspace.∇ℓxλ
  Jx = jac_op!(nlp, x, workspace.Jv, workspace.Jtv) # workspace.Jx
  compute_lx!(Jx, ∇fx, λ, meta)  # λ = argmin ‖∇f + Jᵀλ‖
  ℓxλ = fx + dot(λ, cx)
  set_solver_specific!(stats, :lagrangian, ℓxλ)
  ∇ℓxλ .= ∇fx .+ Jx' * λ

  dualnorm = norm(∇ℓxλ)
  primalnorm = norm(cx)
  set_residuals!(stats, primalnorm, dualnorm)

  # Regularization
  γ = zero(T)
  δ = zero(T)

  # Allocate the sparse structure of K = [H + γI  [Jᵀ]; J -δI]
  rows, cols, vals = workspace.rows, workspace.cols, workspace.vals
  n, m = nlp.meta.nvar, nlp.meta.ncon
  rows .= zero(Int)
  cols .= zero(Int)
  vals .= zero(T)
  regularized_coo_saddle_system!(nlp, rows, cols, vals, γ = γ, δ = δ)

  LDL = COO(n + m, rows, cols, vals)

  Δℓₜ = T(Inf)
  Δℓₙ = zero(T)
  ℓᵣ = T(Inf)

  start_time = time()
  set_time!(stats, 0.0)

  ϵd = meta.atol + meta.rtol * max(dualnorm, norm(∇fx))
  ϵp = meta.ctol

  ρmax = max(ϵp, 5 * primalnorm, 50 * dualnorm)
  ρ = T(NaN) #not needed at iteration 0

  Δtg = meta.tan_Δ

  #stopping statuses
  solved = primalnorm < ϵp && dualnorm < ϵd
  infeasible = false
  small_step_rescue = false
  stalled = false

  set_iter!(stats, 0)

  verbose > 0 && @info log_header(
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
  verbose > 0 && @info log_row(
    Any[
      "init",
      stats.iter,
      evals(nlp),
      fx,
      ℓxλ,
      dualnorm,
      primalnorm,
      ρmax,
      ρ,
      Symbol,
      Float64,
      Float64,
    ],
  )

  status = get_status(nlp, stats, meta, solved, fx, stalled, infeasible)
  set_status!(stats, status)

  callback(nlp, workspace, stats)

  done = stats.status != :unknown

  while !done
    # Trust-cylinder Normal step: find z such that ||h(z)|| ≤ ρ
    z, cz, fz, ℓzλ, ∇ℓzλ, ρ, primalnorm, dualnorm, normal_status = normal_step!(
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
      meta.max_time - stats.elapsed_time,
      meta.max_iter_normal_step,
      meta,
      workspace,
      (verbose > 0 && mod(stats.iter, verbose) == 0),
    )
    # Convergence test
    set_residuals!(stats, primalnorm, dualnorm)
    solved = primalnorm < ϵp && (dualnorm < ϵd || fz < meta.unbounded_threshold)
    infeasible = normal_status == :infeasible
    if solved || infeasible || (normal_status ∉ (:init_success, :success))
      set_time!(stats, time() - start_time)
      x, fx = z, fz
      set_solution!(stats, x)
      set_objective!(stats, fx)
      status = get_status(nlp, stats, meta, solved, fx, stalled, infeasible)
      set_status!(stats, status)
      done = true
      break
    end

    Δℓₙ = ℓzλ - ℓxλ
    if Δℓₙ ≥ (ℓᵣ - ℓxλ) / 2
      ρmax = max(ϵp, ρmax / 2)
      ρ = min(ρ, ρmax)
    else #we don't let ρmax too far from the residuals
      ρmax = min(ρmax, max(ϵp, 5 * primalnorm, 50 * dualnorm))
      ρ = min(ρ, ρmax)
    end
    if Δℓₙ > -Δℓₜ / 2
      ℓᵣ = ℓzλ
    end

    #Update matrix system
    @views hess_coord!(nlp, z, λ, vals[1:(nlp.meta.nnzh)])
    @views jac_coord!(nlp, z, vals[nlp.meta.nnzh .+ (1:(nlp.meta.nnzj))])
    if γ != 0.0
      γ = max(γ * meta.decrease_γ, √eps(T))
      vals[nlp.meta.nnzh .+ nlp.meta.nnzj .+ (1:(nlp.meta.nvar))] .= γ
    end

    gBg = compute_gBg(nlp, rows, cols, vals, ∇ℓzλ)

    rmng_time = meta.max_time - (time() - start_time)
    x, cx, fx, tg_status, Δtg, Δℓₜ, γ, δ = tangent_step!(
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
      workspace,
      (verbose > 0 && mod(stats.iter, verbose) == 0),
      Δ = Δtg,
      max_eval = meta.max_eval,
      max_time = rmng_time,
    )
    set_solution!(stats, x)
    set_objective!(stats, fx)
    if tg_status == :tired
      set_time!(stats, time() - start_time)
      set_status!(stats, :max_time)
      done = true
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
        done = true
      end
      #Or stop!
    else
      #success :)
    end

    #increase the trust-region paramter
    Δtg = min(meta.increase_Δtg * Δtg, 1 / √eps(T))

    if tg_status == :unknown #nothing happened in tangent_step
      # skip some computations z, cz, fz, ℓzλ,  ∇ℓzλ
      #@warn "Pass here sometimes?"
    end

    grad!(nlp, x, ∇fx)
    Jx = jac_op!(nlp, x, workspace.Jv, workspace.Jtv) # workspace.Jx
    compute_lx!(Jx, ∇fx, λ, meta)
    ℓxλ = fx + dot(λ, cx) #differs from the tangent step as λ is different
    set_solver_specific!(stats, :lagrangian, ℓxλ)
    ∇ℓxλ .= ∇fx .+ Jx' * λ

    primalnorm = norm(cx)
    dualnorm = norm(∇ℓxλ)
    set_residuals!(stats, primalnorm, dualnorm)

    verbose > 0 &&
      mod(stats.iter, verbose) == 0 &&
      @info log_row(
        Any[
          "T",
          stats.iter,
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
    set_iter!(stats, stats.iter + 1)
    solved = primalnorm < ϵp && (dualnorm < ϵd || fx < meta.unbounded_threshold)
    set_time!(stats, time() - start_time)

    status = get_status(nlp, stats, meta, solved, fx, stalled, infeasible)
    set_status!(stats, status)

    callback(nlp, workspace, stats)

    done = stats.status != :unknown
  end

  stats
end

function evals(nlp::AbstractNLPModel)
  return neval_obj(nlp) + neval_cons(nlp)
end

function get_status(nlp::AbstractNLPModel, stats, meta, solved, fx, stalled, infeasible)
  status = if solved
    fx < meta.unbounded_threshold ? :unbounded : :first_order
  elseif evals(nlp) > meta.max_eval
    :max_eval
  elseif stats.elapsed_time > meta.max_time
    :max_time
  elseif stats.iter > meta.max_iter
    :max_iter
  elseif stalled
    :small_step
  elseif infeasible
    :infeasible
  else
    :unknown
  end
  return status
end
