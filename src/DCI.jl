module DCI

using LinearAlgebra, Logging

using Krylov, LinearOperators, NLPModels, SolverTools, SparseArrays, SymCOOSolverInterface

export dci

include("dci_normal.jl")
include("dci_tangent.jl")

const solver_correspondence = Dict(:ma57 => MA57Struct, 
                                   :ldlfact => LDLFactorizationStruct)
"""
    dci(nlp; kwargs...)

This method implements the Dynamic Control of Infeasibility for equality-constrained
problems described in

    Dynamic Control of Infeasibility in Equality Constrained Optimization
    Roberto H. Bielschowsky and Francisco A. M. Gomes
    SIAM J. Optim., 19(3), 1299–1325.
    https://doi.org/10.1137/070679557

"""
function dci(nlp :: AbstractNLPModel;
             atol = 1e-6,
             rtol = 1e-6,
             ctol = 1e-6,
             linear_solver = :ma57,
             max_eval = 50000,
             max_time = 60
            )
  if !equality_constrained(nlp)
    error("DCI only works for equality constrained problems")
  end
  if !(linear_solver ∈ keys(solver_correspondence))
    @warn "linear solver $linear_solver not found in $(collect(keys(solver_correspondence))). Using :ldlfact instead"
    linear_solver = :ldlfact
  end

  evals(nlp) = neval_obj(nlp) + neval_cons(nlp)

  f(x) = obj(nlp, x)
  ∇f(x) = grad(nlp, x)
  c(x) = cons(nlp, x)
  J(x) = jac_op(nlp, x)

  x = nlp.meta.x0
  z = copy(x)
  fz = fx = f(x)
  ∇fx = ∇f(x)
  cx = c(x)
  Jx = J(x)
  # λ = argmin ‖∇f + Jᵀλ‖
  λ = cgls(Jx', -∇fx)[1]

  # Regularization
  γ = 0.0
  δmin = 1e-8
  δ = 0.0

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
  vals[nnz_idx] .= 0.0
  # -δI (nvar .+ 1:ncon, nvar .+ 1:ncon)
  nnz_idx = nlp.meta.nnzh .+ nlp.meta.nnzj .+ nlp.meta.nvar .+ (1:nlp.meta.ncon)
  rows[nnz_idx] .= nlp.meta.nvar .+ (1:nlp.meta.ncon)
  cols[nnz_idx] .= nlp.meta.nvar .+ (1:nlp.meta.ncon)
  vals[nnz_idx] .= -δ

  LDL = solver_correspondence[linear_solver](nlp.meta.nvar + nlp.meta.ncon, rows, cols, vals)

  #ℓ(x,λ) = f(x) + λᵀc(x)
  ℓxλ = fx + dot(λ, cx)
  ∇ℓxλ = ∇fx + Jx'*λ

  Δℓₜ = Inf
  Δℓₙ = 0.0
  ℓᵣ = Inf

  dualnorm = norm(∇ℓxλ)
  primalnorm = norm(cx)

  ρmax = max(ctol, 5primalnorm, 50dualnorm)
  ρ = compute_ρ(dualnorm, primalnorm, ∇fx, ρmax, ctol)

  start_time = time()
  eltime = 0.0

  ϵd = atol + rtol * dualnorm
  ϵp = ctol

  Δtangent = 1.0

  solved = primalnorm < ϵp && dualnorm < ϵd
  tired = evals(nlp) > max_eval || eltime > max_time
  infeasible = false

  iter = 0

  @info log_header([:stage, :iter, :nf, :fx, :dual, :primal, :ρmax, :ρ, :status],
                   [String, Int, Int, Float64, Float64, Float64, Float64, Float64, String],
                   hdr_override=Dict(:nf => "#f+#c", :fx => "f(x)", :dual => "‖∇L‖", :primal => "‖c(x)‖")
                  )
  @info log_row(Any["init", iter, evals(nlp), fx, dualnorm, primalnorm, ρmax, ρ])

  while !(solved || tired || infeasible)
    # Normal step
    done_with_normal_step = false
    local ℓzλ
    while !done_with_normal_step
      # ngp = dualnorm/(norm(∇fx) + 1)
      ρ = compute_ρ(dualnorm, primalnorm, ∇fx, ρmax, ctol)
      z, cz, normal_status = normal_step(nlp, ϵp, x, cx, Jx, ρ, max_eval=max_eval, max_time=max_time-eltime)
      ∇fz = ∇f(z)
      λ = cgls(Jx', -∇fz)[1]
      fz = f(z)
      ℓzλ = fz + dot(λ, cz)
      primalnorm = norm(cz)
      # ∇fx = ∇f(x)
      ∇ℓxλ = ∇fx + Jx'*λ
      dualnorm = norm(∇ℓxλ)
      @info log_row(Any["N", iter, evals(nlp), fz, dualnorm, primalnorm, ρmax, ρ, normal_status])
      eltime = time() - start_time
      tired = evals(nlp) > max_eval || eltime > max_time
      infeasible = normal_status == :infeasible
      done_with_normal_step = primalnorm ≤ ρ || tired || infeasible 
    end

    # Convergence test
    solved = primalnorm < ϵp && dualnorm < ϵd

    if solved || tired || infeasible
      break
    end

    # @info("",
    #   fx,
    #   fz,
    #   ℓxλ,
    #   ℓzλ
    # )
    Δℓₙ = ℓzλ - ℓxλ
    if Δℓₙ ≥ (ℓᵣ - ℓxλ) / 2
      ρmax /= 2
    end
    if Δℓₙ > -Δℓₜ / 2
      ℓᵣ = ℓzλ
    end

    gBg = 0.0
    for k = 1:nlp.meta.nnzh
      i, j, v = rows[k], cols[k], vals[k]
      gBg += v * ∇ℓxλ[i] * ∇ℓxλ[j] * (i == j ? 1 : 2)
    end

    @views hess_coord!(nlp, z, λ, vals[1:nlp.meta.nnzh])
    # TODO: Don't compute every time
    @views jac_coord!(nlp, z, vals[nlp.meta.nnzh .+ (1:nlp.meta.nnzj)])
    # TODO: Update γ and δ here
    x, tg_status, Δtangent, Δℓₜ, γ, δ = tangent_step(nlp, z, λ, LDL, vals, ∇ℓxλ, ℓzλ, gBg, ρ, γ, δ,
                                                Δ=Δtangent, max_eval=max_eval, max_time=max_time-eltime)
    #=
    if tg_status != :success
      tired = true
      continue
    end
    =#
    
    γ = γ / 10
    Δtangent *= 10

    fx = obj(nlp, x)
    cx = c(x)
    ∇fx = ∇f(x)
    Jx = J(x)
    # λ = cgls(Jx', -∇fx)[1]
    ℓxλ = fx + dot(λ, cx)
    ∇ℓxλ = ∇fx + Jx'*λ
    primalnorm = norm(cx)
    dualnorm = norm(∇ℓxλ)
    @info log_row(Any["T", iter, evals(nlp), fx, dualnorm, primalnorm, ρmax, ρ, tg_status])
    iter += 1
    solved = primalnorm < ϵp && dualnorm < ϵd
    eltime = time() - start_time
    tired = evals(nlp) > max_eval || eltime > max_time
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

  return GenericExecutionStats(status, nlp, solution=z, objective=fz, dual_feas=dualnorm, primal_feas=primalnorm, elapsed_time=eltime)
end

function compute_ρ(dualnorm, primalnorm, ∇fx, ρmax, ϵ)
  ngp = dualnorm / (norm(∇fx) + 1)
  ρ = max(ngp, 0.75) * ρmax
  if ρ < ϵ && primalnorm > 100ctol
    ρ = primalnorm / 10
  elseif ngp ≤ 5ϵ
    ρ = ϵ
  end
  return ρ
end

end # module
