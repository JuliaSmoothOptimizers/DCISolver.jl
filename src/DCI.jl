module DCI

using LinearAlgebra, Logging

using Krylov, LinearOperators, NLPModels, SolverTools

export dci

include("dci_normal.jl")
include("dci_tangent.jl")

function dci(nlp :: AbstractNLPModel;
             atol = 1e-8,
             rtol = 1e-6,
             ctol = 1e-6,
             max_eval = 1000,
             max_time = 60
            )
  if !equality_constrained(nlp)
    error("DCI only works for equality constrained problems")
  end

  f(x) = obj(nlp, x)
  ∇f(x) = grad(nlp, x)
  H(x,y) = hess_op(nlp, x, y)
  c(x) = cons(nlp, x)
  J(x) = jac_op(nlp, x)

  x = nlp.meta.x0
  fz = fx = f(x)
  ∇fx = ∇f(x)
  cx = c(x)
  Jx = J(x)
  # λ = argmin ‖∇f + Jᵀλ‖
  λ = cgls(Jx', -∇fx)[1]

  #ℓ(x,λ) = f(x) + λᵀc(x)
  ℓxλ = fx + dot(λ, cx)
  ∇ℓxλ = ∇fx + Jx'*λ
  Bx = hess_op(nlp, x, λ)

  ρmax = 1.0
  ρ = 1.0

  dualnorm = norm(∇ℓxλ)
  primalnorm = norm(cx)

  start_time = time()
  eltime = 0.0

  ϵd = atol + rtol * dualnorm
  ϵp = atol + rtol * primalnorm

  solved = primalnorm < ϵp && dualnorm < ϵd
  tired = neval_obj(nlp) > max_eval || eltime > max_time

  iter = 0

  @info log_header([:stage, :iter, :nf, :dual, :primal, :ρ, :status],
                   [String, Int, Int, Float64, Float64, Float64, String],
                   hdr_override=Dict(:nf => "#f", :dual => "‖∇L‖", :primal => "‖c(x)‖")
                  )
  @info log_row(Any["init", iter, neval_obj(nlp), dualnorm, primalnorm, ρ])

  ngp = dualnorm/(norm(∇fx) + 1)
  z, cz, ρ, normal_status = normal_step(nlp, ctol, x, cx, Jx, ρmax, ngp, max_eval=max_eval, max_time=max_time-eltime)
  @assert cons(nlp, z) == cz
  ℓzλ = f(z) + dot(λ, cz)
  primalnorm = norm(cz)
  dualnorm = norm(∇ℓxλ)
  @info log_row(Any["N0", iter, neval_obj(nlp), dualnorm, primalnorm, ρ, normal_status])

  solved = primalnorm < ϵp && dualnorm < ϵd
  tired = neval_obj(nlp) > max_eval || eltime > max_time
  infeasible = normal_status == :infeasible

  while !(solved || tired || infeasible)
    x, tg_status = tangent_step(nlp, z, λ, Bx, ∇ℓxλ, Jx, ℓzλ, ρ, max_eval=max_eval, max_time=max_time-eltime)
    #=
    if tg_status != :success
      tired = true
      continue
    end
    =#
    fx = obj(nlp, x)
    cx = c(x)
    ∇fx = ∇f(x)
    Jx = J(x)
    λ = cgls(Jx', -∇fx)[1]
    ℓxλ = fx + dot(λ, cx)
    ∇ℓxλ = ∇fx + Jx'*λ
    Bx = hess_op(nlp, x, λ)
    primalnorm = norm(cx)
    dualnorm = norm(∇ℓxλ)
    @info log_row(Any["T", iter, neval_obj(nlp), dualnorm, primalnorm, ρ])
    iter += 1
    solved = primalnorm < ϵp && dualnorm < ϵd
    tired = neval_obj(nlp) > max_eval || eltime > max_time

    # Normal step
    ngp = dualnorm/(norm(∇fx) + 1)
    z, cz, ρ, normal_status = normal_step(nlp, ctol, x, cx, Jx, ρmax, ngp, max_eval=max_eval, max_time=max_time-eltime)
    println("z = $z, ‖cz‖ = $(norm(cz)), ρ = $ρ")
    println("ϵp = $ϵp, ϵd = $ϵd")
    @assert cons(nlp, z) == cz
    fz = f(z)
    ℓzλ = fz + dot(λ, cz)
    primalnorm = norm(cz)
    dualnorm = norm(∇ℓxλ)
    @info log_row(Any["N", iter, neval_obj(nlp), dualnorm, primalnorm, ρ, normal_status])

    eltime = time() - start_time
    solved = primalnorm < ϵp && dualnorm < ϵd
    tired = neval_obj(nlp) > max_eval || eltime > max_time
    infeasible = normal_status == :infeasible
  end

  status = if solved
    :first_order
  elseif tired
    if neval_obj(nlp) > max_eval
      :max_eval
    elseif eltime > max_time
      :max_time
    else
      :tired
    end
  elseif infeasible
    :infeasible
  else
    :unknown
  end

  return GenericExecutionStats(status, nlp, solution=z, objective=fz, dual_feas=dualnorm, primal_feas=primalnorm, elapsed_time=eltime)
end

end # module
