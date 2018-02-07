using NLPModels, LinearOperators, Krylov

include("dci_normal.jl")
include("dci_tangent.jl")

function dci(nlp :: AbstractNLPModel;
             atol = 1e-8,
             rtol = 1e-6,
             ctol = 1e-6,
             max_f = 1000,
             max_time = 60,
             verbose = false
            )
  if !equality_constrained(nlp)
    error("DCI only works for equality constrained problems")
  end

  f(x) = obj(nlp, x)
  ∇f(x) = grad(nlp, x)
  H(x,y) = hess_op(nlp, x, y=y)
  c(x) = cons(nlp, x)
  J(x) = jac_op(nlp, x)

  x = nlp.meta.x0
  fx = f(x)
  ∇fx = ∇f(x)
  cx = c(x)
  Jx = J(x)
  # λ = argmin ‖∇f + Jᵀλ‖
  λ = cgls(Jx', -∇fx)[1]

  #ℓ(x,λ) = f(x) + λᵀc(x)
  ℓxλ = fx + dot(λ, cx)
  ∇ℓxλ = ∇fx + Jx'*λ
  Bx = hess_op(nlp, x, y=λ)

  ρmax = 1.0
  ρ = 1.0

  dualnorm = norm(∇ℓxλ)
  primalnorm = norm(cx)

  start_time = time()
  eltime = 0.0

  ϵd = atol + rtol * dualnorm
  ϵp = atol + rtol * primalnorm

  solved = primalnorm < ϵp && dualnorm < ϵd
  tired = sum_counters(nlp) > max_f || eltime > max_time

  iter = 0

  if verbose
    @printf("NT %5s  %8s  %8s  %8s\n",
            "Iter", "‖∇ℓxλ‖", "‖c(x)‖", "ρ")
    @printf("   %5d  %8.2e  %8.2e  %8.2e\n",
            iter, dualnorm, primalnorm, ρ)
  end

  ngp = dualnorm/(norm(∇fx) + 1)
  z, cz, ρ = normal_step(nlp, ctol, x, cx, Jx, ρmax, ngp)
  @assert cons(nlp, z) == cz
  ℓzλ = f(z) + dot(λ, cz)
  primalnorm = norm(cz)
  dualnorm = norm(∇ℓxλ)
  verbose && @printf("N  %5d  %8.2e  %8.2e  %8.2e\n",
                     iter, dualnorm, primalnorm, ρ)

  solved = primalnorm < ϵp && dualnorm < ϵd
  tired = sum_counters(nlp) > max_f || eltime > max_time

  while !(solved || tired)
    x = tangent_step(nlp, z, λ, Bx, ∇ℓxλ, Jx, ℓzλ, ρ)
    fx = obj(nlp, x)
    cx = c(x)
    ∇fx = ∇f(x)
    Jx = J(x)
    λ = cgls(Jx', -∇fx)[1]
    ℓxλ = fx + dot(λ, cx)
    ∇ℓxλ = ∇fx + Jx'*λ
    Bx = hess_op(nlp, x, y=λ)
    primalnorm = norm(cx)
    dualnorm = norm(∇ℓxλ)
    verbose && @printf("T  %5d  %8.2e  %8.2e  %8.2e\n",
                       iter, dualnorm, primalnorm, ρ)
    iter += 1
    solved = primalnorm < ϵp && dualnorm < ϵd
    tired = sum_counters(nlp) > max_f || eltime > max_time

    ngp = dualnorm/(norm(∇fx) + 1)
    z, cz, ρ = normal_step(nlp, ctol, x, cx, Jx, ρmax, ngp)
    @assert cons(nlp, z) == cz
    fx = f(x)
    ℓzλ = fx + dot(λ, cz)
    primalnorm = norm(cz)
    dualnorm = norm(∇ℓxλ)
    verbose && @printf("N  %5d  %8.2e  %8.2e  %8.2e\n",
                       iter, dualnorm, primalnorm, ρ)

    solved = primalnorm < ϵp && dualnorm < ϵd
    tired = sum_counters(nlp) > max_f || eltime > max_time
  end

  return z, fx, dualnorm, primalnorm, eltime, solved, tired
end
