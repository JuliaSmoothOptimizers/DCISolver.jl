# Trust-cylinder Normal step: find z such that ||h(z)|| ≤ ρ
function normal_step(nlp        :: AbstractNLPModel, 
                     x          :: AbstractVector{T}, 
                     cx         :: AbstractVector{T}, 
                     Jx         :: LinearOperator{T},  
                     ∇fx        :: AbstractVector{T}, 
                     λ          :: AbstractVector{T}, 
                     ℓxλ        :: T, 
                     ∇ℓxλ       :: AbstractVector{T}, 
                     dualnorm   :: T, 
                     primalnorm :: T, #norm(cx)
                     ρmax       :: T, 
                     ctol       :: T, 
                     ϵp         :: T, 
                     max_eval   :: Int, 
                     max_time   :: AbstractFloat, 
                     eltime     :: AbstractFloat, 
                     start_time :: AbstractFloat) where T
  #assign z variable:
  z, cz, Jz, ∇fz = x, cx, Jx, ∇fx
  norm∇fz        = norm(∇fx) #can be avoided if we use dualnorm
  ℓzλ, ∇ℓzλ      = ℓxλ, ∇ℓxλ

  infeasible = false
  tired      = false

  #Initialize ρ at x
  ρ = compute_ρ(dualnorm, primalnorm, norm∇fz, ρmax, ctol, 0)

  done_with_normal_step = primalnorm ≤ ρ
  iter_normal_step      = 0

  while !done_with_normal_step

    #primalnorm = norm(cz)
    z, cz, primalnorm, Jz, normal_status = feasibility_step(nlp, z, cz, primalnorm,
                                                            Jz, ρ, ϵp, 
                                                            max_eval = max_eval, 
                                                            max_time = max_time - eltime)

    fz, ∇fz    = objgrad(nlp, z)
    norm∇fz    = norm(∇fz) #can be avoided if we use dualnorm
    compute_lx!(Jz, ∇fz, λ)
    ℓzλ        = fz + dot(λ, cz)
    ∇ℓzλ       = ∇fz + Jz'*λ
    dualnorm   = norm(∇ℓzλ)

    #update rho
    iter_normal_step += 1
    ρ = compute_ρ(dualnorm, primalnorm, norm∇fz, ρmax, ctol, iter_normal_step)

    @info log_row(Any["N", iter_normal_step, neval_obj(nlp) + neval_cons(nlp), 
                           fz, dualnorm, primalnorm, ρmax, ρ, normal_status])

    eltime     = time() - start_time
    many_evals = neval_obj(nlp) + neval_cons(nlp) > max_eval
    tired      = many_evals || eltime > max_time
    infeasible = normal_status == :infeasible

    done_with_normal_step = primalnorm ≤ ρ || tired || infeasible 
  end

  return z, ℓzλ,  ∇ℓzλ, ρ, primalnorm, dualnorm, tired, infeasible
end

#Theory asks for ngp ρmax 10^-4 < ρ <= ngp ρmax
#Should really check whether 3/4ρmax < ngp ρmax 10^-4 ?
#No evaluations of functions here.
# ρ = O(||g_p(z)||) and 
#in the paper ρ = ν n_p(z) ρ_max with n_p(z) = norm(g_p(z)) / (norm(g(z)) + 1)
function compute_ρ(dualnorm   :: T, 
                   primalnorm :: T, 
                   norm∇fx    :: T, 
                   ρmax       :: T, 
                   ϵ          :: T, 
                   iter       :: Int) where T
  if iter > 100
    return 0.75 * ρmax
  end
  ngp = dualnorm / (norm∇fx + 1)
  ρ = min(ngp, 0.75) * ρmax
  if ρ < ϵ && primalnorm > 100ϵ
    ρ = primalnorm / 10
  elseif ngp ≤ 5ϵ
    ρ = ϵ
  end

  return ρ
end