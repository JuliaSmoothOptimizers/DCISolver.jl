# Trust-cylinder Normal step: find z such that ||h(z)|| ≤ ρ
function normal_step(nlp        :: AbstractNLPModel, 
                     x          :: AbstractVector{T}, 
                     cx         :: AbstractVector{T},
                     Jx         :: LinearOperator{T},  
                     fx         :: T,
                     ∇fx        :: AbstractVector{T}, 
                     λ          :: AbstractVector{T}, 
                     ℓxλ        :: T, 
                     ∇ℓxλ       :: AbstractVector{T}, 
                     dualnorm   :: T, 
                     primalnorm :: T, #norm(cx)
                     ρmax       :: T, 
                     ϵp         :: T,
                     meta       :: MetaDCI;
                     max_eval   :: Int = 1_000,
                     max_time   :: AbstractFloat = 1_000.,
                     max_iter   :: Int = typemax(Int64),
                     feas_step  :: Function = feasibility_step
                     ) where T

  #assign z variable:
  z, cz, Jz, fz, ∇fz = x, cx, Jx, fx, ∇fx
  norm∇fz        = norm(∇fx) #can be avoided if we use dualnorm
  ℓzλ, ∇ℓzλ      = ℓxλ, ∇ℓxλ

  infeasible  = false
  restoration = false
  tired       = false
  start_time  = time()
  eltime      = 0.0

  #Initialize ρ at x
  ρ = compute_ρ(dualnorm, primalnorm, norm∇fz, ρmax, ϵp, 0)

  done_with_normal_step = primalnorm ≤ ρ
  iter_normal_step      = 0

  while !done_with_normal_step

    #primalnorm = norm(cz)
    z, cz, primalnorm, Jz, normal_status = eval(meta.feas_step)(nlp, z, cz,
                                                            primalnorm,
                                                            Jz, ρ, ϵp, meta,
                                                            max_eval = max_eval, 
                                                            max_time = max_time)

    fz, ∇fz    = objgrad(nlp, z)
    norm∇fz    = norm(∇fz) #can be avoided if we use dualnorm
    compute_lx!(Jz, ∇fz, λ)
    ℓzλ        = fz + dot(λ, cz)
    ∇ℓzλ       = ∇fz + Jz'*λ
    dualnorm   = norm(∇ℓzλ)

    #update rho
    iter_normal_step += 1
    ρ = compute_ρ(dualnorm, primalnorm, norm∇fz, ρmax, ϵp, iter_normal_step)

    @info log_row(Any["N", iter_normal_step, neval_obj(nlp) + neval_cons(nlp), 
                           fz, ℓzλ, dualnorm, primalnorm, 
                           ρmax, ρ, normal_status, Float64, Float64])

    eltime     = time() - start_time
    many_evals = neval_obj(nlp) + neval_cons(nlp) > max_eval
    tired      = many_evals || eltime > max_time || iter_normal_step > max_iter
    infeasible = normal_status == :infeasible

    if infeasible && !restoration && !(primalnorm ≤ ρ || tired) 
    #Enter restoration phase to avoid infeasible stationary points.
    #Heuristic that forces a random move from z
      restoration, infeasible = true, false
      perturbation_length = min(primalnorm, √ϵp) / norm(z) #sqrt(ϵp)/norm(z)
      z += (2 .* rand(T, nlp.meta.nvar) .- 1) * perturbation_length
      cz = cons(nlp, z)
      Jz = jac_op(nlp, z)
      primalnorm = norm(cz)
      ρ = compute_ρ(dualnorm, primalnorm, norm∇fz, ρmax, ϵp, 0)
    end

    done_with_normal_step = primalnorm ≤ ρ || tired || infeasible 
  end

  status = if primalnorm ≤ ρ && iter_normal_step == 0
      :init_success
    elseif primalnorm ≤ ρ
      :success
    elseif tired
      if neval_obj(nlp) + neval_cons(nlp) > max_eval
        :max_eval
      elseif eltime > max_time
        :max_time
      elseif iter_normal_step > max_iter
        :max_iter
      else
        :unknown_tired
      end
    elseif infeasible
      :infeasible
    else
      :unknown
    end

  return z, cz, fz, ℓzλ,  ∇ℓzλ, ρ, primalnorm, dualnorm, status
end

#Theory asks for ngp ρmax 10^-4 < ρ <= ngp ρmax
#No evaluations of functions here.
# ρ = O(‖g_p(z)‖) and 
#in the paper ρ = ν n_p(z) ρ_max with n_p(z) = norm(g_p(z)) / (norm(g(z)) + 1)
#
# T.M., 2021 Feb. 5th: what if dualnorm is excessively small ?
#            Feb. 8th: don't let ρ decrease too crazy
function compute_ρ(dualnorm   :: T, 
                   primalnorm :: T, 
                   norm∇fx    :: T, 
                   ρmax       :: T, 
                   ϵ          :: T, #ctol
                   iter       :: Int) where T
  if iter > 100
    return 0.75 * ρmax
  end
  ngp = dualnorm / (norm∇fx + 1)
  ρ = max(min(ngp, 0.75) * ρmax, ϵ)
  if ρ ≤ ϵ && primalnorm > 100ϵ
    ρ = primalnorm * 0.90 #/ 10
  #elseif ngp ≤ 5ϵ
  #  ρ = ϵ
  end

  return ρ
end
