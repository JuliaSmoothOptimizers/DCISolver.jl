#December, 9th, T.M. comments:
# iii) as suggest in 2008 paper, maybe don't update Jx if we reduced 
# the Infeasibility by 10%.
# iv) analyze the case with 3 consecutive_bad_steps
# vi) regroup the algorithmic parameters at the beginning
# vii) typing
# viii) the precision of lsmr should depend on the other parameters?
# viv) fix the evaluation counter.
"""    feasibility_step(nls, x, cx, Jx)

Approximately solves min ‖c(x)‖.

Given xₖ, finds min ‖cₖ + Jₖd‖
"""
function feasibility_step(nlp             :: AbstractNLPModel, 
                          x               :: AbstractVector{T}, 
                          cx              :: AbstractVector{T}, 
                          normcx          :: T,
                          Jx              :: Union{LinearOperator{T}, AbstractMatrix{T}}, 
                          ρ               :: T,
                          ctol            :: AbstractFloat;
                          η₁              :: AbstractFloat = 1e-3, 
                          η₂              :: AbstractFloat = 0.66, 
                          σ₁              :: AbstractFloat = 0.25, 
                          σ₂              :: AbstractFloat = 2.0,
                          Δ0              :: AbstractFloat = one(T),
                          max_eval        :: Int = 1_000, 
                          max_time        :: AbstractFloat = 60.,
                          max_normal_iter :: Int = typemax(Int64), #try something smarter?
                          TR_compute_step :: Function = TR_lsmr #dogleg
                          ) where T
  
  z      = x
  cz     = cx
  Jz     = Jx
  normcz = normcx # cons(nlp, x) = normcx = normcz for the first z
  Δ = Δ0

  normal_iter = 0
  consecutive_bad_steps = 0 # Bad steps are when ‖c(z)‖ / ‖c(x)‖ > 0.95
  failed_step_comp = false         

  start_time = time()
  el_time    = 0.0
  tired      = neval_obj(nlp) + neval_cons(nlp) > max_eval || el_time > max_time
  infeasible = false
  status     = :unknown
  
  while !(normcz ≤ ρ || tired || infeasible)
    
    #Compute the a direction satisfying the trust-region constraint
    d, Jd, infeasible, solved = TR_compute_step(cz, Jz, ctol, Δ, normcz)

    if infeasible #the direction is too small
      failed_step_comp = true #too small step
      status = :too_small
    else
      zp      = z + d
      czp     = cons(nlp, zp)
      normczp = norm(czp)

      Pred = T(0.5) * (normcz^2 - norm(Jd + cz)^2)
      Ared = T(0.5) * (normcz^2 - normczp^2)

      if Ared/Pred < η₁
        Δ = max(T(1e-8), Δ * σ₁)
        status = :reduce_Δ
      else #success
        z  = zp
        Jz = jac_op(nlp, z)
        cz = czp
        normcz = normczp
        status = :success
        if Ared/Pred > η₂ && norm(d) >= T(0.99) * Δ
          Δ *= σ₂
        end
      end
    end

    if normcz / normcx > T(0.95)
      consecutive_bad_steps += 1
    else
      consecutive_bad_steps = 0
    end

    # Safeguard AKA agressive normal step - Loses robustness, doesn't seem to fix any
    #maybe also if infeasible is true, to verify that we still have a =0.
    if normcz > ρ && (consecutive_bad_steps ≥ 3 || failed_step_comp)
        (d, stats) = cg(hess_op(nlp, z, cz, obj_weight=0.0) + Jz' * Jz, Jz' * cz)
        zp   = z - d
        czp  = cons(nlp, zp)
        nczp = norm(czp)
        if norm(d) < ctol * min(nczp, one(T))
          infeasible = true
          status = :agressive_fail
        elseif nczp < normcz #even if d is small we keep going
          infeasible = false
          status = :agressive
          z, cz  = zp, czp
          normcz = nczp
          Jz     = jac_op(nlp, z)
          if !stats.solved
            @warn "Fail cg in feasibility_step: $(stats.status)"
          end
        end
    end

    @info log_row(Any["F", normal_iter, neval_obj(nlp) + neval_cons(nlp), 
                           NaN, NaN, normcz, NaN, NaN, status, norm(d), Δ])

    el_time      = time() - start_time
    normal_iter += 1
    many_evals   = neval_obj(nlp) + neval_cons(nlp) > max_eval
    iter_limit   = normal_iter > max_normal_iter
    tired        = many_evals || el_time > max_time || iter_limit
  end

  status = if normcz ≤ ρ
    :success
  elseif tired
    if neval_obj(nlp) + neval_cons(nlp) > max_eval
      :max_eval
    elseif el_time > max_time
      :max_time
    elseif normal_iter > max_normal_iter
      :max_iter
    else
      :unknown_tired
    end
  elseif infeasible
    :infeasible
  else
    :unknown
  end

  return z, cz, normcz, Jz, status
end

"""    feasibility_step(nls, x, cx, Jx)

Compute a direction d such that
min ‖cₖ + Jₖd‖ s.t. ||d|| ≤ Δ
using a dogleg.

Also checks if problem is infeasible.

Returns 4 entries:
(d, Jd, solved, infeasible)
"""
function dogleg(cz     :: AbstractVector, 
                Jz     :: Union{LinearOperator{T}, AbstractMatrix{T}}, 
                ctol   :: AbstractFloat, 
                Δ      :: AbstractFloat, 
                normcz :: AbstractFloat) where T

  infeasible, solved = false, true

  d     = -Jz' * cz
  nd2   = dot(d, d)
  Jd    = Jz * d
    
  if √nd2 < ctol * min(normcz, one(T)) #norm(d) < ctol * normcz
    infeasible = true
    return d, Jd, solved, infeasible
  end
    
  t     = nd2 / dot(Jd, Jd) #dot(d, d) / dot(Jd, Jd)
  dcp   = t * d
  ndcp2 = t^2 * nd2 #dot(dcp, dcp)
 
  if √ndcp2 > Δ
    d   = dcp * Δ / √ndcp2  #so ||d||=Δ
    Jd  = Jd * t * Δ / √ndcp2  #avoid recomputing Jd
  else
    (dn, stats)  = lsmr(Jz, -cz)
    solved =stats.solved
    if !stats.solved #stats.status ∈ ("maximum number of iterations exceeded")
      @warn "Fail lsmr in dogleg: $(stats.status)"
    end

    if norm(dn) <= Δ
      d = dn
    else
      v = dn - dcp 
      #τ = (-dot(dcp, v) + sqrt(dot(dcp, v)^2 + 4 * dot(v, v) * (Δ^2 - dot(dcp, dcp)))) / dot(v, v)
      nv2  = dot(v, v)
      dcpv = dot(dcp, v)
      τ    = (-dcpv + sqrt(dcpv^2 + 4 * nv2 * (Δ^2 - ndcp2))) / nv2
      d    = dcp + τ * v
    end
    Jd      = Jz * d # d has been updated
  end

  return d, Jd, infeasible, solved
end

function TR_lsmr(cz     :: AbstractVector, 
                 Jz     :: Union{LinearOperator{T}, AbstractMatrix{T}}, 
                 ctol   :: AbstractFloat, 
                 Δ      :: AbstractFloat, 
                 normcz :: AbstractFloat) where T

  (d, stats)  = lsmr(Jz, -cz, radius = Δ)

  infeasible = norm(d) < ctol * min(normcz, one(T))
  solved = stats.solved
  if !solved
      @warn "Fail lsmr in TR_lsmr: $(stats.status)"
  end

  Jd = Jz * d #lsmr doesn't return this information

  return d, Jd, infeasible, solved
end
