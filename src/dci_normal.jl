#December, 9th, T.M. comments:
# iii) as suggest in 2008 paper, maybe don't update Jx if we reduced 
# the Infeasibility by 10%.
# iv) analyze the case with 3 consecutive_bad_steps
# vi) regroup the algorithmic parameters at the beginning
# vii) typing
# viii) the precision of lsmr should depend on the other parameters?
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
                          max_eval        :: Int = 1_000, 
                          max_time        :: AbstractFloat = 60.,
                          max_normal_iter :: Int = typemax(Int64), #try something smarter?
                          ) where T
  
  z      = x
  cz     = cx
  Jz     = Jx
  normcz = normcx # cons(nlp, x) = normcx = normcz for the first z

  Δ = one(T)

  normal_iter = 0
  consecutive_bad_steps = 0 # Bad steps are when ‖c(z)‖ / ‖c(x)‖ > 0.95         

  start_time = time()
  el_time    = 0.0
  tired      = neval_obj(nlp) + neval_cons(nlp) > max_eval || el_time > max_time
  infeasible = false
  
  while !(normcz ≤ ρ || tired || infeasible)
      
    d     = -Jz' * cz
    nd2   = dot(d, d)
    
    if sqrt(nd2) < ctol * normcz #norm(d) < ctol * normcz
      infeasible = true
      break
    end
    
    Jd    = Jz * d
    
    t     = nd2 / dot(Jd, Jd) #dot(d, d) / dot(Jd, Jd)
    dcp   = t * d
    ndcp2 = t^2 * nd2 #dot(dcp, dcp)
    
    if sqrt(ndcp2) > Δ
      d   = dcp * Δ / sqrt(ndcp2) #so ||d||=Δ
    else
      dn  = lsmr(Jz, -cz)[1]
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
    end
    
    Jd      = Jz * d
    zp      = z + d
    czp     = cons(nlp, zp)
    normczp = norm(czp)

    Pred = T(0.5) * (normcz^2 - norm(Jd + cz)^2)
    Ared = T(0.5) * (normcz^2 - normczp^2)

    if Ared/Pred < η₁
      Δ = max(T(1e-8), Δ * σ₁)
    else #success
      z  = zp
      Jz = jac_op(nlp, z)
      cz = czp
      normcz = normczp
      if Ared/Pred > η₂ && norm(d) >= T(0.99) * Δ
        Δ *= σ₂
      end
    end

    if normcz / normcx > T(0.95)
      consecutive_bad_steps += 1
    else
      consecutive_bad_steps = 0
    end

    # Safeguard AKA agressive normal step - Loses robustness, doesn't seem to fix any
    # if normcz > ρ && consecutive_bad_steps ≥ 3
    #   d = cg(hess_op(nlp, z, cz, obj_weight=0.0), Jx' * cz)[1]
    #   z -= d
    #   cz = cons(nlp, z)
    #   normcz = norm(cz)
    # end

    el_time = time() - start_time
    normal_iter += 1
    tired   = neval_obj(nlp) + neval_cons(nlp) > max_eval || el_time > max_time || normal_iter > max_normal_iter
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
