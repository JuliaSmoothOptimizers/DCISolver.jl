"""    cannoles_step(nls, x, cx, Jx)

Approximately solves min ‖c(x)‖.

such that |‖c(x)‖ ≤ ρ
"""
function cannoles_step(nlp      :: AbstractNLPModel, 
                       x        :: AbstractVector{T}, 
                       cx       :: AbstractVector{T}, 
                       normcx   :: T,
                       Jx       :: Union{LinearOperator{T}, AbstractMatrix{T}}, 
                       ρ        :: T,
                       ctol     :: AbstractFloat;
                       max_eval :: Int = 1_000, 
                       max_time :: AbstractFloat = 60.,
                       ) where T

  if normcx ≤ ρ
    status = :success
    return x, cx, normcx, Jx, status
  end

  nls = FeasibilityResidual(nlp) #feasibility model from the NLP
  #We now use any solver giving ‖c(x)‖ ≤ ρ
  #The drawback is that we lose the cx and Jx evaluations
  #See  https://github.com/tmigot/CaNNOLeS.jl/blob/master/src/CaNNOLeS.jl
  stats = cannoles(nls, x = x, max_time = max_time, 
                               max_f = max_eval, 
                               ϵtol = 2 * ρ^2,
                               linsolve = :ldlfactorizations,
                               method = :Newton_noFHess) #:Newton, :LM)

  z  = stats.solution
  cz = cons(nlp, z)
  Jz = jac_op(nlp, z)
  normcz = norm(cz)

  status = if stats.status == :first_order && normcz ≤ ρ
      :success
    elseif stats.status == :first_order # && normcz > ρ
      :infeasible
    elseif stats.status ∈ (:max_eval, :max_iter, :max_time)
      :stats.status
    else
      :unknown
    end

  return z, cz, normcz, Jz, status
end
