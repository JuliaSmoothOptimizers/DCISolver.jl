module DCISolverCaNNOLeSExt

import DCISolver
using DCISolver: MetaDCI, cons_norhs!, FeasibilityResidual
using CaNNOLeS: cannoles
using LinearAlgebra: norm
using NLPModels: AbstractNLPModel, jac_op!, neval_obj, neval_cons
using SolverCore: log_row

function DCISolver.feasibility_step_cannoles(
  nlp::AbstractNLPModel,
  x::AbstractVector{T},
  cx::AbstractVector{T},
  normcx::T,
  Jx,
  ρ::T,
  ctol::AbstractFloat,
  meta::MetaDCI,
  workspace,
  verbose::Bool;
  max_eval::Int = 1_000,
  max_time::AbstractFloat = 60.0,
  max_iter::Int = typemax(Int64),
  cannoles_options = Dict{Symbol, Any}(),
) where {T}
  nls = FeasibilityResidual(nlp, x)

  default_options = Dict{Symbol, Any}(
    :atol => ctol,
    :rtol => ctol,
    :Fatol => ctol,
    :Frtol => ctol,
    :max_eval => max_eval,
    :max_time => max_time,
    :max_iter => max_iter,
    :verbose => verbose ? 1 : 0,
  )

  options = merge(default_options, cannoles_options)

  start_time = time()
  stats = cannoles(nls; options...)
  el_time = time() - start_time

  z = stats.solution
  DCISolver.cons_norhs!(nlp, z, workspace.cz)
  cz = workspace.cz
  normcz = norm(cz)
  Jz = jac_op!(nlp, z, workspace.Jv, workspace.Jtv)

  status = if stats.status == :first_order && normcz ≤ ρ
    :success
  elseif stats.status == :first_order || stats.status == :acceptable
    normcz ≤ ρ ? :success : :unknown
  elseif stats.status == :max_eval
    :max_eval
  elseif stats.status == :max_time
    :max_time
  elseif stats.status == :max_iter
    :max_iter
  elseif stats.status == :infeasible
    :infeasible
  else
    :unknown
  end

  verbose && @info log_row(
    Any[
      "F-CaNNOLeS",
      stats.iter,
      neval_obj(nlp) + neval_cons(nlp),
      Float64,
      Float64,
      Float64,
      normcz,
      Float64,
      ρ,
      status,
      Float64,
      Float64,
      el_time,
    ],
  )

  return z, cz, normcz, Jz, status
end

end # module
