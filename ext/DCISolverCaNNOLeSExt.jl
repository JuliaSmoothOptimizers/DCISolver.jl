module DCISolverCaNNOLeSExt

import DCISolver
using DCISolver: MetaDCI, cons_norhs!
using CaNNOLeS: cannoles
using LinearAlgebra: norm
using NLPModelsModifiers: FeasibilityResidual
using NLPModels: AbstractNLPModel, jac_op!, neval_obj, neval_cons
using SolverCore: log_row

"""
    feasibility_step_cannoles(nlp, x, cx, normcx, Jx, ρ, ctol, meta, workspace, verbose; kwargs...)

Approximately solves `min ‖c(x)‖` using the CaNNOLeS solver as an alternative to the
trust-region Levenberg-Marquardt method implemented in [`DCISolver.feasibility_step`](@ref).

CaNNOLeS is a solver for equality-constrained nonlinear least-squares problems, used here
to find a feasible point by minimizing the constraint violation ‖c(x)‖.

# Arguments
- `max_eval::Int = 1_000`: maximum number of evaluations, counted over the whole `dci` call
  (not just this feasibility step).
- `max_time::AbstractFloat = 60.0`: remaining time budget, in seconds, for this call.
- `max_iter::Int = typemax(Int64)`: maximum number of iterations for CaNNOLeS.
- `cannoles_options`: Additional options to pass to the CaNNOLeS solver, overriding the
  defaults derived from `ctol`, `max_eval`, `max_time` and `max_iter`.

# Output
- `z`, `cz`, `normcz`, `Jz`: the new iterate, and updated evaluations.
- `status`: Computation status. Possible outcomes are: `:success`, `:max_eval`, `:max_time`, `:max_iter`, `:infeasible`, `:unknown`.
"""
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
  # Allocates a new NLS wrapper around `nlp` on every call.
  nls = FeasibilityResidual(nlp)

  # `max_eval` is the budget for the whole `dci` solve (see `normal_step!`), so it must be
  # turned into a number of evaluations remaining for this call.
  current_eval = neval_obj(nlp) + neval_cons(nlp)
  remaining_eval = max(0, max_eval - current_eval)

  # Unlike `max_eval`, `max_time` already is the time remaining until the global time limit
  # of the `dci` solve, computed by the caller (see `normal_step!` and `SolverCore.solve!`),
  # so it can be forwarded to CaNNOLeS as is.
  if remaining_eval ≤ 0
    return x, cx, normcx, Jx, :max_eval
  elseif max_time ≤ 0
    return x, cx, normcx, Jx, :max_time
  end

  # Allocates a new `Dict` on every call.
  default_options = Dict{Symbol, Any}(
    :atol => ctol,
    :rtol => ctol,
    :Fatol => ctol,
    :Frtol => ctol,
    :max_eval => remaining_eval,
    :max_time => max_time,
    :max_iter => max_iter,
    :verbose => verbose ? 1 : 0,
  )

  options = merge(default_options, cannoles_options)

  # Ensure the CaNNOLeS solver starts from the current iterate `x` unless
  # the caller explicitly provided a starting point.
  if !haskey(options, :x)
    options = merge(Dict(:x => x), options)
  end

  start_time = time()
  stats = cannoles(nls; options...)
  el_time = time() - start_time

  z = stats.solution
  DCISolver.cons_norhs!(nlp, z, workspace.cz)
  cz = workspace.cz
  normcz = norm(cz)
  Jz = jac_op!(nlp, z, workspace.Jv, workspace.Jtv)

  status = if stats.status == :first_order || stats.status == :acceptable
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

  # `fx`, `lag`, `dual` and `ρmax` are not tracked by CaNNOLeS, and there is no trust-region
  # radius `Δ` to report either: mirror `feasibility_step`'s convention of passing the type
  # instead of a number so `log_row` prints "-" for these missing values (see SolverCore's
  # `log_row` docstring).
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
      norm(z - x),
      Float64,
      el_time,
    ],
  )

  return z, cz, normcz, Jz, status
end

end # module
