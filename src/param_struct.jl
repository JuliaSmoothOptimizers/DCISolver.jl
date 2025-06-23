"""
    `comp_λ_cgls(m, n, ::DataType; kwargs...)`

Keyword arguments correspond to input parameters of `cgls` from `Krylov.jl` used in the computation of the Lagrange multipliers.
Returns a `comp_λ_cgls` structure.
"""
struct comp_λ_cgls{T <: AbstractFloat, S <: AbstractVector{T}}
  comp_λ_solver::CglsSolver{T, T, S}
  M # =I,
  λ::T # =zero(T),
  atol::T # =√eps(T),
  rtol::T # =√eps(T),
  #radius :: T=zero(T),
  itmax::Int # =0,
  #verbose :: Int=0,
  #history :: Bool=false
end

function comp_λ_cgls(
  m,
  n,
  ::Type{S};
  M = I,
  λ::T = zero(T),
  atol::T = √eps(T),
  rtol::T = √eps(T),
  itmax::Int = 5 * (m + n),
) where {T, S <: AbstractVector{T}}
  comp_λ_solver = CglsSolver(m, n, S)
  return comp_λ_cgls(comp_λ_solver, M, λ, atol, rtol, itmax)
end

"""
    comp_λ_solvers = Dict(:cgls => comp_λ_cgls)

Dictonary of the possible structures for the computation of the Lagrange multipliers.
"""
const comp_λ_solvers = Dict(:cgls => comp_λ_cgls)

"""
    solver_correspondence = Dict(:ma57 => MA57Struct, :ldlfact => LDLFactorizationStruct)

Dictonary of the possible structures for the factorization.
"""
const solver_correspondence = if isdefined(HSL, :libhsl_ma57)
  Dict(:ma57 => MA57Struct, :ldlfact => LDLFactorizationStruct)
else
  Dict(:ldlfact => LDLFactorizationStruct)
end

"""
    `TR_lsmr_struct(m, n, ::DataType; kwargs...)`

Keyword arguments correspond to input parameters of `lsmr` from `Krylov.jl` used in the computation of the trust-region step.
Returns a `TR_lsmr_struct` structure.
"""
struct TR_lsmr_struct{T <: AbstractFloat, S <: AbstractVector{T}}
  lsmr_solver::LsmrSolver{T, T, S}
  M # =I,
  #N=I, #unnecessary
  #sqd :: Bool=false, #unnecessary
  λ::T # =zero(T),
  axtol::T # =√eps(T),
  btol::T # =√eps(T),
  atol::T # =zero(T),
  rtol::T # =zero(T),
  etol::T # =√eps(T),
  #window :: Int=5, #unnecessary
  itmax::Int # =0,  #m + n (set in the code if itmax==0)
  #conlim :: T=1/√eps(T), #set conditioning upper limit
  #radius :: T=zero(T),  #unnecessary
  #verbose :: Int=0,  #unnecessary
  #history :: Bool=false #unnecessary
end

function TR_lsmr_struct(
  m,
  n,
  ::Type{S};
  M = I,
  λ::T = zero(T),
  axtol::T = √eps(T),
  btol::T = √eps(T),
  atol::T = zero(T),
  rtol::T = zero(T),
  etol::T = √eps(T),
  itmax::Int = m + n,
) where {T, S <: AbstractVector{T}}
  lsmr_solver = LsmrSolver(n, m, S)
  return TR_lsmr_struct(lsmr_solver, M, λ, axtol, btol, atol, rtol, etol, itmax)
end

"""
    `TR_dogleg_struct(m, n, ::DataType; kwargs...)`

Keyword arguments correspond to input parameters of `lsmr` from `Krylov.jl` used in the computation of the dogleg for the trust-region step.
Returns a `TR_dogleg_struct` structure.
"""
struct TR_dogleg_struct{T <: AbstractFloat, S <: AbstractVector{T}}
  lsmr_solver::LsmrSolver{T, T, S} # There is another lsmr call here
end

function TR_dogleg_struct(m, n, ::Type{S}; kwargs...) where {T, S <: AbstractVector{T}}
  lsmr_solver = LsmrSolver(n, m, S)
  return TR_dogleg_struct(lsmr_solver)
end

"""
    TR_solvers = Dict(:TR_lsmr => TR_lsmr_struct, :TR_dogleg => TR_dogleg_struct)

Dictonary of the possible structures for the trust-region step.
"""
const TR_solvers = Dict(:TR_lsmr => TR_lsmr_struct, :TR_dogleg => TR_dogleg_struct)

"""
    MetaDCI(x, y; kwargs...)
    MetaDCI(nlp::AbstractNLPModel, x = nlp.meta.x0, y = nlp.meta.y0; kwargs...)

Structure containing all the parameters used in the [`dci`](@ref) call.
`x` is an intial guess, and `y` is an initial guess for the Lagrange multiplier.
Returns a `MetaDCI` structure.

# Arguments
The keyword arguments may include:
- `atol::T=T(1e-5)`: absolute tolerance.
- `rtol::T=T(1e-5)`: relative tolerance.
- `ctol::T=T(1e-5)`: feasibility tolerance.
- `unbounded_threshold::T=T(-1e5)`: below this threshold the problem is unbounded.
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.
- `max_eval::Integer=50000`: maximum number of cons + obj evaluations.
- `max_time::Float64=120.0`: maximum number of seconds.
- `max_iter::Integer=500`: maximum number of iterations.
- `max_iter_normal_step::Integer=typemax(Int)`: maximum number of iterations in normal step.
- `λ_struct::comp_λ_cgls=comp_λ_cgls(length(x0), length(y0), S)`.
- `linear_solver::Symbol=:ldlfact`: Solver for the factorization. options: `:ma57`.
- `decrease_γ::T=T(0.1)`: Regularization for the factorization: reduce `γ` if possible, `> √eps(T)`, between tangent steps.
- `increase_γ::T=T(100.0)`: Regularization for the factorization: up `γ` if possible, `< 1/√eps(T)`, during the factorization.
- `δmin::T=√eps(T)`: Regularization for the factorization: smallest value of `δ` used for the regularization.
- `feas_step::Symbol=:feasibility_step`: Normal step.
- `feas_η₁::T=T(1e-3)`: Feasibility step: decrease the trust-region radius when `Ared/Pred < η₁`.
- `feas_η₂::T=T(0.66)`: Feasibility step: increase the trust-region radius when `Ared/Pred > η₂`.
- `feas_σ₁::T=T(0.25)`: Feasibility step: decrease coefficient of the trust-region radius.
- `feas_σ₂::T=T(2.0)`: Feasibility step: increase coefficient of the trust-region radius.
- `feas_Δ₀::T=one(T)`: Feasibility step: initial radius.
- `bad_steps_lim::Integer=3`: Feasibility step: consecutive bad steps before using a second order step.
- `feas_expected_decrease::T=T(0.95)`: Feasibility step: bad steps are when `‖c(z)‖ / ‖c(x)‖ >feas_expected_decrease`.
- `TR_compute_step::Symbol=:TR_lsmr`: Compute the direction in feasibility step: options: `:TR_dogleg`.
- `TR_struct::Union{TR_lsmr_struct, TR_dogleg_struct}=TR_lsmr_struct(length(x0), length(y0), S)`.
- `compρ_p1::T=T(0.75)`: update ρ as `ρ = max(min(ngp, p1) * ρmax, ϵ)`.
- `compρ_p2::T=T(0.90)`: update ρ as `ρ = primalnorm * p2` if not sufficiently feasible.
- `ρbar::T=T(2.0)`: radius of the larger cylinder is `ρbar * ρ`.
- `tan_Δ::T=one(T)`: Tangent step trust-region parameters: initial trust-region radius.
- `tan_η₁::T=T(1e-2)`: Tangent step trust-region parameters: decrease the trust-region radius when `Ared/Pred < η₁`.
- `tan_η₂::T=T(0.75)`: Tangent step trust-region parameters: increase the trust-region radius when `Ared/Pred > η₂`.
- `tan_σ₁::T=T(0.25)`: Tangent step trust-region parameters: decrease coefficient of the trust-region radius.
- `tan_σ₂::T=T(2.0)`: Tangent step trust-region parameters: increase coefficient of the trust-region radius.
- `tan_small_d::T=eps(T)`: Tangent step trust-region parameters: `||d||` is too small.
- `increase_Δtg::T=10`: Tangent step trust-region parameters: increase if possible, `< 1 / √eps(T)`, the `Δtg` between tangent steps.

For more details, we refer to the package documentation [fine-tuneDCI.md](https://juliasmoothoptimizers.github.io/DCISolver.jl/dev/fine-tuneDCI/).
"""
struct MetaDCI{
  T <: AbstractFloat,
  In <: Integer,
  COO <: SymCOOSolver,
  CGLSStruct <: comp_λ_cgls,
  TRStruct <: Union{TR_lsmr_struct, TR_dogleg_struct},
}

  #Tolerances on the problem:
  atol::T
  rtol::T # ϵd = atol + rtol * dualnorm
  ctol::T # feasibility tolerance
  unbounded_threshold::T

  verbose::Int

  #Evaluation limits
  max_eval::In # max number of cons + obj evals
  max_time::Float64
  max_iter::In
  max_iter_normal_step::In

  #Compute Lagrange multipliers
  λ_struct::CGLSStruct
  #λ_struct_rescue #one idea is to have a 2nd set in case of emergency
  #good only if we can make a warm-start.

  # Solver for the factorization
  linear_solver::Symbol # = :ldlfact,#:ma57,
  fact_type::Val{COO}
  ## regularization of the factorization
  decrease_γ::T
  increase_γ::T
  δmin::T

  # Normal step
  feas_step::Symbol #:feasibility_step
  ## Feasibility step (called inside the normal step)
  feas_η₁::T
  feas_η₂::T
  feas_σ₁::T
  feas_σ₂::T
  feas_Δ₀::T
  bad_steps_lim::In
  feas_expected_decrease::T
  agressive_cgsolver::CgSolver # second-order correction
  ## Compute the direction in feasibility step
  TR_compute_step::Symbol #:TR_lsmr, :TR_dogleg
  TR_compute_step_struct::TRStruct

  # Parameters updating ρ (or redefine the function `compute_ρ`)
  compρ_p1::T
  compρ_p2::T
  ρbar::T

  #Tangent step TR parameters
  tan_Δ::T
  tan_η₁::T
  tan_η₂::T
  tan_σ₁::T
  tan_σ₂::T
  tan_small_d::T
  increase_Δtg::T
end

function MetaDCI(nlp::AbstractNLPModel, x = nlp.meta.x0, y = nlp.meta.y0; kwargs...)
  return MetaDCI(x, y; kwargs...)
end

function MetaDCI(
  x0::S,
  y0::AbstractVector{T};
  atol::T = T(1e-5),
  rtol::T = T(1e-5),
  ctol::T = T(1e-5),
  unbounded_threshold::T = -T(1e5),
  verbose::Union{Integer, Bool} = 0,
  max_eval::Integer = 50000,
  max_time::Float64 = 120.0,
  max_iter::Integer = 500,
  max_iter_normal_step::Integer = typemax(Int),
  λ_struct::comp_λ_cgls = comp_λ_cgls(length(x0), length(y0), S),
  linear_solver::Symbol = :ldlfact,
  decrease_γ::T = T(0.1),
  increase_γ::T = T(100.0),
  δmin::T = √eps(T),
  feas_step::Symbol = :feasibility_step,
  feas_η₁::T = T(1e-3),
  feas_η₂::T = T(0.66),
  feas_σ₁::T = T(0.25),
  feas_σ₂::T = T(2.0),
  feas_Δ₀::T = one(T),
  bad_steps_lim::Integer = 3,
  feas_expected_decrease::T = T(0.95),
  TR_compute_step::Symbol = :TR_lsmr,
  TR_struct::Union{TR_lsmr_struct, TR_dogleg_struct} = TR_lsmr_struct(length(x0), length(y0), S),
  compρ_p1::T = T(0.75),
  compρ_p2::T = T(0.90),
  ρbar::T = T(2.0),
  tan_Δ::T = one(T),
  tan_η₁::T = T(1e-2),
  tan_η₂::T = T(0.75),
  tan_σ₁::T = T(0.25),
  tan_σ₂::T = T(2.0),
  tan_small_d::T = eps(T),
  increase_Δtg::T = T(10),
) where {T <: AbstractFloat, S <: AbstractVector{T}}
  if !(linear_solver ∈ keys(solver_correspondence))
    @warn "linear solver $linear_solver not found in $(collect(keys(solver_correspondence))). Using :ldlfact instead"
    linear_solver = :ldlfact
  end

  n = length(x0)
  agressive_cgsolver = CgSolver(n, n, typeof(x0))

  return MetaDCI(
    atol,
    rtol,
    ctol,
    unbounded_threshold,
    convert(Int, verbose),
    max_eval,
    max_time,
    max_iter,
    max_iter_normal_step,
    λ_struct,
    linear_solver,
    Val(solver_correspondence[linear_solver]),
    decrease_γ,
    increase_γ,
    δmin,
    feas_step,
    feas_η₁,
    feas_η₂,
    feas_σ₁,
    feas_σ₂,
    feas_Δ₀,
    bad_steps_lim,
    feas_expected_decrease,
    agressive_cgsolver,
    TR_compute_step,
    TR_struct,
    compρ_p1,
    compρ_p2,
    ρbar,
    tan_Δ,
    tan_η₁,
    tan_η₂,
    tan_σ₁,
    tan_σ₂,
    tan_small_d,
    increase_Δtg,
  )
end
