"""
`comp_λ_cgls{<: AbstractFloat}` attributes correspond to input parameters of `cgls` used in the computation of Lagrange multipliers.
"""
struct comp_λ_cgls{T <: AbstractFloat, S <: AbstractVector{T}}
  comp_λ_solver::CglsSolver{T, S}
  M # =I,
  λ::T # =zero(T), 
  atol::T # =√eps(T), 
  rtol::T # =√eps(T),
  #radius :: T=zero(T), 
  itmax::Integer # =0, 
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
  itmax::Integer = 5 * (m + n),
) where {T, S <: AbstractVector{T}}
  comp_λ_solver = CglsSolver(m, n, S)
  return comp_λ_cgls(comp_λ_solver, M, λ, atol, rtol, itmax)
end

const comp_λ_solvers = Dict(:cgls => comp_λ_cgls)

const solver_correspondence = if isdefined(HSL, :libhsl_ma57)
  Dict(:ma57 => MA57Struct, :ldlfact => LDLFactorizationStruct)
else
  Dict(:ldlfact => LDLFactorizationStruct)
end

struct TR_lsmr_struct{T <: AbstractFloat, S <: AbstractVector{T}}
  lsmr_solver::LsmrSolver{T, S}
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
  itmax::Integer = m + n,
) where {T, S <: AbstractVector{T}}
  lsmr_solver = LsmrSolver(n, m, S)
  return TR_lsmr_struct(lsmr_solver, M, λ, axtol, btol, atol, rtol, etol, itmax)
end

struct TR_dogleg_struct{T <: AbstractFloat, S <: AbstractVector{T}}
  # :-)
  # There is another lsmr call here
  lsmr_solver::LsmrSolver{T, S}
end

function TR_dogleg_struct(m, n, ::Type{S}; kwargs...) where {T, S <: AbstractVector{T}}
  lsmr_solver = LsmrSolver(n, m, S)
  return TR_dogleg_struct(lsmr_solver)
end

const TR_solvers = Dict(:TR_lsmr => TR_lsmr_struct, :TR_dogleg => TR_dogleg_struct)

struct MetaDCI

  #dci function call:
  #Tolerances on the problem:
  atol::AbstractFloat # = 1e-5,
  rtol::AbstractFloat # = 1e-5, #ϵd = atol + rtol * dualnorm
  ctol::AbstractFloat # = 1e-5, #feasibility tolerance

  unbounded_threshold::AbstractFloat # = -1e5

  #Evaluation limits
  max_eval::Integer # = 50000,
  max_time::AbstractFloat # = 60.
  max_iter::Integer #:: Int = 500
  max_iter_normal_step::Integer #:: Int = typemax(Int) 

  #Compute Lagrange multipliers
  comp_λ::Symbol
  λ_struct::comp_λ_cgls
  #λ_struct_rescue #one idea is to have a 2nd set in case of emergency 
  #good only if we can make a warm-start.

  #Solver for the factorization
  linear_solver::Symbol # = :ldlfact,#:ma57,
  # regularization of the factorization
  decrease_γ::AbstractFloat # reduce gamma if possible (> √eps(T)) between tangent steps = 0.1
  increase_γ::AbstractFloat # up gamma if possible (< 1/√eps(T)) in factorization = 100.0
  δmin::AbstractFloat # smallest value (only one > 0) used for regularization δ = √eps(T),

  #Normal step
  feas_step::Symbol #:feasibility_step (add CaNNOLes)
  #Feasibility step (called inside the normal step)
  feas_η₁::AbstractFloat # = 1e-3,
  feas_η₂::AbstractFloat # = 0.66,
  feas_σ₁::AbstractFloat # = 0.25,
  feas_σ₂::AbstractFloat # = 2.0,
  feas_Δ₀::AbstractFloat # = one(T),
  bad_steps_lim::Integer # = 3,
  feas_expected_decrease::AbstractFloat # =0.95, # Bad steps are when ‖c(z)‖ / ‖c(x)‖ >feas_expected_decrease
  #Feasibility step in the normal step
  TR_compute_step::Symbol #:TR_lsmr, :TR_dogleg
  TR_compute_step_struct::Union{TR_lsmr_struct, TR_dogleg_struct}
  # Parameters updating ρ (or redefine the function `compute_ρ`)
  compρ_p1::Real # Float64 = 0.75
  compρ_p2::Real # Float64 = 0.90
end

function MetaDCI(
  x0::S,
  y0::AbstractVector{T};
  atol::AbstractFloat = T(1e-5),
  rtol::AbstractFloat = T(1e-5),
  ctol::AbstractFloat = T(1e-5),
  unbounded_threshold::AbstractFloat = -T(1e5),
  max_eval::Integer = 50000,
  max_time::AbstractFloat = 120.0,
  max_iter::Integer = 500,
  max_iter_normal_step::Integer = typemax(Int),
  comp_λ::Symbol = :cgls!,
  λ_struct::comp_λ_cgls = comp_λ_cgls(length(x0), length(y0), S),
  linear_solver::Symbol = :ldlfact,
  decrease_γ::AbstractFloat = 0.1,
  increase_γ::AbstractFloat = 100.0,
  δmin::AbstractFloat = √eps(T),
  feas_step::Symbol = :feasibility_step,
  feas_η₁::AbstractFloat = 1e-3,
  feas_η₂::AbstractFloat = 0.66,
  feas_σ₁::AbstractFloat = 0.25,
  feas_σ₂::AbstractFloat = 2.0,
  feas_Δ₀::T = one(T),
  bad_steps_lim::Integer = 3,
  feas_expected_decrease::T = T(0.95),
  TR_compute_step::Symbol = :TR_lsmr, #:TR_dogleg
  TR_struct::Union{TR_lsmr_struct, TR_dogleg_struct} = TR_lsmr_struct(length(x0), length(y0), S),
  compρ_p1::Real = 0.75,
  compρ_p2::Real = 0.90,
) where {T <: AbstractFloat, S <: AbstractVector{T}}
  if !(linear_solver ∈ keys(solver_correspondence))
    @warn "linear solver $linear_solver not found in $(collect(keys(solver_correspondence))). Using :ldlfact instead"
    linear_solver = :ldlfact
  end

  return MetaDCI(
    atol,
    rtol,
    ctol,
    unbounded_threshold,
    max_eval,
    max_time,
    max_iter,
    max_iter_normal_step,
    comp_λ,
    λ_struct,
    linear_solver,
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
    TR_compute_step,
    TR_struct,
    compρ_p1,
    compρ_p2,
  )
end
