"""
`comp_λ_cgls{<: AbstractFloat}` attributes correspond to input parameters of `cgls` used in the computation of Lagrange multipliers.
"""
struct comp_λ_cgls{T <: AbstractFloat}
  M # =opEye(), 
  λ :: T # =zero(T), 
  atol :: T # =√eps(T), 
  rtol :: T # =√eps(T),
  #radius :: T=zero(T), 
  itmax :: Integer # =0, 
  #verbose :: Int=0, 
  #history :: Bool=false
end

function comp_λ_cgls(m, n, :: T; M = opEye(), 
                                 λ     :: T=zero(T), 
                                 atol  :: T = √eps(T), 
                                 rtol  :: T = √eps(T),
                                 itmax :: Integer = 5 * (m + n)
                                 ) where T
  return comp_λ_cgls(M, λ, atol, rtol, itmax)
end

const comp_λ_solvers = Dict(:cgls => comp_λ_cgls)

const solver_correspondence = if isdefined(HSL, :libhsl_ma57)
    Dict(:ma57 => MA57Struct, 
         :ldlfact => LDLFactorizationStruct)
  else
    Dict(:ldlfact => LDLFactorizationStruct)
  end

struct TR_lsmr_struct{T  <: AbstractFloat}
  M # =opEye(), 
  #N=opEye(), #unnecessary
  #sqd :: Bool=false, #unnecessary
  λ     :: T # =zero(T), 
  axtol :: T # =√eps(T), 
  btol  :: T # =√eps(T),
  atol  :: T # =zero(T), 
  rtol  :: T # =zero(T),
  etol  :: T # =√eps(T), 
  #window :: Int=5, #unnecessary
  itmax :: Int # =0,  #m + n (set in the code if itmax==0)
  #conlim :: T=1/√eps(T), #set conditioning upper limit
  #radius :: T=zero(T),  #unnecessary
  #verbose :: Int=0,  #unnecessary
  #history :: Bool=false #unnecessary
end

function TR_lsmr_struct(m, n, :: T; 
                        M = opEye(), 
                        λ     :: T = zero(T), 
                        axtol :: T = √eps(T), 
                        btol  :: T = √eps(T), 
                        atol  :: T = zero(T), 
                        rtol  :: T = zero(T), 
                        etol  :: T = √eps(T), 
                        itmax :: Integer = m+n, 
                        ) where T
  return TR_lsmr_struct(M, λ, axtol, btol, atol, rtol, etol, itmax)
end

struct TR_dogleg_struct
  # :-)
  # There is another lsmr call here
end

function TR_dogleg_struct(args...; kwargs...)
  return TR_dogleg_struct()
end

const TR_solvers = Dict(:TR_lsmr   => TR_lsmr_struct,
                        :TR_dogleg => TR_dogleg_struct)

  struct MetaDCI

  #dci function call:
    #Tolerances on the problem:
    atol :: AbstractFloat # = 1e-5,
    rtol :: AbstractFloat # = 1e-5, #ϵd = atol + rtol * dualnorm
    ctol :: AbstractFloat # = 1e-5, #feasibility tolerance

    #Evaluation limits
    max_eval :: Integer # = 50000,
    max_time :: AbstractFloat # = 60.
    max_iter :: Integer #:: Int = 500

    #Compute Lagrange multipliers
    comp_λ :: Symbol
    λ_struct :: comp_λ_cgls
    #λ_struct_rescue #one idea is to have a 2nd set in case of emergency 
    #good only if we can make a warm-start.
    
    #Solver for the factorization
    linear_solver :: Symbol # = :ldlfact,#:ma57,

  #Normal step
    feas_step :: Symbol #:feasibility_step (add CaNNOLes)
  #Feasibility step in the normal step
    TR_compute_step :: Symbol #:TR_lsmr, :TR_dogleg
    TR_compute_step_struct :: Union{TR_lsmr_struct, TR_dogleg_struct}

  end


function MetaDCI(x0                  :: AbstractVector{T},
                 y0                  :: AbstractVector{T}; 
                 atol                :: AbstractFloat = T(1e-5),
                 rtol                :: AbstractFloat = T(1e-5),
                 ctol                :: AbstractFloat = T(1e-5),
                 max_eval            :: Integer = 50000,
                 max_time            :: AbstractFloat = 120.,
                 max_iter            :: Integer = 500,
                 comp_λ              :: Symbol = :cgls,
                 λ_struct            :: comp_λ_cgls = comp_λ_cgls(length(x0), length(y0), atol),
                 linear_solver       :: Symbol = :ldlfact,
                 feas_step           :: Symbol = :feasibility_step,
                 TR_compute_step     :: Symbol = :TR_lsmr, #:TR_dogleg
                 TR_struct           :: Union{TR_lsmr_struct, TR_dogleg_struct} = TR_lsmr_struct(length(x0), length(y0), atol),
                ) where T <: AbstractFloat

  if !(linear_solver ∈ keys(solver_correspondence))
    @warn "linear solver $linear_solver not found in $(collect(keys(solver_correspondence))). Using :ldlfact instead"
    linear_solver = :ldlfact
  end

 return MetaDCI(atol, rtol, ctol, max_eval, max_time, max_iter,
                comp_λ, λ_struct, 
                linear_solver,
                feas_step,
                TR_compute_step, TR_struct)
end   
