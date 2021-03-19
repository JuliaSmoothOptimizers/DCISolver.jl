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

function TR_lsmr_struct(m, n, :: T, args...; M=opEye(), λ :: T=zero(T), 
                                    axtol :: T=√eps(T), btol :: T=√eps(T), 
                                    atol :: T=zero(T), rtol :: T=zero(T), 
                                    etol :: T=√eps(T), itmax :: Int=m+n, 
                                    kwargs...) where T
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
    max_eval :: Int # = 50000,
    max_time :: AbstractFloat # = 60.
    max_iter :: Int #:: Int = 500
    
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
                 max_eval            :: Int = 50000,
                 max_time            :: AbstractFloat = 60.,
                 max_iter            :: Int = 500,
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
                linear_solver,
                feas_step,
                TR_compute_step, TR_struct)
end   
