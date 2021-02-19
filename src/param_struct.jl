const solver_correspondence = if isdefined(HSL, :libhsl_ma57)
    Dict(:ma57 => MA57Struct, 
         :ldlfact => LDLFactorizationStruct)
  else
    Dict(:ldlfact => LDLFactorizationStruct)
  end

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
                ) where T <: AbstractFloat

  if !(linear_solver ∈ keys(solver_correspondence))
    @warn "linear solver $linear_solver not found in $(collect(keys(solver_correspondence))). Using :ldlfact instead"
    linear_solver = :ldlfact
  end

 return MetaDCI(atol, rtol, ctol, max_eval, max_time, max_iter, 
                linear_solver)
end   
