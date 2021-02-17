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

    #Compute Lagrange multiplier min_λ ‖Jx' λ - ∇fx‖
    solver :: Function #cgls by default, lsqr
    λatol  :: AbstractFloat #√eps(T)
    λrtol  :: AbstractFloat #√eps(T)
    niter  :: Int # 2(n+m)
    λreg   :: AbstractFloat # zero(T)

    #List of intermediary functions:
    feas_step           :: Function #minimize ‖c(x)‖  #dogleg or TR_lsmr
    TR_computation_step :: Function #min_d ‖cₖ + Jₖd‖ s.t. ||d|| ≤ Δ #feasibility_step or cannoles_step
    #Infeasibility formula:
    rel_infeasible      :: AbstractFloat #infeasible = norm(d) < ctol * min(normcz, one(T))

    #Trust-region parameters in feasibility_step:
    max_feas_iter :: Int # = typemax(Int64) # maximum number of iteration
    feas_η₁ :: AbstractFloat # = 1e-3, 
    feas_η₂ :: AbstractFloat # = 0.66, 
    feas_σ₁ :: AbstractFloat # = 0.25, 
    feas_σ₂ :: AbstractFloat # = 2.0,
    feas_Δ0 :: AbstractFloat # = one(T),

    #Parameters in tangent step
    Δ       :: AbstractFloat # = one(T), #trust-region radius
    η₁      :: AbstractFloat # = T(1e-2),
    η₂      :: AbstractFloat # = T(0.75),
    σ₁      :: AbstractFloat # = T(0.25), #decrease trust-region radius
    σ₂      :: AbstractFloat # = T(2.0), #increase trust-region radius after success
    small_d :: AbstractFloat # = eps(T)  #below this threshold on ||d|| the step is too small.
    Δtg_inc :: AbstractFloat # ≥ 1 #increase trust-region radius after a tangent step 
    Δtg_max :: AbstractFloat # largest possible value of TR-radius # 1/√eps(T)

    #Parameters for the regularization of the factorization in tangent step
    solve_Newton_system :: Function
    linear_solver :: Symbol # = :ldlfact,#:ma57,
    δmin          :: AbstractFloat # = √eps(T), > 0
    γmin          :: AbstractFloat # > 0
    γred          :: AbstractFloat #parameter reducing 0 < γ < 1
    γinc          :: AbstractFloat #γ = min(max(100γ, √eps(T)), 1/√eps(T))

    #Strategies to handle ρ
    #Initialization of ρmax : ρmax = max(ctol, 5primalnorm, 50dualnorm)
    #we don't let ρmax too far from the residuals
    #ρmax = min(ρmax, max(ctol, 5primalnorm, 50dualnorm))

  end


function MetaDCI(x0                  :: AbstractVector{T},
                 y0                  :: AbstractVector{T}; 
                 atol                :: AbstractFloat = T(1e-5),
                 rtol                :: AbstractFloat = T(1e-5),
                 ctol                :: AbstractFloat = T(1e-5),
                 max_eval            :: Int = 50000,
                 max_time            :: AbstractFloat = 60.,
                 max_iter            :: Int = 500,
                 solver              :: Function = cgls, #Krylov.cgls
                 λatol               :: AbstractFloat = √eps(T),
                 λrtol               :: AbstractFloat = √eps(T),
                 niter               :: Int = length(x0) + length(y0),
                 λreg                :: AbstractFloat = zero(T),
                 feas_step           :: Function = TR_lsmr, #minimize ‖c(x)‖  #dogleg or TR_lsmr
                 TR_computation_step :: Function = lsmr,
                 rel_infeasible      :: AbstractFloat = ctol,
                 max_feas_iter       :: Int = typemax(Int64),
                 feas_η₁             :: AbstractFloat = T(1e-3), 
                 feas_η₂             :: AbstractFloat = T(0.66), 
                 feas_σ₁             :: AbstractFloat = T(0.25), 
                 feas_σ₂             :: AbstractFloat = T(2.0),
                 feas_Δ0             :: AbstractFloat = one(T),
                 Δ                   :: AbstractFloat = one(T), #trust-region radius
                 η₁                  :: AbstractFloat = T(1e-2),
                 η₂                  :: AbstractFloat = T(0.75),
                 σ₁                  :: AbstractFloat = T(0.25), #decrease trust-region radius
                 σ₂                  :: AbstractFloat = T(2.0), #increase trust-region radius after success
                 small_d             :: AbstractFloat = eps(T),  #below this threshold on ||d|| the step is too small.
                 Δtg_inc             :: AbstractFloat = T(10.),#≥ 1 #increase trust-region radius after a tangent step 
                 Δtg_max             :: AbstractFloat = 1/√eps(T),# largest possible value of TR-radius # 1/√eps(T)
                 linear_solver       :: Symbol = :ldlfact,
                 solve_Newton_system :: Function = _compute_newton_step!,
                 δmin                :: AbstractFloat  = √eps(T), #> 0
                 γmin                :: AbstractFloat  = 0.0,# > 0
                 γred                :: AbstractFloat  = 0.1,#parameter reducing 0 < γ < 1
                 γinc                :: AbstractFloat  = 100. #γ = min(max(100γ, √eps(T)), 1/√eps(T))
                ) where T <: AbstractFloat

  if !(linear_solver ∈ keys(solver_correspondence))
    @warn "linear solver $linear_solver not found in $(collect(keys(solver_correspondence))). Using :ldlfact instead"
    linear_solver = :ldlfact
  end

 return MetaDCI(atol, rtol, ctol, max_eval, max_time, max_iter, 
                solver, λatol, λrtol, niter, λreg, 
                feas_step, TR_computation_step, rel_infeasible, max_feas_iter,
                feas_η₁, feas_η₂, feas_σ₁, feas_σ₂, feas_Δ0,
                Δ, η₁, η₂, σ₁, σ₂, small_d, Δtg_inc, Δtg_max,
                solve_Newton_system, linear_solver, δmin, γmin, γred, γinc)
end   
