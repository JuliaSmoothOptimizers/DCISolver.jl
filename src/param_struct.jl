const solver_correspondence = if isdefined(HSL, :libhsl_ma57)
    Dict(:ma57 => MA57Struct, 
         :ldlfact => LDLFactorizationStruct)
  else
    Dict(:ldlfact => LDLFactorizationStruct)
  end

  struct MetaDCI

  #dci function call:
    #Tolerances on the problem:
    atol # = 1e-5,
    rtol # = 1e-5, #ϵd = atol + rtol * dualnorm
    ctol # = 1e-5, #feasibility tolerance

    #Evaluation limits
    max_eval # = 50000,
    max_time # = 60.
    max_iter #:: Int = 500

    #Compute Lagrange multiplier min_λ ‖Jx' λ - ∇fx‖
    solver :: Symbol #cgls by default, ε = atol + rtol * ArNorm
    λatol  :: AbstractFloat #√eps(T)
    λrtol  :: AbstractFloat #√eps(T)

    #List of intermediary functions:
    feas_step :: Symbol #minimize ‖c(x)‖  #dogleg or TR_lsmr
    TR_computation_step :: Symbol #min_d ‖cₖ + Jₖd‖ s.t. ||d|| ≤ Δ #feasibility_step or cannoles_step
    #Infeasibility formula:
    #infeasible = norm(d) < ctol * min(normcz, one(T))

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
    linear_solver # = :ldlfact,#:ma57,
    solve_Newton_system :: Function
    δmin     :: AbstractFloat # = √eps(T), > 0
    γmin     :: AbstractFloat # > 0
    γred     :: AbstractFloat #parameter reducing 0 < γ < 1

    #Strategies to handle ρ
    #Initialization of ρmax : ρmax = max(ctol, 5primalnorm, 50dualnorm)
    #we don't let ρmax too far from the residuals
    #ρmax = min(ρmax, max(ctol, 5primalnorm, 50dualnorm))

  end

#=
function MetaDCI(; atol = 1e-5,
                   rtol = 1e-5,
                   ctol = 1e-5,
                   max_eval = 50000,
                   max_time = 60,
                   max_iter = 500,
                   linear_solver :: Symbol = :ldlfact,
                   δmin :: AbstractFloat = 1.0e-8,
                   γmin :: AbstractFloat = 1.0e-8,
                   γred :: AbstractFloat = 0.1,
                   feas_η₁ :: AbstractFloat = 1e-3, 
                   feas_η₂ :: AbstractFloat = 0.66, 
                   feas_σ₁ :: AbstractFloat = 0.25, 
                   feas_σ₂ :: AbstractFloat = 2.0,
                   feas_Δ0 :: AbstractFloat = one(T),)
=#
