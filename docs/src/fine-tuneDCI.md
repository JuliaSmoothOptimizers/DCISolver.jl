# Advanced-usage of DCI

## Contents

```@contents
Pages = ["fine-tuneDCI.md"]
```

The main function exported by this package is the function `dci` whose basic usage has been illustrated previously.
It is also possible to fine-tune the parameters used in the implementation in two different ways.

## Examples

DCISolver.jl exports the function `dci`:
```
dci(nlp :: AbstractNLPModel, x :: AbstractVector{T}, meta :: MetaDCI) where T
```
where `MetaDCI` is a structure handling all the parameters used in the algorithm.

It is therefore possible to either call `dci(nlp, x, kwargs...)` and the keywords arguments are passed to the `MetaDCI` constructor or build an instance of `MetaDCI` directly.

```@example
using ADNLPModels, DCISolver

nlp = ADNLPModel(
  x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, 
  [-1.2; 1.0],
  x->[x[1] * x[2] - 1], 
  [0.0], [0.0],
  name = "Rosenbrock with x₁x₂=1"
)

m, n = nlp.meta.ncon, nlp.meta.nvar
meta = DCISolver.MetaDCI(
  nlp.meta.x0, nlp.meta.y0, 
  max_time = 600., 
  linear_solver = :ldlfact, 
  TR_compute_step = :TR_lsmr
)
stats = dci(nlp, nlp.meta.x0, meta)

#The alternative would be:
stats2 = dci(
  nlp, nlp.meta.x0, 
  max_time = 600., 
  linear_solver = :ldlfact, 
  TR_compute_step = :TR_lsmr
)
```

## List of possible options

Find below a list of the main options of `dci`:
```
# Tolerances on the problem: in general, we use `ϵ = atol + rtol * dualnorm`
atol :: AbstractFloat # default: 1e-5 ; absolute tolerance.
rtol :: AbstractFloat # default: 1e-5 ; relative tolerance.
ctol :: AbstractFloat # default: 1e-5 ; feasibility tolerance.

unbounded_threshold :: AbstractFloat # default: -1e5 ; below this threshold the problem is unbounded.

# Evaluation limits
max_eval :: Integer # default: 50000 ; maximum number of cons + obj evaluations.
max_time :: AbstractFloat # default: 120 ; maximum number of seconds.
max_iter :: Integer # default: 500 ; maximum number of iterations.
max_iter_normal_step :: Integer # default: typemax(Int) ; maximum number of iterations in normal step.

# Compute Lagrange multipliers
comp_λ :: Symbol # default: :cgls ; eval(comp_λ) is used to compute Lagrange multipliers.
λ_struct :: comp_λ_cgls # default: comp_λ_cgls(length(x0), length(y0), typeof(x0)) ; companion structure of `comp_λ`.
   
# Tangent step
## Solver for the factorization
linear_solver :: Symbol # default: :ldlfact, options: :ma57.
## Regularization for the factorization
decrease_γ :: AbstractFloat # default: 0.1 ; reduce γ if possible, > √eps(T), between tangent steps.
increase_γ :: AbstractFloat # default: 100.0 ; up γ if possible, < 1/√eps(T), during the factorization.
δmin :: AbstractFloat # default: √eps(T) ; smallest value of δ used for the regularization.
## Tangent step trust-region parameters
tan_Δ :: AbstractFloat # default: 1.0 ; initial trust-region radius.
tan_η₁ :: AbstractFloat # default: 1e-2 ; decrease the trust-region radius when Ared/Pred < η₁.
tan_η₂ :: AbstractFloat # default: 0.75 ; increase the trust-region radius when Ared/Pred > η₂.
tan_σ₁ :: AbstractFloat # default: 0.25 ; decrease coefficient of the trust-region radius.
tan_σ₂ :: AbstractFloat # default: 2.0 ; increase coefficient of the trust-region radius.
tan_small_d :: AbstractFloat # default: eps(T) ; ||d|| is too small.
increase_Δtg :: AbstractFloat # default: 10.0 ; increase if possible, < 1 / √eps(T), the Δtg between tangent steps.

# Normal step
feas_step :: Symbol # default: :feasibility_step
## Feasibility step
feas_η₁ :: AbstractFloat # default: 1e-3 ; decrease the trust-region radius when Ared/Pred < η₁.
feas_η₂ :: AbstractFloat # default: 0.66 ; increase the trust-region radius when Ared/Pred > η₂.
feas_σ₁ :: AbstractFloat # default: 0.25 ; decrease coefficient of the trust-region radius.
feas_σ₂ :: AbstractFloat # default: 2.0 ; increase coefficient of the trust-region radius.
feas_Δ₀ :: AbstractFloat # default: 1.0 ; initial radius.
feas_expected_decrease :: AbstractFloat # default: 0.95 ; bad steps are when ‖c(z)‖ / ‖c(x)‖ >feas_expected_decrease.
bad_steps_lim :: Integer # default: 3 ; consecutive bad steps before using a second order step.
## Compute the direction in feasibility step
TR_compute_step :: Symbol # default: :TR_lsmr, options: :TR_dogleg.
TR_compute_step_struct :: Union{TR_lsmr_struct, TR_dogleg_struct} # default: TR_lsmr_struct(length(x0), length(y0), typeof(x0)), options: TR_dogleg_struct(length(x0), length(y0), typeof(x0)).

# Parameters updating ρ (or redefine the function `compute_ρ`)
compρ_p1 :: AbstractFloat # default: 0.75 ; update ρ as `ρ = max(min(ngp, p1) * ρmax, ϵ)`.
compρ_p2 :: AbstractFloat # default: 0.90 ; update ρ as `ρ = primalnorm * p2` if not sufficiently feasible.
ρbar :: AbstractFloat # default: 2.0 ; radius of the larger cylinder is `ρbar * ρ`.
#Computation of ρ can be modified by importing `compute_ρ(dualnorm, primalnorm, norm∇fx, ρmax, ϵ, iter, meta::MetaDCI)`
```
