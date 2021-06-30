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

# Compute Lagrange multipliers
comp_λ :: Symbol # default: :cgls ; eval(comp_λ) is used to compute Lagrange multipliers.
λ_struct :: comp_λ_cgls # default: comp_λ_cgls(length(x0), length(y0), typeof(x0)) ; companion structure of `comp_λ`.
   
# Tangent step
# Solver for the factorization
linear_solver :: Symbol # default: :ldlfact, options: :ma57.

# Normal step
feas_step :: Symbol # default: :feasibility_step
# Feasibility step in the normal step
TR_compute_step :: Symbol # default: :TR_lsmr, options: :TR_dogleg.
TR_compute_step_struct :: Union{TR_lsmr_struct, TR_dogleg_struct} # default: TR_lsmr_struct(length(x0), length(y0), typeof(x0)), options: TR_dogleg_struct(length(x0), length(y0), typeof(x0)).
```
