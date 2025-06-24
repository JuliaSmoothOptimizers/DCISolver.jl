# Advanced-usage of DCI

## Contents

```@contents
Pages = ["fine-tuneDCI.md"]
```

The main function exported by this package is the function `dci` whose basic usage has been illustrated previously.
It is also possible to fine-tune the parameters used in the implementation in two different ways.

## Examples

DCISolver.jl exports the function `dci`:

```julia
   dci(nlp :: AbstractNLPModel)
   dci(nlp :: AbstractNLPModel, x :: AbstractVector)
   dci(nlp :: AbstractNLPModel, meta :: MetaDCI, x :: AbstractVector)
   solve!(workspace :: DCIWorkspace, nlp :: AbstractNLPModel)
   solve!(workspace :: DCIWorkspace, nlp :: AbstractNLPModel, stats :: GenericExecutionStats)
```

where `MetaDCI` is a structure handling all the parameters used in the algorithm, and `DCIWorkspace` pre-allocates all the memory used during the iterative process.

It is therefore possible to either call `dci(nlp, x, kwargs...)` and the keywords arguments are passed to the `MetaDCI` constructor or build an instance of `MetaDCI` directly.

```@example ex1
using ADNLPModels, DCISolver

nlp = ADNLPModel(
  x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2,
  [-1.2; 1.0],
  x->[x[1] * x[2] - 1],
  [0.0], [0.0],
  name = "Rosenbrock with x₁x₂=1"
)

#The alternative would be:
stats = dci(
  nlp, nlp.meta.x0,
  max_time = 600.,
  linear_solver = :ldlfact,
  TR_compute_step = :TR_lsmr
)
```

The alternative would be:

```@example ex1
meta = DCISolver.MetaDCI(
  nlp.meta.x0, nlp.meta.y0,
  max_time = 600.,
  linear_solver = :ldlfact,
  TR_compute_step = :TR_lsmr
)
stats = dci(nlp, meta, nlp.meta.x0)
```

The `DCIWorkspace` allows to reuse the same memory if one would re-solve a problem of the same dimension.

```@example ex1
workspace = DCISolver.DCIWorkspace(nlp, meta, nlp.meta.x0)
stats = DCISolver.solve!(workspace, nlp)
workspace.x0 .= ones(2) # change the initial guess, and resolve
stats = DCISolver.solve!(workspace, nlp)
```

## List of possible options

Find below a list of the main options of `dci`.

### Tolerances on the problem

We use `ϵ = atol + rtol * dualnorm`.

```markdown
| Parameters           | Type          | Default      | Description                                    |
| -------------------- | ------------- | ------------ | ---------------------------------------------- |
| atol                 | AbstractFloat | 1e-5         | absolute tolerance.                            |
| rtol                 | AbstractFloat | 1e-5         | relative tolerance.                            |
| ctol                 | AbstractFloat | 1e-5         | feasibility tolerance.                         |
| unbounded_threshold  | AbstractFloat | -1e5         | below this threshold the problem is unbounded. |
| max_eval             | Integer       | 50000        | maximum number of cons + obj evaluations.      |
| max_time             | AbstractFloat | 120.         | maximum number of seconds.                     |
| max_iter             | Integer       | 500          | maximum number of iterations.                  |
| max_iter_normal_step | Integer       | typemax(Int) | maximum number of iterations in normal step.   |
```

### Compute Lagrange multipliers

```markdown
| Parameters  | Type        | Default                                         | Description                                           |
| ----------- | ----------- | ----------------------------------------------- | ----------------------------------------------------- |
| comp_λ      | Symbol      | :cgls                                           | eval(comp_λ) is used to compute Lagrange multipliers. |
| λ_struct    | comp_λ_cgls | comp_λ_cgls(length(x0), length(y0), typeof(x0)) | companion structure of `comp_λ`.                      |
```

### Tangent step

```markdown
| Parameters    | Type          | Default  | Description                                                                                               |
| ------------- | ------------- | -------- | --------------------------------------------------------------------------------------------------------- |
| linear_solver | Symbol        | :ldlfact | Solver for the factorization. options: :ma57.                                                             |
| decrease_γ    | AbstractFloat | 0.1      | Regularization for the factorization: reduce γ if possible, > √eps(T), between tangent steps.             |
| increase_γ    | AbstractFloat | 100.0    | Regularization for the factorization: up γ if possible, < 1/√eps(T), during the factorization.            |
| δmin          | AbstractFloat | √eps(T)  | Regularization for the factorization: smallest value of δ used for the regularization.                    |
| tan_Δ         | AbstractFloat | 1.0      | Tangent step trust-region parameters: initial trust-region radius.                                        |
| tan_η₁        | AbstractFloat | 1e-2     | Tangent step trust-region parameters: decrease the trust-region radius when Ared/Pred < η₁.               |
| tan_η₂        | AbstractFloat | 0.75     | Tangent step trust-region parameters: increase the trust-region radius when Ared/Pred > η₂.               |
| tan_σ₁        | AbstractFloat | 0.25     | Tangent step trust-region parameters: decrease coefficient of the trust-region radius.                    |
| tan_σ₂        | AbstractFloat | 2.0      | Tangent step trust-region parameters: increase coefficient of the trust-region radius.                    |
| tan_small_d   | AbstractFloat | eps(T)   | Tangent step trust-region parameters: ||d|| is too small.                                                 |
| increase_Δtg  | AbstractFloat | 10.0     | Tangent step trust-region parameters: increase if possible, < 1 / √eps(T), the Δtg between tangent steps. |
```

### Normal step

```markdown
| Parameters             | Type                                    | Default                                            | Description                                                                                               |
| ---------------------- | --------------------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| feas_step              | Symbol                                  | :feasibility_step                                  | Normal step                                                                                               |
| feas_η₁                | AbstractFloat                           | 1e-3                                               | Feasibility step: decrease the trust-region radius when Ared/Pred < η₁.                                   |
| feas_η₂                | AbstractFloat                           | 0.66                                               | Feasibility step: increase the trust-region radius when Ared/Pred > η₂.                                   |
| feas_σ₁                | AbstractFloat                           | 0.25                                               | Feasibility step: decrease coefficient of the trust-region radius.                                        |
| feas_σ₂                | AbstractFloat                           | 2.0                                                | Feasibility step: increase coefficient of the trust-region radius.                                        |
| feas_Δ₀                | AbstractFloat                           | 1.0                                                | Feasibility step: initial radius.                                                                         |
| feas_expected_decrease | AbstractFloat                           | 0.95                                               | Feasibility step: bad steps are when ‖c(z)‖ / ‖c(x)‖ >feas_expected_decrease.                             |
| bad_steps_lim          | Integer                                 | 3                                                  | Feasibility step: consecutive bad steps before using a second order step.                                 |
| TR_compute_step        | Symbol                                  | :TR_lsmr                                           | Compute the direction in feasibility step: options: :TR_dogleg.                                           |
| TR_compute_step_struct | Union{TR_lsmr_struct, TR_dogleg_struct} | TR_lsmr_struct(length(x0), length(y0), typeof(x0)) | Compute the direction in feasibility step: options: TR_dogleg_struct(length(x0), length(y0), typeof(x0)). |
```

### Parameters updating ρ (or redefine the function `compute_ρ`)

```markdown
| Parameters  | Type          | Default | Description                                                     |
| ----------- | ------------- | ------- | --------------------------------------------------------------- |
| compρ_p1    | AbstractFloat | 0.75    | update ρ as `ρ = max(min(ngp, p1) * ρmax, ϵ)`.                  |
| compρ_p2    | AbstractFloat | 0.90    | update ρ as `ρ = primalnorm * p2` if not sufficiently feasible. |
| ρbar        | AbstractFloat | 2.0     | radius of the larger cylinder is `ρbar * ρ`.                    |
```

The computation of ρ can also be modified by importing `compute_ρ(dualnorm, primalnorm, norm∇fx, ρmax, ϵ, iter, meta::MetaDCI)`
