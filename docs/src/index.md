# DCISolver - Dynamic Control of Infeasibility Solver

DCI is a solver for equality-constrained nonlinear problems, i.e.,
optimization problems of the form

```math
    \min_x \ f(x) \quad \text{s.t.} \quad  c(x) = 0,
```

based on the paper

> Bielschowsky, R. H., & Gomes, F. A.
> Dynamic control of infeasibility in equality constrained optimization.
> SIAM Journal on Optimization, 19(3), 1299-1325 (2008).
> [10.1137/070679557](https://doi.org/10.1137/070679557)

`DCISolver` is a JuliaSmoothOptimizers-compliant solver. It takes an [`AbstractNLPModel`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) as an input and returns a [`GenericExecutionStats`](https://github.com/JuliaSmoothOptimizers/SolverCore.jl/blob/16fc349908f46634f2c9acdddddb009b23634b71/src/stats.jl#L60).

We refer to [jso.dev](https://jso.dev) for tutorials on the NLPModel API. This framework allows the usage of models from Ampl (using [AmplNLReader.jl](https://github.com/JuliaSmoothOptimizers/AmplNLReader.jl)), CUTEst (using [CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl)), JuMP (using [NLPModelsJuMP.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl)), PDE-constrained optimization problems (using [PDENLPModels.jl](https://github.com/JuliaSmoothOptimizers/PDENLPModels.jl)) and models defined with automatic differentiation (using [ADNLPModels.jl](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl)).

> Migot, T., Orban D., & Siqueira A. S.
> DCISolver. jl: A Julia Solver for Nonlinear Optimization using Dynamic Control of Infeasibility.
> Journal of Open Source Software 70(7), 3991 (2022).
> [10.21105/joss.03991](https://doi.org/10.21105/joss.03991)

## Installation

`DCISolver` is a registered package. To install this package, open the Julia REPL (i.e., execute the julia binary), type `]` to enter package mode, and install `DCISolver` as follows
```
add DCISolver
```

The DCI algorithm is an iterative method that has the flavor of a projected gradient algorithm and could be characterized as
a relaxed feasible point method with dynamic control of infeasibility. It is a combination of two steps: a tangent step and a feasibility step.
It uses [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl) by default to compute the factorization in the tangent step. [HSL.jl](https://github.com/JuliaSmoothOptimizers/HSL.jl) provides alternative linear solvers if [libHSL](https://licences.stfc.ac.uk/product/libhsl) can be downloaded.
The feasibility steps are factorization-free and use iterative methods from [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl).

## Example

We consider in this example the minization of the Rosenbrock function over an equality constraint.
```math
    \min_x \ 100 * (x₂ - x₁²)² + (x₁ - 1)² \quad \text{s.t.} \quad  x₁x₂=1,
```
The problem is modeled using `ADNLPModels.jl` with `[-1.2; 1.0]` as default initial point, and then solved using `dci`.
```@example
using DCISolver, ADNLPModels, Logging
nlp = ADNLPModel(
  x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, 
  [-1.2; 1.0],
  x -> [x[1] * x[2] - 1], 
  [0.0], [0.0],
  name = "Rosenbrock with x₁x₂=1"
)
stats = dci(nlp, verbose = 0)

println(stats)
```

# Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/DCISolver.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.
