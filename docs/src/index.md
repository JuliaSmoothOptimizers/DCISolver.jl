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
> [10.1007/s10589-020-00201-2](https://doi.org/10.1007/s10589-020-00201-2)

`DCISolver` is a JuliaSmoothOptimizers-compliant solver. It takes an [`AbstractNLPModel`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) as an input and returns a [`GenericExecutionStats`](https://github.com/JuliaSmoothOptimizers/SolverCore.jl/blob/16fc349908f46634f2c9acdddddb009b23634b71/src/stats.jl#L60).

## Installation

`DCISolver` is a registered package. To install this package, open the Julia REPL (i.e., execute the julia binary), type ] to enter package mode, and install `DCISolver` as follows
```
add DCISolver
```

It uses [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl) by default to compute the factorization in the tangent step. Follow [HSL.jl](https://github.com/JuliaSmoothOptimizers/HSL.jl)'s `MA57` installation for an alternative.

The feasibility steps are factorization-free and use iterative methods from [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl).

## Example

```@example
using DCISolver, ADNLPModels, Logging
nlp = ADNLPModel(
  x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, 
  [-1.2; 1.0],
  x -> [x[1] * x[2] - 1], 
  [0.0], [0.0],
  name = "Rosenbrock with x₁x₂=1"
)
stats = with_logger(NullLogger()) do
  dci(nlp, nlp.meta.x0)
end

println(stats)
```