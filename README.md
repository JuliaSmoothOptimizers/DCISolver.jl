# DCISolver - Dynamic Control of Infeasibility Solver

| **Documentation** | **CI** | **Coverage** | **Release** | **DOI** |
|:-----------------:|:------:|:------------:|:-----------:|:-------:|
| [![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] | [![build-ci][build-ci-img]][build-ci-url] | [![codecov][codecov-img]][codecov-url] | [![release][release-img]][release-url] | [![doi][doi-img]][doi-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://JuliaSmoothOptimizers.github.io/DCISolver.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://JuliaSmoothOptimizers.github.io/DCISolver.jl/dev
[build-ci-img]: https://github.com/JuliaSmoothOptimizers/DCISolver.jl/workflows/CI/badge.svg?branch=main
[build-ci-url]: https://github.com/JuliaSmoothOptimizers/DCISolver.jl/actions
[codecov-img]: https://codecov.io/gh/JuliaSmoothOptimizers/DCISolver.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaSmoothOptimizers/DCISolver.jl
[release-img]: https://img.shields.io/github/v/release/JuliaSmoothOptimizers/DCISolver.jl.svg?style=flat-square
[release-url]: https://github.com/JuliaSmoothOptimizers/DCISolver.jl/releases
[doi-img]: https://joss.theoj.org/papers/10.21105/joss.03991/status.svg
[doi-url]: https://doi.org/10.21105/joss.03991

DCI is a solver for equality-constrained nonlinear problems, i.e.,
optimization problems of the form

    min f(x)     s.t.     c(x) = 0.

It uses other JuliaSmoothOptimizers packages for development.
In particular, [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) is used for defining the problem, and [SolverCore](https://github.com/JuliaSmoothOptimizers/SolverCore.jl) for the output.
It uses [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl) by default to compute the factorization in the tangent step. [HSL.jl](https://github.com/JuliaSmoothOptimizers/HSL.jl) provides alternative linear solvers if [libHSL](https://licences.stfc.ac.uk/product/libhsl) can be downloaded.
The feasibility steps are factorization-free and use iterative methods from [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl)

## References

> Bielschowsky, R. H., & Gomes, F. A.
> Dynamic control of infeasibility in equality constrained optimization.
> SIAM Journal on Optimization, 19(3), 1299-1325 (2008).
> [10.1137/070679557](https://doi.org/10.1137/070679557)

> Migot, T., Orban D., & Siqueira A. S.
> DCISolver. jl: A Julia Solver for Nonlinear Optimization using Dynamic Control of Infeasibility.
> Journal of Open Source Software 70(7), 3991 (2022).
> [10.21105/joss.03991](https://doi.org/10.21105/joss.03991)

## How to Cite

If you use DCISolver.jl in your work, please cite using the format given in [CITATION.cff](https://github.com/JuliaSmoothOptimizers/DCISolver.jl/blob/main/CITATION.cff).

## Installation

1. [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl) is used by default. Follow [HSL.jl](https://github.com/JuliaSmoothOptimizers/HSL.jl)'s `MA57` installation for an alternative.
2. `pkg> add DCISolver`

## Example

```julia
using DCISolver, ADNLPModels

# Rosenbrock
nlp = ADNLPModel(x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, [-1.2; 1.0])
stats = dci(nlp)

# Constrained
nlp = ADNLPModel(x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, [-1.2; 1.0],
                 x->[x[1] * x[2] - 1], [0.0], [0.0])
stats = dci(nlp)
```

# Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/DCISolver.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.
