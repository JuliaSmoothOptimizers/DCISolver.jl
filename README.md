# DCISolver.jl - Dynamic Control of Infeasibility Solver

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
[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.4742979.svg
[doi-url]: https://doi.org/10.5281/zenodo.4742979

DCI is a solver for equality-constrained nonlinear problems, i.e.,
optimization problems of the form

    min f(x)     s.t.     c(x) = 0.

It uses other JuliaSmoothOptimizers packages for development.
In particular, [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) is used for defining the problem, and [SolverCore](https://github.com/JuliaSmoothOptimizers/SolverCore.jl) for the output.
It uses [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl) by default to compute the factorization in the tangent step. Follow [HSL.jl](https://github.com/JuliaSmoothOptimizers/HSL.jl)'s `MA57` installation for an alternative.
The feasibility steps are factorization-free and use iterative methods from [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl)

## References

> Bielschowsky, R. H., & Gomes, F. A.
> Dynamic control of infeasibility in equality constrained optimization.
> SIAM Journal on Optimization, 19(3), 1299-1325 (2008).
> [10.1007/s10589-020-00201-2](https://doi.org/10.1007/s10589-020-00201-2)

## How to Cite

If you use DCISolver.jl in your work, please cite using the format given in [CITATION.bib](https://github.com/JuliaSmoothOptimizers/DCISolver.jl/blob/main/CITATION.bib).

## Installation

1. [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl) is used by default. Follow [HSL.jl](https://github.com/JuliaSmoothOptimizers/HSL.jl)'s `MA57` installation for an alternative.
2. `pkg> add DCISolver`

## Example

```julia
using DCISolver, NLPModels

# Rosenbrock
nlp = ADNLPModel(x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, [-1.2; 1.0])
stats = dci(nlp, nlp.meta.x0)

# Constrained
nlp = ADNLPModel(x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, [-1.2; 1.0],
                 x->[x[1] * x[2] - 1], [0.0], [0.0])
stats = dci(nlp, nlp.meta.x0)
```