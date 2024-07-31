# DCISolver

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSmoothOptimizers.github.io/DCISolver.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSmoothOptimizers.github.io/DCISolver.jl/dev)
[![Test workflow status](https://github.com/JuliaSmoothOptimizers/DCISolver.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/JuliaSmoothOptimizers/DCISolver.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaSmoothOptimizers/DCISolver.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSmoothOptimizers/DCISolver.jl)
[![Lint workflow Status](https://github.com/JuliaSmoothOptimizers/DCISolver.jl/actions/workflows/Lint.yml/badge.svg?branch=main)](https://github.com/JuliaSmoothOptimizers/DCISolver.jl/actions/workflows/Lint.yml?query=branch%3Amain)
[![Docs workflow Status](https://github.com/JuliaSmoothOptimizers/DCISolver.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/JuliaSmoothOptimizers/DCISolver.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.03991/status.svg)](https://doi.org/10.21105/joss.03991)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![All Contributors](https://img.shields.io/github/all-contributors/JuliaSmoothOptimizers/DCISolver.jl?labelColor=5e1ec7&color=c0ffee&style=flat-square)](#contributors)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

Dynamic Control of Infeasibility Solver (DCI) is a solver for equality-constrained nonlinear problems, i.e.,
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
>
> Migot, T., Orban D., & Siqueira A. S.
> DCISolver. jl: A Julia Solver for Nonlinear Optimization using Dynamic Control of Infeasibility.
> Journal of Open Source Software 70(7), 3991 (2022).
> [10.21105/joss.03991](https://doi.org/10.21105/joss.03991)

## How to Cite

If you use DCISolver.jl in your work, please cite using the reference given in [CITATION.cff](https://github.com/JuliaSmoothOptimizers/DCISolver.jl/blob/main/CITATION.cff).

## Contributing

If you want to make contributions of any kind, please first that a look into our [contributing guide directly on GitHub](docs/src/90-contributing.md) or the [contributing page on the website](https://JuliaSmoothOptimizers.github.io/DCISolver.jl/dev/90-contributing/)

---

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
