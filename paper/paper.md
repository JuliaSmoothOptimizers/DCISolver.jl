---
title: 'DCISolver.jl: A Julia Solver for Nonlinear Optimization using Dynamic Control of Infeasibility'
tags:
  - Julia
  - nonlinear optimization
  - numerical optimization
  - large-scale optimization
  - constrained optimization
  - nonlinear programming
authors:
  - name: Tangi Migot^[corresponding author]
    orcid: 0000-0001-7729-2513
    affiliation: 1
  - name: Dominique Orban
    orcid: 0000-0002-8017-7687
    affiliation: 1
  - name: Abel Soares Siqueira
    orcid: 0000-0003-4451-281X
    affiliation: 2
affiliations:
 - name: GERAD and Department of Mathematics and Industrial Engineering, Polytechnique Montréal, QC, Canada.
   index: 1
 - name: Netherlands eScience Center, Amsterdam, NL
   index: 2
date: 05 November 2021
bibliography: paper.bib

---

# Summary

`DCISolver.jl` is a new Julia implementation of the Dynamic Control of Infeasibility method (DCI), introduced by @bielschowsky2008dynamic, for solving nonlinear optimization models
with equality constraints:
\begin{equation}\label{eq:nlp}
    \underset{x \in \mathbb{R}^n}{\text{minimize}} \quad f(x) \quad \text{subject to} \quad h(x) = 0,
\end{equation}
where  $f:\mathbb{R}^n \rightarrow \mathbb{R}$ and  $h:\mathbb{R}^n \rightarrow \mathbb{R}^m$ are twice continuously differentiable.
DCI is an iterative method that aims to compute a local minimum of \eqref{eq:nlp} using first and second-order derivatives.

The DCI algorithm is a two-step process that first minimizes the problem with a relaxed feasibility constraint (the tangent step) and then recenters toward a trust cylinder (the normal step). 
The idea of trust cylinders is to keep infeasibility under control, contrary to penalty methods, which encourage feasibility.
Every time the trust cylinder is violated during the tangent step, the normal step reduces the infeasibility.
The radius of the trust cylinder decreases with the iterations, so a feasible and optimal point will be obtained.
For details and theoretical convergence, we refer the reader to the original paper [@bielschowsky2008dynamic].

One of the significant advantages of our implementation is that the normal step is factorization-free, i.e., it uses second-order information via Hessian-vector products but does not need access to the Hessian matrix.
This makes `DCISolver.jl` a valuable asset for large-scale problems, for instance to solve PDE-constrained optimization problems [@migot-orban-siqueira-pdenlpmodels-2021].

`DCISolver.jl` is built upon the JuliaSmoothOptimizers' (JSO) tools [@jso]. JSO is an academic organization containing a collection of Julia packages for nonlinear optimization software development, testing, and benchmarking. It provides tools for building models, accessing repositories of problems, and solving subproblems. `DCISolver.jl` takes as input an `AbstractNLPModel`, JSO's general model API defined in `NLPModels.jl` [@orban-siqueira-nlpmodels-2020], that represents flexible data types to evaluate objective and constraint functions, their derivatives, and to provide any information that a solver might request from a model. The user can code derivatives themselves or with automatic differentiation, or use JSO-converters for classical mathematical optimization modeling languages such as Ampl [@fourer2003ampl] or JuMP [@jump]. Moreover, the API handles sparse matrices and operators for matrix-free implementations. We exploit here Julia's multiple dispatch facilities to specialize instances to different contexts efficiently.

Internally, `DCISolver.jl` combines cutting-edge numerical linear algebra solvers. The normal step relies heavily on iterative methods for linear algebra from `Krylov.jl` [@montoison-orban-krylov-2020], which provides more than 25 implementations of standard and novel Krylov methods, and they all can be used with Nvidia GPU via CUDA.jl [@besard2018juliagpu]. 
The tangent step is computed using the LDL factorization of a SQD matrix via `LDLFactorizations.jl` [@orban-ldlfactorizations-2020], or the well-known Fortran code `MA57` [@duff-2004] from the HSL [@HSL], via `HSL.jl` [@orban-hsl-2021].

`DCISolver.jl` returns a structure containing the available information at the end of the execution, including a solver status, the objective function value, the norm of the constraint function, the elapsed time, and a dictionary of solver specifics. All in all, with a few lines of codes, one can solve large-scale problems or benchmark `DCISolver.jl` against other JSO-compliant solvers using `SolverBenchmark.jl` [@orban-siqueira-solverbenchmark-2020].
We refer to the website \href{https://juliasmoothoptimizers.github.io/}{juliasmoothoptimizers.github.io} for tutorials.

# Statement of need

Julia has been designed to efficiently implement software and algorithms fundamental to the field of operations research, particularly in mathematical optimization [@lubin2015computing], and has become a natural choice for solvers such as `DCISolver.jl`.
Low-level languages, such as C++ and Fortran, have a rather steep learning curve and long write-compile-link-debug cycles.
Julia has a high-level syntax inspired by other well-known languages, such as Matlab and Python, and it uses just-in-time compilation to achieve high performance.
One of Julia's aspects is the ability to access C, Fortran, or Python code without sacrificing speed natively, which helps tackle the two language problems -- prototype on high-level, reimplement in low-level.
In Julia, one can create a prototype just as quickly as other high-level languages, but the resulting prototype is considerably more efficient [@bezanson2017julia], which is of great importance for methods like DCI that are still under research development.
Furthermore, the prototype can be improved instead of moving the code to a low-level language until a competitive version is obtained.
Additionally, solvers coded in pure Julia do not require external compiled dependencies and work with multiple input data types, while solvers in Fortran are limited to simple and double precisions.

There exist solutions to compute local minima of \eqref{eq:nlp} in Julia.
A classical approach is to model the problem using `JuMP` [@jump], from the JuMP-dev organization, and then pass the model to state-of-the-art solvers via `MathOptInterface.jl` [@legat2021mathoptinterface], a thin wrapper to solvers that are typically available in C or Fortran.
Another organization for nonlinear optimization is JuliaNLSolvers with `Optim.jl` [@mogensen2018optim], a package for univariate and multivariate optimization of functions. For instance, it implements in Julia an interior-point primal-dual Newton algorithm for solving \eqref{eq:nlp}.
Finally, JSO also offers alternative solutions such as thin wrappers to the solvers Artelys Knitro [@byrd2006k] via `NLPModelsKnitro.jl` [@orban-siqueira-nlpmodelsknitro-2020] and Ipopt [@wachter2006implementation] via `NLPModelsIpopt.jl` [@orban-siqueira-nlpmodelsipopt-2020],
but also `Percival.jl` [@percival-jl], a pure Julia implementation of an augmented Lagrangian penalization method.
The main advantage of using JSO-compliant solvers is the flexibility in the origin of the inputted problem.
To the best of our knowledge, there is no currently maintained available open-source implementation of DCI. Hence, we offer an interesting alternative to augmented Lagrangian or interior-point methods that are regularly used in the references mentioned above. 

`DCISolver.jl` is designed to help application experts quickly solve real-world problems and help researchers improve, compare and analyze new techniques without writing algorithms themselves.
The user benefits from JuliaSmoothOptimizers's framework to solve nonlinear optimization problems of diverse nature in an accessible fashion, which makes it very suitable for numerical optimization courses.
Last but not least, the documentation of this package includes benchmarks on classical test sets, like [@cutest], showing that this implementation is also very competitive.

# Acknowledgements

Tangi Migot is supported by IVADO and the Canada First Research Excellence Fund / Apogée,
and Dominique Orban is partially supported by an NSERC Discovery Grant.

# References
