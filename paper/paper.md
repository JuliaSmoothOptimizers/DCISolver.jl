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
 - name: GERAD, Department of Mathematics and Industrial Engineering, École Polytechnique, Montréal, QC, Canada.
   index: 1
 - name: Netherlands eScience Center, Amsterdam, NL
   index: 2
date: 05 November 2021
bibliography: paper.bib

---

# Summary

DCISolver.jl is a new Julia implementation of the Dynamic Control of Infeasibility (DCI), introduced in [@bielschowsky2008dynamic], for solving nonlinear optimization models
with equality constraints:
\begin{equation}\label{eq:nlp}
    \min_{x \in \mathbb{R}^n} f(x) \quad \text{subject to} \quad h(x) = 0,
\end{equation}
where  $f:\mathbb{R}^n \rightarrow \mathbb{R}$ and  $h:\mathbb{R}^n \rightarrow \mathbb{R}^m$ are twice continuously differentiable.
As often in nonlinear continuous optimization, DCI is an iterative method that aims at computing a local minimum, or at least a stationary point, of \autoref{eq:nlp} using the information on first and second-order derivatives.

This iterative method uses the idea of trust cylinders to keep infeasibility under control, contrary to penalization methods that only encourage feasibility.
Each time the trust cylinder is violated, a restoration step is called, and the infeasibility level is reduced. 
The radius of the trust cylinder has a nonincreasing update scheme, so a feasible and optimal point is obtained.

The DCI algorithm is a two-step process that minimizes the problem with a relaxed feasibility constraint and then recenters toward a tighter trust cylinder. One of the significant advantages of this implementation is that the restoration step is matrix-free, i.e., it uses second-order information but does not need access directly to the Hessian matrix as it only requires Hessian-vector products. This makes DCISolver a great asset for large-scale problems, where the equality constraint is hard to handle.

DCISolver relies on JuliaSmoothOptimizers' (JSO) tools. JSO is an academic organization containing a collection of Julia packages for nonlinear optimization software development, testing, and benchmarking. It provides solvers and tools for building models, accessing repositories of problems, solving subproblems, and linear algebra. DCISolver takes as an input an `AbstractNLPModel`, JSO's general consistent API defined in `NLPModels.jl` [@orban-siqueira-nlpmodels-2020], which provides flexible data types to represent the objective and constraint functions, to evaluate their derivatives, and to provide essentially any information that a solver might request from a model. The user can provide derivatives themselves, request that they are calculated using automatic differentiation, or use JSO-converters from classical mathematical optimization modeling languages such as Ampl or JuMP. Moreover, the API handles sparse Hessian/Jacobian matrices or operators for matrix-free implementations. We exploit Julia's multiple dispatch facilities to specialize instances to different contexts efficiently.

DCISolver combines cutting-edge numerical linear algebra solvers. The restoration step heavily relies on iterative methods for linear algebra from `Krylov.jl` [@montoison-orban-krylov-2020].
This package provides more than 25 implementations of standard and novels Krylov methods, and they all can be used with Nvidia GPU via CUDA.jl [@besard2018juliagpu]. The other step in DCI algorithm is computed using a regularized LDL factorization that can be handled either with `LDLFactorizations.jl` [@orban-ldlfactorizations-2020], or the well-known Fortran code `MA57` [@duff-2004] from HSL [@HSL], via `HSL.jl` [@orban-hsl-2021].

Finally, DCISolver returns a structure containing the available information at the end of the execution, including a solver status, the objective function value, the norm of the constraint function, the elapsed time, and a dictionary of solver specifics. All in all, with a few lines of codes, one can solve large-scale problems or benchmark DCISolver against other JSO-compliant solvers using `SolverBenchmark.jl` [@orban-siqueira-solverbenchmark-2020].
We refer to the website \href{https://juliasmoothoptimizers.github.io/}{juliasmoothoptimizers.github.io} for tutorials.

# Statement of need

C++ and Fortran are often languages of choice for large-scale optimization solvers, e.g., COIN-OR [@lougee2003common], GALAHAD [@gould2003galahad], NLopt [@johnson2014nlopt], Opt++ [@optcpp], PETcs-TAO [@petsc-user-ref] to cite some of the main organizations. However, such low-level languages have a rather steep learning curve and long write-compile-link-debug cycles.
Hence, practitioners and researchers have often turned to higher-level languages such as Python and Matlab.
Python has an extensive standard library including for optimization, see, CVXOPT [@cvxopt], GEKKO Optimization Suite [@beal2018gekko], NLP.py [@nlppy], Opal [@opal], PulP [@pulp], PyGMO [@izzo2012pygmo], Pyomo [@pyomo], Pyopt [@pyopt], SciPy [@virtanen2020scipy].
Julia has a high-level syntax inspired by other well-known languages, such as Matlab and Python, and it uses just-in-time compilation to achieve high performance.
One of Julia's aspects is the ability to access C, Fortran, or Python code without sacrificing speed natively, which helps tackle the two language problems -- prototype on high-level, reimplement in low-level.
In Julia, one can create a prototype just as quickly as other high-level languages, but the resulting prototype is considerably more efficient [@bezanson2017julia].
Furthermore, the prototype can be improved instead of moving the code to a low-level language until a competitive version is obtained.
Julia has been designed to efficiently implement software and algorithms fundamental to the field of operations research, particularly in mathematical optimization [@lubin2015computing]. Indeed, [@lubin2015computing] showed cross-language benchmarks suggesting that Julia is capable of obtaining state-of-the-art performance.

To solve \autoref{eq:nlp} in Julia, one can also consider using MathOptInterface.jl [@legat2021mathoptinterface] that solves JuMP models [@jump] using state-of-the-art solvers, or Optim.jl [@mogensen2018optim]. JSO also provides thin wrapper to existing solvers such as Artelys Knitro [@byrd2006k] via `NLPModelsKnitro.jl` [@orban-siqueira-nlpmodelsknitro-2020] or Ipopt [@wachter2006implementation] via `NLPModelsIpopt.jl` [@orban-siqueira-nlpmodelsipopt-2020], and pure Julia implementations such as Percival [@percival-jl].
To the best of our knowledge, there is no available open-source implementation of the DCI algorithm, which offers a very interesting alternative to interior-point methods or augmented Lagrangian methods that are extremely popular in the references mentioned above. Moreover, tools coded in pure Julia do not require external compiled dependencies and work with multiple input data types.

DCISolver is designed to help application experts quickly solve real-world problems and help researchers improve, compare and analyze new techniques to handle constraints without writing algorithms themselves.
The user benefits from JuliaSmoothOptimizers's framework to solve nonlinear optimization problems of diverse nature in an accessible fashion, which makes it very suitable for numerical optimization courses.

# Acknowledgements

Tangi Migot is supported by IVADO and the Canada First Research Excellence Fund / Apogée,
and Dominique Orban is partially supported by an NSERC Discovery Grant.

# References
