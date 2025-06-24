#module SymCOOSolverInterface

import LinearAlgebra: isposdef
import Base: success

#export factorize!, symcoo_solve!, success, isposdef, num_neg_eig

"""
An `SymCOOSolver` is an interface to allow simple usage of different solvers.
Ideally, having `rows`, `cols`, `vals` and the dimension `ndim` of a symmetric matrix should allow the user to call
    M = LinearSolver(ndim, rows, cols, vals)
    factorize!(M)
    symcoo_solve!(x, M, b) # Also x = M \\ b
Only the lower triangle of the matrix should be passed.
"""
abstract type SymCOOSolver end

"""
    factorize!(M :: SymCOOSolver)
Calls the factorization of the symmetric solver given by `M`.
Use `success(M)` to check whether the factorization was successful.
"""
function factorize! end

"""
    symcoo_solve!(x, M :: SymCOOSolver, b)
Solve the system ``M x = b``. `factorize!(M)` should be called first.
"""
function symcoo_solve! end

"""
    success(M :: SymCOOSolver)
Returns whether `factorize!(M)` was successful.
"""
function success end

"""
    isposdef(M :: SymCOOSolver)
Returns whether `M` is positive definite.
"""

"""
    num_neg_eig(M :: SymCOOSolver)
Returns the number of negative eigenvalues of `M`.
"""
function num_neg_eig end

include("hsl.jl")
include("ldlfactorizations.jl")

#end # module
