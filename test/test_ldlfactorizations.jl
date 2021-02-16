using Pkg
Pkg.activate("envs/ldlfact")
using LDLFactorizations #v0.7.0, February 14th
using Test, LinearAlgebra, SparseArrays

function _test_factorization(A :: Union{Symmetric{T,SparseMatrixCSC{T,Ti}}, Array{T,2}},
                             S   :: LDLFactorizations.LDLFactorization{T, I, I, I}; 
                             tol :: AbstractFloat = 1e-15) where {I,T,Ti}

  __P = zeros(10,10); for i=1:10 __P[S.P[i],i] = 1. end
  In = spdiagm(0 => ones(10))
  nrm = norm(__P * (S.L + In) * S.D * (S.L + In)' * __P' - A, Inf)
  test = nrm ≤ tol

 return test
end

  A = [ 1.7     0     0     0     0     0     0     0   .13     0
          0    1.     0     0   .02     0     0     0     0   .01
          0     0   1.5     0     0     0     0     0     0     0
          0     0     0   1.1     0     0     0     0     0     0
          0   .02     0     0   2.6     0   .16   .09   .52   .53
          0     0     0     0     0   1.2     0     0     0     0
          0     0     0     0   .16     0   1.3     0     0   .56
          0     0     0     0   .09     0     0   1.6   .11     0
        .13     0     0     0   .52     0     0   .11   1.4     0
          0   .01     0     0   .53     0   .56     0     0   3.1 ]
  S = ldl(Symmetric(triu(A),:U))
  @test _test_factorization(A, S; tol = 1e-15)

  Aup = Symmetric(triu(sparse(A)), :U)
  S2 = ldl_analyze(Aup)
  ldl_factorize!(Aup, S2)
  @test _test_factorization(Aup, S2; tol = 1e-15)

  A = [0.   0.   0.   0.   0.   0.   0.   0.   4.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   5.   0.
       2.   4.   5.   -2   4.   1.   2.   2.   2.   0.
       0.   0.   0.   0.   1.   9.   9.   1.   7.   1.
       0.   0.   0.   0.   0.   0.   0.   0.   1.   0.
       1.   3.   2.   1.   4.   3.   1.   0.   0.   7.
       -3.  8.   0.   0.   0.   0.   -2.  0.   0.   1.
       0.   0.   0.   5.   7.   9.   0.   2.   7.   1.
       3.   2.   0.   0.   0.   0.   1.   3.   3.   2.
       0.   0.   0.   0.  -3   -4    0.   0.   0.   0. ]
  ϵ = sqrt(eps(eltype(A)))
  M = A * A'  # det(A) = 0 => M positive semidefinite
  b = M * ones(10)
  x = copy(b)
  S = ldl_analyze(Symmetric(triu(M), :U))
  S.r1 = -ϵ
  S.r2 = ϵ
  S.tol = ϵ
  S.n_d = 0
  S = ldl_factorize!(Symmetric(triu(M), :U), S)
  x = ldiv!(S, x)
  r = M * x - b
  @test norm(r) ≤ sqrt(eps()) * norm(b)
  @test _test_factorization(Aup, S2; tol = ϵ)

  #=
  Tangi: February, 15th.
  Let's now say we got the lower triangular matrix...
  =#
  M = tril(sparse(A * A'))
  (rows, cols, vals) = findnz(M)
  M2 = sparse(cols, rows, vals, 10, 10)
  Symmetric(M, :L) == Symmetric(M2,:U)
  S = ldl_analyze(Symmetric(M2,:U))
    S.r1 = -ϵ
    S.r2 = ϵ
    S.tol = ϵ
    S.n_d = 0
    S = ldl_factorize!(Symmetric(M2,:U), S)
    @test _test_factorization(Aup, S2; tol = ϵ)
