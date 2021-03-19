using HSL, LinearAlgebra, Random, SparseArrays, Test

using DCI: factorize!, solve!, success, isposdef, num_neg_eig

function tests()
  if isdefined(HSL, :libhsl_ma57)
    test_solver(MA57Struct)
  else
    @info("libhsl_ma57 not defined.")
  end
  test_solver(LDLFactorizationStruct)
end

function _test_factorization(A, S; tol = 1e-15)
  (n, n) = size(A)
  __P = zeros(n, n); for i=1:n __P[S.P[i],i] = 1. end
  In  = spdiagm(0 => ones(n))
  nrm = norm(__P * (S.L + In) * S.D * (S.L + In)' * __P' - A, Inf)
  test = nrm ≤ tol

 return test
end

function test_solver(LinearSolver)
  Random.seed!(0)
  nvar, ncon = 5, 3
  Q = spdiagm(nvar, nvar, 0 => 2 * ones(nvar), -1 => -ones(nvar-1), 1 => -ones(nvar-1))
  Qt = tril(Q)
  A = spdiagm(ncon, nvar, (i => ones(ncon) for i = (0:nvar-ncon))...)

  @testset "Positive definite systems" begin
    rows, cols, vals = findnz(Qt)
    M = LinearSolver(nvar, rows, cols, vals)
    sol = rand(nvar)
    rhs = Q * sol
    x = zeros(nvar)

    factorize!(M)
    solve!(x, M, rhs)
    @test norm(x - sol) ≤ 1e-8 * norm(sol)
    @test success(M)
    @test isposdef(M)
  end

  @testset "Indefinite systems" begin
    λ = sort(eigen(Matrix(Q)).values)
    α = (λ[1] + λ[2]) / 2
    rows, cols, vals = findnz(Qt - α * I)
    M = LinearSolver(nvar, rows, cols, vals)
    sol = rand(nvar)
    rhs = Q * sol - α * sol
    x = zeros(nvar)

    factorize!(M)
    solve!(x, M, rhs)
    @test norm(x - sol) ≤ 1e-8 * norm(sol)
    @test success(M)
    @test !isposdef(M)
    @test num_neg_eig(M) == 1
  end

  @testset "Failed factorization" begin
    rows = [1, 5, 3]
    cols = [1, 3, 5]
    vals = ones(3)
    M = LinearSolver(nvar, rows, cols, vals)

    factorize!(M)
    @test !success(M)
    @test !isposdef(M)
  end

  @testset "SQD system" begin
    rows, cols, vals = findnz([[Qt; A] zeros(nvar + ncon, ncon)])
    M = LinearSolver(nvar + ncon, rows, cols, vals)
    sol = rand(nvar + ncon)
    rhs = [Q A'; A zeros(ncon, ncon)] * sol
    x = zeros(nvar + ncon)

    factorize!(M)
    solve!(x, M, rhs)
    @test norm(x - sol) ≤ 1e-8 * norm(sol)
    @test success(M)
    @test !isposdef(M)
    @test num_neg_eig(M) == ncon
  end
end

tests()

@testset "LDLFactorizations with regularization" begin
  B = [0.   0.   0.   0.   0.   0.   0.   0.   4.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   5.   0.
       2.   4.   5.   -2   4.   1.   2.   2.   2.   0.
       0.   0.   0.   0.   1.   9.   9.   1.   7.   1.
       0.   0.   0.   0.   0.   0.   0.   0.   1.   0.
       1.   3.   2.   1.   4.   3.   1.   0.   0.   7.
       -3.  8.   0.   0.   0.   0.   -2.  0.   0.   1.
       0.   0.   0.   5.   7.   9.   0.   2.   7.   1.
       3.   2.   0.   0.   0.   0.   1.   3.   3.   2.
       0.   0.   0.   0.  -3   -4    0.   0.   0.   0. ]
  A = B * B'
  (rows, cols, vals) = findnz(tril(sparse(A)))
  ϵ = sqrt(eps(eltype(A)))
  M = LDLFactorizationStruct(10, rows, cols, vals)
  factorize!(M)
  @test !success(M)

  Me = LDLFactorizationStruct(10, rows, cols, vals, tol = ϵ, r2 = ϵ)
  factorize!(Me)
  @test success(Me)
  @test _test_factorization(A, Me.factor; tol = ϵ)
end