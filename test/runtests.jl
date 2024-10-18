# stdlib
using LinearAlgebra, Logging, Test
# JSO
using ADNLPModels, Krylov, NLPModels, SolverCore, SolverTest
# This package
using DCISolver

#using SymCOOSolverInterface #tests
include("symcoo_runtests.jl")

if v"1.8.0" <= VERSION
  include("allocs.jl")
end

@testset "Test callback" begin
  f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
  nlp = ADNLPModel(f, [-1.2; 1.0])
  X = [nlp.meta.x0[1]]
  Y = [nlp.meta.x0[2]]
  function cb(nlp, solver, stats)
    x = solver.x
    push!(X, x[1])
    push!(Y, x[2])
    if stats.iter == 8
      stats.status = :user
    end
  end
  stats = with_logger(NullLogger()) do
    dci(nlp, callback = cb)
  end
  @test stats.iter == 8
end

@testset "Re-solve with a different initial guess" begin
  nlp = ADNLPModel(
    x -> (x[1] - 1)^2,
    [-1.2; 1.0],
    x -> [10 * (x[2] - x[1]^2)],
    zeros(1),
    zeros(1),
    name = "HS6",
  )
  stats = GenericExecutionStats(nlp)

  meta = DCISolver.MetaDCI(nlp, atol = 1e-7, rtol = 1e-7, verbose = 0)
  solver = DCISolver.DCIWorkspace(nlp, meta)
  stats = solve!(solver, nlp, stats)
  @test isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)
  @test stats.status == :first_order

  nlp.meta.x0 .= 10.0
  reset!(solver)

  stats = solve!(solver, nlp, stats)
  @test isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)
  @test stats.status == :first_order
end

@testset "Re-solve with a different problem" begin
  nlp = ADNLPModel(
    x -> (x[1] - 1)^2,
    [-1.2; 1.0],
    x -> [10 * (x[2] - x[1]^2)],
    zeros(1),
    zeros(1),
    name = "HS6",
  )
  stats = GenericExecutionStats(nlp)

  x = nlp.meta.x0
  meta = DCISolver.MetaDCI(nlp, x, atol = 1e-7, rtol = 1e-7, verbose = 0)
  solver = DCISolver.DCIWorkspace(nlp, meta, x)
  stats = solve!(solver, nlp, stats)
  @test isapprox(stats.solution, [1.0; 1.0], rtol = 1e-6)
  @test stats.status == :first_order

  nlp = ADNLPModel(
    x -> x[1]^2,
    [-1.2; 1.0],
    x -> [10 * (x[2] - x[1]^2)],
    zeros(1),
    zeros(1),
    name = "shifted HS6",
  )
  reset!(solver, nlp)

  stats = solve!(solver, nlp, stats)
  @test isapprox(stats.solution, [0.0; 0.0], atol = 1e-6)
  @test stats.status == :first_order
end

@testset "Unbounded tests" begin
  nlp = ADNLPModel(x -> sum(x), zeros(2))
  stats = dci(nlp)
  @test stats.status == :unbounded
end

@testset "Unconstrained tests" begin
  unconstrained_nlp(nlp -> dci(nlp, atol = 1e-6, rtol = 1e-6))
end

#The first four were used in Percival.jl
@testset "Small equality constrained problems" begin
  n = 10
  test_set = [
    ADNLPModel(
      x -> 2x[1]^2 + x[1] * x[2] + x[2]^2 - 9x[1] - 9x[2],
      [1.0; 2.0],
      x -> [4x[1] + 6x[2]],
      10 * ones(1),
      10 * ones(1),
      name = "Simple linear-quadratique problem",
    ),
    ADNLPModel(
      x -> (x[1] - 1)^2,
      [-1.2; 1.0],
      x -> [10 * (x[2] - x[1]^2)],
      zeros(1),
      zeros(1),
      name = "HS6",
    ),
    ADNLPModel(
      x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
      [-1.2; 1.0],
      x -> [(x[1] - 2)^2 + (x[2] - 2)^2 - 2],
      zeros(1),
      zeros(1),
      name = "Rosenbrock with (x₁-2)²+(x₂-2)²=2",
    ),
    ADNLPModel(
      x -> -x[1],
      [0.5; 1 / 3],
      x -> [
        (4x[1])^2 + (3x[2])^2 - 25
        4x[1] * 3x[2] - 12
      ],
      zeros(2),
      zeros(2),
      name = "scaled HS8",
    ),
    ADNLPModel(
      x -> dot(x, x),
      zeros(n),
      x -> [sum(x) - n],
      zeros(1),
      zeros(1),
      name = "||x||² s.t. ∑x = n",
    ),
    ADNLPModel(
      x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2,
      [-1.2; 1.0],
      x -> [sum(x) - 2],
      [0.0],
      [0.0],
      name = "Rosenbrock with ∑x = 2",
    ),
    ADNLPModel(
      x -> (1 - x[1])^2,
      [-1.2; 1.0],
      x -> [10 * (x[2] - x[1]^2)],
      [0.0],
      [0.0],
      name = "HS6",
    ),
  ]
  for nlp in test_set
    stats = with_logger(NullLogger()) do
      dci(nlp, atol = 1e-6, rtol = 0.0, ctol = 1e-6)
    end
    sol = ones(nlp.meta.nvar)
    @test isapprox(stats.solution, sol, rtol = 1e-6)
    @test stats.primal_feas < 1e-6
    @test stats.dual_feas < 1e-6
    @test stats.status == :first_order
  end
end

mutable struct DummyModel{T,S} <: AbstractNLPModel{T,S}
  meta::NLPModelMeta{T,S}
end

nlp = DummyModel(NLPModelMeta(1, minimize = false))
@test_throws ErrorException("DCI only works for minimization problem") dci(nlp, zeros(1))

#Test if it has equality constraints
nlp = ADNLPModel(x -> dot(x, x), zeros(5), zeros(5), ones(5))
@test_throws ErrorException("DCI only works for equality constrained problems") dci(
  nlp,
  zeros(5),
)

@testset "Small equality constrained problems II" begin
  tol = 1e-6

  @testset "HS7" begin
    nlp = ADNLPModel(
      x -> log(1 + x[1]^2) - x[2],
      [2.0; 2.0],
      x -> [(1 + x[1]^2)^2 + x[2]^2 - 4],
      [0.0],
      [0.0],
    )

    stats = with_logger(NullLogger()) do
      dci(nlp, atol = 1e-6, rtol = 0.0, ctol = 1e-6)
    end
    dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
    @test norm(stats.solution - [0, sqrt(3)]) < tol
    @test dual < tol
    @test primal < tol
    @test status == :first_order
  end

  @testset "HS8" begin
    nlp = ADNLPModel(
      x -> -1.0,
      [2.0; 1.0],
      x -> [x[1]^2 + x[2]^2 - 25; x[1] * x[2] - 9],
      zeros(2),
      zeros(2),
    )

    stats = with_logger(NullLogger()) do
      dci(nlp, atol = 1e-6, rtol = 0.0, ctol = 1e-6)
    end
    dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
    @show stats.solution
    #@test norm(stats.solution - ones(nlp.meta.nvar)) < tol
    @test dual < tol
    @test primal < tol
    @test status == :first_order
  end

  @testset "HS9" begin
    nlp = ADNLPModel(
      x -> sin(π * x[1] / 12) * cos(π * x[2] / 16),
      zeros(2),
      x -> [4 * x[1] - 3 * x[2]],
      [0.0],
      [0.0],
    )

    stats = with_logger(NullLogger()) do
      dci(nlp, atol = 1e-6, rtol = 0.0, ctol = 1e-6)
    end
    dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
    @test dual < tol
    @test primal < tol
    @test status == :first_order
  end

  @testset "HS26" begin
    nlp = ADNLPModel(
      x -> (x[1] - x[2])^2 + (x[2] - x[3])^4,
      [-2.6; 2.0; 2.0],
      x -> [(1 + x[2]^2) * x[1] + x[3]^4 - 3],
      [0.0],
      [0.0],
    )
    stats = with_logger(NullLogger()) do
      dci(nlp, atol = 1e-6, rtol = 0.0, ctol = 1e-6)
    end
    dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
    @show stats.solution
    #@test norm(stats.solution - ones(nlp.meta.nvar)) < tol
    @test dual < tol
    @test primal < tol
    @test status == :first_order
  end

  @testset "HS27" begin
    nlp = ADNLPModel(
      x -> 0.01 * (x[1] - 1)^2 + (x[2] - x[1]^2)^2,
      [2.0; 2.0; 2.0],
      x -> [x[1] + x[3]^2 + 1.0],
      [0.0],
      [0.0],
    )
    stats = with_logger(NullLogger()) do
      dci(nlp, max_eval = 10_000, atol = 1e-6, rtol = 0.0, ctol = 1e-6)
    end
    dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
    @show stats.solution
    #@test norm(stats.solution - []) < tol
    @test dual < tol
    @test primal < tol
    @test status == :first_order
  end
end

include("test-normal-step.jl")
