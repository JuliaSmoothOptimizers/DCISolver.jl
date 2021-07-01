# stdlib
using LinearAlgebra, Logging, Test
# JSO
using ADNLPModels, Krylov, NLPModels, SolverTest
# This package
using DCISolver

#using SymCOOSolverInterface #tests
include("symcoo_runtests.jl")

@testset "Unbounded tests" begin
  nlp = ADNLPModel(x -> sum(x), zeros(2))
  stats = dci(nlp, nlp.meta.x0)
  @test stats.status == :unbounded
end

@testset "Unconstrained tests" begin
  unconstrained_nlp(nlp -> dci(nlp, nlp.meta.x0, atol = 1e-6, rtol = 1e-6))
end

#The first four were used in Percival.jl
@testset "Small equality constrained problems" begin
  n = 10
  test_set = [
    ADNLPModel(
      x -> 2x[1]^2 + x[1] * x[2] + x[2]^2 - 9x[1] - 9x[2],
      [1.0; 2.0],
      x -> [4x[1] + 6x[2] - 10],
      zeros(1),
      zeros(1),
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
      dci(nlp, nlp.meta.x0, atol = 1e-6, rtol = 0.0, ctol = 1e-6)
    end
    sol = ones(nlp.meta.nvar)
    @test isapprox(stats.solution, sol, rtol = 1e-6)
    @test stats.primal_feas < 1e-6
    @test stats.dual_feas < 1e-6
    @test stats.status == :first_order
  end
end
##################################################################################
mutable struct DummyModel{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
end

function test_dci(; tol = 1e-6)

  #Test if it has equality constraints
  nlp = ADNLPModel(x -> dot(x, x), zeros(5), zeros(5), ones(5))
  @test_throws ErrorException("DCI only works for equality constrained problems") dci(nlp, zeros(5))

  nlp = DummyModel(NLPModelMeta(1, minimize = false))
  @test_throws ErrorException("DCI only works for minimization problem") dci(nlp, zeros(1))

  @testset "HS7" begin
    nlp = ADNLPModel(
      x -> log(1 + x[1]^2) - x[2],
      [2.0; 2.0],
      x -> [(1 + x[1]^2)^2 + x[2]^2 - 4],
      [0.0],
      [0.0],
    )

    stats = with_logger(NullLogger()) do
      dci(nlp, nlp.meta.x0, atol = 1e-6, rtol = 0.0, ctol = 1e-6)
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
      dci(nlp, nlp.meta.x0, atol = 1e-6, rtol = 0.0, ctol = 1e-6)
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
      dci(nlp, nlp.meta.x0, atol = 1e-6, rtol = 0.0, ctol = 1e-6)
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
      dci(nlp, nlp.meta.x0, atol = 1e-6, rtol = 0.0, ctol = 1e-6)
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
      dci(nlp, nlp.meta.x0, max_eval = 10_000, atol = 1e-6, rtol = 0.0, ctol = 1e-6)
    end
    dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
    @show stats.solution
    #@test norm(stats.solution - []) < tol
    @test dual < tol
    @test primal < tol
    @test status == :first_order
  end
end

test_dci()

include("test-normal-step.jl")
