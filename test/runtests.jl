# stdlib
using LinearAlgebra, Logging, Test

# JSO
using NLPModels

# This package
using Main.DCI

function test_dci(;tol = 1e-5)
  @testset "Simple problem" begin
    n = 10
    nlp = ADNLPModel(x->dot(x, x), zeros(n),
                     x->[sum(x) - 1], zeros(1), zeros(1))

    stats = with_logger(NullLogger()) do
      dci(nlp, rtol = 0.0)
    end
    x, dual, primal, status = stats.solution, stats.dual_feas, stats.primal_feas, stats.status
    @test norm(n * x - ones(n)) < tol
    @test dual < tol
    @test primal < tol
    @test status == :first_order
  end

  @testset "Rosenbrock with ∑x = 1" begin
    nlp = ADNLPModel(x->(x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0],
                     x->[sum(x)-1], [0.0], [0.0])

    stats = with_logger(NullLogger()) do
      dci(nlp, rtol = 0.0)
    end
    dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
    @test dual < tol#1e-6
    @test primal < tol
    @test status == :first_order
  end

  @testset "HS6" begin
    nlp = ADNLPModel(x->(1 - x[1])^2, [-1.2; 1.0],
                     x->[10 * (x[2] - x[1]^2)], [0.0], [0.0])

    stats = with_logger(NullLogger()) do
      dci(nlp, rtol = 0.0)
    end
    dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
    @test dual < tol
    @test primal < tol
    @test status == :first_order
  end

  @testset "HS7" begin
    nlp = ADNLPModel(x->log(1 + x[1]^2) - x[2], [2.0; 2.0],
                     x->[(1 + x[1]^2)^2 + x[2]^2 - 4], [0.0], [0.0])

    stats = with_logger(NullLogger()) do
      dci(nlp, rtol = 0.0)
    end
    dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
    @test dual < tol
    @test primal < tol
    @test status == :first_order
  end
#=
#tangent step never converges
  @testset "HS8" begin
    nlp = ADNLPModel(x->-1.0, [2.0; 1.0],
                     x->[x[1]^2 + x[2]^2 - 25; x[1] * x[2] - 9], zeros(2), zeros(2))

    stats = with_logger(NullLogger()) do
      dci(nlp, rtol = 0.0)
    end
    dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
    @test dual < tol
    @test primal < tol
    @test status == :first_order
  end
=#
  @testset "HS9" begin
    nlp = ADNLPModel(x->sin(π * x[1] / 12) * cos(π * x[2] / 16), zeros(2),
                     x->[4 * x[1] - 3 * x[2]], [0.0], [0.0])

    stats = with_logger(NullLogger()) do
      dci(nlp, rtol = 0.0)
    end
    dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
    @test dual < tol
    @test primal < tol
    @test status == :first_order
  end

  @testset "HS26" begin
    nlp = ADNLPModel(x->(x[1] - x[2])^2 + (x[2] - x[3])^4, [-2.6; 2.0; 2.0],
                     x->[(1 + x[2]^2) * x[1] + x[3]^4 - 3], [0.0], [0.0])
    stats = with_logger(NullLogger()) do
      dci(nlp, rtol = 0.0)
    end
    dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
    @test dual < tol
    @test primal < tol
    @test status == :first_order
  end

  @testset "HS27" begin
    nlp = ADNLPModel(x->0.01 * (x[1] - 1)^2 + (x[2] - x[1]^2)^2, [2.0; 2.0; 2.0],
                     x->[x[1] + x[3]^2 + 1.0], [0.0], [0.0])
    stats = with_logger(NullLogger()) do
      dci(nlp, max_eval=10_000, rtol = 0.0)
    end
    dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
    @test dual < tol
    @test primal < tol
    @test status == :first_order
  end
  
end

test_dci()

include("test-normal-step.jl")
