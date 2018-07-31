using Base.Test
using NLPModels, CUTEst

include("dci.jl")

function test_dci()
  @testset "Simple problem" begin
    n = 10
    nlp = ADNLPModel(x->dot(x, x), zeros(n),
                     c=x->[sum(x) - 1], lcon=zeros(1), ucon=zeros(1))

    x, fx, dual, primal, eltime, status = dci(nlp)
    @test norm(n * x - ones(n)) < 1e-6
    @test dual < 1e-6
    @test primal < 1e-6
    @test status == :first_order
  end

  @testset "Rosenbrock with ∑x = 1" begin
    nlp = ADNLPModel(x->(x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2,
                     [-1.2; 1.0], c=x->[sum(x)-1], lcon=zeros(1),
                     ucon=zeros(1))

    x, fx, dual, primal, eltime, status = dci(nlp)
    @test dual < 1e-6
    @test primal < 1e-6
    @test status == :first_order
  end

  @testset "HS6" begin
    nlp = ADNLPModel(x->(1 - x[1])^2, [-1.2; 1.0],
                     c=x->[10 * (x[2] - x[1]^2)],
                     lcon=zeros(1), ucon=zeros(1))

    x, fx, dual, primal, eltime, status = dci(nlp)
    @test dual < 1e-6
    @test primal < 1e-6
    @test status == :first_order
  end

  @testset "HS7" begin
    nlp = ADNLPModel(x->log(1 + x[1]^2) - x[2], [2.0; 2.0],
                     c=x->[(1 + x[1]^2)^2 + x[2]^2 - 4],
                     lcon=zeros(1), ucon=zeros(1))

    x, fx, dual, primal, eltime, status = dci(nlp, verbose=false)
    @test dual < 1e-6
    @test primal < 1e-6
    @test status == :first_order
  end

  @testset "HS8" begin
    nlp = ADNLPModel(x->-1.0, [2.0; 1.0],
                     c=x->[x[1]^2 + x[2]^2 - 25; x[1] * x[2] - 9],
                     lcon=zeros(2), ucon=zeros(2))

    x, fx, dual, primal, eltime, status = dci(nlp, verbose=false)
    @test dual < 1e-6
    @test primal < 1e-6
    @test status == :first_order
  end

  @testset "HS9" begin
    nlp = ADNLPModel(x->sin(π * x[1] / 12) * cos(π * x[2] / 16), zeros(2),
                     c=x->[4 * x[1] - 3 * x[2]],
                     lcon=zeros(1), ucon=zeros(1))

    x, fx, dual, primal, eltime, status = dci(nlp, verbose=false)
    @test dual < 1e-6
    @test primal < 1e-6
    @test status == :first_order
  end

  @testset "HS26" begin
    nlp = ADNLPModel(x->(x[1] - x[2])^2 + (x[2] - x[3])^4, [-2.6; 2.0; 2.0],
                     c=x->[(1 + x[2]^2) * x[1] + x[3]^4 - 3],
                     lcon=zeros(1), ucon=zeros(1))
    x, fx, dual, primal, eltime, status = dci(nlp, verbose=false)
    @test dual < 1e-4
    @test primal < 1e-6
    @test status == :first_order
  end

  @testset "HS27" begin
    nlp = ADNLPModel(x->0.01 * (x[1] - 1)^2 + (x[2] - x[1]^2)^2, [2.0; 2.0; 2.0],
                     c=x->[x[1] + x[3]^2 + 1.0],
                     lcon=zeros(1), ucon=zeros(1))
    x, fx, dual, primal, eltime, status = dci(nlp, verbose=false)
    @test dual < 1e-4
    @test primal < 1e-6
    @test status == :first_order
  end
end

test_dci()
