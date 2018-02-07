using Base.Test
using NLPModels, CUTEst

include("dci.jl")

function test_dci()
  @testset "Simple problem" begin
    n = 10
    nlp = ADNLPModel(x->dot(x, x), zeros(n),
                     c=x->[sum(x) - 1], lcon=zeros(1), ucon=zeros(1))

    x, fx, dual, primal, eltime, solved, tired = dci(nlp)
    @test norm(n * x - ones(n)) < 1e-6
    @test dual < 1e-6
    @test primal < 1e-6
    @test solved
    @test !tired
  end

  @testset "Rosenbrock with ∑x = 1" begin
    nlp = ADNLPModel(x->(x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2,
                     [-1.2; 1.0], c=x->[sum(x)-1], lcon=zeros(1),
                     ucon=zeros(1))

    x, fx, dual, primal, eltime, solved, tired = dci(nlp)
    @test dual < 1e-6
    @test primal < 1e-6
    @test solved
    @test !tired
  end

  @testset "HS6" begin
    nlp = ADNLPModel(x->(1 - x[1])^2, [-1.2; 1.0],
                     c=x->[10 * (x[2] - x[1]^2)],
                     lcon=zeros(1), ucon=zeros(1))

    x, fx, dual, primal, eltime, solved, tired = dci(nlp)
    @test dual < 1e-6
    @test primal < 1e-6
    @test solved
    @test !tired
  end

  @testset "HS7" begin
    nlp = ADNLPModel(x->log(1 + x[1]^2) - x[2], [2.0; 2.0],
                     c=x->[(1 + x[1]^2)^2 + x[2]^2 - 4],
                     lcon=zeros(1), ucon=zeros(1))

    x, fx, dual, primal, eltime, solved, tired = dci(nlp, verbose=true)
    @test dual < 1e-6
    @test primal < 1e-6
    @test solved
    @test !tired
  end
end

test_dci()
