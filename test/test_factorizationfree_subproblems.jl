#=
We test here the computation of a direction d such that
````
 \minimize_d q(d):=∇f(z)'d + 1/2 d'B(z)d \st ||d|| ≤ Δ, ||∇h(z)d|| ≤ Δ²,
 where B represents the Lagrangian Hessian.

 #Solution 1: we consider infinity norm and solve a linear program:
=#
using Krylov, LinearAlgebra, NLPModels, NLPModelsIpopt, NLPModelsKnitro, QuadraticModels

#@testset "Example 1" begin
    nlp = ADNLPModel(
        x->0.01 * (x[1] - 1)^2 + (x[2] - x[1]^2)^2,
        [2.0; 2.0; 2.0],
        #x->[x[1]^2 + x[3]^2 + 1.0],
        x->[x[1]^2 - x[3]^2 - 1.0],
        zeros(1),
        zeros(1)
    )

    x  = [0.; 1.; 0.]
    λ  = rand(1)
    B  = hess(nlp, x, λ)
    cx = cons(nlp, x)
    Jx = jac(nlp, x)

    Δ = .5

    #=
    Hrows, Hcols = hess_structure(nlp)
    Hvals = hess_coord(nlp, x)
    lcon = -Δ^2 * ones(nlp.meta.nvar)
    ucon = -lcon
    lvar = -Δ * ones(nlp.meta.nvar)
    uvar = -lvar
    Arows, Acols = jac_structure(nlp)
    Avals = jac_coord(nlp, x)
    c    = grad(nlp, x)
    lqp = QuadraticModel(c, Hrows, Hcols, Hvals, 
                            Arows=Arows, Acols=Acols, Avals=Avals, 
                            lcon=lcon, ucon=ucon,
                            lvar=lvar, uvar=uvar,
                            name="feasibility_step subproblem")
    @time stats = knitro(lqp)
    #@time stats2 = ipopt(lqp)
    =#

    #Is there a problem here?
    c = grad(nlp, x)
    (μ, stats) = lsmr(jac_op(nlp, x)', -c, M=B, verbose = true)
    @test μ == [0.]
    r = - c - jac_op(nlp, x)' * μ
    nres = sqrt(r' * B *r)
    @test [nres] == stats.residuals
    @test [B Jx'; Jx  zeros(1,1)] * vcat(r, μ) - vcat(-c, zeros(1)) != zeros(4)
    @test [inv(B) Jx'; Jx  zeros(1,1)] * vcat(r, μ) - vcat(-c, zeros(1)) != zeros(4)
    @test [I Jx'; Jx  zeros(1,1)] * vcat(r, μ) - vcat(-c, zeros(1)) == zeros(4)

#end