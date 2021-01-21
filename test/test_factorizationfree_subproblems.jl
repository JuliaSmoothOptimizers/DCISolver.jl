#=
We test here the computation of a direction d such that
````
 \minimize_d q(d):=∇f(z)'d + 1/2 d'B(z)d \st ||d|| ≤ Δ, ||∇h(z)d|| ≤ Δ²,
 where B represents the Lagrangian Hessian.

 #Solution 1: we consider infinity norm and solve a linear program:
 It is easy to find an interior point here as 0 works.
=#
using Krylov, LinearAlgebra, NLPModels, NLPModelsIpopt, NLPModelsKnitro, QuadraticModels, Test

#@testset "Example 1" begin
    nlp = ADNLPModel(
        x-> (x[1] - 1)^2 + (x[2] - 2)^2 + (x[3] - 3)^2,
        [2.0; 2.0; 2.0],
        #x->[x[1]^2 - x[3]^2 - 1.0],
        x->[x[1] - x[3] - 1.0],
        zeros(1),
        zeros(1)
    )

    x  = rand(3)
    λ  = rand(1)
    B  = hess(nlp, x, λ)# + Δ^2 * I 
    cx = cons(nlp, x)
    Jx = jac(nlp, x)

    Δ = .5

    #Is there a problem here?
    c = grad(nlp, x)
    (μ, stats) = lsmr(jac_op(nlp, x)', -c, M=B, verbose = true)
    @show stats

    r = - c - jac_op(nlp, x)' * μ
    nres = sqrt(r' * B *r)
    @test nres ≈ stats.residuals[end] atol = 1e-7

    @show norm([B Jx'; Jx  zeros(1,1)] * vcat(r, μ) - vcat(-c, zeros(1)))
    @show norm([inv(B) Jx'; Jx  zeros(1,1)] * vcat(r, μ) - vcat(-c, zeros(1)))
    @show norm([I Jx'; Jx  zeros(1,1)] * vcat(r, μ) - vcat(-c, zeros(1)))

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
    =#

#end