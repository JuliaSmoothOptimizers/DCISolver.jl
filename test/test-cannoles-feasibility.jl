using CaNNOLeS
using DCISolver

function make_test_nlp()
  ADNLPModel(
    x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2,
    [-1.2; 1.0],
    x -> [x[1] * x[2] - 1],
    [0.0],
    [0.0],
  )
end

@testset "DCI with CaNNOLeS option" begin
  nlp = make_test_nlp()

  stats = dci(
    nlp,
    nlp.meta.x0,
    feas_step = :feasibility_step_cannoles,
    atol = 1e-5,
    ctol = 1e-5,
    rtol = 1e-5,
    max_time = 60.0,
    max_iter = 100,
  )

  @test stats.status in [:first_order, :acceptable, :max_iter]
  @test isfinite(stats.objective)

  x_sol = stats.solution
  c_sol = cons(nlp, x_sol)
  @test norm(c_sol) <= 1e-5
  @test isfinite(norm(x_sol))
end

@testset "DCI with CaNNOLeS vs trust-region comparison" begin
  nlp_cannoles = make_test_nlp()

  stats_cannoles = dci(
    nlp_cannoles,
    nlp_cannoles.meta.x0,
    feas_step = :feasibility_step_cannoles,
    atol = 1e-5,
    ctol = 1e-5,
    rtol = 1e-5,
    max_time = 60.0,
    max_iter = 100,
  )

  nlp_default = make_test_nlp()

  stats_default = dci(
    nlp_default,
    nlp_default.meta.x0,
    feas_step = :feasibility_step,
    atol = 1e-5,
    ctol = 1e-5,
    rtol = 1e-5,
    max_time = 60.0,
    max_iter = 100,
  )

  @test stats_cannoles.status in [:first_order, :acceptable, :max_iter]
  @test stats_default.status in [:first_order, :acceptable]

  @test norm(cons(nlp_default, stats_default.solution)) <= 1e-5
  @test norm(cons(nlp_cannoles, stats_cannoles.solution)) <= 1e-5
  @test isfinite(stats_cannoles.objective)
  @test isfinite(stats_default.objective)
end
