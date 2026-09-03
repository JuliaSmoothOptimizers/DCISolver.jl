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

@testset "feasibility_step_cannoles direct calls" begin
  nlp = make_test_nlp()
  x0 = nlp.meta.x0
  meta = DCISolver.MetaDCI(nlp, x0, nlp.meta.y0)
  workspace = DCISolver.DCIWorkspace(nlp, meta, x0)
  cx = cons(nlp, x0)
  normcx = norm(cx)
  Jx = jac_op(nlp, x0)
  ρ = 1e-3

  @testset "verbose success path" begin
    nlp_v = make_test_nlp()
    ws = DCISolver.DCIWorkspace(nlp_v, meta, nlp_v.meta.x0)
    cxv = cons(nlp_v, nlp_v.meta.x0)
    z, cz, normcz, Jz, status = DCISolver.feasibility_step_cannoles(
      nlp_v,
      nlp_v.meta.x0,
      cxv,
      norm(cxv),
      jac_op(nlp_v, nlp_v.meta.x0),
      1.0, # large ρ so it succeeds immediately
      1e-5,
      meta,
      ws,
      true, # verbose = true, exercises the log_row branch
    )
    @test status in [:success, :unknown]
    @test isfinite(normcz)
  end

  @testset "max_eval exhausted" begin
    z, cz, normcz, Jz, status = DCISolver.feasibility_step_cannoles(
      nlp,
      x0,
      cx,
      normcx,
      Jx,
      ρ,
      1e-5,
      meta,
      workspace,
      false;
      max_eval = 0,
    )
    @test status == :max_eval
    @test z === x0
  end

  @testset "max_time exhausted" begin
    z, cz, normcz, Jz, status = DCISolver.feasibility_step_cannoles(
      nlp,
      x0,
      cx,
      normcx,
      Jx,
      ρ,
      1e-5,
      meta,
      workspace,
      false;
      max_time = 0.0,
    )
    @test status == :max_time
    @test z === x0
  end

  @testset "internal cannoles max_eval status" begin
    nlp_e = make_test_nlp()
    ws_e = DCISolver.DCIWorkspace(nlp_e, meta, nlp_e.meta.x0)
    cx_e = cons(nlp_e, nlp_e.meta.x0)
    z, cz, normcz, Jz, status = DCISolver.feasibility_step_cannoles(
      nlp_e,
      nlp_e.meta.x0,
      cx_e,
      norm(cx_e),
      jac_op(nlp_e, nlp_e.meta.x0),
      1e-8,
      1e-8,
      meta,
      ws_e,
      false;
      cannoles_options = Dict{Symbol, Any}(:max_eval => 1),
    )
    @test status == :max_eval
  end

  @testset "internal cannoles max_iter status" begin
    nlp_i = make_test_nlp()
    ws_i = DCISolver.DCIWorkspace(nlp_i, meta, nlp_i.meta.x0)
    cx_i = cons(nlp_i, nlp_i.meta.x0)
    z, cz, normcz, Jz, status = DCISolver.feasibility_step_cannoles(
      nlp_i,
      nlp_i.meta.x0,
      cx_i,
      norm(cx_i),
      jac_op(nlp_i, nlp_i.meta.x0),
      1e-8,
      1e-8,
      meta,
      ws_i,
      false;
      cannoles_options = Dict{Symbol, Any}(:max_iter => 0),
    )
    @test status == :max_iter
  end

  @testset "explicit x0 in cannoles_options" begin
    z, cz, normcz, Jz, status = DCISolver.feasibility_step_cannoles(
      nlp,
      x0,
      cx,
      normcx,
      Jx,
      1.0,
      1e-5,
      meta,
      workspace,
      false;
      cannoles_options = Dict{Symbol, Any}(:x => x0),
    )
    @test status in [:success, :unknown]
  end
end
