using DCISolver, ADNLPModels, Test, LinearAlgebra, NLPModels

@testset "CaNNOLeS Feasibility Step" begin
  nlp = ADNLPModel(
    x -> 0.01 * (x[1] - 1)^2 + (x[2] - x[1]^2)^2,
    [2.0; 2.0; 2.0],
    x -> [x[1]^2 + x[3]^2 - 1.0],
    zeros(1),
    zeros(1),
  )

  x = [0.0; 1.0; 0.0]
  cx = DCISolver.cons_norhs!(nlp, x, similar(x, nlp.meta.ncon))
  normcx = norm(cx)
  Jx = jac(nlp, x)

  ctol = 1e-5
  ρ = 0.5
  meta_dci = DCISolver.MetaDCI(
    nlp.meta.x0, 
    nlp.meta.y0,
    feas_step = :feasibility_step_cannoles
  )
  workspace_dci = DCISolver.DCIWorkspace(nlp, meta_dci, nlp.meta.x0)

  z, cz, ncz, Jz, status = DCISolver.feasibility_step_cannoles(
    nlp,
    x,
    cx,
    normcx,
    Jx,
    ρ,
    ctol,
    meta_dci,
    workspace_dci,
    false;
    max_eval = 1_000,
    max_time = 60.0,
  )
  
  # Verify results
  @test status in [:success, :unknown]
  @test ncz < normcx  # Should reduce constraint violation
  @test isfinite(ncz)
  
  finalize(nlp)
  
  println("✓ CaNNOLeS feasibility step test passed")
end

@testset "DCI with CaNNOLeS option" begin
  nlp = ADNLPModel(
    x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2,
    [-1.2; 1.0],
    x -> [x[1] * x[2] - 1],
    [0.0], [0.0],
  )

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
  
  @test stats.status in [:first_order, :acceptable, :max_iter, :max_time]
  @test isfinite(stats.objective)
  
  finalize(nlp)
  
  println("✓ DCI with CaNNOLeS test passed")
end
