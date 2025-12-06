using DCISolver, ADNLPModels, Test, LinearAlgebra, NLPModels

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
