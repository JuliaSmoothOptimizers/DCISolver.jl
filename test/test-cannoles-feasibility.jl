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
  
  # Verify successful convergence
  @test stats.status in [:first_order, :acceptable]
  @test isfinite(stats.objective)
  
  # Verify solution quality
  x_sol = stats.solution
  c_sol = cons(nlp, x_sol)
  @test norm(c_sol) ≤ 1e-5  # Constraints should be satisfied
  @test norm(x_sol - [1.0; 1.0]) ≤ 1e-4  # Solution should be near optimum
  
  finalize(nlp)
  
end

@testset "DCI with CaNNOLeS vs trust-region comparison" begin
  nlp = ADNLPModel(
    x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2,
    [-1.2; 1.0],
    x -> [x[1] * x[2] - 1],
    [0.0], [0.0],
  )
  
  # Solve with CaNNOLeS feasibility step
  stats_cannoles = dci(
    nlp,
    nlp.meta.x0,
    feas_step = :feasibility_step_cannoles,
    atol = 1e-5,
    ctol = 1e-5,
    rtol = 1e-5,
    max_time = 60.0,
    max_iter = 100,
  )
  
  finalize(nlp)
  nlp = ADNLPModel(
    x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2,
    [-1.2; 1.0],
    x -> [x[1] * x[2] - 1],
    [0.0], [0.0],
  )
  
  # Solve with default trust-region feasibility step
  stats_default = dci(
    nlp,
    nlp.meta.x0,
    feas_step = :feasibility_step,
    atol = 1e-5,
    ctol = 1e-5,
    rtol = 1e-5,
    max_time = 60.0,
    max_iter = 100,
  )
  
  # Both should converge successfully
  @test stats_cannoles.status in [:first_order, :acceptable]
  @test stats_default.status in [:first_order, :acceptable]
  
  # Both should find similar solutions
  @test norm(stats_cannoles.solution - stats_default.solution) ≤ 1e-3
  @test abs(stats_cannoles.objective - stats_default.objective) ≤ 1e-3
  
  finalize(nlp)
  
  println("✓ CaNNOLeS vs trust-region comparison test passed")
end
