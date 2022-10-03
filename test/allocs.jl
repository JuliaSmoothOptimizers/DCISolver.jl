using NLPModelsTest, BenchmarkTools

for (problem, allocs) in ((:MGH01Feas, 7464), (:HS6, 7160))
  nlp = eval(problem)()
  stats = GenericExecutionStats(nlp)
  x = nlp.meta.x0
  meta = DCISolver.MetaDCI(x, nlp.meta.y0)
  solver = DCISolver.DCIWorkspace(nlp, meta, x)
  b = @ballocated solve!($solver, $nlp, $stats)
  @test b ≤ allocs
end

#=
using Profile, PProf
Profile.Allocs.clear()
NLPModels.reset!(nlp)
@time solve!(solver, nlp, stats)
@time Profile.Allocs.@profile sample_rate=1 solve!(solver, nlp, stats)
PProf.Allocs.pprof(from_c = false)
=#
