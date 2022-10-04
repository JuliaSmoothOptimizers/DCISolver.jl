using NLPModelsTest, BenchmarkTools

for (problem, allocs) in ((:MGH01Feas, 7512), (:HS6, 7208))
  nlp = eval(problem)()
  stats = GenericExecutionStats(nlp)
  meta = DCISolver.MetaDCI(nlp)
  solver = DCISolver.DCIWorkspace(nlp, meta)
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
