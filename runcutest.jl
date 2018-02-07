using CUTEst

include("dci.jl")

function runcutest()
  cutest_problems = readlines("paper-problems.list")
  cutest_problems = cutest_problems[1:3]
  @printf("%-12s  %9s  %8s  %8s  %8s  %6s  %5s\n",
          "Problem", "f(x)", "‖ℓ(x,λ)‖", "‖c(x)‖",
          "time", "status", "eval")
  for pname in cutest_problems
    nlp = CUTEstModel(pname)
    x, fx, dual, primal, t, solved, tired = dci(nlp)
    @printf("%-12s  %+8.2e  %8.2e  %8.2e  %8.2e  %6s  %5d\n",
            pname, fx, dual, primal, t,
            solved ? "solved" : (tired ? "tired" : "???"),
            sum_counters(nlp))
    finalize(nlp)
  end
end

runcutest()
