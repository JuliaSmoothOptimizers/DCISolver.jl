using Gridap, PDENLPModels

include("Burger1d.jl")

n= 512
(nlp, sol) = Burger1d(n = n) #by default n=512

using DCI, NLPModelsIpopt

#First run, for testing:
tol = 1e-7

stats_ipopt = ipopt(nlp, tol = tol, x0 = nlp.meta.x0)
@show norm(stats_ipopt.solution - sol, 1)/nlp.meta.nvar
@show nlp.counters
reset!(nlp)

stats_dci = dci(nlp, atol = tol, rtol = tol, ctol = tol) 
@show norm(stats_dci.solution - sol, 1)/nlp.meta.nvar
nlp.counters
reset!(nlp)

using GR
gr()

t = 0.:1/n:1.
plot(t, vcat(0.,stats_ipopt.solution[1:n-1], 1.))

tc = 0.:1/(nlp.meta.nvar-n):1.
plot(tc, stats_ipopt.solution[n:end])