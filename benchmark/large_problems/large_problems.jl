#=
List of reasonably large problems:

pnames = CUTEst.select(max_var=3000, min_var=1000, min_con=1, only_free_var=true, only_equ_con=true, objtype=3:6)
9-element Array{String,1}:
 "SPINOP" --> work with the agressive step :-)
 "EIGENA2" #easy :)
 "MSS3" --> cycle / Fail cgls computation Lagrange multiplier: maximum number of iterations exceeded
 "EIGENBCO" - not after 10 minutes, but it is decreasing (only 40 iterations)
 "LCH" --> works ! Youhou, about 270 iterations
                    many unnecessary factorizations though (reduce γ increase γ)
 "EIGENB2" --> ran for 30minutes but stuck
 "EIGENC2" --> 25 itérations, 250 secondes :)
 "EIGENCCO" --> not after 10 minutes :( (but slowly getting there I think) - too many unnecessary factorizations (reduce γ increase γ)
                isposdef(Symmetric(hess(nlp, nlp.meta.x0),:L)) is false 
"EIGENACO" easy :) 

Check that our limits are size dependent.

#Our formula for Infeasibility:
infeasible = norm(d) < ctol * ρ * min(normcz, one(T)) #should probably depend on Δ also?
#Alternative: scaling
#Unproductive steps

There are two annoying points:
i) infeasible stationary points of the feasibility problem. (-> move from the current point)
ii) infeasible stationary points of the optimization problem. (-> need to move from the current point AND change ρ)
=#

#=
pnames = CUTEst.select(max_var=10000, min_var=3000, min_con=1, only_free_var=true, only_equ_con=true, objtype=3:6)
 ("LUKVLE8", 10000, 9998, 30000, 29994) -> stuck in the factorization
 ("LUKVLE7", 10000, 4, 10003, 14) - works :)
 ("GRIDNETH", 7564, 3844, 16838, 15128) - works :)
 ("ORTHREGC", 5005, 2500, 22505, 17500) - works :)
 ("LUKVLE17", 9997, 7497, 14995, 17493) infeasible (quick)
 ("LUKVLE15", 9997, 7497, 19993, 22491) - time [looks like a scaling issue]
 ("ORTHRDS2", 5003, 2500, 22506, 12500) - works :)
 ("LUKVLE1", 10000, 9998, 19999, 29994) - works :)
 ("GRIDNETE", 7564, 3844, 15127, 15128) - works :)
 ("LUKVLE6", 9999, 4999, 69972, 14997) - time [looks like a scaling issue]
 ("LCH", 3000, 1, 16000, 2000) - works :)
 ("LUKVLE2", 10000, 4999, 19998, 34993) -unbounded below ? (no test for that yet)
 ("LUKVLE16", 9997, 7497, 14995, 17493) - works :)
 ("LUKVLE14", 9998, 6664, 13330, 19991) - works :)
 ("ORTHREGA", 8197, 4096, 36869, 28672) - time
 ("ORTHREGD", 5003, 2500, 22506, 12500) - works :)
 ("LUKVLE9", 10000, 6, 15002, 30) - works :)
 ("LUKVLE12", 9997, 7497, 22492, 19992) - works :)
 ("LUKVLE10", 10000, 9998, 15000, 29994) - works :)
 ("LUKVLE13", 9998, 6664, 16662, 26656)  - works :)
 ("GRIDNETB", 7564, 3844, 7564, 15128) - works :)
 ("ORTHRDM2", 8003, 4000, 36006, 20000) - works :)
 ("LUKVLE4", 10000, 4999, 19999, 14997) - unbounded? small_step
 ("LUKVLE3", 10000, 2, 24998, 4) - works :)
 ("ORTHRGDS", 5003, 2500, 22506, 12500) - works :)
 ("LUKVLE11", 9998, 6664, 21105, 19992) - works :)
 ("LUKVLE18", 9997, 7497, 14995, 17493) - infeasible at (1.2e-05)

19/27 problems (3000-10 000) + 5/8 1000-3000 (hors LCH) = 24/35 :-)
=#

using NLPModels, CUTEst, DCISolver
# Solved: SPINOP, EIGENA2, EIGENBCO, LCH, EIGENB2, EIGENACO
# Why don't cgls work for MSS3?
# EIGENC2 no real progress
# EIGENCCO Maybe with more time?
nlp = CUTEstModel("LCH") #LUKVLE11
#EIGENBCO is a good example where HSL-DCI has to recompute the factorization
@show nlp.meta.nvar, nlp.meta.ncon, nlp.meta.nnzh, nlp.meta.nnzj

stats = dci(nlp, nlp.meta.x0, linear_solver = :ldlfact, max_time = 160., max_iter = 1000)

@show nlp.counters
@show (stats.status, stats.elapsed_time, neval_obj(nlp))

reset!(nlp)

stats = dci(nlp, nlp.meta.x0, linear_solver = :ma57, max_time = 160., max_iter = 1000)

@show nlp.counters
@show (stats.status, stats.elapsed_time, neval_obj(nlp))

finalize(nlp)

#=
problems = ["SPINOP", "EIGENA2", "MSS3", "EIGENBCO", "LCH", "EIGENB2", "EIGENC2", "EIGENCCO", "EIGENACO"]
for p in pnames
  nlp = CUTEstModel(p)
  @show (nlp.meta.nvar, nlp.meta.ncon, nlp.meta.nnzh, nlp.meta.nnzj)
  finalize(nlp)
end
=#
