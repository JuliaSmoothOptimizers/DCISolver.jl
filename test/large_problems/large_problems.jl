#=
List of reasonably large problems:

pnames = CUTEst.select(max_var=3000, min_var=1000, min_con=1, only_free_var=true, only_equ_con=true, objtype=3:6)
9-element Array{String,1}:
 "SPINOP" --> work with the agressive step :-)
 "EIGENA2" #easy :)
 "MSS3" --> feasibility issue -> move a bit but stuck in the T/Tr step
 "EIGENBCO" - tough
 "LCH" --> looks like a problem in the scaling ρ/dL = 1000 (only 1 constraint)
 "EIGENB2" --> small tangent step (finish too soon)
 "EIGENC2" --> same
 "EIGENCCO" --> same
 "EIGENACO" easy :)

Check that our limits are size dependent.

#Our formula for Infeasibility:
infeasible = norm(d) < ctol * ρ * min(normcz, one(T)) #should probably depend on Δ also?
#Alternative: scaling
#Unproductive steps

There are two annoying points:
i) infeasible stationary points of the feasibility problem. (-> move from the current point)
ii) infeasible stationary points of the optimization problem. (-> need to move from the current point AND change ρ)

Small tangent step: "Execution stats: unhandled exception"
What is the value of λ when it is stalling?
=#

using NLPModels, CUTEst, DCI

nlp = CUTEstModel("EIGENA2")

stats = dci(nlp, nlp.meta.x0, linear_solver = :ma57, max_time = 1200.)

finalize(nlp)
