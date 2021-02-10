#=
List of reasonably large problems:

pnames = CUTEst.select(max_var=3000, min_var=1000, min_con=1, only_free_var=true, only_equ_con=true, objtype=3:6)
9-element Array{String,1}:
 "SPINOP" --> feasibility issue (work with the agressive step)
 "EIGENA2" #easy :)
 "MSS3" --> feasibility issue
 "EIGENBCO" - tough
 "LCH" --> looks like a problem in the scaling ρ/dL = 1000 (only 1 constraint)
 "EIGENB2" --> small tangent step (finish too soon)
 "EIGENC2" --> same
 "EIGENCCO" --> same
 "EIGENACO" easy :)

Check that our limits are size dependent.

"SPINOP" is nice as gradient of Lagrangian is way too small for where we were:
[ Info:           stage    iter   #f+#c      f(x)      ‖∇L‖    ‖c(x)‖      ρmax         ρ           status  
[ Info:            init       0       2   1.2e+03   1.9e+01   3.7e+02   1.8e+03       NaN
[ Info:               T       0       4   1.2e+03   1.8e+01   3.7e+02   1.8e+03   4.9e+02          success
[ Info:               T       1       6   9.0e+02   5.6e+00   4.4e+02   1.8e+03   4.7e+02          success
[ Info:               N       1      10   1.1e+03   1.7e+01   6.8e+01   1.8e+03   4.5e+02          success
[ Info:               T       2      12   1.1e+03   7.9e+00   6.7e+01   1.8e+03   4.5e+02          success
[ Info:               T       3      14   1.0e+03   5.0e+00   6.7e+01   1.8e+03   2.2e+02          success
[ Info:               T       4      16   1.0e+03   3.7e+00   6.6e+01   1.8e+03   1.4e+02          success
[ Info:               T       5      18   1.0e+03   3.0e+00   6.6e+01   1.8e+03   1.0e+02          success
[ Info:               T       6      20   1.0e+03   2.6e+00   6.6e+01   1.8e+03   8.6e+01          success
[ Info:               T       7      22   1.0e+03   2.3e+00   6.6e+01   1.8e+03   7.3e+01          success
[ Info:               N       1      24   1.0e+03   3.1e-03   2.9e+01   1.8e+03   1.0e-05          success
[ Info:               N       2      53   4.6e-02   2.0e-08   1.3e-03   1.8e+03   1.0e-05       infeasible
[ Info:               N       3      58   5.0e-02   4.9e-08   1.4e-03   1.8e+03   1.0e-05       infeasible
"Execution stats: problem may be infeasible"

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

nlp = CUTEstModel("MSS3")

stats = dci(nlp, nlp.meta.x0)

finalize(nlp)
