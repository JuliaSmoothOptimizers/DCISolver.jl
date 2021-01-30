using Gridap, PDENLPModels

include("Burger1d.jl")

(nlp, sol) = Burger1d(n = 10) #by default n=512

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

function run(n)

  (nlp, sol) = Burger1d(n = n)

  tol = min(1/n^2, 1e-7)

  stats_ipopt = ipopt(nlp, tol = tol, x0 = nlp.meta.x0)
  stats_dci  = dci(nlp, atol = tol, rtol = tol, ctol = tol)
  
  err_ipopt = norm(stats_ipopt.solution - sol)
  err_dci   = norm(stats_dci.solution - sol)
  
  return (err_ipopt, err_dci)
end

function conv_test(ns)

  err_ipopts = Float64[]
  err_dcis = Float64[]
  hs = Float64[]
  
  for n in ns
  
    (err_ipopt, err_dci) = run(n)
    h = 1.0/n
  
    push!(err_ipopts, err_ipopt)
    push!(err_dcis, err_dci)
    push!(hs, h)
  
  end
  
  return (err_ipopts, err_dcis, hs)
end
  
(err_ipopts, err_dcis, hs) = conv_test([8, 16, 32, 64])#, 128, 256, 512, 1024]);
  
using GR
gr()
  
  plot(hs,[err_ipopts, err_dcis],
      xaxis=:log, yaxis=:log,
      label=["Ipopt" "DCI"],
      shape=:auto,
      xlabel="h", ylabel="error norm")
  #=
      function slope(hs,errors)
        x = log10.(hs)
        y = log10.(errors)
        linreg = hcat(fill!(similar(x), 1), x) \ y
        linreg[2]
      end
  
  slope(hs,err_ipopts)
  slope(hs,err_dcis)
  =#
  