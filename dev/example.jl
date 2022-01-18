using Gridap, PDENLPModels

n = 20
domain = (-1, 1, -1, 1)
partition = (n, n)
model = CartesianDiscreteModel(domain, partition)

reffe = ReferenceFE(lagrangian, Float64, 2)
Xpde = TestFESpace(model, reffe; conformity = :H1, dirichlet_tags = "boundary")
y0(x) = 0.0
Ypde = TrialFESpace(Xpde, y0)

reffe_con = ReferenceFE(lagrangian, Float64, 1)
Xcon = TestFESpace(model, reffe_con; conformity = :H1)
Ycon = TrialFESpace(Xcon)
Y = MultiFieldFESpace([Ypde, Ycon])

trian = Triangulation(model)
degree = 1
dΩ = Measure(trian, degree)

yd(x) = -x[1]^2
α = 1e-2
function f(y, u)
  ∫(0.5 * (yd - y) * (yd - y) + 0.5 * α * u * u) * dΩ
end

ω = π - 1 / 8
h(x) = -sin(ω * x[1]) * sin(ω * x[2])
function res(y, u, v)
  ∫(∇(v) ⊙ ∇(y) - v * u - v * h) * dΩ
end
op = FEOperator(res, Y, Xpde)

npde = Gridap.FESpaces.num_free_dofs(Ypde)
ncon = Gridap.FESpaces.num_free_dofs(Ycon)
x0 = zeros(npde + ncon);

nlp = GridapPDENLPModel(x0, f, trian, Ypde, Ycon, Xpde, Xcon, op, name = "Control elastic membrane")

(nlp.meta.nvar, nlp.meta.ncon)

using JSOSolvers, NLPModelsModifiers

nls = FeasibilityResidual(nlp)
stats_trunk = trunk(nls)

norm(cons(nlp, stats_trunk.solution))

using NLPModelsIpopt

stats_ipopt = ipopt(nlp, x0 = stats_trunk.solution, tol = 1e-5, print_level = 0)

stats_ipopt.counters

reset!(nlp);

using DCISolver, Logging

stats_dci = with_logger(NullLogger()) do
    dci(nlp, stats_trunk.solution, atol = 1e-5, rtol = 0.0)
end

stats_dci.counters

stats_ipopt.elapsed_time, stats_dci.elapsed_time

(stats_ipopt.objective, stats_ipopt.primal_feas, stats_ipopt.dual_feas), (stats_dci.objective, stats_dci.primal_feas, stats_dci.dual_feas)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

