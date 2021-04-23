using Pkg;
Pkg.activate("bench");
using Gridap, PDENLPModels, NLPModelsIpopt, DCISolver

function Burger1d(; n::Int = 512, kwargs...)

  #Domain
  domain = (0, 1)
  partition = n
  model = CartesianDiscreteModel(domain, partition)

  #Definition of the spaces:
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "diri1", [2])
  add_tag_from_tags!(labels, "diri0", [1])

  Xpde = TestFESpace(
    reffe = :Lagrangian,
    conformity = :H1,
    valuetype = Float64,
    model = model,
    labels = labels,
    order = 1,
    dirichlet_tags = ["diri0", "diri1"],
  )

  uD0 = VectorValue(0)
  uD1 = VectorValue(-1)
  Ypde = TrialFESpace(Xpde, [uD0, uD1])

  Xcon = TestFESpace(
    reffe = :Lagrangian,
    order = 1,
    valuetype = Float64,
    conformity = :L2,
    model = model,
  )
  Ycon = TrialFESpace(Xcon)

  #Integration machinery
  trian = Triangulation(model)
  degree = 1
  quad = CellQuadrature(trian, degree)

  #Now we move to the optimization:
  yd(x) = -x[1]^2
  α = 1e-2
  #objective function:
  f(y, u) = 0.5 * (yd - y) * (yd - y) + 0.5 * α * u * u
  function f(yu) #:: Union{Gridap.MultiField.MultiFieldFEFunction, Gridap.CellData.GenericCellField}
    y, u = yu
    f(y, u)
  end

  #Definition of the constraint operator
  h(x) = 2 * (nu + x[1]^3)
  @law conv(u, ∇u) = (∇u ⋅ one(∇u)) ⊙ u
  c(u, v) = v ⊙ conv(u, ∇(u))
  nu = 0.08
  function res(yu, v)
    y, u = yu
    v

    -nu * (∇(v) ⊙ ∇(y)) + c(y, v) - v * u - v * h
  end
  t_Ω = FETerm(res, trian, quad)
  op = FEOperator(Ypde, Xpde, t_Ω) # or FEOperator(Y, Xpde, t_Ω)

  nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
  nvar_con = Gridap.FESpaces.num_free_dofs(Ycon)
  x0 = zeros(nvar_pde + nvar_con)
  nlp = GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, op, name = "Burger1d")
  #=
      #The solution is just  y = yd and u=0. 
      cell_xs = get_cell_coordinates(trian)
      #Create a function that given a cell returns the middle.
      midpoint(xs) = sum(xs)/length(xs)
      cell_xm = apply(midpoint, cell_xs)
      cell_y = apply(x -> yd(x), cell_xm) #this is a vector of size num_cells(trian)
      #Warning: `interpolate(fs::SingleFieldFESpace, object)` is deprecated, use `interpolate(object, fs::SingleFieldFESpace)` instead.
      soly = get_free_values(Gridap.FESpaces.interpolate(nlp.Ypde, cell_y))
      sol = vcat(soly, zeros(eltype(nlp.meta.x0), ncon))
  =#
  return nlp
end

nlp = Burger1d()

@time stats = ipopt(nlp)

@show nlp.counters

reset!(nlp)
m, n = nlp.meta.ncon, nlp.meta.nvar
meta = DCISolver.MetaDCI(
  nlp.meta.x0,
  nlp.meta.y0,
  max_time = 600.0,
  linear_solver = :ldlfact,
  TR_compute_step = :TR_lsmr,
  TR_struct = DCISolver.TR_lsmr_struct(m, n, 1.0, itmax = 4 * (m + n)),
)
@time stats2 = dci(nlp, nlp.meta.x0, meta)

@show nlp.counters

#=
julia> include("pde_test.jl")
 Activating environment at `~/cvs/JSO/DCI.jl/benchmark/bench/Project.toml`
This is Ipopt version 3.13.4, running with linear solver mumps.
NOTE: Other linear solvers might be more efficient (see Ipopt documentation).

Number of nonzeros in equality constraint Jacobian...:     4086
Number of nonzeros in inequality constraint Jacobian.:        0
Number of nonzeros in Lagrangian Hessian.............:    10224

Total number of variables............................:     1535
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        0
                     variables with only upper bounds:        0
Total number of equality constraints.................:      511
Total number of inequality constraints...............:        0
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  9.9269167e-02 4.07e+01 3.31e-04  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
MUMPS returned INFO(1) = -9 and requires more memory, reallocating.  Attempt 1
  Increasing icntl[13] from 1000 to 2000.
MUMPS returned INFO(1) = -9 and requires more memory, reallocating.  Attempt 2
  Increasing icntl[13] from 2000 to 4000.
MUMPS returned INFO(1) = -9 and requires more memory, reallocating.  Attempt 3
  Increasing icntl[13] from 4000 to 8000.
   1  4.1299027e-03 2.45e-01 1.02e-02  -1.0 1.60e+00  -4.0 1.00e+00 1.00e+00h  1
   2  2.3273757e-03 1.95e-04 2.67e-04  -1.7 6.79e-01  -4.5 1.00e+00 1.00e+00h  1
   3  6.0521829e-04 6.88e-05 1.66e-05  -5.7 5.28e-01  -5.0 1.00e+00 1.00e+00h  1
   4  4.2788845e-04 1.10e-05 1.09e-06  -5.7 2.95e-01  -5.4 1.00e+00 1.00e+00h  1
   5  4.2141185e-04 3.79e-07 9.21e-08  -8.6 7.46e-02  -5.9 1.00e+00 1.00e+00h  1
   6  4.2135337e-04 2.47e-09 3.03e-09  -8.6 7.36e-03  -6.4 1.00e+00 1.00e+00h  1

Number of Iterations....: 6

                                   (scaled)                 (unscaled)
Objective...............:   4.2135336945288123e-04    4.2135336945288123e-04
Dual infeasibility......:   3.0294861701782725e-09    3.0294861701782725e-09
Constraint violation....:   2.4664157660669872e-09    2.4664157660669872e-09
Complementarity.........:   0.0000000000000000e+00    0.0000000000000000e+00
Overall NLP error.......:   3.0294861701782725e-09    3.0294861701782725e-09

Number of objective function evaluations             = 7
Number of objective gradient evaluations             = 7
Number of equality constraint evaluations            = 7
Number of inequality constraint evaluations          = 0
Number of equality constraint Jacobian evaluations   = 7
Number of inequality constraint Jacobian evaluations = 0
Number of Lagrangian Hessian evaluations             = 6
Total CPU secs in IPOPT (w/o function evaluations)   =      9.323
Total CPU secs in NLP function evaluations           =      9.607

EXIT: Optimal Solution Found.
 19.186016 seconds (25.04 M allocations: 1.312 GiB, 4.48% gc time)
nlp.counters =   Counters:
             obj: █████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 7                 grad: ██████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 8                 cons: █████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 7     
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ████████████████████ 16    
           jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ███████████████⋅⋅⋅⋅⋅ 12    
           hprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     

┌ Warning: Fail cgls computation Lagrange multiplier: maximum number of iterations exceeded
└ @ DCI ~/.julia/packages/DCI/EGXI9/src/DCI.jl:107
[ Info:           stage    iter   #f+#c      f(x)         ℓ      ‖∇L‖    ‖c(x)‖      ρmax         ρ           status       ‖d‖         Δ  
[ Info:            init       0       2   9.9e-02   4.4e-02   1.7e-02   4.1e+01   2.0e+02       NaN                -         -         -
[ Info:               F       0       3         -         -         -   4.1e+00         -         -          success   1.0e+00   2.0e+00
[ Info:               F       1       4         -         -         -   4.8e-01         -         -          success   2.0e+00   4.0e+00
[ Info:               N       1       5   9.1e-02  -2.0e-01   8.2e-03   4.8e-01   2.0e+02   1.6e+00          success         -         -
[ Info:              Tr       0       7   8.4e-02  -2.0e-01         -   4.8e-01         -         -          success   1.0e+00   2.0e+00
[ Info:               T       0       7   8.4e-02  -1.9e-01   7.6e-03   4.8e-01   2.4e+00   1.6e+00          success         -         -
[ Info:               F       0       8         -         -         -   2.7e-01         -         -          success   1.0e+00   2.0e+00
[ Info:               F       1       9         -         -         -   1.2e-01         -         -          success   2.0e+00   4.0e+00
[ Info:               F       2      10         -         -         -   4.4e-02         -         -          success   4.0e+00   8.0e+00
[ Info:               F       3      11         -         -         -   1.3e-02         -         -          success   8.0e+00   1.6e+01
[ Info:               N       1      12   1.1e-03   2.4e-03   7.1e-04   1.3e-02   2.4e+00   1.7e-03          success         -         -
[ Info:               F       0      13         -         -         -   2.9e-03         -         -          success   1.0e+00   2.0e+00
[ Info:               F       1      14         -         -         -   1.6e-04         -         -          success   8.7e-01   2.0e+00
[ Info:               N       2      15   4.6e-03   4.7e-03   1.7e-03   1.6e-04   2.4e+00   4.1e-03          success         -         -
[ Info:           stage       -       -         γ         δ      δmin         -     slope         -                -  
[ Info:            Fact       -       -   0.0e+00   0.0e+00   1.5e-08         -       NaN         -       regularize   0.0e+00         -
[ Info:            Fact       -       -   1.5e-08   1.5e-08   1.5e-08         -  -8.9e-03         -          success   5.3e+00         -
[ Info:              Tr       0      17   5.0e-04   2.1e-04         -   9.1e-04         -         -          success   5.3e+00   2.0e+01
[ Info:               T       1      17   5.0e-04   4.8e-04   1.5e-04   9.1e-04   1.2e+00   4.1e-03          success         -         -
[ Info:               F       0      18         -         -         -   1.4e-05         -         -          success   1.1e-01   1.0e+00
[ Info:               N       1      19   4.9e-04   4.9e-04   1.9e-04   1.4e-05   1.2e+00   2.2e-04          success         -         -
[ Info:           stage       -       -         γ         δ      δmin         -     slope         -                -  
[ Info:            Fact       -       -   1.5e-08   1.5e-08   1.5e-08         -  -1.3e-04         -          success   1.2e+00         -
[ Info:              Tr       0      21   4.2e-04   4.2e-04         -   2.6e-05         -         -          success   1.2e+00   2.0e+02
[ Info:               T       2      21   4.2e-04   4.2e-04   2.1e-06   2.6e-05   9.4e-03   2.2e-04          success         -         -
[ Info:               F       0      22         -         -         -   7.7e-07         -         -          success   6.5e-03   1.0e+00
[ Info:               N       1      23   4.2e-04   4.2e-04   6.8e-06   7.7e-07   9.4e-03   1.0e-05          success         -         -
  3.150181 seconds (2.02 M allocations: 166.342 MiB, 2.79% gc time)
nlp.counters =   Counters:
             obj: █████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 9                 grad: █████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 9                 cons: ██████████████⋅⋅⋅⋅⋅⋅ 14    
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ████████████████████ 20    
           jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ██████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 6     
           hprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     

  Counters:
             obj: █████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 9                 grad: █████████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 9                 cons: ██████████████⋅⋅⋅⋅⋅⋅ 14    
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ████████████████████ 20    
           jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ██████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 6     
           hprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
=#
