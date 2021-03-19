using Pkg; Pkg.activate("bench")
using Gridap, PDENLPModels, NLPModelsIpopt, DCI

function Burger1d(;n :: Int = 512, kwargs...)

    #Domain
    domain = (0,1)
    partition = n
    model = CartesianDiscreteModel(domain,partition)

    #Definition of the spaces:
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels,"diri1",[2])
    add_tag_from_tags!(labels,"diri0",[1])

    Xpde = TestFESpace(
      reffe=:Lagrangian, conformity=:H1, valuetype=Float64,
      model=model, labels=labels, order=1, dirichlet_tags=["diri0","diri1"])

    uD0 = VectorValue(0)
    uD1 = VectorValue(-1)
    Ypde = TrialFESpace(Xpde,[uD0,uD1])

    Xcon = TestFESpace(
            reffe=:Lagrangian, order=1, valuetype=Float64,
            conformity=:L2, model=model)
    Ycon = TrialFESpace(Xcon)

    #Integration machinery
    trian = Triangulation(model)
    degree = 1
    quad = CellQuadrature(trian,degree)

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
    h(x) = 2*(nu + x[1]^3)
    @law conv(u,∇u) = (∇u ⋅one(∇u))⊙u
    c(u,v) = v⊙conv(u,∇(u))
    nu = 0.08
    function res(yu, v)
     y, u = yu
     v

     -nu*(∇(v)⊙∇(y)) + c(y,v) - v * u - v * h
    end
    t_Ω = FETerm(res,trian,quad)
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
meta = DCI.MetaDCI(nlp.meta.x0, nlp.meta.y0,
               max_time = 600., 
               linear_solver = :ldlfact, 
               TR_compute_step = :TR_lsmr, 
               TR_struct = DCI.TR_lsmr_struct(m, n, 1., itmax = 4*(m+n)))
@time stats2 = dci(nlp, nlp.meta.x0, meta)

@show nlp.counters