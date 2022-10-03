"""
    DCIWorkspace(nlp, meta, x)

Pre-allocate the memory used during the [`dci`](@ref) call.
Returns a `DCIWorkspace` structure.
"""
struct DCIWorkspace{T, S <: AbstractVector{T}, Si <: AbstractVector{<:Integer}, Op, In, COO} <: AbstractOptimizationSolver
  x0::S
  x::S
  ∇fx::S # grad(nlp, x)
  cx::S # cons(nlp, x)
  Jx::Op # jac_op!(nlp, x, Jv, Jtv)
  Jv::S
  Jtv::S
  λ::S # λ = argmin ‖∇f + Jᵀλ‖
  ∇ℓxλ::S # ∇fx + Jx' * λ
  # From the feasibility/normal step
  z::S
  ∇fz::S
  cz::S
  ∇ℓzλ::S
  zp::S
  czp::S
  Jd::S
  # Allocate the sparse structure of K = [H + γI  [Jᵀ]; J -δI]
  rows::Si # zeros(Int, nnz)
  cols::Si # zeros(Int, nnz)
  vals::S # zeros(nnz)
  # LDL::COO # would need the broadcast LDL
  xtan::S
  dtan::S
  tr::TrustRegion
  # factorization computation
  dζ::S
  dn::S
  dcp::S
  rhs::S
  meta::MetaDCI{T, In, COO}
end

function DCIWorkspace(
  nlp::AbstractNLPModel{T, S},
  meta::MetaDCI{T, In, COO},
  x0::S,
) where {T, S <: AbstractVector{T}, In <: Integer, COO <: SymCOOSolver}
  n, m = nlp.meta.nvar, nlp.meta.ncon
  nnz = nlp.meta.nnzh + nlp.meta.nnzj + n + m
  Jx = jac_op(nlp, x0)
  rows, cols = Vector{Int}(undef, nnz), Vector{Int}(undef, nnz)
  vals = S(undef, nnz)

  return DCIWorkspace(
    x0,
    S(undef, n),
    S(undef, n),
    S(undef, m),
    Jx,
    S(undef, m),
    S(undef, n),
    S(undef, m),
    S(undef, n),
    S(undef, n),
    S(undef, n),
    S(undef, m),
    S(undef, n),
    S(undef, n),
    S(undef, m),
    S(undef, m),
    rows,
    cols,
    vals,
    # LDL,
    S(undef, n),
    S(undef, n),
    TrustRegion(n, zero(T)),
    S(undef, m + n),
    S(undef, n),
    S(undef, n),
    S(undef, m + n),
    meta,
  )
end

get_meta(workspace::DCIWorkspace{T, S, Si, Op, In, COO}) where {T, S, Si, Op, In, COO} = workspace.meta
