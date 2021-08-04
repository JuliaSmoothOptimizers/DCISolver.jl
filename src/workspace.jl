mutable struct DCIWorkspace{T, S <: AbstractVector{T}, Si <: AbstractVector{<:Integer}, Op}
  x0::S
  x::S
  ∇fx::S # grad(nlp, x)
  cx::S # cons(nlp, x)
  Jx::Op # jac_op(nlp, x)
  λ::S # λ = argmin ‖∇f + Jᵀλ‖
  ∇ℓxλ::S # ∇fx + Jx' * λ
  # From the feasibility step
  z::S
  ∇fz::S
  cz::S
  ∇ℓzλ::S
  # Allocate the sparse structure of K = [H + γI  [Jᵀ]; J -δI]
  rows::Si # zeros(Int, nnz)
  cols::Si # zeros(Int, nnz)
  vals::S # zeros(nnz)
  # LDL # would need the broadcast LDL
  xtan::S
  dtan::S
  tr::TrustRegion
  # factorization computation
  dζ::S
  dn::S
  dcp::S
  rhs::S
end

function DCIWorkspace(nlp::AbstractNLPModel{T, S}, meta, x0::S) where {S, T}
  n, m = nlp.meta.nvar, nlp.meta.ncon
  nnz = nlp.meta.nnzh + nlp.meta.nnzj + n + m
  Jx = jac_op(nlp, x0)
  rows, cols = Vector{Int}(undef, nnz), Vector{Int}(undef, nnz)
  vals = S(undef, nnz)
  # LDL = solver_correspondence[meta.linear_solver](n + m, rows, cols, vals)

  xtan = Array{T}(undef, n)
  dtan = Array{T}(undef, n)

  tr = TrustRegion(n, zero(T))
  dζ = Array{T}(undef, m + n)
  dn = Array{T}(undef, n)
  dcp = Array{T}(undef, n)
  rhs = Array{T}(undef, m + n)
  return DCIWorkspace{T, S, Vector{Int}, typeof(Jx)}(
    x0,
    S(undef, n),
    S(undef, n),
    S(undef, m),
    Jx,
    S(undef, m),
    S(undef, n),
    S(undef, n),
    S(undef, n),
    S(undef, m),
    S(undef, n),
    rows,
    cols,
    vals,
    # LDL,
    xtan,
    dtan,
    tr,
    dζ,
    dn,
    dcp,
    rhs,
  )
end