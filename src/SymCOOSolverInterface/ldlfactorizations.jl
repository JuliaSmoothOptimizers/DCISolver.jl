using LDLFactorizations, LinearAlgebra, SparseArrays

export LDLFactorizationStruct

# LDLFactorizations
mutable struct LDLFactorizationStruct{T <: AbstractFloat, Ti <: Int} <: SymCOOSolver
  ndim::Int
  rows::Vector{Ti}
  cols::Vector{Ti}
  vals::Vector{T}
  factorized::Bool #factor.__factorized is for memory allocation
  factor #specify the structure
end

#r1, r2, tol, n_d parameters for the dynamic regularization
function LDLFactorizationStruct(
  ndim::Int,
  rows::AbstractVector{Ti},
  cols::AbstractVector{Ti},
  vals::AbstractVector{T};
  r1::Real = zero(T),
  r2::Real = zero(T),
  tol::Real = zero(T),
  n_d::Int = 0,
) where {T, Ti}
  A = sparse(cols, rows, vals, ndim, ndim)
  S = ldl_analyze(Symmetric(A, :U))
  S.r1 = r1  #-ϵ
  S.r2 = r2  # ϵ
  S.tol = tol #ϵ
  S.n_d = n_d #0
  LDLFactorizationStruct(ndim, rows, cols, vals, false, S)
end

function factorize!(M::LDLFactorizationStruct)
  A = Symmetric(sparse(M.cols, M.rows, M.vals, M.ndim, M.ndim), :U)
  M.factor = ldl_factorize!(A, M.factor)
  M.factorized = factorized(M.factor)
  return M.factorized
end

function solve!(x, M::LDLFactorizationStruct, b)
  ldiv!(x, M.factor, b)
end

function success(M::LDLFactorizationStruct)
  !isnothing(M.factor) && M.factorized
end

function isposdef(M::LDLFactorizationStruct{T, Ti}) where {T, Ti}
  ϵ = eps(T)
  success(M) && count(M.factor.d .≤ -ϵ) == 0
end

function num_neg_eig(M::LDLFactorizationStruct{T, Ti}) where {T, Ti}
  ϵ = eps(T)
  count(M.factor.d .≤ -ϵ)
end
