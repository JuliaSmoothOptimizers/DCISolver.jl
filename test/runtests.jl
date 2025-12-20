# stdlib
using LinearAlgebra, Logging, Test
# JSO
using ADNLPModels, Krylov, NLPModels, SolverCore, SolverTest
# This package
using DCISolver

#=
Don't add your tests to runtests.jl. Instead, create files named

    test-title-for-my-test.jl

The file will be automatically included inside a `@testset` with title "Title For My Test".
=#
for (root, dirs, files) in walkdir(@__DIR__)
  for file in files
    if isnothing(match(r"^test-.*\.jl$", file))
      continue
    end
    if file == "test-cannoles-feasibility.jl"
      continue
    end
    title = titlecase(replace(splitext(file[6:end])[1], "-" => " "))
    @testset "$title" begin
      include(file)
    end
  end
end

if Base.find_package("CaNNOLeS") !== nothing
  @testset "Cannoles Feasibility" begin
    include("test-cannoles-feasibility.jl")
  end
else
  @info "Skipping CaNNOLeS feasibility tests: CaNNOLeS not installed"
end
