ENV["GKSwstype"] = "100"
using ADNLPModels
using Documenter
using Printf
using DCISolver
using Literate

EXAMPLE = joinpath(@__DIR__, "assets", "example.jl")
OUTPUT = joinpath(@__DIR__, "src")

Literate.markdown(EXAMPLE, OUTPUT)
Literate.notebook(EXAMPLE, OUTPUT)
Literate.script(EXAMPLE, OUTPUT)

pages = [
  "Introduction" => "index.md",
  "Benchmark" => "benchmark.md",
  "Fine-tune DCI" => "fine-tuneDCI.md",
  "Large-scale example" => "example.md",
  "Reference" => "reference.md",
]

makedocs(
  sitename = "DCISolver.jl",
  format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
  modules = [DCISolver],
  pages = pages,
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/DCISolver.jl.git",
  push_preview = true,
  devbranch = "main",
)
