using ADNLPModels
using Documenter
using Printf
using DCISolver

pages = [
  "Introduction" => "index.md",
  "Fine-tune DCI" => "fine-tuneDCI.md",
  "Reference" => "reference.md",
]

makedocs(
  sitename = "DCISolver.jl",
  format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
  modules = [DCISolver],
  pages = pages,
)

deploydocs(repo = "github.com/JuliaSmoothOptimizers/DCISolver.jl.git", push_preview = true, devbranch = "main")
