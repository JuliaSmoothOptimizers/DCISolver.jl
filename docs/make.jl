ENV["GKSwstype"] = "100"
using ADNLPModels
using Documenter
using Printf
using DCISolver
using Literate

EXAMPLE = joinpath(@__DIR__, "assets", "example.jl")
OUTPUT = joinpath(@__DIR__, "src")

# Generate markdown
binder_badge = "# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JuliaSmoothOptimizers/DCISolver.jl/gh-pages?labpath=dev%2Fexample.ipynb)"
function preprocess_docs(content)
  return string(binder_badge, "\n\n", content)
end

Literate.markdown(EXAMPLE, OUTPUT; preprocess = preprocess_docs, codefence = "```julia" => "```")
Literate.notebook(EXAMPLE, OUTPUT, documenter = false, execute = false)
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
