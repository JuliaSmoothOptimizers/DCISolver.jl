using ADNLPModels
using Documenter
using Printf
using DCISolver

# Add index.md file as introduction to navigation menu
pages = ["Introduction"=> "index.md",
         "Fine-tune DCI" => "fine-tuneDCI.md",
         "Reference" => "reference.md"]

makedocs(
    sitename = "DCISolver.jl",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [DCISolver],
    pages = pages
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/JuliaSmoothOptimizers/DCISolver.jl.git",
    push_preview = true
)
