using DCISolver
using Documenter

DocMeta.setdocmeta!(DCISolver, :DocTestSetup, :(using DCISolver); recursive = true)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
    file for file in readdir(joinpath(@__DIR__, "src")) if
    file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(;
    modules = [DCISolver],
    authors = "Abel Soares Siqueira <abel.s.siqueira@gmail.com> and Tangi Migot <tangi.migot@gmail.com>",
    repo = "https://github.com/JuliaSmoothOptimizers/DCISolver.jl/blob/{commit}{path}#{line}",
    sitename = "DCISolver.jl",
    format = Documenter.HTML(; canonical = "https://JuliaSmoothOptimizers.github.io/DCISolver.jl"),
    pages = ["index.md"; numbered_pages],
)

deploydocs(; repo = "github.com/JuliaSmoothOptimizers/DCISolver.jl")
