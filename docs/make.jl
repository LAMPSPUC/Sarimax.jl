using Documenter
using TimeSeries
include("../src/Sarimax.jl")

# DocMeta.setdocmeta!(Sarimax, :DocTestSetup, :(using ..Sarimax); recursive=true)

makedocs(;
    modules=[Sarimax],
    doctest=false,
    clean=true,
    checkdocs=:none,
    format=Documenter.HTML(; mathengine=Documenter.MathJax2()),
    sitename="Sarimax.jl",
    authors="Luiz Fernando Duarte",
    pages=[
        "Home" => "index.md",
        "API Reference" => "reference.md",
        "Tutorial" => "tutorial.md",
    ],
)

deploydocs(; repo="github.com/LAMPSPUC/Sarimax.jl.git", push_preview=true)
