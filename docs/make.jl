using Documenter
using TimeSeries
include("../src/SARIMAX.jl")

# DocMeta.setdocmeta!(SARIMAX, :DocTestSetup, :(using ..SARIMAX); recursive=true)

makedocs(;
    modules=[SARIMAX],
    doctest=false,
    clean=true,
    checkdocs=:none,
    format=Documenter.HTML(; mathengine=Documenter.MathJax2()),
    sitename="SARIMAX.jl",
    authors="Luiz Fernando Duarte",
    pages=[
        "Home" => "index.md",
        "API Reference" => "reference.md",
        "Tutorial" => "tutorial.md",
    ],
)

deploydocs(; repo="github.com/LAMPSPUC/SARIMAX.jl.git", push_preview=true)
