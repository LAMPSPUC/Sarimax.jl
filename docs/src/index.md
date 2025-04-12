```@raw html
<div style="width:100%; height:150px;border-width:4px;border-style:solid;padding-top:25px;
        border-color:#000;border-radius:10px;text-align:center;background-color:#99DDFF;
        color:#000">
    <h3 style="color: black;">Star us on GitHub!</h3>
    <a class="github-button" href="https://github.com/LAMPSPUC/Sarimax.jl" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star LAMPSPUC/Sarimax.jl on GitHub" style="margin:auto">Star</a>
    <script async defer src="https://buttons.github.io/buttons.js"></script>
</div>
```

# Sarimax.jl Documentation

## Introduction

Sarimax.jl is a groundbreaking Julia package that revolutionizes SARIMA (Seasonal Autoregressive Integrated Moving Average) modeling by seamlessly integrating with the JuMP framework — a powerful optimization modeling language. Unlike traditional SARIMA implementations, Sarimax.jl leverages JuMP's optimization capabilities to provide precise and highly customizable SARIMA models.

### Key Features

* Fit models using various objective functions:
  * Mean Squared Errors
  * Maximum Likelihood estimation
  * Bilevel objective function
* Auto SARIMA model selection
* Support for exogenous variables (Sarimax)
* Scenario simulation capabilities
* Time series integration and differentiation
* Model evaluation criteria (AIC, AICc, BIC)

## Installation

Sarimax.jl can be installed using Julia's built-in package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia
pkg> add Sarimax
```

Or, you can install it by using `Pkg` directly:

```julia
using Pkg
Pkg.add("Sarimax")
```

To use the development version, you can install directly from the GitHub repository:

```julia
Pkg.add(url = "https://github.com/LAMPSPUC/Sarimax.jl.git")
```

## Quick Start

To start using Sarimax.jl, simply import the package:

```julia
using Sarimax
```

Check out our [Tutorial](#tutorial) section for detailed examples of how to use the package.

## License

Sarimax.jl is licensed under the [MIT License](https://opensource.org/licenses/MIT). This means you are free to use, modify, and distribute the code, subject to the terms and conditions of the MIT license.

## Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/LAMPSPUC/Sarimax.jl). Pull requests for bug fixes and new features are also appreciated.

For more detailed information about the package functionality, please refer to the following sections:

```@contents
Pages = [
    "tutorial.md",
    "api.md",
    "examples.md"
]
Depth = 2
```

