# NonconvexUtils

[![CI](https://github.com/JuliaNonconvex/NonconvexUtils.jl/workflows/CI/badge.svg?branch=main)](https://github.com/JuliaNonconvex/NonconvexUtils.jl/actions?query=workflow%3ACI)
[![Coverage](https://codecov.io/gh/JuliaNonconvex/NonconvexUtils.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaNonconvex/NonconvexUtils.jl)

Useful hacks for use in Nonconvex.jl.

## Hack #1: `AbstractDiffFunction` and `ForwardDiffFunction`

[`Nonconvex.jl`](https://github.com/JuliaNonconvex/Nonconvex.jl) uses [`Zygote.jl`](https://github.com/FluxML/Zygote.jl) for automatic differentiation (AD). In order to force the use of another AD package for a function `f`, one can specify any AD `backend` from [`AbstractDifferentiation.jl`](https://github.com/JuliaDiff/AbstractDifferentiation.jl) in the following way:
```julia
g = AbstractDiffFunction(f, backend)
```

If you want to use [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) to differentiate the function `f`, you can also use
```julia
g = ForwardDiffFunction(f)
```
which is short for to:
```julia
AbstractDiffFunction(f, AbstractDifferentiation.ForwardDiffBackend())
```

## Hack #2: `TraceFunction`

Often one may want to store intermediate solutions, function values and gradients for visualisation or post-processing. This is currently not possible with `Nonconvex.jl` as not all solvers support a callback mechanism. To workround this, `TraceFunction` can be used to store input, output and optionally gradient values
during the optimization:
```julia
g = TraceFunction(f; on_call = false, on_grad = true)
```
If the `on_call` keyword argument is set to true, the input and output values are stored every time the function `g` is called. If the `on_grad` keyword argument is set to true, the input, output and gradient values are stored every time the function `g` is differentiated with a `ChainRules`-compatible AD package such as `Zygote.jl` which is used by `Nonconvex.jl`. The history is stored in `f.trace`.

## Hack #3: `CustomGradFunction`

Often a function `f` can have analytic an gradient function `∇f` that is more efficient than using AD on `f`. The way to make use of this gradient function in `Nonconvex.jl` has been to define an `rrule` for the function `f`. Now the following can be used instead. This will work for scalar-valued or vector-valued functions `f` where `∇f` is either the gradient function or Jacobian function respectively.
```julia
g = CustomGradFunction(f, ∇f)
```

## Hack #4: `CustomHessianFunction` and Hessian-vector products

Similar to `CustomGradFunction` if a function `f` has a custom gradient function `∇f` and a custom Hessian function `∇²f`, they can be used to force Zygote to use them in the following code:
```julia
g = CustomHessianFunction(f, ∇f, ∇²f)
Zygote.gradient(f, x)
Zygote.jacobian(x -> Zygote.gradient(f, x)[1], x)
```
It is on the user to ensure that the custom Hessian is always a symmetric matrix.

Note that one has to use `Zygote` for both levels of differentiation for this to work which makes it currently impossible to use in Nonconvex.jl directly, e.g. with IPOPT, because Nonconvex.jl uses ForwardDiff.jl for the second order differentiation, but this will be fixed soon by making more use of `AbstractDifferentiation` when it gets a `ZygoteBackend` implemented.

If instead of `∇²f`, you only have access to a Hessian-vector product function `hvp` which takes 2 inputs: `x` (the input to `f`) and `v` (the vector to multiply the Hessian `H` by), and returns `H * v`, you can use this as follows:
```julia
g = CustomHessianFunction(f, ∇f, hvp; hvp = true)
```
