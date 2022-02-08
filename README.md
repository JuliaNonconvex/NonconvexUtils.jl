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
which is short for:
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

## Hack #5: `ImplicitFunction`

### Explicit parameters

Differentiating implicit functions efficiently using the implicit function theorem has many applications including:
- Nonlinear partial differential equation constrained optimization
- Differentiable optimization layers in deep learning (aka deep declarative networks)
- Differentiable fixed point iteration algorithms for optimal transport (e.g. the Sinkhorn methods)
- Gradient-based bi-level and robust optimization (aka anti-optimization)
- Multi-parameteric programming (aka optimization sensitivity analysis)

There are 4 components to any implicit function:
1. The parameters `p`
2. The variables `x`
3. The residual `f(p, x)` which is used to define `x(p)` as the `x` which satisfies `f(p, x) == 0` for a given value `p`
4. The algorithm used to evaluate `x(p)` satisfying the condition `f(p, x) == 0`

In order to define a differentiable implicit function using `NonconvexUtils`, you have to specify the "forward" algorithm which finds `x(p)`. For instance, consider the following example:
```julia
using SparseArrays, NLsolve, Zygote, NonconvexUtils

N = 10
A = spdiagm(0 => fill(10.0, N), 1 => fill(-1.0, N-1), -1 => fill(-1.0, N-1))
p0 = randn(N)

f(p, x) = A * x + 0.1 * x.^2 - p
function forward(p)
  # Solving nonlinear system of equations
  sol = nlsolve(x -> f(p, x), zeros(N), method = :anderson, m = 10)
  # Return the zero found (ignore the second returned value for now)
  return sol.zero, nothing
end
```
`forward` above solves for `x` in the nonlinear system of equations `f(p, x) == 0` given the value of `p`. In this case, the residual function is the same as the function `f(p, x)` used in the forward pass. One can then use the 2 functions `forward` and `f` to define an implicit function using:
```julia
imf = ImplicitFunction(forward, f)
xstar = imf(p0)
```
where `imf(p0)` solves the nonlinear system for `p = p0` and returns the zero `xstar` of the nonlinear system. This function can now be part of any arbitrary Julia function differentiated by Zygote, e.g. it can be part of an objective function in an optimization problem using gradient-based optimization:
```julia
obj(p) = sum(imf(p))
g = Zygote.gradient(obj, p0)[1]
```

In the implicit function's adjoint rule definition, the partial Jacobian `∂f/∂x` is used according to the implicit function theorem. Often this Jacobian or a good approximation of it might be a by-product of the `forward` function. For example when the `forward` function does an optimization using a BFGS-based approximation of the Hessian of the Lagrangian function, the final BFGS approximation can be a good approximation of `∂f/∂x` where the residual `f` is the gradient of the Lagrangian function wrt `x`. In those cases, this Jacobian by-product can be returned as the second argument from `forward` instead of `nothing`.

### Implicit parameters

In some cases, it may be more convenient to avoid having to specify `p` as an explicit argument in `forward` and `f`. The following is also valid to use and will give correct gradients with respect to `p`:
```julia
function obj(p)
  N = length(p)
  f(x) = A * x + 0.1 * x.^2 - p
  function forward()
    # Solving nonlinear system of equations
    sol = nlsolve(f, zeros(N), method = :anderson, m = 10)
    # Return the zero found (ignore the second returned value for now)
    return sol.zero, nothing
  end
  imf = ImplicitFunction(forward, f)
  return sum(imf())
end
g = Zygote.gradient(obj, p0)[1]
```
Notice that `p` was not an explicit argument to `f` or `forward` in the above example and that the implicit function is called using `imf()`. Using some explicit parameters and some implicit parameters is also supported.

### Arbitrary data structures

Both `p` and `x` above can be arbitrary data structures, not just arrays of numbers.

### Tolerance

The implicit function theorem assumes that some conditions `f(p, x) == 0` is satisfied. In practice, this will only be approximately satisfied. When this condition is violated, the gradient reported by the implicit function theorem cannot be trusted since its assumption is violated. The maximum tolerance allowed to "accept" the solution `x(p)` and the gradient is given by the keyword argument `tol` (default value is `1e-5`). When the norm of the residual function `f(p, x)` is greater than this tolerance, `NaN`s  are returned for the gradient instead of the value computed via the implicit function theorem. If additionally, the keyword argument `error_on_tol_violation` is set to `true` (default value is `false`), an error is thrown if the norm of the residual exceeds the specified tolerance `tol`.
