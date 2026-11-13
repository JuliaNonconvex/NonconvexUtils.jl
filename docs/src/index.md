# NonconvexUtils.jl

> **This package has been absorbed into
> [NonconvexCore.jl](https://github.com/JuliaNonconvex/NonconvexCore.jl)
> as of NonconvexCore 2.0.0.**

The utilities that used to live here (`diffiy`, `TraceFunction`,
`CustomGradFunction`, `CustomHessianFunction`, `LazyJacobian`,
`ImplicitFunction`, `symbolify`, `sparsify`, ...) are now part of
NonconvexCore:

- the generic utilities are in NonconvexCore proper, and
- `symbolify` is provided by NonconvexCore's `NonconvexCoreSymbolicsExt`
  extension (load Symbolics to activate it).

This package now only re-exports NonconvexCore, so existing code with
`using NonconvexUtils` continues to work unchanged. New code should use
`using NonconvexCore` directly. See the
[Nonconvex.jl documentation](https://julianonconvex.github.io/Nonconvex.jl/).
