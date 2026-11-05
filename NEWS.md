# NEWS

## 1.0.1

- Fixed Zygote 0.7 compatibility: rrule pullbacks now unthunk thunked
  cotangents (`InplaceableThunk`/`Thunk`) that Zygote 0.7 hands to rules,
  including nested inside `Tangent` fields. This restores `ImplicitFunction`
  gradients and the custom gradient/Hessian wrappers on the latest Julia
  stack.
- `nanlike` is now ChainRules-aware and walks `Tangent` structures instead of
  flattening them.
- Symbolics compatibility widened to 6 and 7 (issue #17); tests pass
  unchanged with Symbolics 7.
- Fixed stale `DifferentiableFlatten` test-only compat bound.
- Requires Julia 1.10+.
