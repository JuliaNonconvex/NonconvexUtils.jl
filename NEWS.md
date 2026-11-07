# NEWS

## 1.1.2

- Added tests for the previously untested `@ForwardDiff_frule` bridge cases:
  vector-valued outputs, struct-valued outputs and callable-struct functors.

## 1.1.1

- `TraceFunction` reverse-mode rrule no longer crashes for scalar or struct
  inputs: the cotangent is unthunked instead of reaching into internal thunk
  fields.
- `TraceFunction` forward-mode frules no longer crash when a primal call is
  followed by AD at the same point: trace entries without a `grad` field are
  no longer merged into.
- `CustomGradFunction` evaluates the user gradient once in the forward pass
  instead of once per pullback application.
- `diffiy` preserves the objective `multiple` like `symbolify` does.

## 1.1.0

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
