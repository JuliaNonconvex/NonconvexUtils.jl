# Migration analysis: AbstractDifferentiation → DifferentiationInterface

This document analyzes what it would take to replace
[`AbstractDifferentiation.jl`](https://github.com/JuliaDiff/AbstractDifferentiation.jl)
(abbreviated AD) with
[`DifferentiationInterface.jl`](https://github.com/JuliaDiff/DifferentiationInterface.jl)
(abbreviated DI) in NonconvexUtils.jl, retiring `abstractdiffy` in favor of a
`diffiy` wrapper. It is a design analysis, not a completed migration.

## Background: the two packages

**AbstractDifferentiation** is an earlier backend-agnostic AD interface. Backend
objects are `AD.ForwardDiffBackend()`, `AD.ReverseDiffBackend()`,
`AD.TrackerBackend()`, `AD.ZygoteBackend()`. Operators include
`AbstractDifferentiation.value_and_jacobian(backend, f, x)`. It is lightly
maintained.

**DifferentiationInterface** is the actively-developed JuliaDiff successor.
Backend objects come from
[`ADTypes.jl`](https://github.com/SciML/ADTypes.jl): `AutoForwardDiff()`,
`AutoReverseDiff()`, `AutoTracker()`, `AutoZygote()`, `AutoEnzyme()`, etc.
Operators include `value_and_jacobian(f, backend, x)`, `gradient`,
`jacobian`, `pushforward`, `pullback`, plus a `prepare_*` mechanism that
returns a `prep` object to speed up repeated differentiation calls. It also
ships `DifferentiateWith(f, backend)`, a function wrapper that defines a chain
rule so the wrapped function is differentiable from an outer AD (Zygote,
ForwardDiff, Mooncake, any ChainRules-importing backend) by calling the
substitute `backend` under the hood — the direct spiritual successor to
`AbstractDiffFunction` + its `rrule`/`frule`.

The DI authors' stated plan is that AD would wrap DI for the simple cases. AD
has not meaningfully done so, which makes direct DI adoption the cleaner path.

## Current state: where AD is used

1. **`src/NonconvexUtils.jl`** — `using AbstractDifferentiation`,
   `const AD = AbstractDifferentiation`.
2. **`src/abstractdiff.jl`** — the core file:
   - `AbstractDiffFunction{F,B}` holding `f` + an AD `backend`.
   - `ForwardDiffFunction(f) = AbstractDiffFunction(f, AD.ForwardDiffBackend())`.
   - `ChainRulesCore.rrule` / `frule` on `AbstractDiffFunction` calling
     `AbstractDifferentiation.value_and_jacobian(f.backend, f.f, x)`.
   - `@ForwardDiff_frule` so ForwardDiff `Dual`s propagate through.
   - `forwarddiffy(f_or_m, x...) = abstractdiffy(f_or_m, AD.ForwardDiffBackend(), x...)`.
   - `abstractdiffy(f, backend, x...)`: flattens arbitrary inputs via
     `NonconvexCore.tovecfunc`, wraps the flat vector function in
     `AbstractDiffFunction`, returns a closure that unflattens the result.
   - `abstractdiffy(model::AbstractModel, backend; ...)`: recurses over a
     Nonconvex model's objective / ineq / eq / sd constraints, wrapping each.
3. **`src/implicit.jl`** — two `# Change this to AbstractDifferentiation`
   comments next to hardcoded `Zygote.jacobian(_conditions_y, flat_ystar)[1]`
   calls (lines 71–72, 129–130). These are pre-existing TODOs from the author
   to route the implicit-function-theorem Jacobian through a backend abstraction
   instead of pinning Zygote.
4. **`Project.toml`** — `AbstractDifferentiation = "0.6"` in `[deps]` and
   `[compat]`.
5. **`test/abstractdiff.jl`** — exercises `forwarddiffy` and `abstractdiffy`
   with `AD.ForwardDiffBackend()`, `AD.ReverseDiffBackend()`,
   `AD.TrackerBackend()`. Crucially, it asserts which number type the inner
   function sees (`ForwardDiff.Dual`, `ReverseDiff.TrackedReal`,
   `Tracker.TrackedReal`) — this is the behavioral spec the migration must
   preserve.
6. **`src/sparse_forwarddiff.jl`** — does *not* use AD; it hand-rolls sparse
   Jacobians/Hessians via `SparseDiffTools` + `Symbolics.jacobian_sparsity`.
   It is commented out of the test suite (`# include("sparse_forwarddiff.jl")`)
   and is effectively dead code. It is relevant only because DI now ships
   built-in sparse AD (`AutoSparse`) that could replace it; see the deferred
   section below.

`src/custom.jl`, `src/lazy.jl`, `src/trace.jl`, `src/symbolic.jl`, and
`src/forwarddiff_frule.jl` do not reference AD and are unaffected.

## What DI offers that maps onto this

| Current (AD) | DI replacement |
| --- | --- |
| `AD.ForwardDiffBackend()` | `AutoForwardDiff()` |
| `AD.ReverseDiffBackend()` | `AutoReverseDiff()` |
| `AD.TrackerBackend()` | `AutoTracker()` |
| `AD.ZygoteBackend()` | `AutoZygote()` |
| `AbstractDifferentiation.value_and_jacobian(backend, f, x)` | `DifferentiationInterface.value_and_jacobian(f, backend, x)` |
| `AbstractDiffFunction` + hand-written `rrule`/`frule` | `DifferentiateWith(f, backend)` (defines the chain rule for you) |
| `abstractdiffy(f, backend, x...)` | a thin `diffiy(f, backend, x...)` keeping the `tovecfunc` flatten/unflatten scaffolding |
| `forwarddiffy(f, x...)` | `diffiy(f, AutoForwardDiff(), x...)` (or keep `forwarddiffy` as an alias) |
| re-compute Jacobian every call | `prepare_jacobian` + `jacobian(f, prep, backend, x)` for repeated calls (opt-in perf) |
| `SparseForwardDiffFunction` / `sparsify` (custom, untested) | `AutoSparse(AutoForwardDiff(); sparsity_detector=TracerSparsityDetector(), coloring_algorithm=GreedyColoringAlgorithm())` + `jacobian`/`hessian` |

## The migration, by concern

### A. `abstractdiff.jl` — the core replacement

**`AbstractDiffFunction` → `DifferentiateWith` (mostly).**

`AbstractDiffFunction`'s job is: wrap `f` with a `backend` and make it
differentiable from any ChainRules-consuming outer AD by defining `rrule`/`frule`
that call `value_and_jacobian` on the backend. `DifferentiateWith(f, backend)`
does exactly this for the reverse-mode case and defines the chain rule
automatically.

Things to reconcile:

1. **Vector-only vs. any input.** `AbstractDiffFunction`'s rules are defined
   for `x::AbstractVector`. `DifferentiateWith` supports single-arg
   `y = f(x)`. The vector restriction carries over cleanly.
2. **`frule` / forward-mode.** `AbstractDiffFunction` defines both `rrule` and
   `frule`, plus `@ForwardDiff_frule` so ForwardDiff `Dual`s propagate through.
   `DifferentiateWith` documents support for outer ForwardDiff (forward-mode),
   but whether it defines an `frule` or relies on ForwardDiff treating the
   wrapper as opaque needs verification. The existing forward-mode tests are
   the check.
3. **The `eltype` tests — the single biggest risk.** `test/abstractdiff.jl`
   asserts that the inner `f` sees the substitute backend's number types
   (`ForwardDiff.Dual` / `ReverseDiff.TrackedReal` / `Tracker.TrackedReal`).
   `AbstractDiffFunction` guarantees this because `value_and_jacobian` runs `f`
   under the backend's number types. `DifferentiateWith`'s rule *might*
   instead compute the derivative via `value_and_pushforward`/`value_and_pullback`
   and evaluate `f` at primals only, in which case `eltype(x)` inside `f` would
   be `Float64` and the tests would fail. This is a *semantic* difference, not
   just a wiring difference. **It must be de-risked with a spike before
   committing to `DifferentiateWith`.** If it differs, `diffiy` should keep the
   custom `rrule`/`frule`-defining wrapper pattern and merely swap
   `AbstractDifferentiation.value_and_jacobian` for
   `DifferentiationInterface.value_and_jacobian` underneath — still a valid
   migration (removes the AD dep) that preserves semantics.

**`abstractdiffy` / `forwarddiffy` → `diffiy`.**

`abstractdiffy` does two things `DifferentiateWith` does not, both of which
`diffiy` must keep:

1. **Flattening arbitrary inputs.** `tovecfunc(f, x...)` produces a flat vector
   function + an unflatten for results; the AD wrapper is applied to the flat
   function; the returned closure unflattens. `DifferentiateWith` only handles
   `f(x::Vector)`, so the flatten/unflatten scaffolding stays.
2. **Model walking.** `abstractdiffy(model::AbstractModel, ...)` recurses over
   objective / ineq / eq / sd constraints. This is Nonconvex-specific and
   orthogonal to either AD library; the body stays, with
   `abstractdiffy(c, backend, x)` → `diffiy(c, backend, x)`.

Sketch:

```julia
function diffiy(f, backend, x...)
    flat_f, _, unflatteny = tovecfunc(f, x...)
    di_flat_f = DifferentiateWith(flat_f, backend)   # or custom wrapper, see risk #3
    return (x...,) -> unflatteny(di_flat_f(flatten(x)[1]))
end
diffiy(f_or_m, x...) = diffiy(f_or_m, AutoForwardDiff(), x...)
```

**Naming / API break.** `abstractdiffy`, `forwarddiffy`, `AbstractDiffFunction`,
`ForwardDiffFunction` are all exported. Retiring them is a breaking change:
either a major version bump, or a deprecation cycle where the old names alias
forward to the new implementation (mapping the three AD backends to their
ADTypes equivalents: `AD.ForwardDiffBackend()`→`AutoForwardDiff()`,
`AD.ReverseDiffBackend()`→`AutoReverseDiff()`, `AD.TrackerBackend()`→`AutoTracker()`).

### B. `implicit.jl` — the `# Change this to AbstractDifferentiation` TODOs

```julia
# Change this to AbstractDifferentiation
dfdy = Zygote.jacobian(_conditions_y, flat_ystar)[1]
```

These compute the Jacobian of the conditions function w.r.t. `y` for the
implicit-function-theorem linear solve. They are hardcoded to Zygote in the
dense branch (the `matrixfree` branch already uses `rrule_via_ad`). DI's
`jacobian(_conditions_y, backend, flat_ystar)` is the direct replacement and
makes the backend configurable (default `AutoZygote()` to preserve current
behavior).

**This is the lowest-risk, highest-value change.** It touches no public API,
just swaps an internal `Zygote.jacobian` for `DifferentiationInterface.jacobian`
with a configurable backend, and it fulfills the original author's TODO. The
only thing to verify is the return-shape convention: `Zygote.jacobian(f, x)`
returns a 1-tuple-ish `(J,)`; DI's `jacobian` returns `J` directly — a one-line
adjustment.

### C. `Project.toml` / deps

- Add `DifferentiationInterface` to `[deps]` and `[compat]`. Add `ADTypes` if
  re-exporting backend constructors.
- Either remove `AbstractDifferentiation` cleanly or keep it for one
  deprecation release.
- DI pulls in `ADTypes` plus extension packages for whichever backends users
  actually load (ForwardDiff, Zygote, etc. — already deps), so the dependency
  footprint does not grow meaningfully.

### D. Tests

`test/abstractdiff.jl` is the spec. Required updates:

- Keep the `eltype` assertions — they validate that the wrapper delegates to
  the substitute backend's number types. If they fail against `DifferentiateWith`,
  that is the signal to fall back to the custom-wrapper strategy (risk #3).
- Rename `abstractdiffy` → `diffiy`, swap `AD.ReverseDiffBackend()` →
  `AutoReverseDiff()`, `AD.TrackerBackend()` → `AutoTracker()`.
- The model test (`forwarddiffy(m)` + IPOPT solve) should keep working if
  `forwarddiffy` is preserved as an alias.

### E. Deferred: `sparse_forwarddiff.jl`

This file (~250 lines) hand-rolls sparse Jacobians/Hessians with
`SparseDiffTools.forwarddiff_color_jacobian` + `Symbolics.jacobian_sparsity`
+ manual caching in a large `struct`, and defines its own
`rrule`/`frule`/`@ForwardDiff_frule`. DI's `AutoSparse` +
`jacobian`/`hessian` does this built-in and backend-agnostically, and could
delete most of the file. But:

- It is commented out of the test suite, so it is effectively untested today.
- A DI-based replacement needs equivalent chain rules, or composition with
  `DifferentiateWith` over a sparse backend.
- The custom caching (`nograd_cache!`, reusing `f.jac`) maps onto DI's `prep`
  objects but the wiring differs.

Worth a follow-up PR on its own, after the core migration. Not a blocker.

## Behavioral / semantic risks to de-risk

1. **Does `DifferentiateWith` make `f` see the substitute backend's number
   types?** The `eltype` tests pin this. Verify by porting one test first.
   This is the go/no-go experiment for using `DifferentiateWith` vs. keeping a
   custom wrapper around `DifferentiationInterface.value_and_jacobian`.
2. **`frule` / ForwardDiff-Dual propagation.** The `@ForwardDiff_frule` macro
   lets `Dual`s flow through `AbstractDiffFunction` by calling its `frule`.
   Confirm `DifferentiateWith` composes correctly with an outer ForwardDiff
   (Dual in → Dual out) via the forward-mode tests.
3. **`DifferentiateWith`'s outer-backend restrictions.** It works when the
   *outer* AD is ForwardDiff, reverse-mode Mooncake, or a ChainRules-importing
   backend (Zygote). The current tests only cover outer Zygote and outer
   ForwardDiff, so the supported set lines up. If anyone relies on
   `abstractdiffy` under an outer Tracker/ReverseDiff, `DifferentiateWith` may
   be transparent (no rule) there — check whether that is a regression.
4. **Preparation / caching.** `AbstractDiffFunction` re-computes the Jacobian
   on every call. DI `prep` objects can prepare once and reuse, but they depend
   on `x`'s type/shape, so a stateless `diffiy` closure would have to store or
   recompute `prep`. This is an optimization for later, not required for parity.

## Recommended phasing

1. **Phase 1 (low risk, internal): `implicit.jl`.** Replace the two
   `Zygote.jacobian` calls with `DifferentiationInterface.jacobian(..., backend, ...)`,
   default `AutoZygote()`, making the backend a configurable
   `ImplicitFunction` keyword. Add `DifferentiationInterface` + `ADTypes` deps.
   No public API change; fulfills the existing TODO. Verify with the existing
   implicit tests.
2. **Phase 2 (de-risk): spike `diffiy` with one test.** Port the
   "Scalar-valued reverse-mode" test to a `diffiy` prototype on
   `DifferentiateWith`. Confirm the `eltype` assertion holds, or determine it
   does not and switch to the custom-wrapper-around-DI strategy.
3. **Phase 3 (public API): `diffiy` + deprecate `abstractdiffy`.** Implement
   `diffiy`, keep `abstractdiffy`/`forwarddiffy`/`AbstractDiffFunction`/
   `ForwardDiffFunction` as deprecation aliases for one release, then remove.
   Bump major version. Rewrite `src/abstractdiff.jl` to use DI under the hood.
4. **Phase 4 (optional, separate PR): `sparse_forwarddiff.jl`.** Replace with
   `AutoSparse` + DI `jacobian`/`hessian`, re-enable in the test suite.

## Estimated effort

- Phase 1: small — a few lines, deps, tests pass. ~1 hour including verification.
- Phase 2: small spike — the deciding experiment. ~1 hour.
- Phase 3: medium — rewrite `abstractdiff.jl` (~100 lines), update tests,
  deprecation aliases, version bump, `[compat]` entries. ~half a day.
- Phase 4: medium-large — rewrite ~250 lines, re-enable tests, validate. ~a day
  or more, and the code is currently untested so validation is mostly "does it
  work at all".

Core migration (Phases 1–3): roughly a day of focused work, dominated by
de-risking the `DifferentiateWith` semantics and updating tests.
