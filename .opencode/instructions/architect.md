# Architecture guidance

Guidance for planning, designing, or strategizing before implementation in
this repository.

## Project scope

NonconvexUtils.jl is a collection of automatic-differentiation utilities for
the Nonconvex.jl ecosystem. It provides function wrappers that make
non-differentiable or awkwardly-differentiable code compose cleanly with
ChainRulesCore-based AD (Zygote, ForwardDiff, Tracker, ReverseDiff, …), plus
helpers for symbolic regression of gradients/Hessians and implicit-function
differentiation through iterative solvers.

- **Language**: Julia (≥ 1.0 per `Project.toml`). The LTS release is currently
  1.10; prefer lower bounds in `[compat]` compatible with the LTS where
  possible.
- See `memory-bank/systemPatterns.md` for the full architecture reference:
  module structure, key design patterns, wrapper catalog, dependencies, and
  usage examples. `memory-bank/` is git-ignored and kept up to date as the
  project evolves.

## Key architectural decisions

- **Wrapper pattern**: each utility is a `struct <: Function` that wraps an
  underlying callable plus AD-relevant metadata (a backend, a custom
  gradient/Hessian, a trace buffer, a forward/conditions pair, …).
  Differentiability is supplied by defining `ChainRulesCore.rrule` and (where
  useful) `frule` on the wrapper, not by intercepting the wrapped callable.
- **Backend-agnostic AD**: `diffiy` wraps a function with a
  `DifferentiationInterface` backend (an ADTypes backend such as
  `AutoForwardDiff()`, `AutoZygote()`, `AutoReverseDiff()`, `AutoTracker()`)
  via `DifferentiationInterface.DifferentiateWith`, so the same wrapper works
  with ForwardDiff, Zygote, Tracker, etc. via ChainRulesCore. `ImplicitFunction`
  likewise takes a `jac_backend` keyword (default `AutoZygote()`) for its
  conditions Jacobian. Do not hardcode a backend where a parameter will do.
- **Implicit function theorem**: `ImplicitFunction` differentiates through a
  fixed-point/iterative solve by solving `df/dy' \ v` with a configurable
  `linear_solver`. The `matrixfree` type parameter switches between a dense
  Jacobian and a `LinearMap`-based operator. The `tol` guard fails fast when
  the residual is too large.
- **Lazy Jacobians**: `LazyJacobian` stores only `jvp`/`jtvp` operators and
  never materializes the full matrix. The `symmetric` type parameter lets one
  operator serve both roles.
- **Flattening integration**: wrappers route arbitrary inputs through
  `NonconvexCore.flatten` / `tovecfunc` so backends that need flat vectors can
  still differentiate functions with structured inputs, with results and
  gradients unflattened back.
- **ForwardDiff Dual composition**: the `@ForwardDiff_frule` macro bridges a
  ChainRulesCore `frule` with ForwardDiff `Dual` propagation, so forward-mode
  works through wrappers that only define an `frule`.

## Design principles

- **Fail-fast** over silently trying to continue. Surface unexpected
  conditions so they can be inspected and understood (e.g. `ImplicitFunction`'s
  tolerance guard, `LazyJacobian`'s "both operators cannot be nothing" check).
- Use American spellings. Avoid jargon and metaphors not widely accepted by
  experts in the field. Do not make technical prose sound like a pitch deck.
- Favor **composability**: every new wrapper should be a `Function` subtype
  with a `ChainRulesCore.rrule` (and `frule` where forward-mode is supported),
  so it composes with the rest of the Nonconvex.jl AD stack and with any AD
  system that consumes ChainRulesCore.
- Prefer **unconstrained type parameters** in `struct` constructors (see
  `code.md` for the full cascade pattern). Do not over-constrain method
  signatures — annotate only as specifically as the implementation requires.
- Preserve **behavioral type parameters** (`matrixfree`, `symmetric`, backend
  type, etc.) across construction and dispatch; they encode contracts the rest
  of the code relies on.
- When adding new packages to the local project, also update the `[compat]`
  section of `Project.toml` to bound the version of the new dependency. After
  editing `Project.toml`, run `Pkg.resolve()`.

## Memory Bank

This project uses a **Memory Bank** (`memory-bank/`, git-ignored) to preserve
context across agent sessions. The agent MUST read ALL memory bank files at
the start of EVERY task. When designing, update `systemPatterns.md` and
`activeContext.md` to reflect architectural decisions. See the root `AGENTS.md`
for the full Memory Bank structure.
