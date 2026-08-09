# Explanation guidance

Guidance for providing explanations, documentation, or answers to technical
questions about this repository.

## Stance

- Use American spellings. Avoid jargon and metaphors not widely accepted by
  experts in the field. Do not make technical prose sound like a pitch deck.
- Favor **fail-fast** over silently trying to continue — when explaining code,
  point out where it silently continues vs. where it surfaces errors.
- Be precise about what the code *does now*, not what it was intended to do or
  what it might do in the future. Cite the actual source.

## How to answer

- Ground explanations in the actual source. Reference files and line numbers
  when describing behavior.
- For architecture questions, consult `memory-bank/systemPatterns.md` (the
  full architecture reference) and the root `AGENTS.md` (project context and
  conventions).
- For Julia-specific questions (indexing, `@inbounds`, style), consult
  `code.md`.
- For debugging questions, consult `debug.md`.
- For design questions, consult `architect.md`.

## Key concepts in NonconvexUtils.jl

- **`AbstractDiffFunction` / `forwarddiffy` / `abstractdiffy`**: wrap a
  function with an `AbstractDifferentiation` backend and expose it via
  `ChainRulesCore.rrule`/`frule` (`src/abstractdiff.jl`).
- **`CustomGradFunction` / `CustomHessianFunction`**: supply a user-provided
  gradient (and Hessian) and wire them into `rrule` so a function with a known
  derivative still composes with AD (`src/custom.jl`).
- **`LazyJacobian`**: store only `jvp`/`jtvp` operators (`symmetric` allows one
  operator to serve both); never materialize the full matrix (`src/lazy.jl`).
- **`TraceFunction`**: record input/output (and optionally gradient) into a
  trace buffer on each call or `rrule` (`src/trace.jl`).
- **`ImplicitFunction`**: differentiate through a fixed-point/iterative solve
  via the implicit function theorem; solves `df/dy' \ v` with a configurable
  `linear_solver`, `matrixfree` switches to a `LinearMap` operator, and a `tol`
  guard fails fast on a too-large residual (`src/implicit.jl`).
- **`SymbolicFunction` / `symbolify`**: build a symbolic expression for the
  function (and gradient/Hessian) via Symbolics.jl, then compile it to an
  evaluatable form (`src/symbolic.jl`).
- **`@ForwardDiff_frule`**: macro that composes a `ChainRulesCore.frule` with
  ForwardDiff `Dual` propagation, so forward-mode works through wrappers that
  only define an `frule` (`src/forwarddiff_frule.jl`).
- **Flattening**: wrappers integrate with `NonconvexCore.flatten` /
  `tovecfunc` so AD backends requiring flat vectors still handle structured
  inputs, with results and gradients unflattened back.

## Julia development context

- Use the local `Project.toml` environment (`--project=.`).
- Find the source for session-loaded packages with `Pkg.pkgdir(M::Module)`.
  For packages not loaded into the session, check the active project's
  `Manifest.toml` for the path.
- Use `Revise` to amortize compilation cost. The MCP server runs
  `Revise.revise()` automatically before every eval.
- Use `Pkg.test()` for a final run only when ready to submit a pull request.
  During debugging, run a specific test file directly:
  `julia --project=. -e "include(\"test/implicit.jl\")"`.

## Memory Bank

This project uses a **Memory Bank** (`memory-bank/`, git-ignored) to preserve
context across agent sessions. The agent MUST read ALL memory bank files at
the start of EVERY task. See the root `AGENTS.md` for the full structure.
