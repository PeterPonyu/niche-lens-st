# NicheLens-ST environment and smoke-test contract

## Packaging decision

The first implementation pass will use `pyproject.toml` (PEP 621) as the single source of truth for runtime deps, the `test` optional-dependency group, and tool pins. A separate `requirements.txt` is not planned.

A pinned environment file is deferred until the first implementation PR. The scaffold phase remains dependency-free.

## Planned runtime deps (not yet pinned)

| Package | Purpose | Planned constraint |
|---|---|---|
| `numpy` | tensors | `>=1.26,<3` |
| `scipy` | sparse graph ops | `>=1.11,<2` |
| `anndata` | input container | `>=0.10,<1` |
| `torch` | model backbone | `>=2.1` |
| `pytest` | tests | `>=8,<9` |

Constraints become enforceable when the first code lands.

## First smoke-test command

    pytest tests/smoke -q

The smoke suite must:

- Import the top-level package with no side effects.
- Build a 100-cell synthetic graph from the generator defined in `docs/SYNTHETIC_BENCHMARK.md`.
- Run one forward pass producing the output schemas defined in `docs/MVP_DESIGN.md`.
- Complete in under 30 seconds on CPU.

The smoke test locks shape and dtype only. No performance metric is asserted.
