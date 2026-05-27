# NicheLens-ST first-pass MVP API

Status: design document. All biology and performance claims remain planned until local tests and validated data exist (see `CLAIM_LEDGER.md`). No third-party source is vendored; CellNiche and scComm are cited only as comparison references (see `BASELINE_REFERENCES.md`).

## Inputs

| Object | Shape | Type | Notes |
|---|---|---|---|
| `X` cell features | `(n_cells, n_genes)` | float32, dense or CSR | Per-cell expression for one section. |
| `coords` | `(n_cells, 2)` or `(n_cells, 3)` | float32 | Section-local coordinates. |
| `section_id` | `(n_cells,)` | int | Section/sample index. |
| `edges` | `(2, n_edges)` | int64 COO | k-NN or Delaunay graph over `coords`, built per section. |

## Outputs

| Object | Shape | Type | Notes |
|---|---|---|---|
| `H` niche embedding | `(n_cells, d)` | float32 | Cell-centered subgraph embedding; `d` configurable. |
| `prototype_id` | `(n_cells,)` | int | Assignment to a global niche-prototype index. |
| `proto_kind` | `(n_protos,)` | enum `{conserved, sample_specific}` | Per-prototype tag. |
| `marker_table` | DataFrame | str / float | Per-prototype top-k marker genes with scores. |
| `interaction_summary` | DataFrame | str / float | Per-prototype ligand-receptor or interaction score summary. |

## Subgraph contract

For cell `i`, the cell-centered subgraph is the induced subgraph of `edges` over the k-hop neighborhood of `i` together with `i`. Default `k = 1`; configurable.

## Acceptance matrix

| Output | Claim-ledger row | Future test hook |
|---|---|---|
| `H` | "reproducible niche prototypes" | `tests/test_embedding_reproducibility.py` |
| `prototype_id`, `proto_kind` | "conserved vs sample-specific" | `tests/test_prototype_tagging.py` |
| `marker_table` | "improves interpretability over generic clustering" | `tests/test_marker_enrichment.py` |
| `interaction_summary` | "improves interpretability" | `tests/test_interaction_summary.py` |

## Out of scope for MVP

- Histology fusion.
- 3D reconstruction.
- Multi-modal (ATAC, protein) input.
- Performance comparison against baselines (deferred until validated; see `BASELINE_REFERENCES.md`).
