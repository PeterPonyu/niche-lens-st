# NicheLens-ST synthetic niche-recovery benchmark

Status: spec document. Defines how evidence will be collected. No performance is reported here.

## Generator

Synthetic instances have `K_conserved` niche prototypes shared across all sections and `J_specific` variants present in a subset of sections.

### Generator parameters

| Param | Default | Range | Notes |
|---|---|---|---|
| `n_sections` | 4 | 2-16 | Sections per instance. |
| `n_cells_per_section` | 2000 | 500-10000 | Cells per section. |
| `n_genes` | 500 | 200-2000 | Synthetic gene panel size. |
| `K_conserved` | 6 | 2-20 | Prototypes shared across all sections. |
| `J_specific` | 2 | 0-10 | Variants present in a subset of sections. |
| `noise_sigma` | 0.5 | 0-2 | Gaussian noise on expression. |
| `seed` | 0 | any int | Deterministic regeneration. |

### Saved artifacts (per instance)

| Path | Contents |
|---|---|
| `X.h5` | Cell-by-gene expression. |
| `coords.npy` | Per-cell 2D coordinates. |
| `section_id.npy` | Per-cell section index. |
| `edges.npy` | k-NN edges (COO). |
| `truth/prototype_id.npy` | Ground-truth per-cell prototype. |
| `truth/proto_kind.json` | `conserved` / `sample_specific` tag per prototype. |
| `truth/marker_genes.json` | Per-prototype marker panel. |

## Metrics

| Metric | What it locks | Pass gate |
|---|---|---|
| ARI(`prototype_id`, truth) | Niche recovery vs ground truth. | set when first run lands |
| Moran's I over `prototype_id` | Spatial coherence of recovered niches. | set when first run lands |
| Section-overlap rate per `proto_kind` | Conserved vs sample-specific tagging accuracy. | set when first run lands |
| Marker recall@k vs truth | Marker recovery. | set when first run lands |

## Planned test placeholders

- `tests/synth/test_generator_determinism.py`
- `tests/synth/test_niche_recovery_metrics.py`
- `tests/synth/test_spatial_coherence_metrics.py`
- `tests/synth/test_proto_kind_metrics.py`

## Out of scope

- Real-data execution.
- Comparison runs against CellNiche or scComm (deferred; see `BASELINE_REFERENCES.md`).
- Performance claims.
