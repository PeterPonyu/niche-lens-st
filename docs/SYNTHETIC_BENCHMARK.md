# NicheLens-ST synthetic niche-recovery benchmark

Status: spec document. Defines how evidence will be collected. No performance is reported here.

## Generator

Synthetic instances have `K_conserved` niche prototypes shared across all sections and `J_specific` variants present in a subset of sections. When a section has at least as many cells as its allowed prototypes, the generator seeds at least one cell for every allowed prototype before random assignment so conserved prototypes are represented in each section. Within each section the allowed prototypes are laid out as Voronoi zones around random centres -- each cell takes its nearest centre's prototype -- so niches are spatially contiguous and `prototype_id` has positive spatial autocorrelation over the kNN graph (issue #59).

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
| `k_nn` | 8 | >=0 | Per-section outgoing nearest neighbors; effective k is clamped to `min(k_nn, cells_in_section - 1)`, so singleton sections emit zero edges. |
| `n_ligrec_pairs` | 0 | >=0 | Known-positive ligand→receptor interactions to plant. `0` (default) = **CCC ground truth OFF**; every base array stays byte-identical for a fixed seed. See "CCC ground truth" below. |
| `n_ligrec_decoys` | `None` | >=0 or `None` | Negative ligand→receptor pairs. `None` defaults to `n_ligrec_pairs`. |
| `ligrec_strength` | 8.0 | >0 | Additive expression boost applied to the ligand gene in source-prototype cells and the receptor gene in target-prototype cells. |

### Ligand-receptor / cell-cell-communication (CCC) ground truth

Opt-in via `n_ligrec_pairs > 0`. The base instance (`X`, `coords`, `section_id`, `edges`, `prototype_id`, `proto_kind`, `marker_genes`, `proto_means`) is built first and the CCC signal is planted afterward, drawing from the RNG only at the end — so with `n_ligrec_pairs == 0` the generator is byte-identical to before for a fixed seed, and enabling it perturbs **only** `X` at the planted `(prototype, gene)` entries.

Each planted interaction is a `(ligand_gene, receptor_gene, source_proto, target_proto)` integer tuple, mirroring the `(ligand, receptor, source, target)` key the real pipeline emits (`communication.INTERACTION_SUMMARY_COLUMNS`). Ligand/receptor genes are disjoint from every marker panel and from one another, so the CCC signal never collides with the niche-recovery ground truth.

- **Positives** (`ligrec_truth`): `source_proto`/`target_proto` are conserved prototypes that are spatially **adjacent** over the kNN graph (≥1 directed `source→target` edge); the ligand gene is up-regulated in source cells and the receptor gene in target cells. A proximity + co-expression detector should recover them.
- **Decoys** (`ligrec_decoys`): negatives a detector should rank **below** the positives. Two flavours — *elevated-but-not-colocated* (same up-regulation but between a **non**-adjacent prototype pair, so spatial co-expression is ~0) and, when contiguous Voronoi zones leave too few non-adjacent pairs, *random unboosted* candidates (distinct genes, no planted signal, baseline score).

Restricting source/target to conserved prototypes keeps them present in every section, so adjacency is stable across sections.

| `SynthInstance` field | Type | Contents |
|---|---|---|
| `ligrec_truth` | `list[tuple[int,int,int,int]] \| None` | Planted positive `(ligand, receptor, source, target)` tuples; `None` when CCC is off. |
| `ligrec_decoys` | `list[tuple[int,int,int,int]] \| None` | Negative `(ligand, receptor, source, target)` tuples; `None` when CCC is off. |

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
| ARI(`prototype_id`, truth) | Niche recovery vs ground truth (pair-counting). | set when first run lands |
| NMI / homogeneity / completeness / V-measure | Information-theoretic agreement. Under the fixed cluster-count mismatch (model `n_prototypes` ≠ truth), homogeneity drops when clusters mix niches and completeness drops when a niche is split across clusters, so the decomposition distinguishes over- vs under-segmentation that ARI alone conflates. | set when first run lands |
| neighbor label agreement over `prototype_id` | Spatial coherence of recovered niches: fraction of graph edges joining same-prototype cells (label-numbering invariant, unlike Moran's I on nominal codes). | set when first run lands |
| proto_kind accuracy vs truth | Predicted conserved/sample_specific tags vs ground-truth `proto_kind` after Hungarian prototype matching (not the circular `section_overlap_rate` self-check). | set when first run lands |
| Section-overlap rate per `proto_kind` | Conserved vs sample-specific tagging accuracy. | set when first run lands |
| Marker recall@k vs truth | Marker recovery. | set when first run lands |
| CCC top-k detection (`synth.ccc.score_ccc_topk`) | Ligand-receptor recovery: ranks a method's detected interactions against the planted `ligrec_truth`. Returns `precision_at_k`, `recall_at_k`, `hit_rate` (== recall), `n_true_recovered`. | set when first run lands |

### CCC top-k detection scorer

`score_ccc_topk(pred_ranked, truth, k)` (in `nichelens_st.synth.ccc`) scores a method's ranked ligand-receptor predictions (best first) against the known positives. Each interaction is a `(ligand, receptor[, source, target])` tuple; the scorer compares normalized tuples, so integer gene/prototype ids (synthetic truth) and string symbols (a real `squidpy.gr.ligrec` run) both work, provided `pred_ranked` and `truth` use the same key arity. Predictions are de-duplicated (first/best occurrence kept) before the top-`k` cut.

Edge-case discipline mirrors `metrics.py` (issue #83 family): empty `truth` or empty `pred_ranked` → all rates `NaN` (never a free `1.0`) with `n_true_recovered == 0`; `k < 1` or malformed entries → `ValueError`. Pure numpy/stdlib, no sklearn.

## Planned test placeholders

- `tests/synth/test_generator_determinism.py`
- `tests/synth/test_niche_recovery_metrics.py`
- `tests/synth/test_spatial_coherence_metrics.py`
- `tests/synth/test_spatial_coherence_of_generated_truth.py`
- `tests/synth/test_proto_kind_metrics.py`
- `tests/synth/test_ccc_truth.py`
- `tests/synth/test_ccc_scorer.py`

## Out of scope

- Real-data execution.
- Comparison runs against CellNiche or scComm (deferred; see `BASELINE_REFERENCES.md`).
- Performance claims.
