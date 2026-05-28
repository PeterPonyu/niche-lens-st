# NicheLens-ST manuscript seed draft

## Working title

NicheLens-ST: contrastive graph learning of conserved and sample-specific cellular niches in spatial transcriptomics

## Draft abstract

Spatial transcriptomics measures cells within tissue neighborhoods, but many analyses still reduce local organization to per-cell clusters or pairwise proximity summaries. NicheLens-ST is a method for learning interpretable cellular niche representations from spatial neighborhood graphs. The model treats each cell-centered neighborhood as a subgraph, learns contrastive embeddings that separate conserved niche prototypes from sample-specific variants, and summarizes per-prototype markers and interactions. The first development phase will benchmark against public-code niche and communication references without vendoring third-party implementations. A reference implementation (a contrastive GraphSAGE encoder with a deterministic prototype assignment and a conserved/sample-specific separation head), a synthetic-benchmark harness, and tests are in place; all biological and performance claims remain pending real-data validation and baseline comparisons (see `CLAIM_LEDGER.md`).

## Proposed core contributions

1. A cell-centered spatial-neighborhood graph contract for ST data.
2. A contrastive niche embedding objective with conserved/private prototype outputs.
3. A planned validation suite for niche reproducibility, spatial coherence, marker enrichment, and communication-aware interpretation.

## Implementation status

Implemented and tested: the contrastive encoder (`src/nichelens_st/encoder.py`), deterministic k-means prototype assignment with the conserved/sample-specific separation head (`src/nichelens_st/model.py`), per-section neighborhood-graph and cell-centered subgraph construction (`src/nichelens_st/graph.py`), a synthetic niche-recovery generator (`src/nichelens_st/synth/`), and benchmark metrics (`src/nichelens_st/metrics.py`). Pending: real-data ingestion, marker and interaction summaries, and baseline comparisons. No biological or performance results are reported yet (see `CLAIM_LEDGER.md`).

## Review questions

- Which public ST datasets should define the first niche-reproducibility test?
- Should prototype discovery prioritize cell-type composition, gene programs, or ligand-receptor summaries first?
- Which metrics best separate true niche structure from generic clustering smoothness?
