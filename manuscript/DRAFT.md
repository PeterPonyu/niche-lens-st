# NicheLens-ST manuscript seed draft

## Working title

NicheLens-ST: contrastive graph learning of conserved and sample-specific cellular niches in spatial transcriptomics

## Draft abstract

Spatial transcriptomics measures cells within tissue neighborhoods, but many analyses still reduce local organization to per-cell clusters or pairwise proximity summaries. NicheLens-ST is a planned method for learning interpretable cellular niche representations from spatial neighborhood graphs. The proposed model treats each cell-centered neighborhood as a subgraph, learns contrastive embeddings that separate conserved niche prototypes from sample-specific variants, and exposes marker and interaction summaries for downstream review. The first development phase will benchmark against public-code niche and communication references without vendoring third-party implementations. All biological and performance claims remain planned until local implementation, tests, and dataset validation are complete.

## Proposed core contributions

1. A cell-centered spatial-neighborhood graph contract for ST data.
2. A contrastive niche embedding objective with conserved/private prototype outputs.
3. A planned validation suite for niche reproducibility, spatial coherence, marker enrichment, and communication-aware interpretation.

## Review questions

- Which public ST datasets should define the first niche-reproducibility test?
- Should prototype discovery prioritize cell-type composition, gene programs, or ligand-receptor summaries first?
- Which metrics best separate true niche structure from generic clustering smoothness?
