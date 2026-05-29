# Literature Connections: NicheLens-ST

NicheLens-ST centers on contrastive graph learning of conserved and sample-specific cellular niches in spatial transcriptomics. The following SOTA (2025-2026) papers in the local collection are directly relevant to NicheLens-ST's subgraph formulation, contrastive objectives, and niche interpretations.

## 🔗 Primary Connections to NicheLens-ST

1. **HEIST (Madhu et al., 2025)**
   - *Topic:* Hierarchical Graph Foundation Model for Spatial Transcriptomics and Proteomics.
   - *Relationship:* HEIST models tissues as hierarchical graphs (high-level cell graphs and low-level co-expression networks). NicheLens-ST uses cell-centered neighborhood subgraphs as niche tokens. HEIST's hierarchical message-passing establishes the SOTA representation contract for subgraphs of this scale.

2. **SToFM (Zhao et al., 2025)**
   - *Topic:* Multi-scale Spatial Transcriptomics Foundation Model.
   - *Relationship:* SToFM extracts macro-scale tissue morphology, micro-scale cellular microenvironments, and gene-scale expression. NicheLens-ST's objective of separating conserved prototypes from sample-specific microenvironment variants directly aligns with SToFM's micro-scale cellular niche modeling.

3. **SAGE-FM (Zhan et al., 2026)**
   - *Topic:* Lightweight and interpretable spatial transcriptomics foundation model.
   - *Relationship:* SAGE-FM uses GCNs trained with a masked-central-spot prediction objective. NicheLens-ST builds on top of GCN/subgraph encoders to construct interpretable markers and neighborhood interaction summaries.

## 📝 BibTeX Keys to Cite in Manuscript
- `Madhu2025HEIST`
- `Zhao2025SToFM`
- `Zhan2026SAGEFM`
