#!/usr/bin/env python
"""Fit the model and score its output against synthetic truth (#81/#82).

Before this fix the harness scored truth-vs-truth and never invoked
``fit_niche_model``. Requires the ``[model]`` extra (torch).
"""

from __future__ import annotations

from nichelens_st.metrics import score_against_truth
from nichelens_st.model import NicheModelConfig, fit_niche_model
from nichelens_st.synth import generate_instance


def main() -> None:
    inst = generate_instance(n_sections=2, n_cells_per_section=50, n_genes=20, seed=0)
    cfg = NicheModelConfig(n_prototypes=len(inst.proto_kind), epochs=10, seed=0)
    result = fit_niche_model(inst.X, inst.coords, inst.section_id, inst.edges, cfg)
    score = score_against_truth(
        pred_prototype_id=result.prototype_id,
        pred_marker_table=result.marker_table,
        pred_proto_kind=result.proto_kind,
        true_prototype_id=inst.prototype_id,
        true_marker_genes=inst.marker_genes,
        section_id=inst.section_id,
        edges=inst.edges,
        k=5,
        true_proto_kind=inst.proto_kind,
    )
    print(f"ARI(model, truth):               {score['ARI']:.3f}")
    print(f"NMI(model, truth):               {score['NMI']:.3f}")
    print(f"homogeneity(model, truth):       {score['homogeneity']:.3f}")
    print(f"completeness(model, truth):      {score['completeness']:.3f}")
    print(f"v_measure(model, truth):         {score['v_measure']:.3f}")
    print(f"label_agreement(model):          {score['label_agreement']:.3f}")
    print(f"proto_kind_accuracy(model):      {score['proto_kind_accuracy']:.3f}")
    print(f"marker_recall@5(model vs truth): {score['marker_recall_at_k']:.3f}")


if __name__ == "__main__":
    main()
