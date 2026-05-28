#!/usr/bin/env python
"""Run a no-claim synthetic metric sanity example."""

from __future__ import annotations

from nichelens_st.metrics import adjusted_rand, marker_recall_at_k, morans_i, section_overlap_rate
from nichelens_st.synth import generate_instance


def main() -> None:
    inst = generate_instance(n_sections=2, n_cells_per_section=50, n_genes=20, seed=0)
    print(f"ARI(truth, truth): {adjusted_rand(inst.prototype_id, inst.prototype_id):.3f}")
    print(f"MoranI(truth): {morans_i(inst.prototype_id, inst.edges):.3f}")
    print(
        "section_overlap_rate(truth): "
        f"{section_overlap_rate(inst.prototype_id, inst.section_id, inst.proto_kind):.3f}"
    )
    print(f"marker_recall@5(truth): {marker_recall_at_k(inst.marker_genes, inst.marker_genes, k=5):.3f}")


if __name__ == "__main__":
    main()
