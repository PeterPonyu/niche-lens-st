"""Generated ground-truth niches must be spatially coherent (issue #59).

The old generator assigned ``prototype_id`` independently of position, so the
kNN graph carried no niche signal (Moran's I ~ 0 on the ground truth). The
Voronoi-zone assignment now makes niches spatially contiguous; lock that as a
generator-level invariant so the benchmark fixture cannot silently regress to
spatially-random labels again.
"""

import numpy as np
import pytest

from nichelens_st.metrics import morans_i
from nichelens_st.synth import generate_instance

PARAMS = dict(
    n_sections=2,
    n_cells_per_section=400,
    n_genes=20,
    K_conserved=4,
    J_specific=2,
    k_nn=8,
)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_generated_prototype_field_is_spatially_autocorrelated(seed):
    inst = generate_instance(**PARAMS, seed=seed)
    e, pid = inst.edges, inst.prototype_id
    # Sound categorical measure: fraction of edges joining same-prototype cells.
    edge_agreement = float((pid[e[0]] == pid[e[1]]).mean())
    assert edge_agreement > 0.5, f"edge agreement {edge_agreement:.3f} too low"
    # The benchmark's locked Moran's-I-over-prototype spatial-coherence invariant.
    moran = morans_i(pid.astype(float), e)
    assert moran > 0.3, f"Moran's I {moran:.3f} below 0.3 (no spatial structure)"


def test_position_independent_labels_would_fail():
    # Sanity check that the assertion is meaningful: a position-independent
    # shuffle of the very same labels destroys the spatial autocorrelation.
    inst = generate_instance(**PARAMS, seed=0)
    rng = np.random.default_rng(0)
    shuffled = inst.prototype_id.copy()
    rng.shuffle(shuffled)
    assert morans_i(shuffled.astype(float), inst.edges) < 0.1
