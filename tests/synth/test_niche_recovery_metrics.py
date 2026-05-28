import numpy as np

from nichelens_st.metrics import adjusted_rand, marker_recall_at_k


def test_adjusted_rand_perfect_and_permuted():
    true = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([2, 2, 0, 0, 1, 1])
    assert adjusted_rand(pred, true) == 1.0


def test_marker_recall_at_k_perfect():
    markers = [[1, 2, 3], [4, 5, 6]]
    assert marker_recall_at_k(markers, markers, k=3) == 1.0
