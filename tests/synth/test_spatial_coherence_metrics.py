import numpy as np

from nichelens_st.metrics import morans_i


def test_morans_i_positive_for_smooth_labels():
    labels = np.array([0, 0, 1, 1], dtype=np.float64)
    edges = np.array([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=np.int64)
    assert morans_i(labels, edges) > 0
