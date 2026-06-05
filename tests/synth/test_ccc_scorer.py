import math

import pytest

from nichelens_st.synth.ccc import score_ccc_topk

# Interactions are keyed as (ligand, receptor[, source, target]); the scorer is
# type-agnostic and compares normalized tuples, so integer gene/prototype ids
# (synthetic truth) and string symbols (a real squidpy ligrec run) both work.
TRUTH4 = [(0, 1, 2, 3), (4, 5, 6, 7)]


def test_perfect_ranking_precision_and_recall_one():
    out = score_ccc_topk(pred_ranked=list(TRUTH4), truth=TRUTH4, k=2)
    assert out["precision_at_k"] == 1.0
    assert out["recall_at_k"] == 1.0
    assert out["hit_rate"] == 1.0
    assert out["n_true_recovered"] == 2


def test_all_wrong_ranking_zero():
    pred = [(9, 9, 9, 9), (8, 8, 8, 8)]
    out = score_ccc_topk(pred_ranked=pred, truth=TRUTH4, k=2)
    assert out["precision_at_k"] == 0.0
    assert out["recall_at_k"] == 0.0
    assert out["hit_rate"] == 0.0
    assert out["n_true_recovered"] == 0


def test_partial_ranking_with_decoy_in_topk():
    pred = [TRUTH4[0], (9, 9, 9, 9), TRUTH4[1]]
    out = score_ccc_topk(pred_ranked=pred, truth=TRUTH4, k=2)
    assert out["precision_at_k"] == 0.5
    assert out["recall_at_k"] == 0.5
    assert out["n_true_recovered"] == 1


def test_empty_truth_returns_nan():
    out = score_ccc_topk(pred_ranked=[(0, 1)], truth=[], k=1)
    assert math.isnan(out["precision_at_k"])
    assert math.isnan(out["recall_at_k"])
    assert math.isnan(out["hit_rate"])
    assert out["n_true_recovered"] == 0


def test_empty_pred_returns_nan():
    out = score_ccc_topk(pred_ranked=[], truth=TRUTH4, k=2)
    assert math.isnan(out["precision_at_k"])
    assert math.isnan(out["recall_at_k"])
    assert out["n_true_recovered"] == 0


def test_k_greater_than_pred_length_graceful():
    out = score_ccc_topk(pred_ranked=[(0, 1)], truth=[(0, 1)], k=5)
    assert out["precision_at_k"] == 1.0
    assert out["recall_at_k"] == 1.0


def test_duplicate_predictions_deduped():
    pred = [TRUTH4[0], TRUTH4[0], TRUTH4[1]]
    out = score_ccc_topk(pred_ranked=pred, truth=TRUTH4, k=2)
    # After dedup the top-2 are the two distinct true pairs -> perfect.
    assert out["precision_at_k"] == 1.0
    assert out["recall_at_k"] == 1.0
    assert out["n_true_recovered"] == 2


def test_string_named_interactions_match():
    truth = [("Tgfb1", "Tgfbr1"), ("Wnt5a", "Fzd1")]
    pred = [("Tgfb1", "Tgfbr1"), ("Foo", "Bar")]
    out = score_ccc_topk(pred_ranked=pred, truth=truth, k=1)
    assert out["precision_at_k"] == 1.0
    assert out["recall_at_k"] == 0.5


def test_k_below_one_raises():
    with pytest.raises(ValueError):
        score_ccc_topk(pred_ranked=[(0, 1)], truth=[(0, 1)], k=0)


def test_malformed_tuple_length_raises():
    with pytest.raises(ValueError):
        score_ccc_topk(pred_ranked=[(0, 1, 2, 3, 4)], truth=[(0, 1)], k=1)


def test_malformed_non_iterable_entry_raises():
    with pytest.raises(ValueError):
        score_ccc_topk(pred_ranked=[123], truth=[(0, 1)], k=1)


def test_malformed_string_entry_raises():
    with pytest.raises(ValueError):
        score_ccc_topk(pred_ranked=["ab"], truth=[(0, 1)], k=1)


def test_none_inputs_raise():
    with pytest.raises(ValueError):
        score_ccc_topk(pred_ranked=None, truth=[(0, 1)], k=1)
    with pytest.raises(ValueError):
        score_ccc_topk(pred_ranked=[(0, 1)], truth=None, k=1)
