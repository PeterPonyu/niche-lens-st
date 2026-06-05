"""Top-k detection scoring for synthetic cell-cell-communication ground truth.

Pairs with the CCC ground truth planted by
:func:`nichelens_st.synth.generator.generate_instance` (``n_ligrec_pairs>0``).
A method's ranked ligand-receptor predictions are scored against the known
positives with precision@k / recall@k / hit-rate.

Interaction key
---------------
Each interaction is keyed as a tuple ``(ligand, receptor[, source, target])`` --
the same ``ligand, receptor, source, target`` vocabulary the real pipeline emits
(:data:`nichelens_st.communication.INTERACTION_SUMMARY_COLUMNS`). The scorer
compares normalized tuples, so it is type-agnostic: the synthetic truth's integer
gene/prototype ids and a real ``squidpy.gr.ligrec`` run's string symbols both work,
as long as ``pred_ranked`` and ``truth`` use the same key arity.

Edge-case discipline mirrors :mod:`nichelens_st.metrics` (issue #83 family):
empty truth or empty predictions yield ``NaN`` (never a free ``1.0``); malformed
inputs raise :class:`ValueError`. Pure numpy/stdlib -- no sklearn.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

__all__ = ["score_ccc_topk"]


def _coerce_scalar(value: Any) -> Any:
    """Reduce a numpy scalar to its python equivalent for stable hashing/equality."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_entry(entry: Any) -> tuple:
    """Normalize one interaction to a ``(ligand, receptor[, source, target])`` tuple."""
    if isinstance(entry, (str, bytes)):
        raise ValueError(
            "interaction entry must be a (ligand, receptor[, source, target]) "
            f"tuple, not a bare string/bytes; got {entry!r}"
        )
    try:
        items = tuple(entry)
    except TypeError as exc:
        raise ValueError(
            "interaction entry must be an iterable tuple/list "
            f"(ligand, receptor[, source, target]); got {type(entry).__name__}"
        ) from exc
    if not 2 <= len(items) <= 4:
        raise ValueError(
            "interaction tuple must have length 2..4 "
            f"(ligand, receptor[, source, target]); got length {len(items)}"
        )
    return tuple(_coerce_scalar(x) for x in items)


def _normalize_many(interactions: Any, name: str) -> list[tuple]:
    """Normalize a collection of interactions, validating the container itself."""
    if interactions is None or not isinstance(interactions, Iterable) or isinstance(
        interactions, (str, bytes)
    ):
        raise ValueError(f"{name} must be an iterable of interaction tuples")
    return [_normalize_entry(e) for e in interactions]


def score_ccc_topk(pred_ranked: Any, truth: Any, k: int) -> dict[str, float]:
    """Score a ranked ligand-receptor prediction list against known positives.

    Parameters
    ----------
    pred_ranked
        A method's ranked detections, best first. Each element is a
        ``(ligand, receptor[, source, target])`` tuple. Duplicates are collapsed
        (keeping first/best occurrence) before scoring.
    truth
        The set/list of known-positive interaction tuples (same key arity as
        ``pred_ranked``), e.g. ``SynthInstance.ligrec_truth``.
    k
        Cutoff depth (``>= 1``). The top ``min(k, n_unique_predictions)``
        predictions are evaluated.

    Returns
    -------
    dict
        ``precision_at_k``, ``recall_at_k``, ``hit_rate`` (== recall, the fraction
        of true positives recovered in the top-k) and ``n_true_recovered`` (int
        count). ``precision_at_k`` is hits / number-of-predictions-considered;
        ``recall_at_k`` is distinct-true-recovered / number-of-true-positives.
        Empty ``truth`` or empty ``pred_ranked`` -> all rates ``NaN`` and
        ``n_true_recovered == 0``.

    Raises
    ------
    ValueError
        If ``k < 1``, or either argument is not an iterable of length-2..4
        interaction tuples.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError(f"k must be an integer; got {type(k).__name__}")
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")

    truth_keys = _normalize_many(truth, "truth")
    pred_keys = _normalize_many(pred_ranked, "pred_ranked")

    nan = float("nan")
    if not truth_keys or not pred_keys:
        # Undefined, not a free perfect score (issue #83 discipline).
        return {
            "precision_at_k": nan,
            "recall_at_k": nan,
            "hit_rate": nan,
            "n_true_recovered": 0,
        }

    truth_set = set(truth_keys)

    seen: set[tuple] = set()
    deduped: list[tuple] = []
    for key in pred_keys:
        if key not in seen:
            seen.add(key)
            deduped.append(key)

    top = deduped[:k]
    hits = [key for key in top if key in truth_set]
    n_recovered = len(set(hits))

    precision = len(hits) / len(top)
    recall = n_recovered / len(truth_set)
    return {
        "precision_at_k": float(precision),
        "recall_at_k": float(recall),
        "hit_rate": float(recall),
        "n_true_recovered": int(n_recovered),
    }
