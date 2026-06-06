"""Niche-abundance to per-sample outcome association (#320 / #339).

The single most common tissue-microenvironment question is *"is this niche
enriched in responders / high-grade / short-survival samples?"* -- i.e. does a
learned niche's **per-sample abundance** associate with a **per-sample** label.
This module turns a per-cell prototype assignment into a (sample x prototype)
relative-abundance matrix and tests each prototype's abundance against a
per-sample label.

Design choices that keep the test honest:

* **The sample is the unit of analysis, never the cell.** Each sample
  contributes one abundance composition, so groups are samples (no per-cell
  pseudo-replication that would inflate significance).
* **Compositional-aware.** Abundances live on the simplex (they sum to 1), so a
  raw per-prototype test is biased by the closure. We apply the centered
  log-ratio (CLR) before testing, the standard compositional transform.
* **Multiple-testing controlled.** Per-prototype p-values are corrected with
  Benjamini-Hochberg FDR across prototypes; the reported ``q_value`` is what a
  claim should gate on.

Everything here is pure ``numpy`` / ``scipy`` and deterministic. The real
biological claim is data-gated (needs a labelled multi-sample cohort); these
helpers make the metric runnable and testable now on planted associations.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sample_prototype_abundance",
    "clr_transform",
    "benjamini_hochberg",
    "niche_outcome_association",
]


def sample_prototype_abundance(
    sample_id: np.ndarray,
    prototype_id: np.ndarray,
    *,
    n_prototypes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce per-cell assignments to a (sample x prototype) abundance matrix.

    Each row is one sample's relative prototype abundance -- the fraction of that
    sample's cells assigned to each prototype -- so rows are compositions that
    sum to 1 (a sample with zero cells, which cannot occur here, would sum to 0).

    ``n_prototypes`` fixes the column count (default ``max(prototype_id) + 1``);
    prototypes that never occur become all-zero columns, so the matrix shape is
    stable across samples and runs. Returns ``(abundance, samples)`` where
    ``samples`` is the sorted unique sample ids aligned to the rows.
    """
    sid = np.asarray(sample_id).reshape(-1)
    proto = np.asarray(prototype_id).reshape(-1)
    if sid.shape[0] != proto.shape[0]:
        raise ValueError(
            f"sample_id and prototype_id must align; got {sid.shape[0]} "
            f"and {proto.shape[0]}"
        )
    proto = proto.astype(np.int64)
    if proto.size and proto.min() < 0:
        raise ValueError("prototype_id must be non-negative integer codes")
    samples, inv = np.unique(sid, return_inverse=True)
    if n_prototypes is None:
        n_prototypes = int(proto.max()) + 1 if proto.size else 0
    n_prototypes = int(n_prototypes)
    if proto.size and proto.max() >= n_prototypes:
        raise ValueError(
            f"n_prototypes={n_prototypes} too small for max code {int(proto.max())}"
        )
    counts = np.zeros((samples.size, n_prototypes), dtype=np.float64)
    np.add.at(counts, (inv, proto), 1.0)
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0.0] = 1.0
    return counts / totals, samples


def clr_transform(comp: np.ndarray, *, pseudocount: float = 0.5) -> np.ndarray:
    """Centered log-ratio of each row (a composition).

    ``clr(x)_i = log(x_i) - mean_j log(x_j)``. A ``pseudocount`` is added before
    the log (then the row is renormalised) so exact-zero abundances stay finite.
    Each output row sums to zero by construction; a uniform composition maps to
    the all-zero row.
    """
    x = np.asarray(comp, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"comp must be 2-D (n_samples, n_prototypes); got {x.shape}")
    if pseudocount < 0:
        raise ValueError("pseudocount must be non-negative")
    x = x + float(pseudocount)
    x = x / x.sum(axis=1, keepdims=True)
    logx = np.log(x)
    return logx - logx.mean(axis=1, keepdims=True)


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values for a 1-D array of p-values.

    Returns q-values in the input order, monotone in rank and clipped to
    ``[0, 1]``. Pure numpy (no statsmodels dependency).
    """
    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    n = p.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    ranks = np.arange(1, n + 1, dtype=np.float64)
    q = ranked * n / ranks
    # enforce monotonicity: q_(i) = min over k >= i of the raw values
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(n, dtype=np.float64)
    out[order] = q
    return out


def _binary_association(Z: np.ndarray, labels: np.ndarray):
    """Per-prototype Mann-Whitney U of CLR abundance between the two label
    groups, with a signed rank-biserial effect size (positive => higher in the
    second/`groups[1]` group)."""
    from scipy import stats

    groups = np.unique(labels)
    if groups.size != 2:
        raise ValueError(
            f"binary label_kind needs exactly 2 distinct labels; got {groups.size}"
        )
    g_hi = labels == groups[1]
    g_lo = labels == groups[0]
    n_hi, n_lo = int(g_hi.sum()), int(g_lo.sum())
    n_proto = Z.shape[1]
    effects = np.zeros(n_proto)
    statistics = np.zeros(n_proto)
    pvals = np.ones(n_proto)
    for j in range(n_proto):
        a = Z[g_hi, j]
        b = Z[g_lo, j]
        if np.ptp(np.concatenate([a, b])) == 0.0:
            # degenerate (all equal) -> no signal
            statistics[j] = n_hi * n_lo / 2.0
            effects[j] = 0.0
            pvals[j] = 1.0
            continue
        res = stats.mannwhitneyu(a, b, alternative="two-sided")
        u = float(res.statistic)
        statistics[j] = u
        # rank-biserial correlation in [-1, 1]; sign follows g_hi vs g_lo
        effects[j] = 2.0 * u / (n_hi * n_lo) - 1.0
        pvals[j] = float(res.pvalue)
    return effects, statistics, pvals, "mannwhitneyu", "rank_biserial"


def _continuous_association(Z: np.ndarray, outcome: np.ndarray):
    """Per-prototype Spearman rank correlation of CLR abundance with a numeric
    per-sample outcome (e.g. a survival proxy). Effect size is rho."""
    from scipy import stats

    y = np.asarray(outcome, dtype=np.float64)
    n_proto = Z.shape[1]
    effects = np.zeros(n_proto)
    pvals = np.ones(n_proto)
    for j in range(n_proto):
        if np.ptp(Z[:, j]) == 0.0:
            effects[j] = 0.0
            pvals[j] = 1.0
            continue
        res = stats.spearmanr(Z[:, j], y)
        rho = float(res.statistic)
        p = float(res.pvalue)
        effects[j] = rho if np.isfinite(rho) else 0.0
        pvals[j] = p if np.isfinite(p) else 1.0
    return effects, effects.copy(), pvals, "spearmanr", "spearman_rho"


def niche_outcome_association(
    abundance: np.ndarray,
    sample_labels: np.ndarray,
    *,
    label_kind: str = "binary",
    transform: str = "clr",
    pseudocount: float = 0.5,
) -> dict:
    """Associate each prototype's per-sample abundance with a per-sample label.

    ``abundance`` is the ``(n_samples, n_prototypes)`` matrix from
    :func:`sample_prototype_abundance`; ``sample_labels`` is one label per row.
    With ``label_kind="binary"`` a two-group Mann-Whitney U is run per prototype
    (effect = signed rank-biserial correlation); with ``label_kind="continuous"``
    a Spearman correlation is run (effect = rho). ``transform="clr"`` (default)
    applies the compositional CLR first; ``"none"`` tests raw abundance.

    Returns a JSON-serialisable dict with one record per prototype
    (``prototype``, ``effect_size``, ``statistic``, ``p_value``,
    Benjamini-Hochberg ``q_value``) plus run metadata.
    """
    A = np.asarray(abundance, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"abundance must be 2-D; got {A.shape}")
    labels = np.asarray(sample_labels).reshape(-1)
    if labels.shape[0] != A.shape[0]:
        raise ValueError(
            f"need one label per sample (row); got {labels.shape[0]} labels "
            f"for {A.shape[0]} samples"
        )
    if A.shape[0] < 2:
        raise ValueError("need >= 2 samples to test an association")

    if transform == "clr":
        Z = clr_transform(A, pseudocount=pseudocount)
    elif transform == "none":
        Z = A.copy()
    else:
        raise ValueError(f"transform must be 'clr' or 'none'; got {transform!r}")

    if label_kind == "binary":
        effects, statistics, pvals, method, effect_name = _binary_association(Z, labels)
    elif label_kind == "continuous":
        effects, statistics, pvals, method, effect_name = _continuous_association(
            Z, labels
        )
    else:
        raise ValueError(
            f"label_kind must be 'binary' or 'continuous'; got {label_kind!r}"
        )

    qvals = benjamini_hochberg(pvals)
    prototypes = [
        {
            "prototype": int(j),
            "effect_size": float(effects[j]),
            "statistic": float(statistics[j]),
            "p_value": float(pvals[j]),
            "q_value": float(qvals[j]),
        }
        for j in range(A.shape[1])
    ]
    return {
        "label_kind": label_kind,
        "transform": transform,
        "method": method,
        "effect_size_name": effect_name,
        "n_samples": int(A.shape[0]),
        "n_prototypes": int(A.shape[1]),
        "prototypes": prototypes,
    }
