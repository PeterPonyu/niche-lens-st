#!/usr/bin/env python
"""Further niche-characterization metrics on the 5,488-cell slice.

This is the "further computational results" path for NicheLens-ST (G3 of the
``further-results-brief``). It loads the on-disk 5,488-cell MERFISH slice plus
the previously fitted niche artifacts (``outputs/niche.npz`` -> learned ``H`` and
``prototype_id``) and APPENDS new computational results to the existing
``results/niche-lens-st/metrics.json`` via the vendored uniform results
contract. It does NOT refit the niche model.

It computes + emits (standard niche-characterization reductions:
spatial-domain clustering / refinement / Hungarian label matching;
``nichelens_st.metrics`` ``morans_i``; ``nichelens_st.graph.build_graph``):

  1. Niche x cell_class composition matrix (per-prototype fraction of each of
     14 cell types) -> full matrix to outputs/, summary stats to metrics.json.
  2. Niche-marker DE genes (``sc.tl.rank_genes_groups`` Wilcoxon by prototype,
     top-k markers per niche) -> outputs/.
  3. Per-niche Moran's I (one-vs-rest indicator per prototype) -> per-prototype
     + summary.
  4. Cell-type co-localization enrichment (cross-cell_class neighbor-pair freq
     over the kNN graph + permutation z-score) -> matrix to outputs/, summary to
     metrics.json.
  5. Clustering agreement of prototype_id vs domain GT: ARI/NMI/AMI/macro-F1/
     homogeneity (Hungarian ``match_labels`` to align 10 prototypes vs 8 GT
     regions). GT source obs['domain'] recorded in run_metadata.
  6. k-stability sweep: re-cluster H at n_prototypes 5..20, silhouette +
     Calinski-Harabasz vs k -> justify k=10.
  7. PCA-baseline silhouette: silhouette of PCA-of-expression under the same
     prototypes vs the learned-H silhouette.

The ARI/NMI vs domain are GT-backed (real ``obs['domain']`` labels); the GT
source is recorded under ``run_metadata.gt_source`` so the verify gate accepts
them. ``conserved_fraction=1.0`` is FLAGGED as degenerate (single section) in
``run_metadata.interpretability`` and is not presented as a finding.

Usage (under the project's conda env):

    conda run --no-capture-output -n dl python scripts/compute_further_niche_metrics.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# Resolve repo root from this file so the script runs from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = _REPO_ROOT.parent / "data"
sys.path.insert(0, str(_REPO_ROOT / "src"))

SLICE_PATH = _DATA_ROOT / "processed" / "niche_merfish_slice" / "anndata.h5ad"
NPZ_PATH = _REPO_ROOT / "results" / "niche-lens-st" / "outputs" / "niche.npz"
METRICS_PATH = _REPO_ROOT / "results" / "niche-lens-st" / "metrics.json"
META_PATH = _REPO_ROOT / "results" / "niche-lens-st" / "run_metadata.json"

CELL_CLASS_COL = "cell_class"
DOMAIN_COL = "domain"
GRAPH_K = 6  # matches run_real_niche.py build_graph(k=6)
TOP_K_MARKERS = 10
N_PERMUTATIONS = 200
K_SWEEP = list(range(5, 21))  # 5..20 inclusive
FITTED_K = 10
SEED = 0


def _to_dense(X) -> np.ndarray:
    import scipy.sparse as sp

    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


def _composition_matrix(prototype_id, cell_class_codes, n_protos, n_types):
    """Per-prototype fraction of each cell type -> (n_protos, n_types)."""
    counts = np.zeros((n_protos, n_types), dtype=np.float64)
    np.add.at(counts, (prototype_id, cell_class_codes), 1.0)
    row_tot = counts.sum(axis=1, keepdims=True)
    frac = np.divide(counts, row_tot, out=np.zeros_like(counts), where=row_tot > 0)
    return counts, frac


def _composition_summary(frac):
    """Dominance + entropy summary of a (n_protos, n_types) fraction matrix."""
    n_types = frac.shape[1]
    dominant = frac.max(axis=1)
    # Per-prototype Shannon entropy (nats), normalized by log(n_types).
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(frac > 0, frac * np.log(frac), 0.0).sum(axis=1)
    norm_ent = ent / np.log(n_types) if n_types > 1 else np.zeros_like(ent)
    return {
        "composition_dominant_fraction_mean": float(dominant.mean()),
        "composition_dominant_fraction_min": float(dominant.min()),
        "composition_dominant_fraction_max": float(dominant.max()),
        "composition_entropy_norm_mean": float(norm_ent.mean()),
        "composition_entropy_norm_min": float(norm_ent.min()),
        "composition_entropy_norm_max": float(norm_ent.max()),
    }


def _de_markers(adata, prototype_id, top_k):
    """Wilcoxon rank_genes_groups per prototype -> top-k marker names + scores."""
    import scanpy as sc

    ad = adata.copy()
    # Normalize for DE (raw counts -> CPM-like + log1p); leaves saved metrics
    # path untouched (operates on a copy).
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    ad.obs["prototype"] = [str(p) for p in prototype_id]
    ad.obs["prototype"] = ad.obs["prototype"].astype("category")
    sc.tl.rank_genes_groups(ad, "prototype", method="wilcoxon")
    rec = ad.uns["rank_genes_groups"]
    groups = list(rec["names"].dtype.names)
    markers: dict[str, list[dict]] = {}
    for g in groups:
        names = rec["names"][g][:top_k]
        scores = rec["scores"][g][:top_k]
        lfc = rec["logfoldchanges"][g][:top_k]
        pvals = rec["pvals_adj"][g][:top_k]
        markers[g] = [
            {
                "gene": str(names[i]),
                "score": float(scores[i]),
                "logfoldchange": float(lfc[i]),
                "pval_adj": float(pvals[i]),
            }
            for i in range(len(names))
        ]
    return markers


def _per_niche_morans_i(prototype_id, edges, n_protos):
    """One-vs-rest indicator Moran's I per prototype."""
    from nichelens_st.metrics import morans_i

    per = {}
    for p in range(n_protos):
        indicator = (prototype_id == p).astype(np.float64)
        per[str(p)] = float(morans_i(indicator, edges))
    vals = np.array(list(per.values()), dtype=np.float64)
    summary = {
        "per_niche_morans_i_mean": float(vals.mean()),
        "per_niche_morans_i_min": float(vals.min()),
        "per_niche_morans_i_max": float(vals.max()),
        "per_niche_morans_i_median": float(np.median(vals)),
    }
    return per, summary


def _colocalization_enrichment(cell_class_codes, edges, n_types, n_perm, seed):
    """Cross-cell_class neighbor-pair frequency over the kNN graph + permutation
    z-score. Returns observed counts, z-score matrix, and a summary.

    Edges are directed COO (src, dst). We symmetrize the type-pair count matrix
    so the enrichment is between unordered cell-type pairs.
    """
    src = edges[0]
    dst = edges[1]
    ts = cell_class_codes[src]
    td = cell_class_codes[dst]

    def pair_counts(a, b):
        m = np.zeros((n_types, n_types), dtype=np.float64)
        np.add.at(m, (a, b), 1.0)
        return m + m.T  # symmetrize unordered pairs

    observed = pair_counts(ts, td)

    rng = np.random.default_rng(seed)
    n_cells = cell_class_codes.shape[0]
    perm_stack = np.zeros((n_perm, n_types, n_types), dtype=np.float64)
    for i in range(n_perm):
        perm = rng.permutation(n_cells)
        labels = cell_class_codes[perm]
        perm_stack[i] = pair_counts(labels[src], labels[dst])
    mean = perm_stack.mean(axis=0)
    std = perm_stack.std(axis=0)
    zscore = np.divide(
        observed - mean, std, out=np.zeros_like(observed), where=std > 0
    )

    # Summary: enrichment of same-type vs cross-type neighbor pairs.
    diag = np.diag(zscore)
    offdiag = zscore[~np.eye(n_types, dtype=bool)]
    summary = {
        "coloc_same_type_z_mean": float(diag.mean()),
        "coloc_cross_type_z_mean": float(offdiag.mean()),
        "coloc_cross_type_z_max": float(offdiag.max()),
        "coloc_z_abs_max": float(np.abs(zscore).max()),
    }
    return observed, zscore, summary


def match_labels(true_labels, predicted_labels, n_classes):
    """Hungarian label alignment between two clusterings.

    Self-contained ``scipy.optimize.linear_sum_assignment`` over the
    label-overlap cost matrix: it relabels ``predicted_labels`` to maximize
    overlap with ``true_labels`` before computing F1/accuracy. Implemented
    inline so this script depends only on numpy/scipy (no torch_geometric /
    torch_sparse chain).
    """
    from scipy.optimize import linear_sum_assignment as linear_assignment

    cost_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            cost_matrix[i, j] = np.sum((true_labels == i) & (predicted_labels == j))
    row_ind, col_ind = linear_assignment(-cost_matrix)
    new_labels = np.copy(predicted_labels)
    for i, j in zip(row_ind, col_ind):
        new_labels[predicted_labels == j] = i
    return new_labels


def _clustering_agreement(prototype_id, domain_codes, n_clusters):
    """ARI/NMI/AMI/macro-F1/homogeneity of prototype_id vs domain GT.

    Uses the Hungarian ``match_labels`` to align the n_prototypes
    predicted labels to the n_GT_regions reference before computing F1/accuracy.
    Permutation-invariant scores (ARI/NMI/AMI/homogeneity) are computed on the
    raw labels; macro-F1/accuracy use the Hungarian-aligned labels.
    """
    from sklearn.metrics import (
        accuracy_score,
        adjusted_mutual_info_score,
        adjusted_rand_score,
        f1_score,
        homogeneity_score,
        normalized_mutual_info_score,
    )

    pred = np.asarray(prototype_id, dtype=int)
    true = np.asarray(domain_codes, dtype=int)
    # match_labels needs a square cost space; size to the larger label space.
    n_match = max(n_clusters, int(true.max()) + 1)
    aligned = match_labels(true, pred, n_match)

    return {
        "domain_ari": float(adjusted_rand_score(true, pred)),
        "domain_nmi": float(normalized_mutual_info_score(true, pred)),
        "domain_ami": float(adjusted_mutual_info_score(true, pred)),
        "domain_homogeneity": float(homogeneity_score(true, pred)),
        "domain_macro_f1": float(f1_score(true, aligned, average="macro")),
        "domain_accuracy": float(accuracy_score(true, aligned)),
    }


def _k_stability_sweep(H, k_values, seed):
    """Re-cluster H over k_values; silhouette + Calinski-Harabasz per k."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, silhouette_score

    per_k = {}
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(H)
        if np.unique(labels).size < 2:
            sil = None
            ch = None
        else:
            sil = float(silhouette_score(H, labels))
            ch = float(calinski_harabasz_score(H, labels))
        per_k[str(k)] = {"silhouette": sil, "calinski_harabasz": ch}
    # Argmax-by-silhouette to justify the fitted k.
    sils = {k: v["silhouette"] for k, v in per_k.items() if v["silhouette"] is not None}
    best_k = max(sils, key=lambda kk: sils[kk]) if sils else None
    summary = {
        "k_sweep_best_k_by_silhouette": float(best_k) if best_k is not None else None,
        "k_sweep_silhouette_at_fitted_k": per_k[str(FITTED_K)]["silhouette"],
        "k_sweep_calinski_harabasz_at_fitted_k": per_k[str(FITTED_K)]["calinski_harabasz"],
    }
    return per_k, summary


def _pca_baseline_silhouette(X, prototype_id, n_components, seed):
    """Silhouette of PCA-of-expression under the SAME prototype labels."""
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    n_comp = int(min(n_components, X.shape[1], X.shape[0] - 1))
    pcs = PCA(n_components=n_comp, random_state=seed).fit_transform(X)
    if np.unique(prototype_id).size < 2:
        return None, n_comp
    return float(silhouette_score(pcs, prototype_id)), n_comp


def main() -> int:
    import scanpy as sc

    from nichelens_st import results_contract
    from nichelens_st.graph import build_graph
    from sklearn.metrics import silhouette_score

    t0 = time.time()

    # --- Load slice + fitted artifacts -------------------------------------
    adata = sc.read_h5ad(str(SLICE_PATH))
    n_obs, n_vars = int(adata.shape[0]), int(adata.shape[1])
    coords = np.ascontiguousarray(adata.obsm["spatial"], dtype=np.float64)
    section_id = np.zeros(n_obs, dtype=np.int64)  # single slice

    npz = np.load(str(NPZ_PATH))
    H = np.asarray(npz["H"], dtype=np.float64)
    prototype_id = np.asarray(npz["prototype_id"], dtype=np.int64)
    n_protos = int(prototype_id.max()) + 1
    if H.shape[0] != n_obs or prototype_id.shape[0] != n_obs:
        raise ValueError(
            f"artifact/slice mismatch: H={H.shape} proto={prototype_id.shape} n_obs={n_obs}"
        )

    import pandas as pd

    cell_class = pd.Series(adata.obs[CELL_CLASS_COL]).astype("category")
    cell_class_codes = cell_class.cat.codes.to_numpy()
    cell_class_names = list(cell_class.cat.categories)
    n_types = len(cell_class_names)

    domain = pd.Series(adata.obs[DOMAIN_COL]).astype("category")
    domain_codes = domain.cat.codes.to_numpy()
    domain_names = list(domain.cat.categories)
    n_regions = len(domain_names)

    edges = build_graph(coords, section_id, k=GRAPH_K, method="knn")

    # --- 1. composition matrix --------------------------------------------
    comp_counts, comp_frac = _composition_matrix(
        prototype_id, cell_class_codes, n_protos, n_types
    )
    comp_summary = _composition_summary(comp_frac)

    # --- 2. DE markers -----------------------------------------------------
    markers = _de_markers(adata, prototype_id, TOP_K_MARKERS)

    # --- 3. per-niche Moran's I -------------------------------------------
    per_niche_mi, mi_summary = _per_niche_morans_i(prototype_id, edges, n_protos)

    # --- 4. co-localization enrichment ------------------------------------
    coloc_obs, coloc_z, coloc_summary = _colocalization_enrichment(
        cell_class_codes, edges, n_types, N_PERMUTATIONS, SEED
    )

    # --- 5. clustering agreement vs domain GT -----------------------------
    agreement = _clustering_agreement(prototype_id, domain_codes, n_protos)

    # --- 6. k-stability sweep ---------------------------------------------
    k_sweep, k_summary = _k_stability_sweep(H, K_SWEEP, SEED)

    # --- 7. PCA-baseline silhouette ---------------------------------------
    X = _to_dense(adata.X)
    pca_sil, pca_n_comp = _pca_baseline_silhouette(X, prototype_id, H.shape[1], SEED)
    learned_h_sil = (
        float(silhouette_score(H, prototype_id))
        if np.unique(prototype_id).size >= 2
        else None
    )

    # --- Assemble appended metrics ----------------------------------------
    new_metrics: dict[str, float | None] = {}
    new_metrics["n_cell_classes"] = float(n_types)
    new_metrics["n_gt_regions"] = float(n_regions)
    new_metrics.update(comp_summary)
    new_metrics.update(mi_summary)
    new_metrics.update(coloc_summary)
    new_metrics.update(agreement)
    new_metrics.update(k_summary)
    new_metrics["pca_baseline_silhouette"] = pca_sil
    new_metrics["learned_h_silhouette"] = learned_h_sil
    new_metrics["learned_vs_pca_silhouette_delta"] = (
        float(learned_h_sil - pca_sil)
        if (learned_h_sil is not None and pca_sil is not None)
        else None
    )

    # --- Merge with existing metrics + run_metadata -----------------------
    prior_metrics = json.loads(METRICS_PATH.read_text())
    prior_meta = json.loads(META_PATH.read_text())
    merged_metrics = dict(prior_metrics.get("metrics", {}))
    merged_metrics.update(new_metrics)

    # Extend interpretability: record GT source + flag degenerate conserved_fraction.
    interp = dict(prior_meta.get("interpretability", {}))
    caveats = list(interp.get("caveats", []))
    degenerate_note = (
        "conserved_fraction=1.0 is DEGENERATE: the 5,488-cell slice is a single "
        "section (slice_id has 1 level), so every prototype trivially appears in "
        "every section; do NOT present conserved_fraction as a finding."
    )
    if degenerate_note not in caveats:
        caveats.append(degenerate_note)
    interp["caveats"] = caveats
    interp["conserved_fraction_degenerate_single_section"] = True

    # GT source for the GT-backed clustering metrics (verify-gate provenance).
    # The vendored results_contract.write_results only passes through a fixed set
    # of run_metadata keys plus ``interpretability``; ``gt_source`` is recorded
    # INSIDE ``interpretability`` so the provenance survives the contract (the
    # contract is byte-identical and must not be edited).
    gt_source = {
        "clustering_agreement": {
            "metrics": [
                "domain_ari",
                "domain_nmi",
                "domain_ami",
                "domain_homogeneity",
                "domain_macro_f1",
                "domain_accuracy",
            ],
            "gt_obs_column": DOMAIN_COL,
            "gt_n_classes": n_regions,
            "gt_class_names": domain_names,
            "gt_is_real_labels": True,
            "alignment": "Hungarian match_labels (scipy linear_sum_assignment) for F1/accuracy only",
            "note": (
                "GT-backed: prototype_id vs REAL obs['domain'] (Moffitt 2018 "
                "MERFISH hypothalamus niche regions); permutation-invariant "
                "scores on raw labels."
            ),
        },
        "composition": {"gt_obs_column": CELL_CLASS_COL, "gt_n_classes": n_types},
    }
    interp["gt_source"] = gt_source

    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    elapsed = time.time() - t0

    # Preserve prior outputs provenance + add the new further-metrics artifacts.
    prior_outputs = dict(prior_meta.get("outputs", {}))
    new_outputs = dict(prior_outputs)
    new_outputs.update(
        {
            "niche_cellclass_composition": "outputs/niche_cellclass_composition.npz",
            "niche_markers": "outputs/niche_markers.json",
            "per_niche_morans_i": "outputs/per_niche_morans_i.json",
            "celltype_colocalization": "outputs/celltype_colocalization.npz",
            "k_stability_sweep": "outputs/k_stability_sweep.json",
        }
    )

    run_metadata = {
        "dataset_paths": [str(SLICE_PATH)],
        "n_obs": n_obs,
        "n_vars": n_vars,
        "seed": SEED,
        "runtime_s": float(prior_meta.get("runtime_s") or 0.0) + elapsed,
        "started_utc": prior_meta.get("started_utc", started_utc),
        "device": prior_meta.get("device", "cpu"),
        "deterministic": prior_meta.get("deterministic", False),
        "num_threads": prior_meta.get("num_threads", 1),
        "reproducibility_level": prior_meta.get("reproducibility_level", "seeded"),
        "normalization": prior_meta.get("normalization", {"applied": False, "method": "none"}),
        "interpretability": interp,
        "notes": prior_meta.get("notes", "")
        + "; further_niche_metrics appended (G3): composition+DE+per_niche_morans"
        + f"+coloc(perm={N_PERMUTATIONS})+domain_agreement+k_sweep(5..20)+pca_baseline"
        + f"; further_runtime_s={elapsed:.2f}",
    }

    paths = results_contract.write_results(
        project="niche-lens-st",
        dataset_card_id=prior_metrics.get("dataset_card_id", "niche_merfish_slice"),
        metrics=merged_metrics,
        outputs=new_outputs,
        run_metadata=run_metadata,
        results_dir=str(_REPO_ROOT / "results"),
    )

    # --- Persist heavy artifacts into outputs/ ----------------------------
    outputs_dir = Path(paths["outputs_dir"])
    np.savez_compressed(
        outputs_dir / "niche_cellclass_composition.npz",
        counts=comp_counts,
        fraction=comp_frac,
        prototype_ids=np.arange(n_protos),
        cell_class_names=np.array(cell_class_names, dtype=object),
    )
    np.savez_compressed(
        outputs_dir / "celltype_colocalization.npz",
        observed_pair_counts=coloc_obs,
        permutation_zscore=coloc_z,
        cell_class_names=np.array(cell_class_names, dtype=object),
        n_permutations=np.array(N_PERMUTATIONS),
    )
    with open(outputs_dir / "niche_markers.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"top_k": TOP_K_MARKERS, "method": "wilcoxon", "markers": markers},
            fh,
            indent=2,
        )
    with open(outputs_dir / "per_niche_morans_i.json", "w", encoding="utf-8") as fh:
        json.dump({"graph_k": GRAPH_K, "per_prototype": per_niche_mi}, fh, indent=2)
    with open(outputs_dir / "k_stability_sweep.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"fitted_k": FITTED_K, "pca_n_components": pca_n_comp, "per_k": k_sweep},
            fh,
            indent=2,
        )

    print(f"slice: {SLICE_PATH}")
    print(f"n_obs={n_obs} n_vars={n_vars} n_protos={n_protos} "
          f"n_cell_classes={n_types} n_gt_regions={n_regions}")
    print(f"appended {len(new_metrics)} further metrics to {paths['metrics']}")
    print(f"new outputs -> {outputs_dir}")
    print(f"further_runtime_s={elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
