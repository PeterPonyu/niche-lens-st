## N-T1 Multi-Method Niche Leaderboard

**Dataset:** `niche_merfish_slice`

> ⚠️ **SINGLE-SECTION FALLBACK DATA** — results are from the downsized 5,488-cell MERFISH section, **NOT** the 124k-cell atlas run. Do NOT cite as atlas-scale results.

> *Fallback note: DOWNSIZED SINGLE-SECTION FALLBACK, NOT THE ATLAS-SCALE RUN. These results are from a 5,488-cell single MERFISH section, used because the primary 124k-cell atlas run is OOM-infeasible (it requires a 232.60 GiB CUDA allocation on a 23.40 GiB GPU; see `notes` below). Being single-section, the conserved*

| method | domain_ari | domain_ami | domain_nmi | domain_macro_f1 | domain_homogeneity | domain_accuracy | embedding_silhouette | niche_morans_i |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| niche-lens-st | 0.1685 | 0.2648 | 0.2669 | 0.3525 | 0.2962 | 0.4012 | 0.1719 | 0.5335 |

*Columns: domain_ari, domain_ami, domain_nmi, domain_macro_f1, domain_homogeneity, domain_accuracy, embedding_silhouette, niche_morans_i. NA = metric absent or not applicable for this method.*
