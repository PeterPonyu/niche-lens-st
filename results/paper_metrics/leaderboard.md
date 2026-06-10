## N-T1 Niche Leaderboard — PRELIMINARY, NOT A VALIDATION (published baseline not run; internal-reimplementation rows; intrinsic metrics only; spatial-domain metrics not emitted)

**Dataset:** `niche_merfish_slice`

> ⚠️ **SINGLE-SECTION FALLBACK DATA** — results are from the downsized 5,488-cell MERFISH section, **NOT** the 124k-cell atlas run. Do NOT cite as atlas-scale results.

> *Fallback note: DOWNSIZED SINGLE-SECTION FALLBACK, NOT THE ATLAS-SCALE RUN (5488-cell single MERFISH section; the conserved/sample-specific distinction is degenerate and the scale is not representative of the target dataset). Do NOT cite as atlas-scale results.*

| method | domain_ari | domain_ami | domain_nmi | domain_macro_f1 | domain_homogeneity | domain_accuracy | embedding_silhouette | niche_morans_i |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| niche-lens-st | NA | NA | NA | NA | NA | NA | 0.2426 | 0.4230 |
| diffusion | NA | NA | NA | NA | NA | NA | 0.1071 | 0.8240 |
| neighborhood | NA | NA | NA | NA | NA | NA | 0.0956 | 0.4214 |
| pca | NA | NA | NA | NA | NA | NA | 0.1568 | 0.2709 |

*Columns: domain_ari, domain_ami, domain_nmi, domain_macro_f1, domain_homogeneity, domain_accuracy, embedding_silhouette, niche_morans_i. NA = NOT COMPUTED for this published artifact (no executed spatial-domain GT scoring) — means 'not run / no GT', NOT 'not applicable'.*
