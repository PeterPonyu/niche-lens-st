## N-T2 Cohort-scale Runtime & Peak-memory Scalability

> ⚠️ **FALLBACK ROW PRESENT** — the fallback-to-slice row is the downsized 5,488-cell MERFISH section, NOT an atlas-scale run. Do NOT cite its scale as representative.

| dataset | n_cells | wall-clock (s) | peak RSS (GB) | throughput (cells/s) | InfoNCE mode | status |
| --- | --- | --- | --- | --- | --- | --- |
| niche_merfish_slice | 5488 | 1.29 | 2.02 | 4241 | full-batch | fallback-to-slice |
| single-section full-batch (#148/#302) | 124938 | — | — | — | full-batch | OOM (full-batch) |
| xenium_prime_5k_cancer | 699000 | — | — | — | minibatch | download-blocked (#300) |
| codex_spleen_goltsev2018 | 734101 | 121.90 | 4.50 | 6022 | full-batch | ok |
| abc_atlas_zhuang_merfish | 4200000 | — | — | — | minibatch | download-blocked (#300) |
| vizgen_ffpe_io_merfish | 9000000 | — | — | — | minibatch | download-blocked (#300) |

*peak RSS in GB (1e9 bytes); — = not measured / not applicable; blocked regimes (OOM, download-blocked) carry status with no fabricated numbers. NOT paper-claim-ready.*
