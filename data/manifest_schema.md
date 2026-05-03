# `openbrain_v1_0_qc_manifest.csv` schema

The QC manifest is one row per case (35,838 rows total). Columns are:

| Column | Type | Description |
|--------|------|-------------|
| `case_id` | string | OpenNeuro-derived case identifier (matches `cases/<case_id>/`) |
| `source` (or `source_dataset`) | string | OpenNeuro dataset accession (e.g., `ds002424`) |
| `Q1` | float | Brain-mask leakage (lower better). NaN if mask unavailable. |
| `Q2` | float | Volumetric outlier fraction (lower better) using per-source 1.5-IQR fences. |
| `Q3` | float | Distant-component voxel fraction (lower better). |
| `Q4` | float | Within-ROI T1w intensity CV, median across ROIs (lower better). |
| `Q5` | float | Committee Dice 25th percentile (higher better). NaN if M2 or M3 missing. |
| `Q6` | float | Mean voxel-level committee entropy (lower better). NaN if M2 or M3 missing. |
| `Q7` | float | Mean per-ROI committee volume CV (lower better). NaN if M2 or M3 missing. |
| `r1`..`r7` | float in [0, 1] | Within-source ECDF percentile risks aligned so that higher = riskier (Q5 is flipped). |
| `R` | float in [0, 1] | Combined risk = mean(r1..r7). |
| `tier` | enum {HighConf, Standard, Flagged} | Confidence tier from K=25%/75% within-source ECDF cuts on R. |
| `image_quality_score` | float | Optional join from OpenMind metadata (NaN where absent). |
| `m1_label_path` | string | Path to the case's whole-brain segmentation, relative to release root. |
| `brain_mask_path` | string | Path to the case's SynthStrip mask, relative to release root. |
| `image_path` | string | Path to the case's revised T1w image. |

A small fraction of cases have NaN in `Q5`, `Q6`, or `Q7` because the M2
or M3 committee prediction was not available for those cases. Those rows
default to `tier = Standard` (since the within-source ECDF on R uses
only the populated entries to determine cut points).

## Sister metadata files

- `openbrain_v1_0_metadata.csv` — the full per-case metadata from
 upstream OpenNeuro (subject demographics where available, BIDS path,
 acquisition parameters where available).
- `openneuro_dataset_audit.csv` — per-source license-class audit and
 provenance.
- `excluded_cases_summary.csv` — case-level exclusion log (cases that
 failed the license / matching / mask QC and were dropped from v1.0).
- `sha256_manifest.csv` — SHA-256 hash of every redistributed file,
 used to verify a local copy.
