# QC reference-free QC

Each case in OpenBrain v1.0 carries seven raw QC values (Q1..Q7), seven
within-source percentile risks (r1..r7), the combined risk
R = mean(r1..r7), and a confidence tier (HighConf / Standard / Flagged).
This package computes all seven axes plus the combiner.

For the full mathematical statement, see Appendix H of the paper.

## Per-axis definitions

| Axis | Module | Direction | Inputs | Definition (one-line) |
|-----:|--------|-----------|--------|------------------------|
| Q1 | `q1_brain_leakage.py` | lower better | M1 label + brain mask | `\|seg>0 ∩ mask=0\| / \|seg>0\|` |
| Q2 | `q2_volume_outlier.py` | lower better | per-source ROI volumes | fraction of labels with V outside per-source 1.5-IQR fences |
| Q3 | `q3_distant_components.py`| lower better | M1 label | voxel fraction in CCs whose centroid > 1.5·V_primary^(1/3) from primary CC |
| Q4 | `q4_intensity_cv.py` | lower better | T1w image + M1 label | median across ROIs of std/mean of T1w intensity |
| Q5 | `q5_committee_dice.py` | higher better| 3 committee labels (M1,M2,M3)| 25th percentile of pairwise per-ROI Dice over 3 model pairs |
| Q6 | `q6_voxel_entropy.py` | lower better | 3 committee labels | mean Shannon entropy of 3-vote distribution over foreground voxels |
| Q7 | `q7_volume_cv.py` | lower better | 3 committee labels | mean across ROIs of std/mean of per-model volumes |

Q5 is "higher is better" (more committee agreement = better); the
combiner flips it to a risk-aligned `r5 = 1 - ECDF_D(Q5)`. All other
axes are lower-better, so `r_k = ECDF_D(Q_k)` directly.

## Combiner (`combine_R.py`)

Given a long DataFrame with columns `case_id, source, Q1..Q7`,
`combine_R.combine(df)` returns a copy with `r1..r7`, `R`, and `tier`
columns. Per-axis ECDFs are computed *within each source* so that a
case's percentile reflects its rank against its own cohort, removing
acquisition-site bias from the risk score.

Default tier cuts are K_high = 0.25 and K_flag = 0.75 on the
within-source ECDF of R, matching the canonical Stage-C release. Both
cuts are exposed as CLI flags.

## Single-case driver

```bash
python -m qc \
 --case-dir $OPENBRAIN_RELEASE_ROOT/sample/cases \
 --m1-dir $OPENBRAIN_RELEASE_ROOT/sample/m1 \
 --m2-dir $OPENBRAIN_RELEASE_ROOT/sample/m2 \
 --m3-dir $OPENBRAIN_RELEASE_ROOT/sample/m3 \
 --include-ids data/excluded_label_ids.txt \
 --output /tmp/sample_q1q7.csv
```

Q2 is intentionally left NaN by the single-case driver because it
requires a cohort: see `q2_volume_outlier.py` for the cohort-wide
fence-fitting workflow, which expects a CSV of per-case ROI volume
vectors as its input.

## Provenance

- Q1 — reconstructed for the public release (the internal pipeline
 inlined this as a lambda in ``the QC pipeline driver``).
- Q3 — faithful port of ``the paper's distant-components QC module``.
- Q4 — faithful port of ``paper/qc_tools/q10_intensity_homogeneity.py``.
- Q5 — distilled from ``paper/scripts/the QC committee module`` (only the canonical
 25-pctile aggregation is exposed; the original script also provided
 `mean` and `min` ablations).
- Q6 — faithful port of ``paper/qc_tools/q11_committee_entropy.py``.
- Q7 — faithful port of ``paper/qc_tools/q12_committee_volume_cv.py``.
- Combiner — re-implementation of the per-source ECDF logic in
 ``the QC pipeline driver`` Pass 3.
