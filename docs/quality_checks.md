# Quality checks

Every case carries seven raw values `Q1` to `Q7`, seven within-source
percentiles `r1` to `r7`, the combined risk `R`, and a tier. The
`openbrain.qc` package computes them. Appendix I of the paper gives the
formal definitions; this page gives the one-line versions.

| | Module | Inputs | Definition | Better |
|---|---|---|---|---|
| Q1 | `leakage.py` | parcellation, brain mask | labeled voxels outside the mask / all labeled voxels | lower |
| Q2 | `volume_outliers.py` | region volumes of all cases from one source | number of regions outside the source's 1.5 IQR fences | lower |
| Q3 | `distant_fragments.py` | parcellation | voxel fraction in connected components whose centroid is farther than 1.5 times the cube root of the main component's volume | lower |
| Q4 | `intensity_cv.py` | T1w image, parcellation | median over regions of std / mean of the image intensity inside the region | lower |
| Q5 | `committee_dice.py` | three parcellations | 25th percentile of pairwise per-region Dice over the three model pairs | higher |
| Q6 | `vote_entropy.py` | three parcellations | mean Shannon entropy of the three-way vote over foreground voxels | lower |
| Q7 | `volume_disagreement.py` | three parcellations | mean over regions of std / mean of the three models' volumes | lower |

`combine.py` turns the raw values into percentiles within each source
dataset (`r5 = 1 - ECDF(Q5)`, the others `r_k = ECDF(Q_k)`), averages them
into `R`, and assigns tiers. In the released manifest the tiers are the
lowest quarter, middle half, and top quarter of `R` over the whole release.
The `combine` module, when run on your own cohort, assigns tiers within each
source instead; both cut points are flags.

Running the checks on a folder of cases:

```bash
python -m openbrain.qc \
    --case-dir <cases> --m1-dir <primary> --m2-dir <committee_2> --m3-dir <committee_3> \
    --include-ids data/excluded_labels.txt --output qc.csv
```

Q2 is left empty by this driver because it needs the whole source cohort;
see `volume_outliers.py` for the cohort step. Q5 to Q7 are left empty when
the two committee parcellations are not given.
