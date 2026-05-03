# Selection: High / Random / Low training arms

The OpenBrain v1.0 paper studies three downstream training arms, each of
size N=1000 cases, drawn from the 35,838-case Stage-C manifest:

- **High** (the paper's training arm selector): maximum-diversity round-robin over the
 ascending-joint-score list. The joint score is `rank(R) + rank(-IQ)`
 where `R` is the combined QC risk and `IQ` is OpenMind's image
 quality score. Cases with low R + high IQ are picked first. The
 round-robin per-source cap (incremented one at a time until the
 budget is filled) maximises the number of unique source datasets in
 the arm.

- **Random** (the paper's training arm selector): random sample of N cases, drawn from a
 randomly chosen pool of 500 source datasets. The two-step random
 draw produces an arm whose source diversity is intentionally
 *moderate* — between the High arm's max-diversity round-robin and
 the Low arm's source-capped greedy.

- **Low** ( / the paper's training arm selector): descending joint-score (high-risk +
 low-IQ first), greedy walk with per-source cap. The cap (default 5
 picks per source) ensures the arm still covers at least N/cap unique
 source datasets, avoiding a degenerate "all from one site" failure
 mode.

## CLI

```bash
# Manifest is the output of core7.combine_R, plus an optional join
# with a metadata CSV that has image_quality_score per case.
python -m selection.select_high_arm \
 --manifest $OPENBRAIN_RELEASE_ROOT/metadata/openbrain_v1_0_qc_manifest.csv \
 --metadata $OPENBRAIN_RELEASE_ROOT/metadata/openbrain_v1_0_metadata.csv \
 --n 1000 \
 --output $OPENBRAIN_ANALYSIS_ROOT/arms/high.csv

python -m selection.select_random_arm \
 --manifest $OPENBRAIN_RELEASE_ROOT/metadata/openbrain_v1_0_qc_manifest.csv \
 --metadata $OPENBRAIN_RELEASE_ROOT/metadata/openbrain_v1_0_metadata.csv \
 --n 1000 --n-sources 500 --seed 42 \
 --output $OPENBRAIN_ANALYSIS_ROOT/arms/random.csv

python -m selection.select_low_arm \
 --manifest $OPENBRAIN_RELEASE_ROOT/metadata/openbrain_v1_0_qc_manifest.csv \
 --metadata $OPENBRAIN_RELEASE_ROOT/metadata/openbrain_v1_0_metadata.csv \
 --n 1000 --per-source-cap 5 \
 --output $OPENBRAIN_ANALYSIS_ROOT/arms/low.csv
```

## Provenance

- `select_high_arm.py` — port of
 the corresponding repo selector script.
- `select_random_arm.py` — port of
 the corresponding repo selector script.
- `select_low_arm.py` — port of
 `paper/scripts/analysis/select_v8_subsets_diverse.py` (E arm only;
 the random/B arm is in `select_random_arm.py`).
