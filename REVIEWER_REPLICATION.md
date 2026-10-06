# Reviewer replication guide

This guide reproduces the OpenBrain v1.0 paper's main numbers from the
released QC manifest plus a held-out evaluation cohort the reviewer
provides. Total wall-clock excluding cohort downloads:

- ~5 minutes for the QC smoke test (CPU only).
- ~25 minutes to reproduce the Stage-C tier histogram and combined-risk
  distribution on the 35,838-case manifest (CPU).
- ~10 minutes to compute 96-label macro Dice on a user-supplied cohort.
- ~5 minutes for the retention sweep and the paired-contrast tables.

## 1. Environment

Python 3.10 / 3.11 / 3.12. Fresh virtualenv recommended:

```bash
git clone <this-repo>.git openbrain && cd openbrain
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Set environment variables for paths (see `.env.example`):

```bash
export OPENBRAIN_RELEASE_ROOT=/path/to/openbrain_v1_0
export OPENBRAIN_DATA_ROOT=/path/to/eval_cohorts
export OPENBRAIN_ANALYSIS_ROOT=/path/to/scratch_outputs
```

## 2. Download the dataset

```bash
mkdir -p $OPENBRAIN_RELEASE_ROOT && cd $OPENBRAIN_RELEASE_ROOT
hf download lanqz7766/OpenBrain \
    --include "metadata/**" --include "croissant.json" --include "LICENSE" \
    --repo-type dataset --local-dir .

# (Optional) Download the 64-case reviewer sample to inspect examples and smoke-test QC:
hf download lanqz7766/OpenBrain \
    --include "sample/**" --repo-type dataset --local-dir .

# (Optional) Download all 35,838 cases (~191 GB):
hf download lanqz7766/OpenBrain \
    --include "cases/**" --repo-type dataset --local-dir .
```

To render a quick image/mask/segmentation preview for one sample case:

```bash
python examples/preview_sample_case.py \
    --case-dir $OPENBRAIN_RELEASE_ROOT/sample/cases/<case_id> \
    --output $OPENBRAIN_ANALYSIS_ROOT/sample_preview.png
```

## 3. Smoke-test the QC pipeline on one case

```bash
python -m qc \
    --image $OPENBRAIN_RELEASE_ROOT/sample/cases/<case_id>/image.nii.gz \
    --label $OPENBRAIN_RELEASE_ROOT/sample/cases/<case_id>/whole_brain_segmentation.nii.gz \
    --brain-mask $OPENBRAIN_RELEASE_ROOT/sample/cases/<case_id>/brain_mask.nii.gz
```

Output: Q1, Q2, Q3, Q4 raw values printed to stdout. Q5, Q6, Q7 require
the committee predictions (released alongside the canonical label in
`m2/`, `m3/` if you elected to download them).

## 4. Reproduce the Stage-C tier histogram (Figure 5)

```bash
python -m eval.tier_histogram \
    --manifest $OPENBRAIN_RELEASE_ROOT/metadata/openbrain_v1_0_qc_manifest.csv \
    --output $OPENBRAIN_ANALYSIS_ROOT/stage_c_tier_hist.csv
```

The counts must match the Stage-C tier composition in the paper
(HighConf 8,960 / Standard 17,918 / Flagged 8,960 in v1.0).

## 5. 96-label macro Dice on a held-out evaluation cohort

Evaluation cohorts (IXI, Mindboggle-101, BrainLat) are NOT bundled with
the release; they ship under their original licenses from upstream
owners. With cohort labels and a set of segmentation predictions:

```bash
python -m eval.macro_dice_96 \
    --pred $OPENBRAIN_DATA_ROOT/<cohort>/preds \
    --gt $OPENBRAIN_DATA_ROOT/<cohort>/labels \
    --label-map data/label_map.json \
    --exclude-ids data/excluded_label_ids.txt \
    --output $OPENBRAIN_ANALYSIS_ROOT/<cohort>_macro_dice_96.csv
```

The output is a long-format CSV (one row per case × ROI).

## 6. Stage-A retention curves (Figure 2)

After steps 4 and 5 across all three external cohorts:

```bash
python -m eval.retention_sweep \
    --risk-dir $OPENBRAIN_ANALYSIS_ROOT/risk \
    --exclude-ids data/excluded_label_ids.txt \
    --output-dir $OPENBRAIN_ANALYSIS_ROOT/retention
```

The script writes per-cohort retained-mean Dice tables and the deltas
plotted in Figure 2 of the paper.

## 7. Stage-B paired contrasts (Table 2 / Figure 4)

Once the High / Random / Low training arms are produced via
`selection/select_*_arm.py` and downstream segmenter predictions exist
on each evaluation cohort:

```bash
python -m eval.stage_b_paired_delta \
    --pred-high $OPENBRAIN_ANALYSIS_ROOT/preds/high \
    --pred-random $OPENBRAIN_ANALYSIS_ROOT/preds/random \
    --pred-low $OPENBRAIN_ANALYSIS_ROOT/preds/low \
    --gt $OPENBRAIN_DATA_ROOT/<cohort>/labels \
    --output $OPENBRAIN_ANALYSIS_ROOT/stage_b_paired.csv
```

Numbers must match the paired-contrast table in the paper.

## 8. Small-structure sensitivity (Section 5)

```bash
python -m eval.small_struct_defA \
    --per-roi-dice $OPENBRAIN_ANALYSIS_ROOT/<cohort>_macro_dice_96.csv \
    --def-a data/def_a_label_ids.json \
    --output $OPENBRAIN_ANALYSIS_ROOT/<cohort>_def_a.csv
```

## Wall-clock budget

| Step | Wall-clock | Hardware |
|------|-----------:|----------|
| Sample QC | 1 min | CPU |
| Stage-C tier histogram (35,838 cases) | ~25 min | CPU |
| Macro Dice 96-label per cohort | ~10 min | CPU |
| Retention sweep (3 cohorts) | ~5 min | CPU |
| Stage-B paired delta | ~5 min | CPU |
| Small-structure sensitivity | ~2 min | CPU |
| **Total (excluding model inference)** | **~50 min** | CPU |

Training new downstream segmenters from scratch on the High / Random /
Low arms requires nnU-Net v2 and is multiple GPU-days; not part of this
target.
