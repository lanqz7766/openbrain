# Evaluation

The eval package contains the analysis entrypoints that produce the
paper's main tables and figures.

| Script | Paper artifact | Inputs |
|--------|----------------|--------|
| `macro_dice_96.py` | per-case 96-label macro Dice (long format) | predicted labels + GT labels |
| `retention_sweep.py` | Stage A retention curves (Fig. 2) | per-cohort risk CSV + macro_dice CSV |
| `per_axis_univariate.py` | per-axis discrimination table | per-cohort risk CSV with macro_dice column |
| `partial_corr_axes.py` | F_axes_partial_corr table | per-cohort risk CSV |
| `small_struct_defA.py` | Def-A 16-ROI sensitivity (Fig. 3 / supp.) | label_map.json + per-cohort risk + macro_dice |
| `stage_b_paired_delta.py` | Stage-B paired contrasts (Table T2 / Fig. 4) | predictions for each arm + GT |
| `tier_histogram.py` | Stage-C tier counts (Table T1) | openbrain_v1_0_qc_manifest.csv |

## Inputs convention

All scripts expect:

- **Per-cohort risk CSV** (`<analysis-root>/risk/<cohort>_risk.csv`):
 one row per case with columns `case_id, source_dataset,
 Q1..Q7, r1..r7, R, tier` (the output of `core7.combine_R`),
 optionally augmented with `macro_dice` and `small_struct_dice`.
- **Per-cohort macro Dice CSV** (`<analysis-root>/dice/<cohort>_macro_dice.csv`):
 long format produced by `eval.macro_dice_96`, with columns
 `case_id, roi_id, dice, gt_voxels, pred_voxels, small_structure_flag`.
- **Predictions / GT** as directories of per-case `*.nii.gz` files.
 Both M1-style and arm-trained (the released training arms) predictions follow the
 same `<case_id>.nii.gz` filename convention.

## Path parameterisation

None of the scripts hardcode paths; all sources, sinks, and
cohort lists are CLI flags. The sample defaults match the canonical
paper analysis tree.
