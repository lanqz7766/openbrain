# Metadata files

All metadata lives under `metadata/` in the Hugging Face dataset.

## `quality.csv`

One row per released case (35,838 rows).

| Column | Meaning |
|---|---|
| `case_id` | case identifier, same as the folder name under `cases/` |
| `source_dataset` | OpenNeuro accession, for example `ds002424` |
| `Q1` | fraction of labeled voxels outside the brain mask |
| `Q2` | number of regions whose volume falls outside the 1.5 IQR fences of the source dataset |
| `Q3` | fraction of labeled voxels in fragments far from their region's main body |
| `Q4` | median over regions of the intensity coefficient of variation inside the region |
| `Q5` | 25th percentile of pairwise per-region Dice among the three models (higher is better) |
| `Q6` | mean voxel-level entropy of the three-model vote |
| `Q7` | mean over regions of the coefficient of variation of the three models' volumes |
| `r1` to `r7` | each `Q` as a percentile within its source dataset; higher is riskier (`Q5` is inverted) |
| `R` | mean of `r1` to `r7` |
| `tier` | `HighConf`, `Standard`, or `Flagged`: the lowest quarter, middle half, and top quarter of `R` over the whole release |

## `cases.csv`

One row per released case with its provenance and whatever the source
dataset provides: `openneuro_dataset_name`, `openneuro_snapshot_tag`,
`archive` (the tar under `cases/` that holds the case), `dataset_doi`,
`dataset_license`, `matched_bids_path`, `openneuro_image_path`, `modality`, `age`, `weight`, `bmi`, `sex`,
`handedness`, `race`, `health_status`, `manufacturer`, `model_name`,
`phase_encoding_direction`, `magnetic_field_strength`, `repetition_time`,
`echo_time`. Missing values are left empty, not imputed.

## Other files

| File | Contents |
|---|---|
| `labels.json` | the 110 region ids and names of the parcellation |
| `sources.csv` | license and provenance check for every OpenNeuro dataset considered |
| `raw_paths.csv` | how each case was matched to its raw OpenNeuro file |
| `excluded.csv` | the 2,620 candidate cases that were not released, by reason |
| `q2_fences.csv` | for every case, whether Q2 used its source's fences or the pooled fences, and the size of the fitting pool |
| `training_selections.csv` | the cases of the paper's lower-risk, random and higher-risk training selections that are in the release (906, 935, 931) |
| `archives.csv` | size, case count and SHA-256 of every tar under `cases/` |
| `checksums.csv` | size and SHA-256 of every file inside the tars |
