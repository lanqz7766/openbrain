# OpenBrain

Whole-brain MRI parcellations for 35,838 public T1-weighted scans, each with a per-case quality record.

Paper: [OpenBrain: An Auditable Generated-Label Release for Whole-Brain MRI Parcellation](https://openreview.net/forum?id=dR9OMtUEEC), NeurIPS 2026 Evaluations & Datasets Track.
Data: [huggingface.co/datasets/lanqz7766/OpenBrain](https://huggingface.co/datasets/lanqz7766/OpenBrain) (CC0).

![Overview](assets/overview.png)

The scans come from 607 OpenNeuro datasets. Each has a brain mask and a
110-region parcellation produced by ModernSurfer, a 3D residual U-Net trained
on FreeSurfer labels. The labels are model output, not manual annotation.
Every scan also carries seven automatic quality checks, a combined risk score
`R`, and a confidence tier, so that users can decide which cases to use. This
repository contains the code for the checks, for building training subsets,
and for the experiments in the paper.

## Results

We trained nnU-Net on 1,000 cases from the low-risk end of the release, 1,000
random cases, and 1,000 from the high-risk end, and evaluated on three
external cohorts.

<p align="center"><img src="assets/training_selection.png" width="640" alt="Macro Dice by training selection"></p>

| Training set | Mindboggle-101 (n=101) | BrainLat (n=394) | IXI (n=563) |
|---|---|---|---|
| 1,000 lower-risk cases | **0.746** | **0.719** | **0.769** |
| 1,000 random cases | 0.710 | 0.687 | 0.733 |
| 1,000 higher-risk cases | 0.456 | 0.458 | 0.490 |
| lower-risk minus random (pp, paired 95% CI) | +3.6 [+2.8, +4.4] | +3.2 [+2.6, +3.8] | +3.7 [+3.2, +4.2] |

Within a cohort, keeping the quarter of cases with the lowest `R` raises macro
Dice by 1.3 points on IXI, 3.7 on BrainLat, and 1.5 on Mindboggle-101, the
cohort with manual labels. The lower-risk and random training sets differ in
source composition as well as in risk, and IXI and BrainLat use
FreeSurfer-derived references; see the Limitations section of the paper.

## Example

A low-risk and a high-risk case from the release: parcellation, voxel
disagreement among the three models, and the seven checks as within-dataset
percentiles.

![A low-risk and a high-risk case](assets/release_examples.png)

## Quality checks

| | Check | High value means |
|---|---|---|
| Q1 | mask leakage | labels outside the brain mask |
| Q2 | volume outliers | region volumes unusual for the source dataset |
| Q3 | distant fragments | pieces of a region far from its main body |
| Q4 | intensity variation | large intensity spread inside a region |
| Q5 | committee overlap | agreement with SwinUNETR and UNETR (high is good) |
| Q6 | vote uncertainty | voxel-level disagreement among the three models |
| Q7 | volume disagreement | region volumes differ among the three models |

Each check is converted to a percentile within its OpenNeuro dataset, with Q5
inverted so that higher always means riskier. `R` is the mean of the seven
percentiles. Sorting the release by `R`, the lowest quarter is HighConf, the
middle half Standard, and the top quarter Flagged. Raw values are released, so
other weightings and cutoffs are possible.

## Release contents

- `image.nii.gz`: skull-stripped T1w, 1 mm isotropic
- `brain_mask.nii.gz`: SynthStrip mask
- `whole_brain_segmentation.nii.gz`: 110-region parcellation
- `metadata/openbrain_v1_0_qc_manifest.csv`: `Q1` to `Q7`, `r1` to `r7`, `R`, tier
- `metadata/openbrain_v1_0_metadata.csv`: OpenNeuro dataset, snapshot, DOI, license, original path, and age, sex, and scanner fields where available

Case ids look like `ds000001__sub-01__ses-DEFAULT__sub-01_T1w`.

## Download

```bash
# metadata and QC manifest, about 55 MB
hf download lanqz7766/OpenBrain --repo-type dataset --local-dir openbrain_v1_0 \
    --include "metadata/**" --include "croissant.json" --include "LICENSE"

# 64-case sample
hf download lanqz7766/OpenBrain --repo-type dataset --local-dir openbrain_v1_0 --include "sample/**"

# full release, about 350 GB
hf download lanqz7766/OpenBrain --repo-type dataset --local-dir openbrain_v1_0
```

```python
import pandas as pd
m = pd.read_csv("openbrain_v1_0/metadata/openbrain_v1_0_qc_manifest.csv")
high_conf = m[m["tier"] == "HighConf"]
strict    = m[(m["R"] < 0.3) & (m["r1"] < 0.8)]
```

## Usage

```bash
git clone https://github.com/lanqz7766/openbrain.git && cd openbrain
pip install -e .

python -m qc --help
python -m qc --case-dir <cases> --m1-dir <primary> --m2-dir <committee_2> --m3-dir <committee_3> --output qc.csv
```

Q2 is fit per dataset and needs more than one scan. Q5 to Q7 need the two
committee outputs; without them those columns are NaN.

`REVIEWER_REPLICATION.md` reproduces the tier counts, retention curves, paired
training-selection contrasts, and small-structure analysis from the released
manifest and an evaluation cohort of your own.

```
qc/          the seven checks, percentile conversion, R and tier
selection/   lower-risk, random, and higher-risk training subsets
eval/        macro Dice, retention sweep, axis correlations, paired contrasts
examples/    sample download and PNG preview
data/        label map, 96-region evaluation set, 16-region small-structure set, manifest schema
```

## Limitations

The parcellations follow the FreeSurfer convention of the training labels.
`R` ranks cases within a dataset and is not a probability of error. Age, sex,
and scanner metadata are incomplete, so the release does not support fairness
analyses. The data and code are for research use, not clinical use.

## Citation

```bibtex
@inproceedings{lan2026openbrain,
  title     = {OpenBrain: An Auditable Generated-Label Release for Whole-Brain MRI Parcellation},
  author    = {Lan, Qizhen and Hsu, Yu-Chun and Wei, Yuxiang and Zhu, Lijing and Sun, Zenan and He, Liang and Yu, Lishan and Jiang, Xiaoqian},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```

Code: MIT. Data: CC0. Source scans are from OpenNeuro under CC0 or equivalent terms.

Contact: Qizhen Lan, Qizhen.Lan@uth.tmc.edu.
