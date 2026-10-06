# OpenBrain v1.0

**OpenBrain: An Auditable Generated-Label Release for Whole-Brain MRI Parcellation**

Accepted to NeurIPS 2026, Evaluations & Datasets Track.

[Paper](https://openreview.net/forum?id=dR9OMtUEEC) · [Dataset](https://huggingface.co/datasets/lanqz7766/OpenBrain)

OpenBrain is a release of brain MRI parcellations with per-case quality
information attached to every label. We ship 35,838 T1-weighted MRI
cases drawn from 607 OpenNeuro source datasets that carry CC0 or
public-domain-compatible licenses. Each case carries a brain mask, an
automated whole-brain parcellation, and a quality manifest that lets a
downstream user judge how much to trust the labels for that case.

The labels are produced by a segmentation model, not by a human. The
release does not claim manual ground-truth quality. What it offers is
*per-case quality evidence* alongside the labels: seven reference-free
quality checks per case, plus a single combined risk score and a
confidence tier, so users can filter, weight, audit, or re-threshold
without rerunning the segmentation model.

## What's in the release

For every released case:

- the preprocessed T1-weighted image,
- a brain mask (SynthStrip),
- an automated whole-brain segmentation,
- seven per-case quality scores (the QC axes below) plus their
  within-source percentile risks and a combined risk score,
- a confidence tier: HighConf, Standard, or Flagged.

The seven QC axes:

| Axis | What it checks | Lower means better? |
|------|----------------|---------------------|
| Q1 | label voxels falling outside the brain mask (leakage) | yes |
| Q2 | regions whose volume is an outlier within the source dataset | yes |
| Q3 | label components disconnected from the main region body | yes |
| Q4 | within-region intensity inhomogeneity in the T1 image | yes |
| Q5 | spatial agreement among three independent segmentation models | no (higher = better) |
| Q6 | voxel-level vote uncertainty among the three models | yes |
| Q7 | volume disagreement among the three models per region | yes |

Each axis is converted to a within-source percentile rank, and the
combined risk is the average of those seven ranks. The HighConf,
Standard, and Flagged tiers are the lowest 25%, middle 50%, and highest
25% of the combined risk within the released cases.

## Quick start

```bash
git clone https://github.com/lanqz7766/openbrain.git
cd openbrain
pip install -e .

# Compute per-case QC values (Q2 is computed across the cohort)
python -m qc --case-dir <path/to/cases> \
  --m1-dir <path/to/primary_parcellations> \
  --m2-dir <path/to/committee_2> --m3-dir <path/to/committee_3> \
  --output qc.csv

# Inspect tier composition of the released manifest
python -c "import pandas as pd; m = pd.read_csv('<path-to>/openbrain_v1_0_qc_manifest.csv'); print(m['tier'].value_counts())"
```

## Reviewer sample

The Hugging Face dataset includes a standalone reviewer preview subset under
`sample/`. It contains 64 source-diverse HighConf cases, each with:

- `image.nii.gz`
- `brain_mask.nii.gz`
- `whole_brain_segmentation.nii.gz`

Download only the sample:

```bash
bash examples/download_sample.sh openbrain_v1_0_sample
```

Create a quick PNG preview:

```bash
python examples/preview_sample_case.py \
  --case-dir openbrain_v1_0_sample/sample/cases/<case_id> \
  --output preview.png
```

The sample is selected for low combined risk and visual cleanliness, so it is
appropriate for reviewer preview and smoke tests. It is still a pseudo-label
sample, not manual ground truth.

## Dataset hosting

OpenBrain v1.0 is publicly available on [Hugging Face](https://huggingface.co/datasets/lanqz7766/OpenBrain).

A machine-readable dataset card following the Croissant 1.1 spec is
available as `croissant.json` in both this repo and the dataset.

## Citation

```bibtex
@inproceedings{lan2026openbrain,
  title = {OpenBrain: An Auditable Generated-Label Release for Whole-Brain MRI Parcellation},
  author = {Lan, Qizhen and Hsu, Yu-Chun and Wei, Yuxiang and Zhu, Lijing and Sun, Zenan and He, Liang and Yu, Lishan and Jiang, Xiaoqian},
  booktitle = {Advances in Neural Information Processing Systems, Evaluations \& Datasets Track},
  year = {2026}
}
```

## License

Code in this repository is released under the MIT license. The dataset
itself is released under CC0 1.0 (see the Hugging Face dataset page).

## Repository layout

- `qc/` — implementation of the seven QC checks and the per-source
  percentile combiner that produces the combined risk and tier for a
  given case. `python -m qc --help` for the command-line driver.
- `selection/` — three selection scripts that turn the released manifest
  into matched-size training subsets: a high-confidence subset, a
  random baseline, and a low-confidence subset.
- `eval/` — evaluation entrypoints used in the paper: 96-label macro
  Dice, retention-quality sweep, partial correlations across QC axes,
  paired contrasts between training subsets, and the small-structure
  sensitivity analysis.
- `examples/` — sample download and NIfTI preview scripts for reviewer
  inspection.
- `data/` — reference files: the 110-class label map, the 16-region
  small-structure subset used in the paper, and a schema description of
  the released QC manifest CSV.
- `croissant.json` — the Croissant 1.1 dataset card.
- `DATASHEET.md` — datasheet following the format of Gebru et al.
- `REVIEWER_REPLICATION.md` — a copy-paste walk-through that reproduces
  the paper's main numbers from the released manifest plus a
  reviewer-supplied evaluation cohort.

## Authors and contact

Qizhen Lan, Yu-Chun Hsu, Yuxiang Wei, Lijing Zhu, Zenan Sun, Liang He, Lishan Yu, Xiaoqian Jiang.

Maintained by Qizhen Lan. For questions about the release, contact [Qizhen.Lan@uth.tmc.edu](mailto:Qizhen.Lan@uth.tmc.edu).
