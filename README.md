# OpenBrain v1.0

Confidence-aware whole-brain MRI parcellation release: 35,838 brain-extracted
T1-weighted MRI cases drawn from 607 OpenNeuro source datasets carrying
CC0 / public-domain-compatible license metadata. Each case ships a SynthStrip
brain mask, an automated whole-brain segmentation, seven reference-free
quality-control axes (Q1 brain leakage, Q2 volume outliers, Q3 distant
components, Q4 intensity coefficient of variation, Q5 committee Dice,
Q6 voxel entropy, Q7 volume coefficient of variation), seven within-source
percentile risks ($r_1$..$r_7$), the combined risk $R = \overline{r_k}$, and a
confidence tier (HighConf / Standard / Flagged) at $K{=}25\%/75\%$ percentile
cuts on $R$. Labels are AI-generated; the release does not claim manual
ground-truth quality. Downstream users can filter, re-weight, audit, or
re-threshold without rerunning inference.

## Quick start

```bash
git clone <this-repo>.git openbrain && cd openbrain
pip install -e .

# compute the QC manifest on a single case
python -m qc --case-dir <path/to/case>

# load the released manifest and inspect tier composition
python -c "import pandas as pd; m = pd.read_csv('<path-to>/openbrain_v1_0_qc_manifest.csv'); print(m['tier'].value_counts())"
```

## Dataset

OpenBrain v1.0 is hosted on Hugging Face during double-blind review:

`https://huggingface.co/datasets/openbrain-anon/openbrain_v1_0`

The release tree mirrors the metadata files in `data/` and is described by
a Croissant 1.0 metadata file (`croissant.json`).

## Citation

```bibtex
@inproceedings{openbrain2026,
  author    = {Anonymous},
  title     = {OpenBrain: A Large-Scale, Multi-Source, and Confidence-Tiered
               Brain MRI Pseudo-Label Release for Population-Scale
               Parcellation and Reference-Free Quality Control},
  booktitle = {Advances in Neural Information Processing Systems
               Evaluations \& Datasets Track},
  year      = {2026}
}
```

## License

- Code in this repository: MIT (see `LICENSE`).
- Dataset (images, masks, labels, manifests) released on Hugging Face: CC0 1.0.

## Repository contents

- `qc/` — the seven per-case QC axes (Q1..Q7), the source-local percentile
  combiner, and the `python -m qc` driver.
- `selection/` — High / Random / Low training-arm selectors that turn the
  released manifest into fixed-size training subsets.
- `eval/` — evaluation entrypoints: 96-label macro Dice, retention-quality
  sweep, partial correlations between QC axes, paired-contrast comparisons,
  small-structure sensitivity.
- `data/` — `label_map.json`, `def_a_label_ids.json` (the 16-ROI
  small-structure subset used in the paper), and `manifest_schema.md`.
- `croissant.json` — machine-readable dataset card.
- `DATASHEET.md` — Gebru-style datasheet.
- `REVIEWER_REPLICATION.md` — copy-paste walk-through to reproduce the
  paper's main numbers.

## Anonymous repository statement

This repository accompanies a NeurIPS 2026 Evaluations & Datasets Track
submission and is provided without author or institutional information for
double-blind review.
