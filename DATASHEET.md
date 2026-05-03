# OpenBrain v1.0 — Datasheet

A Gebru-style datasheet for the OpenBrain v1.0 dataset release. The full
machine-readable record is in `croissant.json`.

---

## 1. Motivation

**For what purpose was the dataset created?**
OpenBrain v1.0 was created to enable population-scale benchmarking of
whole-brain segmentation models on a large, license-clean, multi-source
T1-weighted MRI corpus, with built-in reference-free quality-control (QC)
signals so that researchers can study how downstream metrics depend on
pseudo-label confidence without re-running expensive committee inference.

**Who created the dataset and on behalf of which entity?**
Anonymous (NeurIPS 2026 Datasets & Benchmarks track submission).

**Who funded the creation of the dataset?**
Funding information is omitted for double-blind review. A funding statement
will be added to the camera-ready release.

---

## 2. Composition

**What do the instances represent?**
Each instance is a single brain MRI scan (subject session) of a human
participant, drawn from a public OpenNeuro snapshot. An instance includes:
the revised T1-weighted image, the SynthStrip brain mask, an automated
whole-brain segmentation label, seven raw QC values (Q1..Q7), seven
within-source percentile risks (r1..r7), the combined risk R = mean(r1..r7),
and a confidence tier (HighConf, Standard, or Flagged).

**How many instances are there in total?**
35,838 cases drawn from 607 distinct OpenNeuro source datasets.

**Does the dataset contain all possible instances or is it a sample?**
It is a license-filtered, license-classified subset of the OpenNeuro corpus
restricted to snapshots whose metadata indicates a CC0 or
public-domain-compatible license, and from which we could exactly match the
raw T1w to a verified BIDS path.

**What data does each instance consist of?**
NIfTI images (`image.nii.gz`, `brain_mask.nii.gz`,
`whole_brain_segmentation.nii.gz`) plus row-aligned entries in the
metadata CSVs.

**Is there a label or target associated with each instance?**
Yes. The labels are 110-class anatomical parcellations produced by the M1
segmenter (a Dataset012-trained variant of the ModernSurfer architecture).
**Labels are AI-generated and should not be interpreted as manual ground
truth.** They are intended for benchmarking, not for clinical use.

**Is any information missing?**
A small fraction of cases have partial coverage of the M2 and M3 committee
predictions; for those cases Q5/Q6/Q7 are NaN and the tier defaults to
Standard. This is documented in the manifest schema.

**Are there recommended data splits?**
We deliberately do not impose train/val/test splits. The release ships
per-source provenance so that users can construct cross-source splits
appropriate for their evaluation question.

**Are there errors, sources of noise, or redundancies?**
Yes. The dataset includes pseudo-labels and inherits any biases of the M1
model. Source-cohort imbalance (some OpenNeuro datasets contribute many
sessions, others contribute one) is preserved by design and documented in
`metadata/openneuro_dataset_audit.csv`.

**Is the dataset self-contained?**
The release is self-contained for benchmarking. Downstream evaluation
cohorts (IXI, Mindboggle-101, BrainLat) are NOT bundled and must be
obtained from their upstream owners under their respective licenses.

**Does the dataset contain confidential, offensive, or sensitive data?**
No. All sources are de-identified OpenNeuro snapshots with CC0 or
public-domain-compatible license metadata. Subject IDs are the OpenNeuro
case identifiers; no auxiliary demographic linkage is provided.

---

## 3. Collection process

**How was the data acquired?**
T1w volumes and metadata were pulled from public OpenNeuro snapshots,
subject to license filtering. SynthStrip masks were computed locally from
the revised images. Whole-brain segmentations were produced locally by the
M1 segmenter; an M2 (Swin UNETR) and M3 (UNETR) committee was run on a
subset to compute Q5/Q6/Q7.

**Over what timeframe was the data collected?**
OpenNeuro snapshots were pulled and verified between 2025-08 and 2026-04.

**Were any ethical-review processes conducted?**
The upstream OpenNeuro datasets are released under CC0 / public-domain
licenses by their original IRB-approved source studies. We performed an
additional license audit (see `metadata/openneuro_dataset_audit.csv`).

**Per-source license-class counts:**
| License class | Source datasets |
|---|---:|
| CC0 | 595 |
| Public-domain-compatible (PDDL, Other-Open) | 12 |
| **Total** | **607** |

(Counts derived from `metadata/openneuro_dataset_audit.csv`. Verify the
exact split with `pandas.read_csv(...).license_class.value_counts()`.)

---

## 4. Preprocessing / cleaning / labelling

**Was any preprocessing or cleaning of the data done?**
Yes. The pipeline performed: license-class filtering, exact raw-BIDS
matching, intensity revision (the "revised" T1w convention), SynthStrip
brain extraction, and 110-class whole-brain segmentation. Cases with
ambiguous matches, missing raw T1w, or manual-review license status were
excluded.

**Was the raw data saved in addition to the preprocessed data?**
The OpenNeuro source snapshots remain available upstream. We do not
redistribute raw T1w volumes; we redistribute the revised/brain-extracted
images derived from them under the CC0 source licenses.

**Is the software used to preprocess/clean/label the instances available?**
Yes. The QC computation, retention-sweep, paired-contrast, and dataset
construction code is in this repository. Inference recipes for M1/M2/M3 are
in `inference/`.

---

## 5. Uses

**Has the dataset been used for any tasks already?**
The dataset is the basis for the OpenBrain v1.0 paper, which evaluates:
(a) per-axis Q_k vs macro-Dice retention curves on three external cohorts,
(b) Stage-B paired contrasts among High/Random/Low confidence training arms,
and (c) Stage-C deployment behaviour at population scale.

**Is there a repository that links to papers or systems that use the dataset?**
This GitHub repository is the canonical entry point. The Croissant record
(`croissant.json`) contains the canonical dataset URL.

**What other tasks could the dataset be used for?**
- Reference-free QC method development.
- Studies of pseudo-label noise on downstream segmentation training.
- Source-bias and cross-source generalisation analyses.
- Pretraining for foundation models that condition on brain MRI.

**Is there anything about the composition or use of the dataset that might
impact future uses?**
Users must remember that the labels are AI-generated. Models trained on
OpenBrain v1.0 inherit any systematic biases of the M1 segmenter. The
HighConf tier is *not* a guarantee of label correctness; it is a relative
ranking within OpenBrain.

**Discouraged use case (worked example).**
Do *not* use OpenBrain labels to train a model intended for clinical
neurosurgical planning where ROI boundaries must be sub-millimetre
accurate. The labels are pseudo-labels with a Dice-to-manual ceiling
documented in the paper, and small-structure ROIs (Def-A 16-ROI panel)
in particular have higher residual error than cortical lobes.

**Are there tasks for which the dataset should not be used?**
Tasks requiring manually-verified anatomy. Tasks that re-identify
participants. Any clinical decision-making.

---

## 6. Distribution

**How will the dataset be distributed?**
The dataset is hosted on Hugging Face under an anonymous user during
double-blind review:
`https://huggingface.co/datasets/openbrain-anonymous/openbrain_v1_0`.
The Croissant 1.0 record at `croissant.json` is the canonical machine-
readable description.

**Is there a fee?**
No. Both the code (MIT) and the dataset (CC0) are freely redistributable.

**When will the dataset be distributed?**
Available immediately for review and post-publication.

**Will the dataset be distributed under a copyright or other IP license,
and/or under applicable terms of use (ToU)?**
Dataset: CC0 1.0 Universal (subject to upstream OpenNeuro source licenses,
which are themselves CC0 or public-domain-compatible by construction).
Code: MIT.

**Have any third parties imposed IP-based or other restrictions on the
data?**
No additional restrictions beyond the upstream CC0 / public-domain
compatibility verified in the license audit.

---

## 7. Maintenance

**Who will be supporting/hosting/maintaining the dataset?**
The anonymous authors during the review period; a permanent maintainer
identity will be added at camera-ready.

**How can the owner/curator/manager of the dataset be contacted?**
For double-blind review: `openbrain-anonymous-contact@neurips-anonymous.org`
(placeholder anonymous routing). After camera-ready: a permanent contact
email and a GitHub issues queue.

**Is there an erratum?**
The repository will track known issues in `KNOWN_ISSUES.md` after
publication.

**Will the dataset be updated?**
Yes. We plan minor versions (v1.x) at most quarterly, used to add new
QC axes or correct license-audit errors. Breaking changes (image
respacing, label-space remap, mask redefinition) are reserved for a
v2.0.

**Versioning policy.**
Each release ships a `VERSION` file and a `metadata/sha256_manifest.csv`
that lists every redistributed file's SHA-256. Reviewers can verify a
local copy by re-hashing the files and diffing the manifest.

**Takedown policy.**
If a participant or upstream OpenNeuro maintainer requests removal of a
specific case or source dataset, we will issue a patch release that
drops the affected rows and bumps the patch version. Requests should be
sent to the contact address above.

**Will older versions be supported?**
Yes. v1.0 is the long-lived "submission" release; subsequent v1.x
versions will be additive whenever possible, and we will keep all
historical versions accessible on Hugging Face under their tagged
revisions.

**If others want to extend/augment/build on the dataset, is there a
mechanism for them to do so?**
Yes. Users can fork this repository and contribute new QC axes via
pull request once the repo is de-anonymised. The `core7/` package is
designed to be extensible (each axis is a self-contained module).
