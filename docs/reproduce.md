# Reproducing the paper

Everything below runs on a CPU. The evaluation cohorts (IXI, BrainLat,
Mindboggle-101) are not part of the release; get them from their owners.

## Setup

```bash
git clone https://github.com/lanqz7766/openbrain.git && cd openbrain
python -m venv .venv && source .venv/bin/activate
pip install -e .

export RELEASE=/path/to/openbrain_v1_0     # where the dataset is downloaded
export COHORTS=/path/to/eval_cohorts       # one subfolder per evaluation cohort
export OUT=/path/to/outputs
```

```bash
hf download lanqz7766/OpenBrain --repo-type dataset --local-dir $RELEASE \
    --include "metadata/**" --include "sample/**" --include "LICENSE"
```

## Tier counts

```bash
python -m openbrain.evaluation.tier_counts \
    --manifest $RELEASE/metadata/quality.csv --output $OUT/tiers
```

Re-derives the tiers from `R` with the release cut points and reports how
often they agree with the `tier` column; the global counts should be
8,960 / 17,918 / 8,960.

## Macro Dice on an evaluation cohort

```bash
python -m openbrain.evaluation.macro_dice \
    --pred $COHORTS/<cohort>/preds --gt $COHORTS/<cohort>/labels \
    --label-map data/label_map.json --exclude-ids data/excluded_labels.txt \
    --output $OUT/dice/<cohort>.csv
```

Writes one row per case and region in the 96-region evaluation space.

## Retention curves (Figure 2 of the paper)

With one risk CSV per cohort in `$OUT/risk` and the Dice CSVs from the previous step in `$OUT/dice`:

```bash
python -m openbrain.evaluation.retention \
    --risk-dir $OUT/risk --dice-dir $OUT/dice --cohorts IXI BrainLat Mindboggle-101 \
    --exclude-ids data/excluded_labels.txt --output-dir $OUT/retention
```

## Training-set selection and paired contrasts (Table 3 and Figure 4)

The three 1,000-case training sets are built from the quality manifest:

```bash
for s in lower_risk random_baseline higher_risk; do
  python -m openbrain.selection.$s --manifest $RELEASE/metadata/quality.csv \
      --metadata $RELEASE/metadata/cases.csv --n 1000 --no-iq --output $OUT/$s.csv
done
```

`--no-iq` is needed because the selectors in the paper also used an
image-quality score from the OpenMind metadata, which is not part of this
release; with the flag they rank by `R` alone. Training nnU-Net on each set takes several GPU
days and is not covered here. With predictions from the three models on a
cohort:

```bash
python -m openbrain.evaluation.paired_contrasts \
    --pred-high $OUT/preds/lower_risk --pred-random $OUT/preds/random --pred-low $OUT/preds/higher_risk \
    --gt $COHORTS/<cohort>/labels --output $OUT/<cohort>_paired.csv
```

## Small structures (Appendix E)

```bash
python -m openbrain.evaluation.small_structures \
    --label-map data/label_map.json --risk-dir $OUT/risk --dice-dir $OUT/dice \
    --cohorts IXI BrainLat Mindboggle-101 --output $OUT/small_structures.csv
```

## Sample

```bash
bash openbrain/sample/download.sh openbrain_sample
python openbrain/sample/preview.py --case-dir openbrain_sample/sample/cases/<case_id> --output preview.png
```
