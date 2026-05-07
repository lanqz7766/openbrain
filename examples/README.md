# OpenBrain sample examples

These scripts are for the 64-case reviewer sample hosted under
`sample/` in the Hugging Face dataset repository.

Download the sample:

```bash
bash examples/download_sample.sh openbrain_v1_0_sample
```

Create a quick PNG preview for one case:

```bash
python examples/preview_sample_case.py \
  --case-dir openbrain_v1_0_sample/sample/cases/<case_id> \
  --output openbrain_v1_0_sample/preview.png
```

The sample cases are selected from the HighConf tier using low combined
risk and visual-cleanliness tie-breaks. They are useful for reviewer
inspection and pipeline smoke tests, but the segmentations remain
algorithmic pseudo-labels rather than manual ground truth.
