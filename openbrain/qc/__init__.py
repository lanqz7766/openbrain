# SPDX-License-Identifier: MIT
"""QC reference-free QC pipeline for OpenBrain v1.0.

Each Q axis is a self-contained module with a `compute(...)` entry-point.
The combiner (`combine`) computes percentile risks r_k = ECDF(Q_k) by
ranking each case over the whole input table and the average
R = mean(r_1, ..., r_7), then assigns HighConf / Standard / Flagged tiers
using the 25%/75% quantiles of R.

See `docs/quality_checks.md` and Appendix I of the paper for the
mathematical definitions.
"""
from __future__ import annotations

NUM_LABELS = 111 # 0..110 inclusive (background + 110 anatomical labels)
MIN_ROI_VOXELS = 50 # default minimum ROI size for inclusion in axis
DEFAULT_K_HIGHCONF = 0.25
DEFAULT_K_FLAGGED_LOWER = 0.75

__all__ = [
    "NUM_LABELS",
    "MIN_ROI_VOXELS",
    "DEFAULT_K_HIGHCONF",
    "DEFAULT_K_FLAGGED_LOWER",
]

__version__ = "1.0.0"
