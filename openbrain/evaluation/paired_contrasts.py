# SPDX-License-Identifier: MIT
"""paired contrasts among the lower-risk / random / higher-risk selections.

For each (eval cohort, ROI panel), compute paired deltas between the
three selections on the held-out evaluation cohort:

    delta_HR = mean_c (Dice_high(c) - Dice_random(c))
    delta_HL = mean_c (Dice_high(c) - Dice_low(c))
    delta_RL = mean_c (Dice_random(c) - Dice_low(c))

with 5000-bootstrap 95% CIs on the per-case paired differences and
Wilcoxon signed-rank one-sided p-values for the ``high > random``,
``high > low``, and ``random > low`` directional hypotheses.

Inputs:

    --pred-high directory of per-case predicted labels for the lower-risk selection
    --pred-random directory of per-case predicted labels for the random selection
    --pred-low directory of per-case predicted labels for the higher-risk selection
    --gt directory of ground-truth labels for the eval cohort

The script computes per-case macro Dice on each selection using the same
exclude-id list and label map used by ``openbrain.evaluation.macro_dice``, then
restricts to cases that have predictions in *all three* selections.

This produces the rows of paper Table T2 (Paired contrasts, IXI / BL /
Mb-101 / aggregate) and the deltas plotted in Figure 4.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from openbrain.evaluation.macro_dice import (
    _load_label_ids,
    _load_nifti_int,
    _normalize_key,
    per_roi_dice,
)

def _macro_dice_dir(pred_dir: Path, gt_paths: dict[str, Path],
                    included_ids: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    pred_paths = {_normalize_key(p.name): p for p in pred_dir.glob("*.nii.gz")}
    common = set(pred_paths) & set(gt_paths)
    for cid in sorted(common):
        gt = _load_nifti_int(gt_paths[cid])
        pr = _load_nifti_int(pred_paths[cid])
        if gt.shape != pr.shape:
            continue
        rows = per_roi_dice(gt, pr, included_ids)
        if rows:
            out[cid] = float(np.mean([r["dice"] for r in rows]))
    return out

def _det_seed(*parts) -> int:
    return int.from_bytes(
        hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:4], "big"
    )

def _paired_boot(diffs: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size < 2:
        return float("nan"), float("nan"), float("nan")
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = diffs[rng.integers(0, diffs.size, size=diffs.size)].mean()
    return float(diffs.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-high", type=Path, required=True)
    ap.add_argument("--pred-random", type=Path, required=True)
    ap.add_argument("--pred-low", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--exclude-ids", type=Path, default=None)
    ap.add_argument("--cohort-name", default="cohort")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    excl = _load_label_ids(args.exclude_ids)
    included = sorted([L for L in range(1, 111) if L not in excl])
    gt_paths = {_normalize_key(p.name): p for p in args.gt.glob("*.nii.gz")}
    print(f"[gt] {len(gt_paths)} cases")

    high = _macro_dice_dir(args.pred_high, gt_paths, included)
    rnd = _macro_dice_dir(args.pred_random, gt_paths, included)
    low = _macro_dice_dir(args.pred_low, gt_paths, included)
    common = sorted(set(high) & set(rnd) & set(low))
    print(f"[paired] {len(common)} cases scored on all three selections")
    if not common:
        raise SystemExit("no common cases across selections")

    H = np.array([high[c] for c in common])
    R = np.array([rnd[c] for c in common])
    L = np.array([low[c] for c in common])

    rows = []
    for name, a, b in [("high_minus_random", H, R),
                       ("high_minus_low", H, L),
                       ("random_minus_low", R, L)]:
        diffs = a - b
        rng = np.random.default_rng(_det_seed(args.cohort_name, name))
        d, lo, hi = _paired_boot(diffs, args.n_boot, rng)
        try:
            _, pval = wilcoxon(diffs, alternative="greater")
            pval = float(pval)
        except Exception:
            pval = float("nan")
        rows.append(dict(
            cohort=args.cohort_name, contrast=name, n=len(common),
            delta=d, ci_lo=lo, ci_hi=hi,
            wilcoxon_one_sided_p=pval,
        ))

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))
    print(f"[ok] wrote {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
