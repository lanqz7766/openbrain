# SPDX-License-Identifier: MIT
"""96-label macro Dice on a held-out evaluation cohort.

Given (a) a directory of per-case predicted labels, (b) a directory of
per-case ground-truth labels, (c) the 110-class label map, and (d) a
list of label IDs to exclude from the macro (the canonical 14-label
"unified exclusion" list, leaving 96 foreground labels), compute the
per-case macro Dice as the mean per-ROI Dice over included ROIs that
have at least one voxel in the GT *or* the prediction.

The output is a long-format CSV with columns
``case_id, roi_id, dice, gt_voxels, pred_voxels``. Aggregating by
``case_id`` recovers per-case macro Dice; aggregating with a
``small_structure_flag`` recovers the small-structure macro.

Faithful re-implementation of ``paper/scripts/analysis/compute_78label_dice.py``
(group/Def-A path) plus ``recompute_dice_96.py`` (96-label space). The
two scripts together produce the macro_dice column that
``eval/retention_sweep.py`` consumes.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

def _normalize_key(name: str) -> str:
    k = name[: -len(".nii.gz")] if name.endswith(".nii.gz") else name
    if k.startswith("TestSet_"):
        k = k[len("TestSet_"):]
    return k

def _load_label_ids(path: Path | None) -> set[int]:
    if path is None:
        return set()
    excl: set[int] = {0}
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tok = s.split()[0].split(",")[0].strip()
        if tok.isdigit():
            excl.add(int(tok))
    return excl

def _load_nifti_int(path: Path) -> np.ndarray:
    import SimpleITK as sitk # lazy

    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img).astype(np.int32, copy=False)

def per_roi_dice(
    gt: np.ndarray, pred: np.ndarray, included_ids: list[int]
) -> list[dict]:
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch gt={gt.shape} pred={pred.shape}")
    rows: list[dict] = []
    for L in included_ids:
        gt_mask = gt == L
        pr_mask = pred == L
        gt_n = int(gt_mask.sum())
        pr_n = int(pr_mask.sum())
        if gt_n == 0 and pr_n == 0:
            continue # not applicable
        inter = int((gt_mask & pr_mask).sum())
        denom = gt_n + pr_n
        d = (2.0 * inter / denom) if denom > 0 else 0.0
        rows.append(dict(roi_id=int(L), dice=float(d), gt_voxels=gt_n, pred_voxels=pr_n))
    return rows

def _process_one(task):
    case_id, gt_path, pred_path, included_ids, small_ids = task
    try:
        gt = _load_nifti_int(gt_path)
        pred = _load_nifti_int(pred_path)
    except Exception as e:
        return case_id, None, f"{type(e).__name__}: {e}"
    try:
        rows = per_roi_dice(gt, pred, included_ids)
    except Exception as e:
        return case_id, None, f"{type(e).__name__}: {e}"
    out = []
    for r in rows:
        r["case_id"] = case_id
        r["small_structure_flag"] = bool(r["roi_id"] in small_ids)
        out.append(r)
    return case_id, out, None

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", type=Path, required=True,
                    help="ground-truth label directory (per-case *.nii.gz)")
    ap.add_argument("--pred", type=Path, required=True,
                    help="prediction directory")
    ap.add_argument("--label-map", type=Path, default=Path("data/label_map.json"),
                    help="110-class label map (informational only)")
    ap.add_argument("--exclude-ids", type=Path, default=None,
                    help="path to label-ID exclusion file (background = 0 always excluded)")
    ap.add_argument("--small-ids", type=Path, default=None,
                    help="optional JSON list of small-structure IDs to flag")
    ap.add_argument("--output", type=Path, required=True,
                    help="long-format output CSV (one row per case x ROI)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    excl = _load_label_ids(args.exclude_ids)
    included = sorted([L for L in range(1, 111) if L not in excl])

    small_ids: set[int] = set()
    if args.small_ids and args.small_ids.is_file():
        small_ids = set(int(x) for x in json.loads(args.small_ids.read_text()))

    gts = {_normalize_key(p.name): p for p in args.gt.glob("*.nii.gz")}
    preds = {_normalize_key(p.name): p for p in args.pred.glob("*.nii.gz")}
    common = sorted(set(gts) & set(preds))
    print(f"[files] gt={len(gts)} pred={len(preds)} common={len(common)}")
    if not common:
        raise SystemExit("no overlapping cases between --gt and --pred")

    tasks = [(k, gts[k], preds[k], included, small_ids) for k in common]
    rows: list[dict] = []
    errors: list[tuple[str, str]] = []
    if args.workers <= 1:
        for t in tasks:
            cid, res, err = _process_one(t)
            if err:
                errors.append((cid, err))
            else:
                rows.extend(res or [])
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_process_one, t): t[0] for t in tasks}
            for f in as_completed(futures):
                cid, res, err = f.result()
                if err:
                    errors.append((cid, err))
                else:
                    rows.extend(res or [])

    df = pd.DataFrame(rows).sort_values(["case_id", "roi_id"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[ok] wrote {len(df)} rows -> {args.output}")
    macro = df.groupby("case_id")["dice"].mean()
    print(f"[summary] cases={len(macro)} macro_dice mean={macro.mean():.4f} "
          f"median={macro.median():.4f}")
    if errors:
        err_path = args.output.with_suffix(".errors.log")
        err_path.write_text("\n".join(f"{c}\t{m}" for c, m in errors) + "\n")
        print(f"[warn] {len(errors)} errors -> {err_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
