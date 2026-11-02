# SPDX-License-Identifier: MIT
"""Select the lower-risk training set used in the paper.

For each case in the release quality manifest, compute a joint score
``rank(R) + rank(-IQ)`` and walk the sorted-ascending list with a
round-robin per-source cap. This maximises source diversity in the
selected subset (the published selection covered ~600+ unique source
datasets at N=1000).

The result is a subset manifest CSV ready to be fed to
``inference/build_nnunet_dataset.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from openbrain.selection.common import (
    add_joint_score,
    load_manifest,
    round_robin_select,
    summarise_arm,
)

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="release quality manifest CSV (quality.csv) (output of openbrain.qc.combine)")
    ap.add_argument("--metadata", type=Path, default=None,
                    help="optional metadata CSV with image_quality_score etc.")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--no-iq", action="store_true",
                    help="ignore image_quality_score and rank only by R")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--name", default="lower_risk")
    args = ap.parse_args(argv)

    df = load_manifest(args.manifest, args.metadata, require_iq=not args.no_iq)
    df = add_joint_score(df, use_iq=not args.no_iq)
    df_sorted = df.sort_values("joint_score").reset_index(drop=True)
    print(f"[eligible] {len(df_sorted)} cases, "
          f"sources={df_sorted['source_dataset'].nunique()}")

    arm = round_robin_select(df_sorted, args.n)
    arm["selection"] = args.name
    print(summarise_arm(arm, args.name))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arm.to_csv(args.output, index=False)
    print(f"[ok] wrote {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
