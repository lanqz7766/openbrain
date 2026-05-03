# SPDX-License-Identifier: MIT
"""Select the Low-confidence training arm ( / the paper's training arm selector).

Sort the eligible pool by joint_score = rank(R) + rank(-IQ), in
*descending* order (high-risk + low-IQ first), then walk the sorted
list with a per-source cap. This gives an arm that is biased toward
flagged / low-IQ cases while still covering at least ``n / cap`` unique
source datasets.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from selection._common import (
 add_joint_score,
 greedy_capped_select,
 load_manifest,
 summarise_arm,
)

def main(argv: list[str] | None = None) -> int:
 ap = argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--manifest", type=Path, required=True)
 ap.add_argument("--metadata", type=Path, default=None)
 ap.add_argument("--n", type=int, default=1000)
 ap.add_argument("--per-source-cap", type=int, default=5,
 help="maximum picks from any single source dataset")
 ap.add_argument("--no-iq", action="store_true")
 ap.add_argument("--output", type=Path, required=True)
 ap.add_argument("--arm-name", default="C_LowConf")
 args = ap.parse_args(argv)

 df = load_manifest(args.manifest, args.metadata, require_iq=not args.no_iq)
 df = add_joint_score(df, use_iq=not args.no_iq)
 df_sorted = df.sort_values("joint_score", ascending=False).reset_index(drop=True)
 print(f"[eligible] {len(df_sorted)} cases, "
 f"sources={df_sorted['source_dataset'].nunique()}")

 arm = greedy_capped_select(df_sorted, args.n, args.per_source_cap)
 arm["arm"] = args.arm_name
 print(summarise_arm(arm, args.arm_name))

 args.output.parent.mkdir(parents=True, exist_ok=True)
 arm.to_csv(args.output, index=False)
 print(f"[ok] wrote {args.output}")
 return 0

if __name__ == "__main__":
 raise SystemExit(main())
