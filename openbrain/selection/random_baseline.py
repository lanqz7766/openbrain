# SPDX-License-Identifier: MIT
"""Select the random training set used in the paper.

Sample ``n_sources`` source datasets uniformly at random, then sample
``n`` cases uniformly at random from the union of those sources. This
produces a selection whose source diversity is moderate — by construction
between the lower-risk selection and the higher-risk selection.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openbrain.selection.common import (
    DEFAULT_SOURCE_COL,
    load_manifest,
    summarise_arm,
)

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, default=None)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-sources", type=int, default=500,
                    help="number of source datasets to draw the random pool from")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-iq", action="store_true",
                    help="don't require IQ to be present")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--name", default="random")
    args = ap.parse_args(argv)

    df = load_manifest(args.manifest, args.metadata, require_iq=not args.no_iq)
    rng = np.random.default_rng(args.seed)
    all_sources = df[DEFAULT_SOURCE_COL].unique()
    if args.n_sources < len(all_sources):
        chosen = rng.choice(all_sources, size=args.n_sources, replace=False)
        pool = df[df[DEFAULT_SOURCE_COL].isin(chosen)]
    else:
        pool = df
    print(f"[pool] {len(pool)} cases in {pool[DEFAULT_SOURCE_COL].nunique()} sources")
    if len(pool) < args.n:
        raise SystemExit(f"pool too small: {len(pool)} < {args.n}")
    idx = rng.choice(len(pool), size=args.n, replace=False)
    arm = pool.iloc[idx].copy()
    arm["selection"] = args.name
    print(summarise_arm(arm, args.name))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arm.to_csv(args.output, index=False)
    print(f"[ok] wrote {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
