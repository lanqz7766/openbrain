#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-openbrain_sample}"
REPO_ID="${OPENBRAIN_HF_REPO:-lanqz7766/OpenBrain}"

mkdir -p "${OUT_DIR}"

hf download "${REPO_ID}" \
  --repo-type dataset \
  --include "README.md" \
  --include "LICENSE" \
  --include "croissant.json" \
  --include "sample/**" \
  --local-dir "${OUT_DIR}"

echo "Downloaded OpenBrain sample to ${OUT_DIR}"
echo "Manifest: ${OUT_DIR}/sample/sample_manifest.csv"
echo "Preview one case with:"
echo "  python openbrain/sample/preview.py --case-dir ${OUT_DIR}/sample/cases/<case_id> --output preview.png"
