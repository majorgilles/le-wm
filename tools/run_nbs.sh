#!/bin/bash
# Execute one or more notebooks in place, from the repo root:
#
#     ./tools/run_nbs.sh 00_module 01_jepa
#     ./tools/run_nbs.sh 03_tutorial_pusht     # slow: real training if enabled
#
# Names are given without the .ipynb suffix.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/nbs" || exit 1

status=0
for nb in "$@"; do
  echo "=== $nb ==="
  if uv run jupyter nbconvert --to notebook --execute --inplace \
       --ExecutePreprocessor.timeout=1800 "$nb.ipynb" 2>&1 | tail -6; then
    echo "  ok"
  else
    echo "  FAILED"
    status=1
  fi
done
exit $status
