#!/usr/bin/env bash
# Roll back the constrained integration: revert the patch + reinstall pinned upstream.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/PINNED_VERSION"
: "${MISSALIGN_SRC:?set MISSALIGN_SRC to the patched checkout}"
cd "$MISSALIGN_SRC"
echo "[uninstall] reverting patch + integration module"
rm -f miss_alignment/constrained_integration.py 2>/dev/null || true
git restore . 2>/dev/null || git checkout -- . 2>/dev/null || true
if [ "$MISSALIGN_COMMIT" != "UNPINNED_SET_ON_CLUSTER" ]; then git checkout "$MISSALIGN_COMMIT"; fi
pip install --no-deps -e "$MISSALIGN_SRC"
echo "[uninstall] restored to pinned upstream."
