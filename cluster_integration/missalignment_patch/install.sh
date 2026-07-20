#!/usr/bin/env bash
# Install the constrained MissAlignment integration against the pinned upstream.
# Runs on the cluster (where the MissAlignment source exists). Never edits an
# installed site-package silently: it works on a checked-out, patched source tree
# and prints the resolved import path.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/PINNED_VERSION"

SRC="${MISSALIGN_SRC:-}"
if [ -z "$SRC" ]; then
  : "${MISSALIGN_REPO:?set MISSALIGN_SRC to an existing checkout or MISSALIGN_REPO to a git url}"
  SRC="$(mktemp -d)/miss-alignment"
  git clone "$MISSALIGN_REPO" "$SRC"
fi
echo "[install] MissAlignment source: $SRC"
cd "$SRC"

if [ "$MISSALIGN_COMMIT" != "UNPINNED_SET_ON_CLUSTER" ]; then
  git fetch --all || true
  git checkout "$MISSALIGN_COMMIT"
fi
git rev-parse HEAD > "$HERE/.installed_commit"

# Copy the reference integration module into the package and apply the dispatch patch.
PKG_DIR="$(python -c 'import miss_alignment, os; print(os.path.dirname(miss_alignment.__file__))' 2>/dev/null || true)"
if [ -z "$PKG_DIR" ]; then
  PKG_DIR="$SRC/miss_alignment"
fi
echo "[install] integrating into: $PKG_DIR"
cp "$HERE/constrained_integration.py" "$PKG_DIR/constrained_integration.py"

# Apply the dispatch patch if a concrete diff is present; otherwise instruct.
if ls "$HERE"/patches/*.patch >/dev/null 2>&1; then
  for p in "$HERE"/patches/*.patch; do
    echo "[install] git apply $p"
    git apply --check "$p" && git apply "$p"
  done
else
  echo "[install] NOTE: no concrete .patch present. Wire the dispatcher to call"
  echo "          miss_alignment.constrained_integration.run_constrained_iteration for"
  echo "          alignment in (translation, rigid, similarity). See the template:"
  echo "          $HERE/patches/0001-constrained-dispatch.patch.template"
fi

pip install --no-deps -e "$SRC"
echo "[install] resolved import path:"
python -c "import miss_alignment, miss_alignment.constrained_integration as c; \
print(miss_alignment.__file__); print('supported:', c.SUPPORTED_ALIGNMENTS)"
echo "[install] done. Run probe.sh to validate."
