#!/usr/bin/env bash
# Frees /private/tmp by removing Claude Code session output files,
# then reports other large candidates for cleanup.
# SAFE: never touches Celery, REFOLD, InteractionFormer, or ORACLE.

set -euo pipefail

echo "=== Step 1: Free /private/tmp (Claude session output files) ==="
TASK_DIR="/private/tmp/claude-502"
if [ -d "$TASK_DIR" ]; then
    BEFORE=$(du -sh "$TASK_DIR" 2>/dev/null | cut -f1)
    find "$TASK_DIR" -name "*.output" -delete 2>/dev/null && echo "Deleted *.output files (was $BEFORE)"
    find "$TASK_DIR" -name "*.err"    -delete 2>/dev/null && echo "Deleted *.err files"
    echo "Done. /tmp free: $(df -h /private/tmp | awk 'NR==2{print $4}')"
else
    echo "/private/tmp/claude-502 not found"
fi

echo ""
echo "=== Step 2: Large /tmp files from other processes ==="
find /private/tmp -maxdepth 3 -size +10M -not -path "*/claude-502/*" 2>/dev/null | \
    xargs -I{} du -sh {} 2>/dev/null | sort -rh | head -20 || true

echo ""
echo "=== Step 3: Large cache directories (safe to wipe) ==="
CACHES=(
    "$HOME/Library/Caches/pip"
    "$HOME/.cache/pip"
    "$HOME/.cache/torch/hub"
    "$HOME/.cache/huggingface/datasets"
    "$HOME/.cache/numba"
    "$HOME/.nuxt"
    "$HOME/.npm/_cacache"
    "/opt/homebrew/Caskroom/miniforge/base/pkgs"
    "/opt/homebrew/Caskroom/miniforge/base/conda-bld"
)
echo "Candidate cache dirs (not auto-deleted, review first):"
for c in "${CACHES[@]}"; do
    if [ -d "$c" ]; then
        SZ=$(du -sh "$c" 2>/dev/null | cut -f1)
        echo "  $SZ  $c"
    fi
done

echo ""
echo "=== Step 4: Auto-wipe known safe caches ==="
# pip cache is always safe to delete
if [ -d "$HOME/.cache/pip" ]; then
    pip cache purge 2>/dev/null && echo "pip cache purged" || rm -rf "$HOME/.cache/pip" && echo "pip cache removed"
fi

# Numba cache (regenerated automatically)
if [ -d "$HOME/.cache/numba" ]; then
    rm -rf "$HOME/.cache/numba"
    echo "Numba cache removed"
fi

# __pycache__ dirs OUTSIDE the four protected projects
echo ""
echo "=== Step 5: Remove __pycache__ outside protected projects ==="
PROTECTED="Celery|REFOLD|InteractionFormer|ORACLE"
find "$HOME" -type d -name "__pycache__" 2>/dev/null | \
    grep -vE "$PROTECTED" | \
    while read d; do
        sz=$(du -sh "$d" 2>/dev/null | cut -f1)
        rm -rf "$d"
        echo "  removed $sz  $d"
    done | head -30 || true

# .pyc files outside protected projects
find "$HOME" -name "*.pyc" 2>/dev/null | grep -vE "$PROTECTED" | xargs rm -f 2>/dev/null && \
    echo "Removed stray .pyc files outside protected projects"

echo ""
echo "=== Done. Final /tmp free space: $(df -h /private/tmp | awk 'NR==2{print $4}') ==="
echo "Bash tool should work again now."
