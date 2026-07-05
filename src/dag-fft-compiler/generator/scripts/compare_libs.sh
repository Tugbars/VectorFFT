#!/bin/bash
# compare_libs.sh — byte-compare EVERY codelet the production generator
# (lib/, library vfft_v2, driven by bin/) emits against the refactor
# staging tree (new-lib/, library vfft_v2r, driven by new-bin/).
#
# This is the promotion gate for the new-lib refactor (see
# new-docs/75_refactor_stage_log.md, REPO-TREE section). Two passes:
#
#   1. FULL TREE: both gen_set.exe binaries generate every quadrant
#      (the same coverage.ml walk production regen uses) into two
#      throwaway roots; the trees are diff -r'd. gen_set stamps the
#      LOGICAL per-codelet argv into provenance, so tree files are
#      path-neutral and byte-comparable as-is.
#   2. KNOB-ON SPOT CELLS: direct gen_radix runs that exercise opt-in
#      code paths the knobs-off tree never reaches (deep-collect,
#      collect-m, split-radix, regalloc pin, fence, GH threshold, the
#      oop family, DIF/bwd). Direct runs embed argv[0] in provenance,
#      so both binaries are invoked through the SAME copied neutral
#      path.
#
# Usage:
#   bash compare_libs.sh [--keep] [quadrant ...]
#     quadrants default to "all" (same names regen_codelets.sh takes);
#     --keep preserves the two generated trees + spot dirs for manual
#     inspection (prints their location) instead of deleting them.
#
# Exit: 0 = every file byte-identical; 1 = differences (listed);
#       2 = setup failure (missing tree / build error).
#
# Notes bought by experience (do not "simplify" these away):
#   - DUNE_CACHE=disabled: the global cache has served a STALE exe
#     after an edit; a comparison against a stale binary is worthless.
#   - SCOPED build targets only. A bare `dune build` fires the
#     generated/dune promote rules and rewrites generated/*.h in the
#     source tree.
#   - Work dirs live under WSL-native mktemp -d, not /mnt/c: 2x1074
#     small files are dramatically faster off drvfs, and nothing
#     touches the repo.
set -eu

cd "$(dirname "$0")/.."
GEN=$(pwd)

KEEP=0
TARGETS=()
for a in "$@"; do
  case "$a" in
    --keep) KEEP=1 ;;
    *) TARGETS+=("$a") ;;
  esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(all)

if [ ! -d new-lib ] || [ ! -d new-bin ]; then
  echo "compare_libs: new-lib/ or new-bin/ missing — nothing to compare" >&2
  echo "(if new-lib was promoted into lib/, this harness has done its job)" >&2
  exit 2
fi

export DUNE_CACHE=disabled
command -v dune >/dev/null 2>&1 || export PATH="$HOME/.opam/5.2.0/bin:$PATH"

echo "== build (scoped; both libraries) =="
dune build bin/gen_set.exe bin/gen_radix.exe \
           new-bin/gen_set.exe new-bin/gen_radix.exe

WORK=$(mktemp -d -t cmp_libs.XXXXXX)
cleanup() {
  if [ "$KEEP" = "1" ]; then echo "kept work dir: $WORK";
  else rm -rf "$WORK"; fi
}
trap cleanup EXIT

OLD_ROOT="$WORK/tree_old"; NEW_ROOT="$WORK/tree_new"
mkdir -p "$OLD_ROOT" "$NEW_ROOT"

echo "== pass 1: full tree, quadrants: ${TARGETS[*]} =="
./_build/default/bin/gen_set.exe     --root "$OLD_ROOT" "${TARGETS[@]}" | tail -1
./_build/default/new-bin/gen_set.exe --root "$NEW_ROOT" "${TARGETS[@]}" | tail -1

FAIL=0
if diff -r "$OLD_ROOT" "$NEW_ROOT" > "$WORK/tree.diff" 2>&1; then
  N=$(find "$OLD_ROOT" -name '*.c' | wc -l)
  echo "tree: IDENTICAL ($N files)"
else
  FAIL=1
  ND=$(grep -c '^diff\|^Only in' "$WORK/tree.diff" || true)
  echo "tree: DIFFERENCES in $ND entries:"
  grep '^diff\|^Only in' "$WORK/tree.diff" | head -25
  echo "(full diff: rerun with --keep and read tree.diff in the work dir)"
fi

echo "== pass 2: knob-on spot cells =="
# One cell per opt-in machinery family. Format: "env<TAB>args<TAB>name".
CELLS=$(cat <<'EOF'
VFFT_DEEP_COLLECT=1	13 --emit-c --isa avx2	deep13_avx2
VFFT_COLLECT_M=1	25 --emit-c --isa avx512	collectm25_avx512
VFFT_SPLIT_RADIX=1	32 --emit-c --isa avx512	sr32_avx512
VFFT_PIN_FORCE=1	32 --twiddled --log3 --emit-c --isa avx512	pin32_log3_avx512
VFFT_FORCE_FENCE=1	7 --emit-c --isa avx2	fence7_avx2
VFFT_GH_THRESHOLD=8	64 --twiddled --emit-c --isa avx2 --uarch raptor_lake	gh64_t1_avx2
-	16 --oop --twiddled --oop-load UG --oop-store UG --oop-buffer-oop --emit-c --isa avx2	oop16_ug_ug_avx2
-	20 --twiddled --dif --bwd --emit-c --isa avx2	t1dif20_bwd_avx2
EOF
)

# Provenance neutrality: run both drivers via the same copied path.
DRV="$WORK/gen_radix.exe"
run_cells() { # $1 = built exe to copy, $2 = outdir
  rm -f "$DRV"; cp "$1" "$DRV"
  mkdir -p "$2"
  while IFS=$'\t' read -r env args name; do
    [ -z "$name" ] && continue
    if [ "$env" = "-" ]; then
      $DRV $args > "$2/$name.c"
    else
      env "$env" "$DRV" $args > "$2/$name.c"
    fi
  done <<< "$CELLS"
}
run_cells ./_build/default/bin/gen_radix.exe     "$WORK/spot_old"
run_cells ./_build/default/new-bin/gen_radix.exe "$WORK/spot_new"

if diff -r "$WORK/spot_old" "$WORK/spot_new" > "$WORK/spot.diff" 2>&1; then
  echo "spot cells: IDENTICAL ($(ls "$WORK/spot_old" | wc -l) cells)"
else
  FAIL=1
  echo "spot cells: DIFFERENCES:"
  grep '^diff\|^Only in' "$WORK/spot.diff"
fi

if [ "$FAIL" = "0" ]; then
  echo "== compare_libs: PASS — new-lib output is byte-identical to lib =="
else
  echo "== compare_libs: FAIL — see diffs above ==" >&2
  # keep evidence on failure even without --keep
  KEEP=1
  exit 1
fi
