#!/bin/bash
# Rebuild every gate at the baseline SHA and record HOW each one builds.
#
# The prebuilt .exe files all predate vfft.c, so their results mean nothing
# until rebuilt. Step 1 of docs/design/refactor_migration_plan.md.
#
# Build mode is DISCOVERED, not guessed: a grep for vfft.h/vfft_create
# misclassifies gates that reach the library only through a module header
# (wisdom2_2d_gate calls vfft_wisdom2_2d_gate_run and matches neither).
# So: try standalone, fall back to --vfft, record which worked.
#
# sp_ccol_decode_gate is the one hard exception -- it #includes vfft.c
# textually, so compiling vfft.c beside it is a duplicate-symbol error.
#
# Repointed 2026-09-27: the gates live in build_tuned/benches/ and build with
# gauntlet/build.py (the script used to cd to src/tools/ and glob a
# benches/ folder that no longer exists there). Run from anywhere; it works
# from the repo root. The ISA is build.py's: set VFFT_ISA.
cd "$(dirname "$0")/../../../.." || exit 1
OUT=src/tools/baseline/reference/gates_build.txt
: > "$OUT"
ok=0; fail=0
for g in build_tuned/benches/*gate*.c; do
  n=$(basename "$g" .c)
  if [ "$n" = "sp_ccol_decode_gate" ]; then
    if python3 gauntlet/build.py --src "$g" --compile >/dev/null 2>&1; then
      echo "BUILD_OK   textual  $n" >> "$OUT"; ok=$((ok+1))
    else
      echo "BUILD_FAIL          $n" >> "$OUT"; fail=$((fail+1))
    fi
    continue
  fi
  if python3 gauntlet/build.py --src "$g" --compile >/dev/null 2>&1; then
    echo "BUILD_OK   standalone $n" >> "$OUT"; ok=$((ok+1))
  elif python3 gauntlet/build.py --src "$g" --vfft --compile >/dev/null 2>&1; then
    echo "BUILD_OK   vfft       $n" >> "$OUT"; ok=$((ok+1))
  else
    echo "BUILD_FAIL            $n" >> "$OUT"; fail=$((fail+1))
  fi
done
echo "# built=$ok failed=$fail" >> "$OUT"
