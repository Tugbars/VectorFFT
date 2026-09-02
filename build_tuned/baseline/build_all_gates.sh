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
cd "$(dirname "$0")/.." || exit 1
OUT=baseline/gates_build.txt
: > "$OUT"
ok=0; fail=0
for g in benches/*gate*.c; do
  n=$(basename "$g" .c)
  if [ "$n" = "sp_ccol_decode_gate" ]; then
    if python build.py --src "$g" --compile >/dev/null 2>&1; then
      echo "BUILD_OK   textual  $n" >> "$OUT"; ok=$((ok+1))
    else
      echo "BUILD_FAIL          $n" >> "$OUT"; fail=$((fail+1))
    fi
    continue
  fi
  if python build.py --src "$g" --compile >/dev/null 2>&1; then
    echo "BUILD_OK   standalone $n" >> "$OUT"; ok=$((ok+1))
  elif python build.py --src "$g" --vfft --compile >/dev/null 2>&1; then
    echo "BUILD_OK   vfft       $n" >> "$OUT"; ok=$((ok+1))
  else
    echo "BUILD_FAIL            $n" >> "$OUT"; fail=$((fail+1))
  fi
done
echo "# built=$ok failed=$fail" >> "$OUT"
