#!/bin/bash
# build_codelets.sh - Linux/WSL counterpart of build_codelets.ps1. Compiles the
# in-place AVX2 codelets to ELF .o (-fPIC, required for a shared object) plus a
# response file, so the Linux JIT (and smoke test) link a stable in-repo dir
# instead of /tmp. Run once after checkout / when codelets change:
#   bash jit/build_codelets.sh
# Outputs (gitignored): jit/generated/codelets_linux/*.o + jit/generated/codelets_linux.rsp
set -e
GCC="${GCC:-gcc}"
JIT="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$JIT")"
SRC="$ROOT/codelets/inplace/avx2"
OUT="$JIT/generated/codelets_linux"
RSP="$JIT/generated/codelets_linux.rsp"
mkdir -p "$OUT"
CF="-O3 -mavx2 -mfma -march=haswell -fPIC -Wno-incompatible-pointer-types -Wno-unused-result"
ok=0; fail=0
for f in "$SRC"/*.c; do
  o="$OUT/$(basename "${f%.c}").o"
  if $GCC $CF -c "$f" -o "$o" 2>/dev/null; then ok=$((ok+1)); else echo "FAIL: $(basename "$f")"; fail=$((fail+1)); fi
done
ls "$OUT"/*.o > "$RSP"
echo "compiled $ok codelets ($fail failed) -> $OUT"
echo "wrote $RSP ($(wc -l < "$RSP") entries)"
