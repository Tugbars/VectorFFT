#!/bin/bash
# run_linux_smoke.sh - build + run the JIT smoke test on Linux/WSL (.so + dlopen).
# Proves the Linux JIT path: emit (python3) -> gcc -shared -fPIC -> dlopen.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GCC="${GCC:-gcc}"
CF="-O3 -mavx2 -mfma -march=haswell -Wno-incompatible-pointer-types -Wno-unused-result"
$GCC $CF -c "$ROOT/jit/jit_smoke.c" -o /tmp/jit_smoke.o
$GCC /tmp/jit_smoke.o "@$ROOT/generated/jit/codelets_linux.rsp" -o /tmp/jit_smoke -lm -lpthread -ldl
echo "=== Linux smoke build OK ==="
for F in "4,4,4,4,4,4,4,8" "4,4,4,32,64"; do
  echo "---- factors $F ----"
  /tmp/jit_smoke "$F" 0
done
