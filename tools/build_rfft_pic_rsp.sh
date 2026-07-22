#!/bin/bash
# §6a27: build the rfft/c2r PIC codelet set and extend the jit response file.
# The jit links emitted .so files against @rsp; a .so needs -fPIC objects, so
# the static-archive objects cannot be reused. Run from the repo root.
# Usage: ./tools/build_rfft_pic_rsp.sh /tmp/jitcache/codelets_linux.rsp /tmp/opic_rfft
RSP=${1:-/tmp/jitcache/codelets_linux.rsp}
OUT=${2:-/tmp/opic_rfft}
cp "$RSP" "$RSP.bak_$(date +%s)"
mkdir -p "$OUT"
FAIL=0
for f in src/dag-fft-compiler/codelets/rfft/avx2/*.c src/dag-fft-compiler/codelets/c2r/avx2/*.c; do
  o="$OUT/$(basename "${f%.c}").o"
  gcc -O2 -mavx2 -mfma -mbmi2 -fPIC -c "$f" -o "$o" || { FAIL=$((FAIL+1)); echo "FAIL: $f"; }
done
echo "failures: $FAIL"
ls "$OUT"/*.o >> "$RSP"
sort -u "$RSP" -o "$RSP"
echo "rsp: $(wc -l < "$RSP") lines"
# stale emitted libs must be purged or the resolver reuses them:
# rm -f <jitdir>/rfftjit_*_verN.so
