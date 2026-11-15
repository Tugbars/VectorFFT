#!/bin/sh
# check.sh -- is VectorFFT working on this machine? Run it from anywhere: sh check.sh
# It builds check.c against the library in this folder (it needs gcc) and runs it. On a CPU with AVX-512 it runs
# the check twice: once with the library the CPU gets by itself, once forced to the AVX2 library.
cd "$(dirname "$0")" || exit 2
unset VFFT_ISA
echo "CPU: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | sed 's/.*: //'); AVX-512: $(grep -q avx512f /proc/cpuinfo 2>/dev/null && echo yes || echo no)"
FLOAT=""
[ -f vfftf.h ] && FLOAT="-DCHECK_FLOAT"
gcc -O2 check.c $FLOAT -I. -L. -lvfft -lm -Wl,-rpath,'$ORIGIN' -o check || { echo "check.c did not compile"; exit 2; }
echo "==== the library this CPU gets by itself"
./check > check_result.txt 2>&1
RESULT=$?
cat check_result.txt
if grep -q "library loaded    avx512" check_result.txt; then
    echo
    echo "==== the same CPU, forced to the AVX2 library"
    VFFT_ISA=avx2 ./check || RESULT=1
fi
echo
if [ "$RESULT" = 0 ]; then echo "RESULT: everything passed."; else echo "RESULT: something failed. Send the text above."; fi
exit $RESULT
