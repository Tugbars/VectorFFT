#!/usr/bin/env python3
"""truth_gate.py — accuracy-vs-truth comparator for codelet transforms.

For transforms that are NOT bit-exact (e.g. VFFT_BUTTERFLY_SHARE), the
question that matters is whether error against the TRUE DFT changed.
This gate computes a 40-digit mpmath reference for the verify-harness
input (srand(42) pattern) and compares two output binaries.

Usage: truth_gate.py R IOS input.bin baseline.bin candidate.bin
Input/outputs are the raw re[]||im[] dumps the verify*.c harnesses
write; input.bin from the same srand(42) fill without running the
codelet (see gen_input.c pattern in docs 73).
PASS iff candidate's max-rel and rms error vs truth <= baseline's
(equality allowed; measured on transformed columns only).
"""
import struct, sys
import mpmath as mp
mp.mp.dps = 40
R, ios = int(sys.argv[1]), int(sys.argv[2])
def load(p):
    n = R*ios
    d = struct.unpack(f'{2*n}d', open(p,'rb').read())
    return d[:n], d[n:]
ir, ii = load(sys.argv[3]); br, bi = load(sys.argv[4]); cr, ci = load(sys.argv[5])
cols = [k for k in range(ios)
        if any(br[k+ios*t] != ir[k+ios*t] or bi[k+ios*t] != ii[k+ios*t]
               for t in range(R))]
def dft(col, sign):
    xs = [mp.mpc(ir[col+ios*t], ii[col+ios*t]) for t in range(R)]
    return [sum(xs[t]*mp.e**(sign*2j*mp.pi*t*k/R) for t in range(R))
            for k in range(R)]
def err(outr, outi, ref, col):
    m = mp.mpf(0); s = mp.mpf(0)
    for k in range(R):
        got = mp.mpc(outr[col+ios*k], outi[col+ios*k])
        e = abs(got-ref[k])/max(abs(ref[k]), mp.mpf('1e-300'))
        m = max(m, e); s += e*e
    return m, mp.sqrt(s/R)
c0 = cols[0]
sgn = min(((s, err(br, bi, dft(c0, s), c0)[0]) for s in (-1, 1)),
          key=lambda x: x[1])[0]
bm = bs = cm = cs = mp.mpf(0)
for col in cols:
    ref = dft(col, sgn)
    m1, r1 = err(br, bi, ref, col); m2, r2 = err(cr, ci, ref, col)
    bm = max(bm, m1); cm = max(cm, m2); bs += r1*r1; cs += r2*r2
bs = mp.sqrt(bs/len(cols)); cs = mp.sqrt(cs/len(cols))
print(f"baseline : max {mp.nstr(bm,4)} rms {mp.nstr(bs,4)}")
print(f"candidate: max {mp.nstr(cm,4)} rms {mp.nstr(cs,4)}")
ok = cm <= bm*(1+mp.mpf('1e-6')) and cs <= bs*(1+mp.mpf('1e-6'))
print("TRUTH GATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
