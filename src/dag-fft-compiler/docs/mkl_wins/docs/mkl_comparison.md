# 32x32 OOP versus Intel MKL

This note records a re-measurement of the optimized 32x32 out-of-place engine
against Intel MKL DFTI, after the codelet spill work (stride specialization,
M-project fuse, store-on-compute). The earlier baseline had MKL ahead of the
OOP engine by 15 to 35 percent. After the optimizations the gap closes and, in
the cache-resident regime on the test machine, reverses to a few percent in
VectorFFT's favor. The result is real but regime-sensitive and not yet confirmed
on quiet hardware, so it is documented with its caveats rather than as a clean
win.

Reproduce with `benchmarks/09_compare_vs_mkl.c` (optional, requires MKL; build
instructions are in its header).

## Setup

Three engines, all single-thread, out-of-place, batched K transforms of
N = 1024 forward complex double:

* VectorFFT: the 32x32 log3 one-call engine using the specialized codelets
  (`radix32_*_oop_avx512_spec.c`: stride-specialized + `--fuse 8` +
  `--oop-store-fused`).
* FFTW 3.3.10: `fftw_plan_many_dft`, FFTW_PATIENT.
* MKL DFTI: `DFTI_NOT_INPLACE`, `DFTI_NUMBER_OF_TRANSFORMS = K`, committed once.

Timing is interleaved (one call of each back to back), rdtsc, min of 120,
8 warmups. Thread count is forced to 1 via `MKL_NUM_THREADS=1` and
`OMP_NUM_THREADS=1`; `mkl_set_num_threads()` must not be called (it segfaults in
this build).

Two correctness and fairness checks were necessary before trusting anything:

* MKL output is verified against a fresh non-destructive FFTW ESTIMATE plan
  (relerr 1.2e-16). The first attempt compared MKL against the PATIENT
  `plan_many` output, but PATIENT planning overwrites its input buffer, so that
  comparison was meaningless (it printed inf). MKL itself is correct.
* An early version timed two VectorFFT variants (flat and log3) that shared the
  same output buffers, so log3 ran on cache the flat call had just warmed, an
  advantage MKL did not get. Removing that double-touch left VectorFFT's edge
  intact, so it was not the cause.

## Result

Cache-resident regime, K = 128 (the three buffer sets total about 6 MB, so the
working set fits in cache and the comparison is codelet-bound), five repetitions,
stable:

| ratio | value | meaning |
|---|---|---|
| FFTW / MKL | 1.12 to 1.28 | MKL beats FFTW by 12 to 28 percent |
| MKL / VF_log3 | 1.05 to 1.07 | VectorFFT log3 beats MKL by 5 to 7 percent |

The FFTW-versus-MKL ratio is the stable anchor and matches expectation: MKL is
hand-tuned assembly and is faster than FFTW. The new fact is that the optimized
VectorFFT now sits slightly ahead of MKL in this regime, where before it was
well behind.

The cause is cumulative spill reduction in the codelets. The radix-32 leaf went
from the originally shipped 162 stack spills / 342 vmovapd to 90 / 189 across
the three optimizations. Each optimization was only worth 6 to 10 percent on its
own and was measured that way; against the original codelet the combined effect
is much larger, which is consistent with closing a 30-plus percent gap.

## Caveats

This is reported cautiously on purpose.

* It contradicts an earlier session on the same VM that measured MKL about 37
  percent faster at K = 128. The two cannot be reconciled on this variable
  single-vCPU virtualized host. The robust, regime-independent statement is that
  the gap moved from tens of percent to single digits; the sign of the small
  remaining margin is not settled until it is confirmed on the quiet 14900KF.
* The numbers above are the cache-resident regime. At larger K the footprint
  exceeds cache (at K = 512 the three buffer sets total about 48 MB and FFTW
  alone runs 2.6x slower than in cache). There VectorFFT's lead grows to 25 to
  44 percent, but that is a memory-bandwidth and layout effect (split re/im
  element-major versus MKL's interleaved complex), not codelet quality, so it is
  not counted as a codelet result.
* Very small K (16, 32) is dominated by per-call dispatch overhead and gave wild
  ratios (1.4x); ignore it.
* AVX2 is untested. The register math the optimizations rely on differs with 16
  YMM, so the AVX2 deploy target must be measured separately.

## Bottom line

The spill work moved the 32x32 OOP engine from clearly behind MKL to roughly at
parity or a few percent ahead in the cache-resident regime on this machine.
MKL's old comfortable out-of-place lead is gone. Confirm on quiet hardware
before calling it a win.
