# VectorFFT vs Intel MKL — benchmark results (the wins)

A curated pack of the .md writeups and raw CSV data where VectorFFT met or beat
Intel MKL. Assembled from the vectorfft-oop-engine repo.

## Epistemics (read first)
All ratios are mkl/ours or ours/mkl as labelled, measured in ONE process,
min-of-N, single thread (MKL pinned to 1 thread). MKL is non-deterministic, so
only same-process ratios are trustworthy; absolute timings are thermal noise.
Every speed number was taken only after a correctness gate passed. The
container has no PMU, so cache/bandwidth claims are inferred from behaviour, not
measured directly. Where a result was later withdrawn or qualified, the doc says
so — these are kept because the reasoning is the value.

## The wins, by document

### docs/60_rfft_beats_mkl_hc2c_log3.md   [HEADLINE WIN]
The forward real FFT (r2c-256) beats MKL by ~1.2-1.4x using hc2c + log3
codelets and the (8,32) factorization. This is the central, robust result:
a full transform (not just a codelet) beating MKL on the same hardware.

### docs/61_vfft_vs_mkl_r2c_opcount.md   [WIN, mechanism]
Disassembly-level op-count proof of WHY doc 60 wins: it is the per-kernel
instruction count, not a smarter factorization. The op count and the timing
agree, so the win is real and explained, not a measurement fluke.

### docs/mkl_comparison.md   [WIN]
The 32x32 out-of-place engine: VectorFFT log3 beats MKL by 5-7%
(MKL/VF_log3 = 1.05-1.07). Smaller margin, different shape, same direction.

### docs/mkl_vs_vectorfft_1024_conclusion.md   [WIN, nuanced]
N=1024 double-complex: VectorFFT wins the algorithmic axes. Includes the honest
observation that VectorFFT's lead narrows under sustained thermal load — a
frequency effect, documented rather than hidden.

### docs/63_c2r_K_dependent_bottleneck.md   [PARITY + the optimization story]
The inverse real FFT (c2r). After three optimizations (SIMD mid-column inverse,
folded leaf, ranged interior) c2r reaches PARITY with MKL at small batch
(K=8, ratio 0.99). Also the corrected analysis of the large-K gap (it is shared
with the forward path, not a c2r deficiency) and two falsified predictions kept
on the record because the falsifications narrowed the search correctly.

### docs/59_r2c_vs_mkl_layout_and_fold_journey.md   [THE JOURNEY]
The investigation that preceded doc 60: how an apparent "MKL beats us 1.6x"
result was traced to layout/fold artifacts and ultimately WITHDRAWN, clearing
the path to the doc-60 win. Included for the reasoning trail.

## Raw data (CSV)
- benchmarks/pow2_vs_mkl.csv       — pow2 sizes, AVX-512, ratio_vs_mkl column
- benchmarks/pow2_avx2_vs_mkl.csv  — pow2 sizes, AVX2
- benchmarks/odd_vs_mkl.csv        — odd/mixed-radix sizes
  Columns: N,K,factors,variants,vfft_ns,mkl_ns,vfft_gflops,ratio_vs_mkl
  ratio_vs_mkl > 1 means VectorFFT faster. NOTE: these are CODELET-level
  comparisons against MKL's split-complex path, which is MKL's second-class
  format; the large ratios (3-12x) overstate the advantage versus MKL's
  interleaved path. The FULL-TRANSFORM win is the ~1.2-1.4x in doc 60, which is
  the number to cite. The CSVs are the regression baseline, not the headline.
