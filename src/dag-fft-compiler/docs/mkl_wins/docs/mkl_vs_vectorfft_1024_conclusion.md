# MKL versus VectorFFT on N=1024: the Conclusive Finding

A single-sentence version: for a 1024-point double-complex FFT, Intel MKL uses the
exact same algorithm VectorFFT does, a 32 by 32 radix-32 decomposition at 512-bit,
and VectorFFT is slightly ahead on the merits while MKL closes most of the gap
through better instruction-level scheduling. The performance difference is not
arithmetic, not op mix, and not data layout. It is the schedule.

This was established by disassembling and runtime-tracing the installed Intel
oneMKL 2026 in the working container, not from documentation or reputation.

## 1. What MKL actually runs for 1024

Runtime PC sampling of a tight 1024-FFT loop (gdb attached to the running
process) showed the hot code living in exactly two kernels, both in
`libmkl_avx512.so.3`:

* `mkl_dft_avx512_mg_colbatch_plain_fwd_32_d`, the no-twiddle radix-32 column stage.
* `mkl_dft_avx512_mg_rowbatch_twidl_fwd_032_d`, the twiddled radix-32 row stage.

Each is called exactly once per transform. So MKL computes

```
1024 = 32 x 32
     = [radix-32, no twiddle, column stage]  ->  twiddle  ->  [radix-32, twiddled, row stage]
```

which is identical in structure and radix to VectorFFT's 32x32 engine (the n1
no-twiddle leaf followed by the t1p twiddled stage). Two independent efforts, the
same decomposition and the same radix.

Note on method: the obvious symbol guesses did not fire. The named entries
`noprefetch_step_radix8_fwd`, `cFFTfwd_32`, `cFftFwd_BlkStep`, and friends are
real symbols but belong to other size and configuration paths. The 1024 path uses
the `mg_*batch_*_32_d` kernels above, found only by sampling the program counter
of the running transform.

## 2. The two corrections this forces

Earlier in the investigation I disassembled `STEP_RADIX8` and found it ran at
256-bit (EVEX `ymm`), and I hypothesized that this 256-bit choice explained the
observed narrowing of VectorFFT's lead under sustained thermal load. Both points
need correcting:

* `STEP_RADIX8` is not on the 1024 path. It serves some other size. The 1024
  kernels are 512-bit `zmm` on both sides.
* Therefore the thermal narrowing on 1024 is not a 512-versus-256 frequency
  effect. Both MKL and VectorFFT run 512-bit on this size, so both pay the same
  AVX-512 frequency behavior. Whatever narrows the lead under load on 1024 is
  something else (cache and memory behavior under sustained iteration is the more
  likely candidate, but that was not measured here).

The 256-bit observation remains valid for the radix-8 path; it is simply not the
1024 story.

## 3. The head-to-head, measured

Both implementations run the same algorithm at the same width, so the comparison
is genuinely like-for-like per stage. Counts are static instruction counts from
the disassembly.

| per-stage kernel | FMA | standalone mul | add/sub | shuffle/perm/bcast | stack-spill movs | width |
|---|---|---|---|---|---|---|
| MKL colbatch_plain (no twiddle) | 44 | 34 | 208 | 110 | 21 | 512-bit |
| VectorFFT n1 (no twiddle) | 140 | 6 | 240 | ~0 | ~90 | 512-bit |
| MKL rowbatch_twidl (twiddle) | 52 | 124 | 272 | 386 | 192 | 512-bit |
| VectorFFT t1p log3 (twiddle) | 254 | 120 | 240 | ~0 | low | 512-bit |
| VectorFFT t1p flat (twiddle) | 202 | 68 | 240 | ~0 | low | 512-bit |

Three structural differences, every one of them favoring VectorFFT on paper:

* Fusion. MKL barely fuses. FMA-to-mul is about 1.3 to 1 in the plain stage and
  0.4 to 1 in the twiddle stage. VectorFFT is 23 to 1 in the plain stage and about
  2 to 1 in the twiddled log3 stage. MKL deliberately keeps multiplies standalone.
* Shuffles. MKL pays a heavy permute and shuffle tax, 110 ops in the plain stage
  and 386 in the twiddle stage, because it stores complex interleaved and does the
  32x32 transpose in-register. VectorFFT pays essentially none, because of its
  split real/imag layout and dedicated transpose-reducing codelets. This is the
  single most concrete vindication of a VectorFFT design choice: hundreds of
  shuffle ops per stage that the layout simply deletes.
* Spills. MKL's twiddle stage spills hard, 192 stack moves, where VectorFFT's
  spill-reduction work keeps the no-twiddle leaf near 90 and the twiddle stage low.

## 4. Why op count is not the answer

Because both run radix-32 by radix-32, they do essentially the same arithmetic
work. VectorFFT's measured arithmetic, which is therefore also MKL's to within the
small difference between their radix-32 factorings:

| stage | flops per radix-32 butterfly | flops per point |
|---|---|---|
| n1 (no twiddle) | 526 | 16.4 |
| t1p flat | 712 | 22.3 |
| t1p log3 | 868 | 27.1 |

Full 1024 as two radix-32 stages: 38.7 flops per point on the flat path, 43.6 on
the log3 path, against a split-radix theoretical floor of about 34. MKL sits in
the same range by construction. Neither side has a meaningful op-count advantage,
which is exactly what "same algorithm, same radix" implies.

## 5. The conclusion: it is the schedule

VectorFFT executes fewer total instructions, fuses far more, performs almost no
shuffles, and spills less. Despite all of that it is only about 5 to 7 percent
ahead of MKL in the cache-resident regime. For MKL to carry hundreds of extra
shuffle ops per stage and heavier spilling and still land within single digits,
its per-instruction throughput has to be markedly higher. That is
instruction-level parallelism: port utilization and issue-order quality.

So the standings are clear and, importantly, good for VectorFFT:

* VectorFFT wins the axes that are hard to win generatively: the algorithm, the
  split layout, the transpose elision, the fusion discipline, and the spill
  reduction. The binary proves MKL found nothing structural that VectorFFT missed.
  They chose the same 32x32.
* The one axis where a vendor library with an internal microarchitecture model has
  a real structural edge is the schedule, and in VectorFFT's pipeline that is a
  single isolated, swappable pass (currently selective-unpinning plus a
  Graham-style greedy list scheduler). The remaining distance to "ahead by a lot"
  is one well-scoped scheduler problem, not a pile of mistakes.

## 6. Honest limits

* "Better ILP" is inferred, not measured. It is the most parsimonious explanation
  for "more work in nearly the same time," but the container has no IPC or
  port-utilization counters. A VTune port-utilization run or a one-line IPC
  comparison on real hardware would confirm it directly.
* The exact dynamic flop sum for MKL's 1024 was not pinned to the last digit. The
  twiddle kernel loops, so its static op counts are per-body, not per-transform,
  and there is no `perf`, `vtune`, or `pin` in the container to count executed
  instructions. The reliable figure is the algorithmic one in section 4, which is
  exact for VectorFFT and the same by construction for MKL.
* All timing came from a single-vCPU virtualized host. Interleaved min-of-N
  rankings are reliable; absolute cross-binary times are not. This result also
  contradicts an earlier session that had MKL ahead, so the small VectorFFT lead
  should be confirmed on quiet hardware before it is banked.
* AVX2 is untested, and its 16-register file changes the spill and scheduling math.

## 7. Benchmark results, for reference

Single-thread, out-of-place, batched K transforms of N=1024, interleaved rdtsc
min-of-120, on the noisy single-vCPU VM. Ratios above 1 mean VectorFFT is faster.

Cache-resident regime (K=128, about 6 MB, codelet-bound), five repetitions, stable:

* FFTW / MKL approximately 1.12 to 1.28 (MKL beats FFTW, as expected for a tuned library).
* MKL / VectorFFT log3 approximately 1.05 to 1.07 (VectorFFT a few percent ahead).

Full sweep (first run; large-K rows are memory-bandwidth bound, not codelet quality):

| K | VF_flat | VF_log3 | FFTW | MKL | MKL/VF_log3 | FFTW/MKL |
|---|---|---|---|---|---|---|
| 128 | 772,969 | 749,172 | 961,261 | 803,747 | 1.073 | 1.196 |
| 256 | 1,536,533 | 1,479,209 | 2,337,106 | 1,824,410 | 1.233 | 1.281 |
| 512 | 3,697,460 | 3,460,753 | 6,563,830 | 4,969,721 | 1.436 | 1.321 |
| 1024 | 9,500,518 | 9,504,494 | 14,376,046 | 11,893,825 | 1.251 | 1.209 |

(Cycles. The K>=512 ratios reflect a memory-bound regime, about 48 MB across the
three buffer sets at K=512, where FFTW alone runs 2.6x slower than in cache; those
are layout and bandwidth numbers, not codelet quality. K=128 is the
codelet-representative row.)

## 8. Bottom line

MKL computes 1024 exactly the way VectorFFT does, 32 by 32 radix-32 at 512-bit.
VectorFFT is ahead on instruction count, fusion, layout, and spills, and ends up
slightly ahead overall, but by less than those advantages alone should buy. The
difference being given back is scheduling quality, MKL's instruction-level
parallelism. The algorithmic and structural work is won. The remaining gap is a
single scheduler pass away.

## CORRECTION (2026-06-06)

The FFTW rows in this document are overstated by roughly 1.8x and should
not be quoted. Cause: FFTW was timed in a separate binary from the
VectorFFT numbers, and cross-binary absolute timings on the shared
container drift by up to 2x (documented in the OOP engine sessions). The
authoritative FFTW comparisons are the SAME-BINARY round-robin races in
docs/t1p_radix_extension_fftw_guru.md sections 5, 9, and 20 and in
benchmarks/bench_oop_vs_fftw.c. The MKL rows in this document were
same-binary and stand. Conclusions drawn here about FFTW margins are
superseded; in particular the avx2-vs-FFTW gap is OPEN per section 20.
