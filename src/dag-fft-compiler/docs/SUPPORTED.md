# What VectorFFT's out-of-place path supports

This document describes the output-ordering modes the out-of-place engine
exposes, their measured performance, their maturity, and how to choose between
them. The companion document `OOP_DESIGN.md` explains how the engine is built.

Scope: complex-to-complex, N = 1024, batched over K transforms, out-of-place
(input preserving). The codelets are AVX-512. Everything below is for that
configuration unless stated otherwise.


## Two output-ordering modes

VectorFFT's out-of-place path offers two modes that trade output ordering for
speed. This is the same trade every serious FFT library makes, and the right
choice depends entirely on whether your consumer reads frequency bins by index.

### Mode A: natural order (FFTW-competitive)

`X[k]` is written at element index `k`. Input is preserved. This is the
general-purpose mode and the one to compare against FFTW, because serial FFTW
only ever produces natural order (see the library notes below).

* Engine: the per-block recursive engine. The packaged reference
  (`engine/engine_natural_oop.c`) uses a 64x16 work-buffer structure; the
  in-place-twiddle structure described in `OOP_DESIGN.md` (16x64, no work buffer)
  is the validated improvement and the recommended evolution.
* Correctness: machine precision (relative error about 1e-14), input preserved,
  natural order, no separate bit-reversal pass.
* Use for: anything that reads individual bins (spectral peak detection,
  periodicity at known frequencies, plotting a spectrum), frequency-domain
  filtering where the filter response is defined in natural order, or any
  interface with code that assumes `X[k]` at index `k`.

### Mode B: scrambled / permuted order (MKL-competitive)

The output contains the correct values in a fixed mixed-radix permuted order
rather than natural order. It is numerically exact (Parseval energy matches FFTW
to the last digit), just reordered. This is the fast convolution/filtering mode.

* Engine: the batched stride executor in `core/` (`stride_executor.h`,
  `executor_generic.h`), run with `me = K`. The executor is in-place; for
  out-of-place use, run it on a copy of the input.
* Correctness: exact, but the element order is a fixed permutation determined by
  the factorization. The matching inverse must consume the same permutation.
* Use for: convolution, correlation, and FFT-based filtering (overlap-add,
  overlap-save), where you forward transform, multiply pointwise, and inverse
  transform. The forward permutation and the inverse permutation cancel, so the
  ordering never appears in the result.


## Measured performance

All numbers are versus FFTW (PATIENT plan, out-of-place) on a virtualized
Sapphire/Emerald Rapids Xeon, gcc-13, AVX-512, min-of-many back-to-back. Read
these as rankings and directions, not as absolute ratios for your hardware (see
caveats).

### Natural order (Mode A)

| K     | regime       | per-call 16x64 | one-call 16x64 | one-call 32x32 (t1p) |
|-------|--------------|----------------|----------------|----------------------|
| 128   | L2/L3        | ~0.91x         | ~1.01x         | ~1.08x               |
| 256   | L3-resident  | ~0.93x         | ~1.02x         | ~1.12x               |
| 512   | L3-resident  | ~0.95x         | ~1.01x         | ~1.07 to 1.15x       |
| 1024  | L3/DRAM      | -              | ~1.04x         | ~1.09x               |

The fastest natural-order OOP engine is the balanced **32x32 one-call with a log3
twiddle stage**, at ~1.03 to 1.12x FFTW. The one-call form (each twiddle stage a
single `t1p` call over the whole block, not one call per position) is what lifts
natural-order OOP above FFTW; balancing the two stages at radix-32 then beats the
16x64 split by 6 to 13% (spill grows faster than linearly with radix on 32 ZMM
registers, and the heavy radix-64 twiddle stage of 16x64 was the bottleneck); and
log3 on the radix-32 twiddle stage adds another 2 to 4% by trading 26 broadcast
twiddle loads for FMAs on otherwise-idle FMA ports. 
32x32), the 3-stage 16x16x4, and the log3-vs-flat comparison are all in
docs/OOP_DESIGN.md section 6. Numbers are directional on a noisy VM; the rankings
hold across K.

### Scrambled order (Mode B)

Best small-radix factorization per K, versus FFTW (the optimum shifts with K):

| K     | best factorization | vs FFTW       |
|-------|--------------------|---------------|
| 128   | 4x4x4x16           | ~0.95x        |
| 256   | 4x4x4x4x4          | ~0.94x        |
| 512   | 4x4x4x4x4          | ~1.01x (beats FFTW) |

At K=512 the all-radix-4 plan is at or past FFTW parity. The small-radix
advantage is real here precisely because the batched executor calls codelets with
`me = K`, so tiny radices still do a full batch of work per call. The same
factorizations are slow in the per-block natural-order engine (see `OOP_DESIGN.md`
section 6).


## What MKL and FFTW expose, for comparison

This bears directly on which mode is the fair comparison target.

* **MKL has scrambled output as a first-class option.** Its DFTI descriptor
  parameter `DFTI_ORDERING` accepts `DFTI_BACKWARD_SCRAMBLED`, documented exactly
  for convolution and power-spectrum use where order is unimportant. It is a hint
  ("permit scrambled order if possible"), so MKL only skips the reorder when its
  algorithm and size allow. `DFTI_FORWARD_SCRAMBLED` exists in the headers but is
  not implemented. So Mode B is directly comparable to MKL-scrambled, apples to
  apples.

* **Serial FFTW is natural-order only.** There is no scrambled-output flag for
  `fftw_plan_dft`; serial FFTW always pays the natural-order cost. The scrambled
  mode exists only in FFTW-MPI (`FFTW_MPI_SCRAMBLED_OUT` / `FFTW_MPI_SCRAMBLED_IN`,
  where it saves a distributed transpose). So Mode A is the only mode with a
  meaningful FFTW comparison, and "beats FFTW" only means something in natural
  order.

Practical reading: benchmark Mode B against MKL-scrambled and Mode A against
FFTW. Both are legitimate, and most FFT-based filtering pipelines take Mode B and
never notice the ordering.


## Choosing a mode

A single rule covers almost every case:

* If you ever look at a single bin and ask "what frequency is this," you need
  **natural order** (Mode A).
* If you forward transform, do something order-agnostic (pointwise multiply for
  convolution, or aggregate the spectrum), and inverse transform, the
  permutation cancels and **scrambled** (Mode B) is faster and fine.

For a typical HFT or DSP filtering path built on FFT convolution (overlap-add or
overlap-save), Mode B is the right default.


## Caveats and scope

* **N = 1024 specific.** The per-block natural-order engine is an N=1024 reference
  that pins the architecture. It is not a general-N planner.
* **Benchmark environment.** Numbers come from a noisy single-vCPU VM with
  occasional 2x descheduling spikes. FFTW_MEASURE picks wildly varying plans, so
  all comparisons use FFTW PATIENT and min-of-many. Rankings and directions are
  reliable; absolute ratios are not. Confirm on your target hardware.
* **AVX-512 codelets.** These are AVX-512. The intended deployment target
  (i9-14900KF) is AVX2-only, where none of these codelets run and the comparison
  is entirely different. An AVX2 codelet backend exists in the generator
  (`--isa avx2`).
* **Scrambled output is a fixed permutation.** It is exact but reordered. The
  inverse transform must consume the same permutation for the round trip to be
  correct. Do not feed scrambled output to code that expects natural order.
* **Not built: a recursive natural-order small-radix engine.** Combining small
  radices with natural order would require either an explicit permutation pass
  (whose cost roughly cancels the small-radix gain) or FFTW's full recursive
  natural-order stride machinery. The measured evidence is that this lands at
  about where the in-place-twiddle 16x64 engine already sits, so it is deferred.


## Summary

| | Mode A: natural | Mode B: scrambled |
|---|-----------------|-------------------|
| output order | `X[k]` at index `k` | fixed permutation |
| correctness | machine precision | exact (Parseval) |
| best vs FFTW (this box) | ~0.87 to 0.93x | ~0.94 to 1.01x |
| compare against | FFTW (serial, natural only) | MKL `DFTI_BACKWARD_SCRAMBLED` |
| use for | per-bin work, spectral analysis, natural-order filters | convolution, correlation, FFT filtering round trips |
| engine | per-block recursive (`engine/`) | batched stride executor (`core/`) |
