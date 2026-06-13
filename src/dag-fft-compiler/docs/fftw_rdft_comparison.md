# FFTW 3.3.10 rdft source vs VectorFFT native rfft (section 68)

Source read: the exact tree we benchmark against
(/home/claude/fftw-3.3.10). Files cited by path:line. No benchmarks
in this section; code architecture only.

## 1. The codelet contracts

FFTW (rdft/codelet-rdft.h):
  khc2hc (R *rio, R *iio, const R *W, stride rs,
          INT mb, INT me, INT ms)                      [102]
  khc2c  (R *Rp, R *Ip, R *Rm, R *Im, const R *W,
          stride rs, INT mb, INT me, INT ms)           [131]
  r2cf_N (R *R0, R *R1, R *Cr, R *Ci, stride rs, csr, csi,
          INT v, INT ivs, INT ovs)        [scalar/r2cf/r2cf_16.c:37]

Ours (ABI v2, section 62):
  hc2hc/hc2c: 9-arg per-COLUMN call (in_re, in_im, out_re, out_im,
              tw_re, tw_im, is, os, vl)
  r2cf: 7-arg (in, out_re, out_im, is, os_re, os_im<0, vl)

THE GRANULARITY DELTA, the headline of this comparison: one FFTW
hc2hc call walks the ENTIRE interior column range [mb, me). The
m-loop lives INSIDE the codelet (hf_4.c:41):

  for (m = mb, W = W + ((mb-1)*6); m < me;
       m = m + 1, cr = cr + ms, ci = ci - ms, W = W + 6, ...)

cr ascends, ci DESCENDS — the same mirror twin-stream geometry we
derived independently in section 62 (in_re ascending, in_im
descending). Their twiddles advance as a CONTIGUOUS scalar stream
(2(r-1) doubles per column, consecutive in memory): the twiddle walk
is a perfect linear prefetchable stream, and the call overhead plus
per-call cold-start is amortized over every interior column at once.
We pay one call per column with strided row hops between calls — the
exact spot where sections 64-67 located our context wall.

Their r2cf leaf carries a built-in batch loop (v, ivs, ovs): calls
amortize over the vector dimension too, scalar per element.

## 2. The executor

FFTW hc2hc solver (rdft/hc2hc-direct.c:45 apply):
  per vector element: cld0->apply (k=0 column = a small rdft child;
  matches our k0-is-r2cf corollary), then ONE ego->k call with twin
  pointers IO + ms*mb and IO + (m-mb)*ms (ascending/descending
  mirror bases — our derivation again), then cldm->apply at
  (m/2)*ms — the mid column as a separate small child plan (our
  direct-mid corollary). The decomposition k0 / interior / mid is
  ISOMORPHIC to ours; independently re-derived, structurally
  identical. Differences: their plans are recursive trees and
  in-place over a single IO array; ours is a flat stage loop over
  ping-pong planes.

rdft2 solver (rdft/ct-hc2c.c:31 apply_dit): child rdft computes the
bulk, then a plan_hc2c finishes mirror pairs into natural complex
output. This is design D2 (our pending natural-split terminator)
already shipped in FFTW; the khc2c 4-pointer signature (Rp,Ip,Rm,Im)
is the twin-stream form our D2 ABI sketch matches in spirit.

dft-r2hc.c:43: r2hc VIA A FULL COMPLEX DFT plus an O(n) post-combine
(rop -/+ iom, iop +/- rom...). This is precisely our half-complex
wrapper, existing inside FFTW as a planner-selectable ALTERNATIVE.
FFTW keeps both strategies and lets the planner choose per size.

vrank-geq1.c: howmany handled by a plain loop applying the child
plan one transform at a time.

## 3. The buffered variant — FFTW met our wall and built around it

hc2hc-direct.c:94 apply_buf + dobatch (and the identical machinery in
ct-hc2c-direct.c): copy a batch of columns into a CONTIGUOUS buffer
(cpy2d_ci), run the codelet at unit stride, copy back (cpy2d_co).
And the batch size (hc2hc-direct.c:66):

  /* should not be 2^k to avoid associativity conflicts */
  radix rounded up to a multiple of 4, then +2.

FFTW explicitly engineered around pow2-stride cache-set conflicts in
BOTH real-path solvers. Their remedy for strided column access is
copy-to-contiguous with a deliberately non-pow2 buffer pitch — they
change the ACCESS PATTERN, not the schedule. Notable agreement with
our falsifications: we measured naive lane-blocking (footprint-only)
NEGATIVE and software prefetch NEGATIVE; FFTW uses neither — no
prefetch intrinsics anywhere in the rdft path, and their blocking is
always paired with the contiguous-copy.

## 4. SIMD: three different axes

(a) FFTW core rdft codelets (hf_N, r2cf_N): SCALAR, ILP from the
    unrolled butterfly body. hf_16 = 174 add + 100 mul per column,
    796 lines; r2cf_16 = 58 add + 20 mul.
(b) FFTW SIMD rdft2 stage (simd/common/hc2cfdftv_N.c:37): vectorizes
    ACROSS ADJACENT COLUMNS — m advances by VL, Rp/Ip ascend and
    Rm/Im descend by VL*ms, mirror pairs handled with VFMACONJ,
    twiddles pre-interleaved (TWVL layout). Column-lane SIMD.
(c) Ours: batch-lane SIMD — every codelet AVX-512 across K lanes.

For howmany=K=256 workloads (our product shape), (c) matches the
problem natively while FFTW runs vrank one-at-a-time over small-N
transforms whose ~4KB working set is cache-resident BY CONSTRUCTION.
That is the deep reason the container's memory wall punishes our
schedule and not theirs: their inner loops never leave L1/L2 for
N=256; our planes are 512KB each. The flip side: at K=1 our
batch-lane axis collapses while their column-lane (b) and scalar ILP
(a) still work — relevant to API generality, not to the current
product bench.

## 5. Actionable takeaways (ranked, with effort and precedent)

T1. RANGED MULTI-COLUMN CODELETS (m-loop inside, FFTW-style).
    Generator: emit the column loop inside hc2hc/hc2c with twiddle
    pointer advance; executor: one call per (stage, q) instead of
    per column. Amortizes call overhead AND turns the twiddle walk
    into a linear stream; the column walk's row hops become loop-
    carried (prefetcher-friendlier than call-boundary-separated).
    Effort: moderate (emitter loop wrapper + ABI v3). Test on real
    hardware; the container cannot adjudicate (section 67).
T2. DOBATCH COPY-TO-CONTIGUOUS with non-pow2 buffer pitch for the
    combine stages. Direct FFTW precedent in both real-path solvers;
    addresses stride conflicts by construction. Effort: executor-
    only; composable with T1. Supersedes the falsified naive
    lane-blocking (changes pattern, not just footprint).
T3. KEEP THE HALF-COMPLEX WRAPPER as a planner alternative (FFTW
    does exactly this via dft-r2hc). The wisdom layer should choose
    wrapper vs native per (N, K, machine) instead of native
    unconditionally replacing it.
T4. COLUMN-LANE SIMD (hc2cfdftv-style) as the small-K strategy if
    the API ever needs K < 8. Strategic note only.
T5. Minor: MAKE_VOLATILE_STRIDE (their anti-aliasing barrier for
    stride variables) — gcc stride-aliasing pessimization is a known
    hazard; we have not observed it, no action.

## 6. Scorecard of independent agreement

Derived in sections 60-62 without reading this source, now confirmed
identical in FFTW: packed halfcomplex output map (sym1.sym2 = their
storage), k=0 column = plain small real DFT child, mid column =
separate direct child at m/2, mirror twin-stream addressing with a
descending imaginary pointer, natural-complex terminal stage as a
separate codelet family (hc2c / D2). The two implementations differ
in execution strategy (recursion vs flat, in-place vs ping-pong,
scalar-ILP vs batch-SIMD, ranged vs per-column calls) — not in the
mathematics.
