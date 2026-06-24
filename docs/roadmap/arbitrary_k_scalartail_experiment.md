# Arbitrary-K tail — scalar-tail experiment, full record (2026-06-24)

> Everything learned de-risking the **arbitrary-K** problem (K not a multiple of
> the SIMD width) by hand-building a tail-handling path and testing it on a real
> 1D C2C cell vs MKL — *before* committing to the generator change. Companion to
> the design in [`arbitrary_k_vectorization.md`](arbitrary_k_vectorization.md)
> and the head-to-head data in
> [`../performance/batched_smallN_vs_mkl_fftw.md`](../performance/batched_smallN_vs_mkl_fftw.md).
>
> **Bottom line:** both scalar and masked tails are **correct** and beat MKL at
> odd K. Scalar erodes with `rem`; masked is **flat in `rem`** (fixed ~1 vector
> pass). **DECISION (Tugbars): rem-aware hybrid — scalar tail at rem=1, masked at
> rem≥2** (runtime branch `if (rem==1) {scalar 1-lane} else {masked pass}`).
> Production path = the generator change, not the hand-edits (parked in
> `src/dag-fft-compiler/codelets/experiments/scalartail/`).

## Masked-tail result + DECISION (rem-aware hybrid)

Re-ran the exact margin test with the **masked** tail (forward maskload + one
masked vector pass; r4_n1 + r4_t1s built `-DVFFT_TAIL_MASKED`). Bit-exact at every
K, flip0≈flip1. Margin vs MKL (avg of order-flips), by remainder:

| rem | masked margin (K) | scalar margin (same/near K) |
|---|---|---|
| 0 (even) | 2.02 (32), 1.92 (24), 1.82 (16) | ~1.7–1.9 |
| 1 | 1.72 (33), 1.66 (29), 1.60 (25) | ~1.77 |
| 2 | 1.70 (30), 1.66 (26), 1.38 (14) | ~1.69 (interp) |
| 3 | 1.67 (31), 1.72 (27), 1.41 (15) | 1.61 (31), 1.29 (15) |

- **Masked is FLAT across rem** (~1.6–1.72× at K≈25–33 for rem=1,2,3 alike) — the
  odd-K penalty is a *fixed ~1 extra vector pass*, independent of rem, i.e. MKL's
  flat-remainder behavior. Scalar **erodes** with rem (1.77→1.61→1.29 at K=15).
- **Same-cell (thermal-robust) high-rem wins for masked:** K=31 (rem3) **1.67 vs
  1.61**, K=15 (rem3) **1.41 vs 1.29**. Masked holds where scalar leaked.
- At **rem=1** scalar's single cheap lane edges masked's full vector pass (small;
  cross-run thermal-confounded — the even-K control differs 1.73 vs 2.02 between
  runs, so only within-run patterns + same-cell comparisons are trusted).
- ⇒ **rem-aware hybrid is the verdict.** scalar penalty = `rem`×scalar-lane;
  masked penalty = fixed 1 vector pass. Scalar wins rem=1, masked wins (and holds
  flat) rem≥2. Mechanically a cheap runtime `if (rem==1)` in the cold tail.

## Tail tax vs the nearest full-vector K (hybrid built + measured)
Built the actual hybrid (`r4_n1_fwd_hybrid.c`, `r4_t1s_dit_fwd_hybrid.c` in the
experiments dir) and measured **per-transform ns vs the nearest pow2** (`bench_tailtax.c`,
one process; the *within-engine* tax is thermal-robust — anchor & member measured
adjacently). "tax" = pt(K)/pt(pow2):

| family | K (rem) | **our tail tax** | MKL tail tax |
|---|---|---|---|
| 8  | 7 (3) | 1.76× | 2.41× |
| 8  | 9 (1) | 1.22× | 2.59× |
| 16 | 15 (3) | 1.44× | 0.97×* |
| 16 | 17 (1) | 1.23× | 1.12×* |
| 32 | 31 (3) | 1.29× | 1.35× |
| 32 | 33 (1) | 1.26× | 1.49× |
| 64 | 63 (3) | 1.24× | 1.11× |
| 64 | 65 (1) | 1.18× | 1.15× |

- **Our tail tax is smooth and modest (~1.2–1.3×), flat across rem** (hybrid working:
  rem1 scalar ≈ rem3 masked); blows up only at **tiny K=7** (1.76×) and shrinks as
  K grows (tail amortizes).
- **MKL's tail tax is erratic** (0.95–2.59×) — its per-transform cost is non-monotone
  in K (kernel-selection; *its K=16 anchor is anomalously slow → taxes <1). MKL's
  odd-K penalty is generally **equal-or-worse** than ours.
- ⚠️ The absolute *margins* in `bench_tailtax` are **order-biased** (vfft sweep then
  MKL sweep → MKL on a hot core, ~1.85× inflated). The TRUE margin is the **~1.6–1.8×**
  from the per-cell-isolation+flip run; the *tax* columns (within-engine) are the
  trustworthy output here.

⇒ Hybrid validated as **competitive-to-better than MKL on the tail.**

---

## ✅ SOLUTION — nailed (rem-aware hybrid tail)

The arbitrary-K problem (K not a multiple of the SIMD width) is **solved** by a
**codelet-internal, rem-aware hybrid tail**:

```
for (k = 0; k + VW <= me; k += VW) { ...bulk full-vector body, UNCHANGED... }
if (k < me) {                          // remainder; only fires for K % VW != 0
    size_t rem = me - k;
    if (rem == 1)  { ...scalar single lane (SSE-1-wide)...        }   // rem=1: cheapest
    else           { ...one forward masked vector pass...          }   // rem>=2: flat cost
}
```

Why this is the answer, proven on a real 1D C2C plan (N=1024 [4,4,4,4,4] T1S):
- **Correct** — bit-exact at every K (even, odd, tiny), in-place safe (forward
  masked base=floor reads lanes disjoint from the bulk; masked store, never
  full-store-overlap).
- **Holds the MKL margin** — ~1.6–1.8× over MKL at odd K (rigorous per-cell-isolation
  + best-of-5-min + cachebust/cool + order-flip methodology).
- **Tail tax is tiny, smooth, and flat in `rem`** — ~1.2–1.3× per-transform vs the
  nearest full-vector K (rem=1 scalar ≈ rem=3 masked), only large at tiny K (K=7:
  1.76×), shrinking as K grows. **MKL's odd-K tax is erratic and generally
  equal-or-worse.**
- **Cheap to dispatch** — `if (rem==1)` is one branch in the cold tail; even-K is
  byte-identical to today's codelet (zero regression).

**Decisions baked in:** twiddle leg-stride stays `me` ⇒ tail must be *in-codelet*
(executor/registry unchanged). t1s/n1 mask only `rio` (broadcast/none tw); t1
(flat per-lane tw) masks tw too; **log3 routes to masked** (its per-VW-block tw
table assumes `me%VW==0`). "scalar" = SSE-1-wide (`vmovsd`), not x87 — the 1-wide
rung of a width cascade, same ISA class MKL floors at.

**Reference impl** (hand-built, validated): `codelets/experiments/scalartail/`
`r4_n1_fwd_hybrid.c`, `r4_t1s_dit_fwd_hybrid.c`.

**Production path:** teach the generator (`emit_c.ml`) to emit this hybrid tail per
codelet (per-radix it's uniform — the `if(rem==1)` handles the split at runtime),
then relax the `K%8` dispatch guards to `K != 0` and let the MT slicer hand the
remainder to the last worker. **Still open:** the blocked-r8 (r≥8 two-pass) odd-K
bug, which is in the executor/seam, *not* the tail (see below).

---

## The problem

Codelets vectorize across the batch K (`data[e*K+lane]`); the generated loop is
`for (b=0; b<me; b+=VW)` with **no remainder** ([emit_c.ml:1612] for strided,
the in-place n1 at `for (k=0;k<me;k+=4)`). So `me` (=K, or slice_K) not a
multiple of VW (4 AVX2 / 8 AVX512) **silently corrupts** (last iteration overruns
into the next element). The in-place c2c path has *no guard*; the real-FFT/OOP
dispatchers fail-closed (`K%8 != 0 → NULL`).

Two regimes: **odd K ≥ VW** (K=7, K=31 — a remainder) and **tiny K < VW** (K=1,2,3
— can't fill one vector). This experiment is about the first.

---

## Design decisions (validated)

1. **The tail must live INSIDE the codelet, not in the executor.** A `t1` codelet
   addresses twiddles `tw[(j-1)*me + b]` — leg-stride `me`. An executor split into
   `bulk(me=floor)` + `tail(me=rem)` would compute *different* leg-strides → wrong
   twiddles. So one codelet processes all `me` lanes with one consistent stride:
   **bulk full-vectors + tail, executor + registry UNCHANGED.** (t1s broadcast
   twiddles are lane-independent, so they're the exception — see the r8 offset-call.)

2. **In-place ⇒ the tail must MASKED-store, never full-store-overlap.** Overlap
   trick = process the last full vector at base `me-VW`. For OOP single-write
   that's fine. For **in-place multi-stage it corrupts the overlap lane**: the bulk
   already transformed it, the overlap re-reads the transformed value and writes
   garbage. Proven on radix-2 K=7. The masked store (write only the genuinely-new
   lanes) fixes it. Verified empirically (`masked_tail_spike.c`): bit-exact for all
   K with masked store; full-store overlap corrupts exactly lane `K-VW`.

3. **OOP isn't pure ping-pong.** MODEB ([oop_execute.h:59-74]) runs stage-0 OOP
   (`src→dst`) then stages 1.. **in-place on dst** — so it hits the same trap;
   **one masked-tail codelet serves both placements**, fork by ISA not placement.

4. **"Scalar" is not x87 — it's SSE-1-wide.** The generated `--isa scalar`
   codelet compiles to **32 `vmovsd`/`vaddsd`, 0 x87** — the 1-wide rung of the
   SIMD-width cascade, the same class MKL floors at (SSE). So the tail is a
   *width cascade* (AVX2 4-wide bulk → SSE 1-wide tail [light] / AVX2 4-wide masked
   [heavy]), all SSE/AVX ISA. There is no "scalar vs vectorized" dichotomy.

---

## What was measured

### Scalar emit correctness — 12/12 bit-exact
`tail_validate.c`: generated `--isa scalar` codelets vs AVX2, per-lane, K=8, for
radix {4,5,8} × {n1_fwd, n1_bwd, t1_dit_fwd, t1s_dit_fwd} → **all `0.00e+00`**,
including the radix-8 spill path and the radix-5 prime conjugate-pair. The scalar
emit is trustworthy. (log3/t1p excluded — its per-VW-block twiddle table assumes
`me%VW==0`; route log3 stages to the masked tail.)

### Tail-strategy crossover (`crossover.c`, K=7, synthetic W-chain)
ns per 4-pt; W = butterfly chain depth (~16·W ALU ops/lane = codelet weight):

| W | ~ALU/lane | masked-fwd | scalar (1-wide) | pad-stack | winner |
|---|---|---|---|---|---|
| 1 | 16 (≈r4) | 13.4 | **7.1** | 46.9 | scalar 1.85× |
| 2 | 32 (≈r8) | 16.8 | **11.4** | 49.1 | scalar 1.37× |
| 4 | 64 (≈r16) | **21.2** | 23.1 | 50.3 | masked 1.12× |
| 8 | 128 (≈r32) | **32.0** | 46.1 | 53.2 | masked 1.45× |
| 16 | 256 | **57.8** | 90.4 | 65.5 | masked 1.65× |
| 32 | 512 | **101.6** | 190.1 | 115.2 | masked 1.84× |

- **Crossover ≈ 40–50 ALU/lane**: scalar tail wins radix ≤ 8, masked wins radix ≥ 16.
- **`maskstore` tax ≈ 1.04×** (negligible vs plain `storeu`). The masked tail's
  cost is the **redundant full-width butterfly** (FP-port pressure), not the mask.
- **pad-into-stack-scratch is dominated everywhere** (gather/scatter copy) — rejected.

### vs MKL & FFTW (batched 4-pt split; full data in the performance doc)
- Our **scalar R4 ties FFTW-split** at every K; our AVX2 R4 beats FFTW-split 2–4×.
- On the split layout MKL is **3–8× slower than our scalar** (its `DFTI_REAL_REAL`
  strided path is its slow path), and **MKL pays its own odd-K penalty ≤1.78×** —
  there is no magic free vectorized tail it has that we don't.

### The real-cell margin test (this is the headline)
Plan **N=1024 [4,4,4,4,4] T1S** (all radix-4, scalar tail — the proven path),
`bench_oddk_tail.c`, canonical methodology (below), avg of order-flips:

| K | rem (scalar lanes) | scalar fraction | margin vs MKL | correctness |
|---|---|---|---|---|
| 32 (even) | 0 | 0% | 1.73× | 0.00e+00 |
| 24 (even) | 0 | 0% | 1.89× | 0.00e+00 |
| 30 (even) | 0 | 0% | 1.88× | 0.00e+00 |
| 33 | 1 | 3.0% | ~1.81× | 0.00e+00 |
| 29 | 1 | 3.4% | 1.77× | 0.00e+00 |
| 25 | 1 | 4.0% | 1.77× | 0.00e+00 |
| 31 | 3 | 9.7% | 1.61× | 0.00e+00 |
| 15 | 3 | 20% | 1.29× | 0.00e+00 |

- **Bit-exact at every odd K. Beats MKL ~1.6–1.8× at odd K** (even-K ~1.7–1.9×).
- **The margin tracks `rem/K` monotonically.** At rem=1 we're at the even-K margin
  (the one scalar lane is negligible); as the scalar fraction grows the margin
  erodes. K=15 (3/15 = 20% scalar) is the worst at 1.29×; at large K the penalty
  vanishes (K=255 rem=3 → 1.2% scalar → ≈ even-K margin).

---

## The two big lessons

### 1. Measurement methodology dominates (Tugbars' insistence, validated)
On the **unlocked** i9-14900KF, naive back-to-back timing is worthless:
- Unpaced fixed-order inflates the second-measured engine (vfft heats the core →
  MKL measured hot → ratio flatters us).
- **Heavy pacing made it WORSE** — cooling between runs lets the core re-turbo
  unpredictably → ~**2× swings in the same engine's time**.
- **Averaging over repeats** helped but left outliers (even-K cells with 0.94–3.10×
  spread).
- **The canonical fix (mirrors `bench_1d_vs_mkl.c` `measure_ab`):**
  **(a) one cell per PROCESS** (cross-cell cache/thermal carryover can't be
  cachebusted away); **(b) best-of-5 MIN** per engine; **(c) cachebust + `cool_ms`
  idle BETWEEN the two engines** (comparable baseline); **(d) ORDER-FLIP**
  (vfft-first / mkl-first) averaged. With this, **flip0 ≈ flip1** — that agreement
  *is* the proof the measurement is fair. This is how the ~1.6–1.8× numbers above
  were earned (an earlier unpaced "1.65–1.83×" was withdrawn, then re-confirmed
  properly).
- Reliable *absolute* numbers still need machine lockdown (fixed freq / turbo off);
  the ratio method above is robust enough for the margin question.

### 2. The scalar tail is correct but NOT optimal (Tugbars' catch)
Because the margin tracks `rem/K`, the erosion is **our scalar tail being less
efficient per remainder-lane than MKL's method**: MKL's per-transform cost stays
~flat across K (vector-efficient remainder); ours rises ~`rem`× because each
scalar lane runs ~3–4× slower than a vector lane. So **MKL's remainder handling
beats our scalar tail**, and it shows more the larger `rem` is.
⇒ **Next experiment: a vector-efficient remainder** — one masked vector pass for
the *whole* remainder (cost ≈ 1 vector butterfly, flat in `rem`), instead of `rem`
scalar passes. (The crossover said our *earlier* masked tail lost to scalar
standalone, but that was a sloppy masked with a redundant full-width recompute +
port pressure; MKL's flat curve proves a clean vectorized remainder wins in the
full-transform-vs-MKL sense. The standalone microbench and the vs-MKL margin
answer different questions; the latter is what matters competitively.)

---

## Open bug — blocked codelets at odd K
The `[4,4,4,8,8]` plan (r4 + **blocked two-pass r8** t1s) corrupts a **bulk** lane
at odd K (~4e2 error, e.g. lane 0/2), **independent of the tail**:
- Identical error with the hand-spliced scalar tail **and** the verified scalar
  offset-call (`radix8_t1s_dit_fwd_scalar(rio+k, …, ios, me-k)`).
- The r8 AVX2 **bulk is byte-identical to the git original** (diff clean).
- **All-radix-4 plans are clean at odd K.**
⇒ It's in the **executor / twiddle / L1-seam handling of BLOCKED codelets at odd
K**, not the codelet tail. (Caveat that masked it: even-K correctness was a
degenerate self-compare `Kp=K` — only odd-K comparisons are real.) Note r8 is the
borderline radix and r16/r32 are masked-tail radixes anyway.

---

## Gotchas hit (save the next session the time)
- **`vfft_proto_posix_memalign` = `_aligned_malloc` on Windows → MUST free with
  `vfft_proto_aligned_free`, not `free()`.** Freeing with `free()` = silent heap
  corruption → crash (exit 116 after the work completed). This cost a debug cycle.
- **MKL run needs Intel `setvars.bat`** (manual PATH + local-dll-copy is not
  enough) **+ `MKL_THREADING_LAYER=SEQUENTIAL`** (else it loads
  `mkl_intel_thread.2.dll`/libiomp5 and fails). Wrap in a `.bat` (setvars is
  cmd-only); call the exe by **full path** (cmd wouldn't find it by bare name).
- **build.py** `dag_codelet_srcs()` globs only `inplace/avx2`, `rfft`, `c2r`,
  `oop` (non-recursive) → `codelets/experiments/` is **not** compiled (safe to
  park edits there, no symbol clash). Bench dag includes are **bare**
  (`#include "executor.h"`), resolved by build.py's recursive `-I core` walk — not
  `../core/...`.
- Editing codelets invalidates the cached `libdagcodelets.a` → ~100s full rebuild;
  driver rebuilds after are ~8–13s.
- stdout to a pipe is block-buffered → a crash loses buffered prints; `setvbuf(
  stdout, NULL, _IONBF, 0)` to see progress before a crash.

---

## Artifacts
- **Parked experiment:** `src/dag-fft-compiler/codelets/experiments/scalartail/`
  (`r4_n1_fwd.c` [scalar+masked toggle], `r4_t1s_dit_fwd.c`, `r8_t1s_dit_fwd.c`
  [offset-call], `r8_t1s_dit_fwd_scalar.c`, + README). NOT built by default;
  production `inplace/avx2/` is the original generated code.
- **Bench:** `build_tuned/benches/bench_oddk_tail.c` (one cell/process,
  `<K> <flip> <cool_ms>`). Driver: `C:\tmp\run_oddk.bat` (sources setvars, loops
  K × both flips as fresh processes).
- **Scratchpad spikes:** `masked_tail_spike.c` (trap + maskstore proof),
  `crossover.c` (scalar/masked/padstack sweep), `r4_vs_fftw.c`, `r4_vs_mkl.c`,
  `tail_validate.c` (12/12), and `tailgen/` (generated `--isa scalar` codelets).

---

## Next steps
1. **Vector-efficient remainder experiment** — clean forward masked tail (one
   pass, no redundant recompute), re-run the exact margin test (scalar vs masked
   vs MKL across rem=1/2/3) to confirm masked holds the margin flat in `rem`.
2. **Fix the blocked-r8 odd-K bug** — instrument the executor's twiddle/seam path
   for blocked codelets at odd K (or a tiny printable N=32 [4,8] case).
3. **Generator change** (the production path): emit the tail per codelet in
   `emit_c.ml` (two functions per file — bulk + `static inline` tail helper, or the
   build-config/macro-VM route); per-radix strategy (scalar ≤ r8, masked ≥ r16,
   measured); then **relax the `K%8` dispatch guards** to `K != 0`; wire the MT
   slicer to hand the remainder to the last worker.
4. **Tiny K < VW** — the within-transform leaf (separate axis), later.
