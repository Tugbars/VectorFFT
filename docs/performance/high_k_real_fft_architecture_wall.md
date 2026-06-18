# The High-K Real-FFT Wall: Wide-Batch vs Deep-Single-Transform

**Date:** 2026-06-18
**Hardware:** Intel i9-14900KF (Raptor Lake), AVX2 only, L1d 48KB / L2 2MB, single-thread
**Toolchain:** mingw gcc 15.2, MKL oneAPI (`mkl_set_num_threads(1)`), pinned core 2
**Transform under study:** real-input FFT (r2c), N=256, batch K up to 256, split-complex output

> TL;DR — Our FFT *codelets* beat MKL. The library wins C2C and 2D at every size, including
> K=256. Real-FFT at high K is the **one corner** where we lose (~0.81× MKL). The cause is **not**
> the codelets and **not** the split layout — it is the *wide-K-batch executor orchestration*, which
> is exactly what makes us win everywhere else. Every cheap fix was prototyped and ruled out by
> measurement. Closing the gap requires a fused, L1-resident r2c kernel for that corner only.

---

## 1. The numbers

Decoupled r2c (pack → our c2c(N/2) → Hermitian recombine), best variant, vs single-thread MKL r2c:

| K   | ours (ns) | MKL (ns) | ratio   |
|-----|-----------|----------|---------|
| 32  | 4,903     | 4,298    | 0.877×  |
| 64  | 14,049    | 8,720    | 0.621×  |
| 128 | 27,677    | 16,893   | 0.610×  |
| 256 | 41,790    | 33,888   | **0.811×** |

For reference, the **native packed rfft** cascade is **0.47× MKL** at K=256 — so the decoupled
method is already ~**2× better** than our own native real-FFT path at high K. The loss zone is K≥64;
at low K the native rfft path wins outright (hybrid dispatch handles the crossover).

### Where the time goes (K=256 breakdown)

| stage            | cost     | notes |
|------------------|----------|-------|
| pack             | ~11.6µs  | read real `x`, write packed split-complex `z` |
| inner c2c FFT    | ~17.4µs  | the actual transform — *faster than MKL's FFT* |
| Hermitian recombine | ~12.6µs | read `z`, fold conjugate pairs, write output bins |
| **total**        | ~41.7µs  | vs MKL 33.9µs |

The decisive fact: **recombine AVX2 (12.6µs) is no faster than scalar (12.3µs).** Vectorizing the
math does nothing → the recombine is **100% memory-bandwidth-bound** (~1MB touched in ~12.6µs ≈ 83
GB/s, at the bandwidth floor). The FFT itself is only ~17µs; the pack and recombine — two *dumb
memory passes* — are nearly as expensive, and together they are the entire gap.

---

## 2. What the 0.81× plan is actually made of

Wisdom: `N=128 K=256 → factors (4,4,8), variant T1S`. The codelets:

| step | codelet | role |
|------|---------|------|
| pack (fused) | `r4_n1_oop_avx2` | stage-0 reads real `x` directly via separate in/out strides |
| c2c stage 0 (leaf) | `r4_n1_fwd` | radix-4, no twiddle |
| c2c stage 1 | `r4_t1s_dit_fwd` | radix-4, T1S twiddle |
| c2c stage 2 | `r8_t1s_dit_fwd` | radix-8, T1S twiddle |
| recombine | hand-written AVX2 (mirrors `core/r2c.h:200-274`) | Hermitian fold |

These `n1` / `t1s` codelets are the **same family that wins MKL 238/238 on pure C2C.** The 17.4µs
inner FFT is faster than MKL would transform the same data. **The codelets are not the bottleneck.**

---

## 3. The architecture wall — *which* part, precisely

Our design is a deliberate trio:

1. **Split planes** — separate `re[]` / `im[]` arrays.
2. **Lane-batched layout** — `re[n*K + lane]`; bin *n*'s K batch-lanes are contiguous.
3. **Wide-K-batch, stage-at-a-time execution** — one codelet call sweeps **all K transforms** at
   once, vectorizing AVX2 over the lanes.

This trio is **why we win C2C and 2D**: perfect SIMD across the batch, and **zero transpose**
needed (MKL's interleaved + transform-major layout forces a transpose to vectorize a batch; we
never pay it).

The cost of the trio: the working set is the **whole plane**. At K=256 that is 128×256×8×2 = 512KB,
which is **L2-resident** (< 2MB L2). For pure C2C this is fine — there is only the FFT, and our FFT
beats MKL even running from L2. **We win C2C at K=256.**

Real-FFT is the one corner that breaks, because it needs **two extra full-plane passes** (pack +
Hermitian recombine) wrapped around the FFT. Each one streams the full 512KB plane through L2, and
they are bandwidth-bound. Our wide-batch design *cannot fuse them L1-resident*.

MKL takes the opposite strategy: **deep single-transform.** It runs pack → FFT → recombine on **one
~4KB transform entirely in L1**, then moves to the next of the K batch. It never materializes the
fold passes as separate L2 streams — they happen while the transform's data is already hot.

> **The wall, stated exactly:**
> wide-batch (L2-resident working set; ideal for batched SIMD)
> **vs**
> deep-single-transform (L1-resident working set; ideal for hiding the real-FFT fold passes).

> **Correction (2026-06-18):** an earlier draft called split-vs-interleaved "second-order." That is
> wrong for r2c. The **pack is a first-order split-layout tax** — see §3a. The recombine's extra
> streams (split = re/im as separate planes) genuinely are second-order; the pack is not.

The split/lane-batch layout is the *strength* that wins everywhere else; the real-FFT fold passes
(and, for split specifically, the pack) are the price.

### 3a. The pack is a split-layout tax — and OOP buys it back

The decoupled method packs reals → half-size complex: `z[j] = x[2j] + i·x[2j+1]`.

- **Interleaved (MKL/FFTW):** the real input array *is* the half-size complex array, bit-for-bit
  (`x[2j]`=re, `x[2j+1]`=im). The pack is a **free pointer cast.** In-place r2c needs no pack pass.
- **Split (ours):** `zre[]` / `zim[]` are *separate planes*. Even→`zre`, odd→`zim` is a real
  **de-interleave copy**, and an **in-place codelet cannot avoid it** (one stride for load and store).
  So **in split layout, in-place ⟹ you must pack** (the ~11.6µs pass).
- **OOP recovers the free pack:** the OOP leaf codelet (`r4_n1_oop_avx2`) has *separate* in/out
  strides — stage-0 reads `x` directly (load stride 2K: re from even rows, im from odd rows) and
  writes packed `z`. No separate pack pass.

Clean order-neutralized A/B (`build_tuned/benches/bench_r2c_inplace_vs_oop.c`, flip per trial), K=256:

| path | ns | vs MKL | vs in-place |
|------|----|--------|-------------|
| in-place (separate pack) | 43,933 | 0.792× | — |
| **OOP (no pack pass)** | **38,092** | **0.913×** | **+15% (≈5.8µs)** |
| MKL | 34,784 | 1.000× | — |

So the clean high-K r2c number is **0.913× with OOP**, not 0.81× — the ~5.8µs delta is precisely the
pack. OOP is the **intended** high-K r2c forward path. (The win only appears where the inner stage-0
leaf is a radix with an OOP codelet — at K=256 that's radix-4; for K=64/128 the inner picks (64,2),
so generalizing requires OOP leaf codelets for all leaf radixes. See §6.) The remaining ~9% to parity
is the separate recombine pass (§3 structural wall).

---

## 4. Levers tried and ruled out (measured)

### Lever 1 — natural-order Z via DIF-forward inner — DEAD (two ways)
*Bench: `build_tuned/benches/dif_order_probe.c`*

The recombine reads `Z[k]` and `Z[N/2−k]`; our forward c2c output is digit-reversed, so those are
a scattered row gather. Hypothesis: a DIF-forward inner would emit *natural* order → contiguous,
L1-blockable recombine like MKL's.

- **DIF-forward does NOT give natural order.** It is a *different* strided mixed-radix permutation
  (slot→freq = 0,16,32,48,…,4,20,…), correct (err 9e-13) but not natural. Neither DIT nor DIF hands
  us natural Z for free.
- **It wouldn't matter anyway.** The recombine is bandwidth-bound (AVX2 == scalar), so the scatter
  is *not* the cost. A contiguous access pattern saves nothing.

### Lever 2 — L1-block the K-batch via repeated `execute_fwd` — DEAD
*Bench: `build_tuned/benches/bench_r2c_l1block.c`*

Process the K-batch in blocks of B lanes; run the full pack→FFT→recombine pipeline per block so the
block (16KB at B=8) stays L1-hot. The executor supports lane sub-blocking
(`execute_fwd(plan, re+l0, im+l0, B)`).

Result — **monotonically slower**, optimum is B=K (no blocking):

| K=256 | unblocked | B=4 | B=8 | B=16 | B=32 |
|-------|-----------|-----|-----|------|------|
| ns    | 44,126    | 156,888 | 89,123 | 86,508 | 54,816 |

Why it fails:
1. **The plane already fits L2** (512KB < 2MB). The unblocked passes are L2-resident, not
   DRAM-bound, so the *only* available win is L1-vs-L2 (small) — not L1-vs-DRAM (large, as first
   estimated).
2. **Per-block overhead swamps it.** 32 blocks × (dispatch + codelets inefficient at `slice_K=8`,
   where the fixed per-invocation cost isn't amortized + 127× short recombine loops) ≈ 45µs of pure
   overhead.

You **cannot** emulate MKL's L1-residency by *calling* our wide executor in a loop — the dispatch
*is* the cost. It only works if the block loop is **inlined inside a fused kernel**.

(Side note: K=32 plans containing radix-32 even fail correctness under `slice_K < plan->K` — the
executor doesn't cleanly support sub-block execution for all plan shapes.)

---

## 5. The only remaining lever: a fused r2c kernel (option B)

Closing the corner means adopting MKL's deep-L1-resident strategy **for the r2c fold path only**: a
dedicated kernel that blocks the K-batch internally, with pack + all FFT stages + Hermitian
recombine **inlined** (no per-stage function-pointer dispatch), so each block's working set stays in
L1 across the whole pipeline.

Scope and honest expectations:
- This is **not** an architecture rewrite. The wide-batch C2C/2D engine (which is winning) is
  untouched. It is **one specialized path for the corner the architecture isn't shaped for.**
- The ceiling is **parity**, not a blowout. MKL itself keeps the recombine a separate (L1-blocked)
  pass, and the work is irreducibly memory-bound.
- Effort is real (a new emitted/hand-written fused kernel + its dispatch), for a corner-case win.

---

## 6. Decision — Option A LANDED (2026-06-19)

**Platform directive (2026-06-18):** the platform must offer **both an in-place and an OOP variant for
every feature** (C2C, R2C, C2R, 2D, …). R2C/C2R OOP is in scope now; OOP C2C is deferred.

**Productionized (Option A, 2026-06-19) — high-K r2c flipped 0.50× → 1.01× MKL:**
- The OOP codelet tree was **not needed** — `_r2c_fused_first_stage` already pack-fuses for any leaf radix
  via the regular `n1_fwd` (7-arg separate in/out stride). Build.py unchanged.
- The shape guard was **stale** and is lifted; the perm-driven `_r2c_postprocess` is general (verified 30
  shapes × K, all <1e-9). `stride_execute_r2c_inplace` exposes the in-place placement (matches OOP exactly).
- A latent bug surfaced + fixed: c2c wisdom can pick a **DIF** inner whose output order ≠ DIT
  digit-reversal → wrong recombine; the builder now force-rebuilds DIF inners as DIT (the recombine requires DIT).
- **Hybrid dispatch** (`_vfft_r2c_decouple_min_k`, default 32 = measured N=256 crossover): K≤16 → rfft,
  K≥32 → decoupled stride. Production dispatch vs MKL (SPLIT): K=8 1.07×, K=16 1.03× (rfft); K=32 0.99×,
  K=64 0.68×, K=128 0.66×, **K=256 1.01×** (decoupled, beats MKL) — the old all-rfft path was 0.50× at K=256.

Below is the original decision framing (kept for context):

| Option | What | Outcome |
|--------|------|---------|
| **A — OOP r2c + generalize + hybrid** (recommended) | Make OOP the high-K r2c path. Wire `codelets/oop/{isa}` (~51 n1 leaves) into build.py's `dag_codelet_srcs` so *all* leaf radixes get pack-fusion (not just radix-4). Lift the `core/r2c.h:1161-1191` guard; OOP-fused + in-place variants both exposed; hybrid dispatch (native rfft low-K, decoupled high-K) in `core/r2c_dispatch.h`. | 0.913× at K=256 today (radix-4 cell); generalizes the win + satisfies the in-place/OOP directive. Low–moderate effort. |
| **B — Fused r2c kernel** | Deep-L1-resident kernel for the r2c fold path (pack+FFT+recombine inlined per L1 block). | Targets MKL parity at high K (closes the last ~9% = the recombine pass). Real workstream, parity ceiling. |
| **C — Defer** | Leave prototypes + findings; revisit later. | No change. |

A and B compose: A (OOP, no pack) gets ~0.91×; B (fused recombine) closes the remaining recombine pass
to parity. The C2C and 2D wins are unaffected by this corner. Real-FFT at high K is the single place our
batched-SIMD + split layout pays for the strength that wins it everywhere else — and OOP recovers most
of that price for free.

---

## 6b. Does the split 2× SIMD edge beat the pack tax at high N? (tested — no)

Hypothesis: N=256 is the regime least favorable to us (tiny inner FFT → pack is a big fraction); at high N,
FFT compute O(N·log N) outpaces the O(N) pack, and split's 2× SIMD edge should dominate → win by more.
Tested (`bench_r2c_highN_vs_mkl.c`, calibrated inner wisdom present for all N/2 up to 4096):

| N | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|-----|-----|------|------|------|------|
| mkl/strd, K=256 | **0.97×** | 0.64× | 0.85× | 0.65× | 0.66× | 0.55× |
| mkl/strd, K=64  | 0.71× | 0.78× | 0.70× | 0.69× | 0.65× | 0.57× |

**Falsified** — decoupled/MKL is *best at N=256* and drifts down at high N. The pack-tax-shrinking effect is
real (visible K=64 N=256→512: 0.71→0.78) but is quickly overwhelmed: the **2× SIMD edge is a *compute*
lever, and high-N real-FFT is *memory*-bound** (N/2×K plane ≫ L2; the decoupled method's ~3 passes =
~3× DRAM traffic). When DRAM-bandwidth-bound, instruction efficiency is moot and MKL's fused/cache-blocked
real-FFT wins. N=256 K=256 wins *because* 512KB is L2-resident — the one spot compute still dominates.
Consistency: pure C2C still beats MKL at high N (no extra passes); only the real-FFT pack+recombine passes
lose at scale. ⇒ The sole high-N lever is cutting passes (recombine-fusion / Option B), not more SIMD.

## 7. Follow-ups / TODO

- [ ] **Re-calibrate the hybrid threshold for other N.** `_vfft_r2c_decouple_min_k = 32` is the measured
  crossover for **N=256 only**. Other transform lengths (e.g. 128, 512, 1024, 4096) may cross over at a
  different K — re-run `bench_r2c_dispatch_vs_mkl.c` per N (and per host) and set via the runtime knob
  `vfft_r2c_dispatch_set_decouple_min_k()`. It is a single global today; may need to become N-aware.
- [x] **DIF inner support — DONE (2026-06-19), and measured not worth defaulting to.** Added
  `_r2c_compute_perm_dif` (DIF output order = digit-reversal with **factor order REVERSED**; verified
  `r2c_dif_inner_test.c`, all <1e-9), dispatched by `inner->use_dif_forward` in `stride_r2c_plan`, and gated
  the DIT-leaf pack-fusion off for DIF (DIF leaf is the LAST stage → it takes the explicit-pack path). So
  any inner orientation is now CORRECT (no more silent-wrong-output). **But the dispatcher still force-DITs
  the inner for PERFORMANCE:** pack-fusion is DIT-only, and a DIF inner + explicit pack loses more than DIF's
  c2c edge gains — measured N=256 K=32: DIT+fused **0.99×** vs DIF+explicit **0.87×** MKL. So force-DIT is the
  perf-optimal default; DIF support is a correctness safety net, not a speed lever.
- [ ] **Expose the in-place placement through the unified `vfft_r2c_execute_fwd`** (it exists at the stride
  level as `stride_execute_r2c_inplace`; the top dispatcher currently only does out-of-place).
- [ ] **Commit** (changes are on `dev/OcamlScheduling`, uncommitted).

## Appendix — reproduce

```sh
cd build_tuned
# breakdown (pack / c2c / recombine, scalar vs AVX2):
python build.py --src benches/bench_r2c_opt_inner.c --mkl --compile
# DIF ordering probe (lever 1):
python build.py --src benches/dif_order_probe.c --compile
# L1-blocking sweep (lever 2):
python build.py --src benches/bench_r2c_l1block.c --mkl --compile
# in-place (with pack) vs OOP (no pack), order-neutralized (§3a):
python build.py --src benches/bench_r2c_inplace_vs_oop.c --mkl --compile
# run with MKL bin + C:\mingw152\mingw64\bin on PATH, pinned core 2
```

Related: `src/dag-fft-compiler/docs/rfft_highk_gap_and_mkl_blueprint.md` (native-rfft VTune +
MKL disassembly), and the project memory note `recombine_fusion_resume.md`.
