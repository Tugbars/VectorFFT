# r2c/c2r × IL — design, corrected by measurement (2026-07-15 bench)

*Design target: make single-threaded r2c/c2r consistently beat MKL by deleting
the split-layout boundary taxes, within the committed IL structure (6a16–6a19:
boundary folds on the stride engine, split interior, z only at true
boundaries, NULL-halves public contract). This doc supersedes the pre-bench
sketch; every claim below is anchored to the attribution bench
(`benches/bench_r2c_tax.c`, three-binary stub-delta methodology the tree
itself defines, plus a live model-B activation test).*

## 1. What the bench established (same-process, container)

(512,256), decoupled path, DIT inner, fused-first-stage ACTIVE:

| component | µs | share | method |
|---|---|---|---|
| total (good inner plan) | 244.2 | — | median |
| MKL r2c CCE (row-batched) | 207.4 | 0.849× | |
| stage-0-from-woven-x | 92.9* | ~29% | STUB_PACK delta (*earlier run's plan; ratio stable) |
| postprocess (split recombine) | ~103 | ~42% | prof ratio × total |
| inner stages 1.. | ~29% | | residual |

Non-pow2 low-K (rfft-native path): **(2000,4) = 1.22× MKL, (200,4) = 1.62×**
— we already beat MKL there; the losing corner is HIGH-K DECOUPLED ONLY.

Plan-quality sensitivity: the same (512,256) cell ran 354µs → 244µs purely
from a better inner (256,256) wisdom entry (merged bundle). Inner-cell wisdom
moves r2c by ~30% — r2c cells should participate in wisdom composition the
same way fft3d does (6a19 pattern).

## 2. Corrections to the naive design

**C1. The input-side pack tax is conditional, already mostly dead.** The
fused first stage (`_r2c_fused_first_stage`, woven lattice
`input_leg_stride = elem_per_leg·2·K` — exactly the design's woven view) is
live and primary. It is gated **DIT-inner + B%VW==0** only; DIF inners and
misaligned B pay the full explicit pack. The IL structure's contribution is
therefore COVERAGE, not invention: our fold family already has DIF entry
folds (`t1_dif il_in`) and tail-aware `me` (gated at 65/67) — generalize the
fusion to DIF + arbitrary B. At K=1 the identity x ≡ z(half) makes the
existing `il_in` codelets apply verbatim.

**C2. Model (b) exists, is CORRECT, and is SLOWER — the setter was never the
only gap.** `_r2c_laststage_fused` + the prototype codelet
(`radix256_r2c_term_ls_r8_fwd_avx2`, generated, in the archive) have **no
plan-time setter anywhere in the tree** (`ls_fwd`/`term_fwd` are declared,
NULL-checked, never assigned). Live activation via the bench (same-TU field
set): output match BAD=0 — but **286.6µs vs 244.2µs separate-postprocess
(−17%)**. The caller-side scaffold (scalar group-pair specials loop) eats the
scratch-round-trip savings. Model (b)'s value is real but locked behind
vectorizing that scaffold; it was correctness-verified, never
perf-validated. This is why it was dormant.

**C3. z-store alone ≠ beat.** Post split I/O is 8 streams; z-out makes 6 →
saves ~25–27µs here → ~218µs vs 207 (≈0.95×). Necessary, not sufficient.

## 3. The revised path to "consistently beat", in value order

| # | work | mechanism | expected effect (this cell class) |
|---|---|---|---|
| 1 | inner-cell wisdom hygiene | ship/compose good (N/2,K) entries (6a19 pattern) | up to ~30% total — measured, free |
| 2 | z-store terminator variants (step-2 `term_fwd` + postprocess) | store-pattern flag in the hand-maintained kernels; z contract via `dim==NULL` (6a19 convention) | ~10% of post; enables the MKL-comparable contract |
| 3 | vectorize model-(b) scaffold + write the SETTER + codelet family gen (`r2c_term_ls` beyond the r8 prototype) | kills the post scratch round-trip for real | post 103µs → collapses toward last-stage cost; the actual beat lever |
| 4 | fold coverage: DIF-inner + B-tail via IL entry folds | committed 6a16 machinery | deletes the explicit pack where it still fires (high-K DIF cells) |
| 5 | rfft z-terminator | terminator-variant precedent (rfft.h:176) | extends the z contract to the low-K path we already win |

c2r mirrors 2–4 via the swap-identity route (6a18): the inner bwd = swapped
fwd, so the exit fold reuses fwd OOP stores — no new bwd codelet family.

## 4. Bugs / debts surfaced by the bench

- `rfft.h:523` mid-column loop: `-Waggressive-loop-optimizations` flags
  genuine UB at some (N,K) (iteration-count overflow in the `vl` bound) —
  triggered while the rfft **runtime-jit** compiled
  `rfftjit_n1024_k4_..._ver4` (which also confirms the rfft jit inherited the
  6a17 dual-symbol emission). Fix the bound arithmetic regardless.
- `ls_fwd`/`term_fwd` setter absence (C2) — tracked as the model-(b) wiring
  debt; do NOT wire before the scaffold is vectorized (it is a measured
  regression as-is).
- Public wisdom free/save parity gap (6a19) also applies to any future r2c
  terminator wisdom.

## 5. Bench reproduction

`benches/bench_r2c_tax.c`: includes vfft.c same-TU (prof counters + live
field access), three binaries (`-DVFFT_R2C_PROFILE`, `-DVFFT_R2C_STUB_PACK`,
`-DVFFT_R2C_STUB_POST`), shared wisdom bundle so all arms time identical
plans; model-B activation block self-gates on codelet geometry
(halfN==256, last radix 8, DIT) and checks output equality before timing.
MKL arm: CCE, NOT_INPLACE, row-batched (their home layout — noted in every
comparison). Container ratios carry the usual cross-session instability;
stub deltas and before/after activations are same-process.


## 6. Post-twiddle OOP codelet family — generator spec (Gap A)

The one missing artifact for fused DIF-inner r2c entry. Platform generator
task (DAG-construction change in codelet_oop.ml; the pre-twiddle analog is
`dft_expand_twiddled` ~line 614 — a post mode builds the DFT DAG first, then
multiplies output legs).

**Math**: out_leg0 = DFT_0(in); out_leg_j = W[j-1] ⊙ DFT_j(in) for j=1..R-1
(standard non-conjugate cmul), i.e. the OOP twin of the in-place
`radix{R}_t1_dif_fwd_{isa}`.

**ABI**: match the dormant `stride_t1_oop_fn` exactly —
`(const in_re, const in_im, out_re, out_im, const W_re, const W_im,
size_t is, size_t os, size_t me)`, W layout `[(j-1)*me + m]` (same as t1).
Lanes contiguous (UG both sides), runtime is/os.

**Requirements**: rem-aware tails via the existing anyk_tail machinery
(masked group loads/stores; kill switch VFFT_NO_ANYK_TAIL); a `_log3`
variant for variant-bound parity with the generic tier's per-variant
`st->t1_fwd` binding; symbol MUST carry the orientation (`t1_dif_oop`, not
`t1_oop`) — the current family's `--dif` flag does not change math or name,
a collision hazard documented in §6a23.

**Wiring once the family lands** (all recon done): populate
`st->t1_oop_fwd` at plan time (or a local r2c resolver keyed radix+use_log3,
NULL → explicit fallback continues); add the DIF branch beside the DIT
fused-first dispatch (kb-broadcast grp_tw rows, leg0 untwiddled); extend
`benches/gate_r2c_tail.c` DIF cells to assert the fused path fired. Radices
needed for spike DIF inners: {5, 10, 20, 25} × {plain, log3} × avx2/avx512.


## 7. §6a24 outcome note (claim correction)

The §3 estimate "z terminator saves ~10% of post" was CONSERVATIVE: measured
−13.7% of post (108.3->93.5µs at (512,256)), −3.4% total pure store-variant,
−5.5% at the public boundary (z also skips the split convenience's staging
copy). First contract-equal MKL comparison: z/MKL 0.722× at MKL's best r2c
geometry. Evidence + drift-refutation narrative in
mkl_geometry_contracts.md §6a24.
