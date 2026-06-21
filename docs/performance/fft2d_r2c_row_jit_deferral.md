# 2D R2C/C2R row-pass JIT — deferral analysis

**Status:** the 2D real transforms JIT their **column c2c** pass (fwd + bwd) — wired and
bit-exact. The **row r2c/c2r** pass is **deferred**. This note records *why*, the risk on
both sides, and the expected payoff, so the decision is reproducible and the future
workstream is scoped. Host: i9-14900KF (Raptor Lake, AVX2).

## What is wired vs deferred

| Inner FFT | 2D path | JIT status |
|---|---|---|
| column c2c (fwd) | R2C forward Phase 2 | ✅ wired (`vfft_proto_plan_jit_fwd(plan_col)`) |
| column c2c (bwd) | C2R backward Phase 2 | ✅ wired (`vfft_proto_plan_jit_bwd(plan_col)`) |
| **row r2c** | R2C forward Phase 1 | ⏸ **deferred** |
| **row c2r** | C2R backward Phase 3 | ⏸ **deferred** |

The column pass is a clean **whole-plan** call (`stride_execute_fwd(plan_col, …)` on the
full padded `N1×K_pad` scratch), so it drops straight onto the existing c2c JIT resolver —
identical to how `fft2d.h` JITs the 2D-C2C column pass. The row pass is not.

## Why the row pass is not whole-plan-JIT-able

The row pass is invoked as `_fft2d_r2c_inner_fwd(d->plan_r2c, sr, si, tid)`, a **worker
shim**, not a plan executor. It:

1. casts `plan->override_data → stride_r2c_data_t*` and hand-builds a worker arg carrying a
   **`tid`** — the tile-parallel row pass uses `tid` to select each thread's **own scratch
   slot**;
2. calls `_r2c_worker_fwd` directly, which runs the inner real FFT **fused + sliced**:
   `_r2c_fused_first_stage` (the pack fused into stage 0) then
   `_stride_execute_fwd_slice_from(…, from_stage=1)` for the remaining stages.

The JIT executor ABI is `fn(plan, re, im, K, plan->K, 0)` — it has **no `tid`**, cannot run
a **partial** stage range (`from_stage=1`), and cannot express the **fused** stage 0. It is
also the decoupled-**stride** r2c engine (not the rfft engine), whose own c2c inner is
*also* fused — three nested fused layers. Same structural blocker as the 1D strided-r2c
deferral.

## Risk — two separate questions

**Risk of deferring: ~zero.** The row pass keeps running the existing, correct, already-tuned
fused stride-r2c executor. The fallback path *is* the generic path — nothing is broken or
exposed, no regression. The only cost is opportunity: that pass isn't JIT-*specialized*.

**Risk of wiring it anyway: high — and not a synchronization problem.**
1. **Data races (the serious one).** `tid` exists precisely to give each tile-parallel thread
   a disjoint scratch slot. A JIT executor with no `tid` notion would let two threads write
   the **same** scratch slot → silent corruption, two call-levels down — the class of bug the
   MT-vs-ST gate caught in OOP.
2. **A new emitter shape.** JIT-ing a `from_stage=1` partial run needs a from-stage emitter
   variant plus separate handling of the fused pack-stage — a meaningfully larger, bug-prone
   surface than the clean whole-plan emitters.

### Why NOT mutexes / atomics

The instinct to guard the shared scratch with a mutex/atomic is the **wrong tool** and would
*destroy the MT win*:

- The scratch is the **FFT working buffer**, read/written O(N log N) times per tile per
  thread. A lock around it serializes every thread's inner FFT → single-threaded execution
  **plus** lock overhead → slower than one thread. Atomics are for small shared scalars, not a
  multi-KB streamed buffer.
- VectorFFT's MT advantage *is* that the split/tiled layout makes threading **lock-free**:
  independent lanes/tiles, per-thread scratch, no barriers, no shared mutable state. A lock
  anywhere in the hot path is the regression — it hands MKL back the win.

The correct fix is to **preserve the per-thread scratch (`tid`)**, i.e. a **`tid`-aware JIT
executor** (each thread passes its slot index and writes its own scratch), keeping it
contention-free. Synchronization in the FFT hot path is a red flag that the parallel
decomposition was broken, not a fix for it.

## Expected latency saving (estimate)

**Single digits — ~3–10% on the overall 2D r2c at small N, washing toward in-noise at large
N. It does not close the structural ~0.65× MKL gap.** Reasoning, anchored to measured analogs:

- **Favorable:** the row inner runs at **K = B = 8 (low K)** — the regime where JIT's
  per-stage-dispatch removal pays most (rfft low-K hit 14–36% and flipped a 1D cell past MKL).
- **Caps:**
  1. Only stages 1..n-1 are JIT-able; **stage 0 is fused with the pack** — and the pack is the
     *dominant* cost (the real-FFT pack tax that makes 2D r2c structurally 0.65×). JIT
     specializes everything *except* the dominant term.
  2. The row pass is ~50–65% of total (the col c2c — already JIT'd — is the rest).
  3. Dispatch fraction shrinks with N; the **column-pass JIT we measured showed exactly this:
     +6.7% at 64², in-noise at 128²/256²**.
- Rough arithmetic: ~15% off the JIT-able inner-c2c stages × ~⅓ of a row pass × ~60% of total
  ≈ **~3%** overall; up to ~10% at the smallest cells, less above 128². Same mechanism, same
  ballpark as the measured column-pass result.

## Decision and path forward

Deferred: the trade is a real **data-race + new-emitter-complexity risk** for a **marginal,
structural-gap-bound reward** that still loses to MKL. The high-value JIT targets are the
**clean whole-plan, dispatch-heavy** cases (1D rfft low-K, 1D c2c, 2D c2c inners) — the row
r2c is the opposite profile.

When revisited, do it as its own workstream:
1. **Confirm ROI cheaply first** — VTune one 2D-r2c cell and read the row-pass per-stage
   **dispatch fraction**. If it's >~15% of the row pass, the JIT is worth it; if ~3%, it is
   confirmed not.
2. Only then build a **`tid`-aware, from-stage JIT executor** (preserving lock-free per-thread
   scratch), with its **own MT-vs-ST correctness gate** — not bolted onto this pass.
