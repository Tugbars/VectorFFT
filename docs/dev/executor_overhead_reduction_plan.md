# Executor overhead reduction — design plan

**Status:** v1.1 candidate work, not started.
**Target cell:** N=131072 K=4 (the v1.0 closest-margin cell, 1.17× MKL).
**Goal:** push 1.17× → ~1.30-1.45× MKL by reducing per-stage executor
overhead.
**Reference:** [vtune_n131072_k4_vfft_vs_mkl.md](vtune_n131072_k4_vfft_vs_mkl.md)

---

## Why this matters

The N=131072 K=4 VTune dive established:

- VectorFFT runs **2.43× more instructions per FFT** than MKL
  (41.24 B vs 16.95 B retired).
- We compensate with **2.68× higher IPC** (3.94 vs 1.47).
- Net wall-time margin is 1.17× — the two effects nearly cancel.

The 2.43× instruction-count gap is **architectural cost of multi-stage
plan-driven decomposition.** 80% of our retired instructions are
non-FMA overhead per FMA — function-call boilerplate, per-stage base
pointer compute, scalar twiddle preprocessing, twiddle table walks,
loop bookkeeping.

VTune's function-level breakdown localizes the cost:

| Function | Share | What it is |
|---|---:|---|
| `radix4_t1s_dit_fwd_avx2` | 33.6% | Codelet body — useful work |
| `radix4_n1_fwd_avx2` | 17.4% | Codelet body — useful work |
| `radix8_t1s_dit_fwd_avx2` | 6.9% | Codelet body — useful work |
| **`_stride_execute_fwd_slice_from`** | **21.0%** | **Per-stage executor bookkeeping** |
| **`stride_cmul_scalar_avx2`** | **7.1%** | **Scalar twiddle preprocessing** |

**~28% of CPU time is non-codelet executor work.** That's the
optimization surface.

The "actual ceiling" for this cell is ~3× MKL (per the VTune doc) — the
gap between 1.17× and 3× is roughly the instruction-density gap. Reducing
executor overhead is the path to closing it.

---

## Two design candidates

### B1 — Pre-flattened iteration descriptors

**Idea:** at plan time, walk every `(stage, group)` pair and pack into
a single contiguous array of structs. Hot path becomes one indirect-call
loop with no outer-stage logic.

```c
typedef struct {
    void (*codelet)(double *, double *, const double *, const double *,
                    size_t, size_t);
    int    re_offset;       /* offset into user-provided re buffer */
    int    im_offset;
    const double *tw_re;     /* pre-fetched twiddle pointer */
    const double *tw_im;
    double cf0_re, cf0_im;   /* baked into twiddle at plan time --
                                this field exists for diagnostics */
    size_t stride;
} stride_flat_desc_t;
```

New executor:
```c
void _stride_execute_fwd_flat(const stride_plan_t *plan,
                              double *re, double *im) {
    const stride_flat_desc_t *d = plan->flat_descs;
    int n = plan->n_flat_descs;
    size_t K = plan->K;
    for (int i = 0; i < n; i++) {
        d[i].codelet(re + d[i].re_offset, im + d[i].im_offset,
                     d[i].tw_re, d[i].tw_im, K, d[i].stride);
    }
}
```

**Engineering challenges:**

1. **Codelet signature uniformity.** Today's `stride_n1_fn`,
   `stride_t1_fn`, `stride_t1s_fn` types differ. Three options:
   - (a) Uniform "wrapper" signature with unused params for variants
     that don't need them. Cost: ~5% per call from passing dummy args.
   - (b) Separate descriptor arrays per variant type, chained. Cost:
     extra branch per stage transition.
   - (c) Per-(N,K) cell custom loop with right call types. Edges into
     JIT territory.

   **Pick (a)** — accept 5% per-call overhead in exchange for branch-free
   dispatch. Net should still be a win.

2. **CF0 fold-into-twiddle.** The scalar `cmul_scalar` does
   `tw[i] *= cf0`. To eliminate the separate pass, do this multiplication
   at plan time and store the resulting twiddles per descriptor.
   **Cost: 2× memory for twiddles** (each desc has its own pre-multiplied
   set). For N=131072 K=4 with ~32k descriptors: ~4 MB of pre-baked
   twiddle. Plan size grows.

3. **Backward path duplication.** Build flat descriptors for both fwd
   and bwd. Doubles plan-time cost; not hard.

4. **Blocked executor compatibility.** The blocked path
   (`_stride_execute_fwd_blocked`) has different inner-loop structure.
   **Skip blocked for v1 of this work.** Most cells use the standard
   executor.

**Expected gain:**

The 21% executor overhead at N=131072 K=4 breaks down (estimated):

| Component | Estimated share | B1 effect |
|---|---:|---|
| Per-stage dispatch (switch) | 2-3% | Eliminated |
| Per-group base ptr compute | 5-7% | Constant offset baked |
| Per-group twiddle ptr fetch | 3-4% | Constant ptr in descriptor |
| Scalar cmul preprocessing | 6-8% | Folded into pre-baked twiddle |
| Codelet function call | 2-3% | Same (one call per desc) |

**Net wall-time gain estimate: ~10-13%.** Translates 1.17× → ~1.30× MKL.

**Implementation cost:** ~2-3 days focused.

**Risk:** uniform-signature shim's per-call overhead might eat the
savings. **Mitigate via 1-day spike that benches before committing.**

---

### A1 — Plan-time JIT compilation

**Idea:** at plan time, emit a fully-unrolled function for the cell with
all base pointers, twiddle pointers, and stage parameters baked as
immediates. No descriptor array — the function IS the table, in I-cache.

```c
/* JIT generates this at plan time for cell (N=131072, K=4): */
void plan_4x4x4x4x8x4x4x4_K4_fwd(double *re, double *im) {
    /* Stage 0 — no twiddle */
    radix4_n1_inline(re+0,    im+0);
    radix4_n1_inline(re+16,   im+16);
    /* ... 32766 more, generated */

    /* Stage 1 — twiddle ptrs hard-coded */
    radix4_t1_inline(re+0,  im+0,  TW_S1+0);
    radix4_t1_inline(re+16, im+16, TW_S1+8);
    /* ... */
}
```

**Implementation paths:**

| Path | Plan-time cost | Lib dependency | Maintenance | Notes |
|---|---|---|---|---|
| **A-system**: `system("cl ...") + LoadLibrary` | ~seconds/cell | needs `cl`/`icx` at runtime | high | text-based codegen |
| **A-LLVM**: LLVM ORC | ~50 ms/cell | LLVM (~30 MB) | medium | IR generation |
| **A-asmjit**: header-only x86 emitter | ~5 ms/cell | header-only | high — write x86 yourself | wild-card option |
| **A-AOT**: pre-generate `.c` per cell, compile at lib build time | 0 at runtime | none | low — codegen is build-time | FFTW's actual approach |

**Expected gain:**

Same wins as B1 PLUS:
- No descriptor-array memory cost (function in I-cache instead)
- No descriptor-walk overhead at all (inlined instructions)
- Compiler can optimize the whole function as a unit (cross-stage
  scheduling, peephole opts, register allocation)

Realistic estimate: B1's ~10-13% PLUS another 5-10% from compiler-level
optimization → **~15-25% wall-time win.** Pushes toward 1.40× MKL.

**Trade-off vs B1:**

| Aspect | B1 (descriptor flatten) | A1 (JIT) |
|---|---|---|
| Plan-time cost | ~µs | 5 ms (asmjit) to seconds (cl) |
| Runtime deps | none | varies by path |
| Memory | +~4 MB plan size | +~50 KB I-cache per plan |
| Engineering | 2-3 days | 1-3 weeks depending on path |
| Risk | medium (signature shim) | high (toolchain, codegen complexity) |
| Reversibility | easy (skip plan->use_flat) | hard (codegen embedded) |
| White paper claim | "data layout optimization" | "runtime code generation" |

---

## Recommendation

**v1.1: ship B1.** Predictable ~10-13% gain, no new toolchain, fits the
v1.0 architecture cleanly. Implementation is data-structure work +
uniform-signature shim, both well-understood.

**v2.0+: consider A-AOT (FFTW-style pre-gen).** Bigger gain (~20%),
no runtime deps, but fundamentally different distribution shape — one
library + thousands of pre-compiled cell-specific functions. Repo and
build time balloon. Adopt only if the v1.1 gain isn't enough.

**Don't do A-LLVM or A-asmjit** unless someone strongly wants runtime
JIT specifically. They introduce hard dependencies (LLVM) or hard
maintenance (hand-emitted x86) for marginal gain over A-AOT.

---

## v1.1 implementation plan (B1)

### Phase 0 — spike (1 day)

Goal: verify the 10-13% projected gain actually lands. Build the
minimum to bench, before committing to the full implementation.

1. **Read the current executor inner loop** (`src/core/executor.h`'s
   `_stride_execute_fwd_slice_from`) to confirm the per-stage cost
   breakdown.
2. **Hand-build flat descriptors** for one cell (N=131072 K=4) at
   plan-construction time, hardcoded in a development branch.
3. **Stub uniform-signature shims** for n1/t1/t1s (just adapter
   functions, no codegen yet).
4. **Bench** N=131072 K=4 against current executor. If 5%+ gain → green
   light. If less → investigate before continuing.

### Phase 1 — production (1-2 days)

If the spike confirms a real gain:

1. Promote `stride_flat_desc_t` and `_stride_execute_fwd_flat` from
   development branch to `src/core/executor.h`.
2. Add `plan->use_flat` flag, populate at plan time when applicable.
3. Add `_stride_execute_fwd_auto` dispatch that picks flat over
   standard when the flag is set.
4. Backward path: build flat descriptors for fwd AND bwd.
5. Bench across the full 207-cell grid to ensure no cell regresses.
6. Update `MEMORY.md` profiles for the affected cells.

### Phase 2 — gating (½ day)

Decide which cells get flat:
- All standard-executor cells: yes
- Blocked-executor cells: skip for v1.1, consider later
- Bluestein/Rader inner FFTs: may benefit, but the inner FFT's plan is
  built per-call inside Bluestein execute path — needs care to avoid
  blowing plan-time cost
- Small cells (K=4, K=32): biggest expected gain
- Large cells (K=256): smaller gain but probably still positive

**v1.1 ship criterion:** every cell that opts into flat improves or stays
within 1% of current. No regressions.

---

## What we're NOT doing

- **Full codelet fusion (C1 from the brainstorm).** The 3× MKL ceiling
  needs this, but it requires fundamental codegen work. Defer to v2.0.
- **Register-tiled data-flow fusion** — works only on AVX-512 hosts, where
  the register file fits. Not applicable to Raptor Lake AVX2.
- **Any of the wild ideas** (bytecode VM, helper threads, GPU-assisted
  executor) — interesting on paper, no clear engineering ROI for v1.x.
- **Touching the blocked executor.** That codepath has its own structure;
  optimizing it is a separate project.

---

## Bench cells for verification

When spike + Phase 1 land, validate against this grid:

| Cell | Why |
|---|---|
| **N=131072 K=4** | The target cell, biggest expected absolute gain |
| N=32768 K=4 | Same shape (pow2 K=4), different size — verify the gain scales |
| N=8192 K=4 | Smaller pow2 K=4, executor overhead share is higher → may gain more |
| N=1024 K=256 | Pow2 K=256 — verify K=256 cells don't regress |
| N=4096 K=256 | Larger pow2 K=256 |
| N=243 K=4 | Non-pow2 (3^5), exercises radix-3 codelets |
| N=179 K=256 | Bluestein inner — verify Bluestein path still works |

Use `build_tuned/dev/bench_vtune/` for measurement protocol.

---

## See also

- [vtune_n131072_k4_vfft_vs_mkl.md](vtune_n131072_k4_vfft_vs_mkl.md) —
  the VTune dive that motivated this work
- [wisdom_bridge_predicates.md](wisdom_bridge_predicates.md) — the
  architectural rule about plan-level measurement vs cost-model
  extrapolation
- [executor_prefetch_experiments.md](executor_prefetch_experiments.md) —
  prior failed attempts at executor optimization (prefetch + small-R
  inline dispatch). Part of the historical record so we don't redo
  these without a hypothesis.
