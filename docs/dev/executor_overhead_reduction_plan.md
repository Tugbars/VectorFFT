# Executor overhead reduction — design plan

**Status:** v1.1+ work, not started.
**Target cell:** N=131072 K=4 (the v1.0 closest-margin cell, 1.17× MKL).
**Goal:** push 1.17× → ~1.30-1.45× MKL (v1.1) and toward ~1.40× MKL
(v2.0) by reducing per-stage executor overhead, layered.
**Reference:** [vtune_n131072_k4_vfft_vs_mkl.md](vtune_n131072_k4_vfft_vs_mkl.md)

**Approach:** ship in two layers atop a shared intermediate
representation:
- **Layer 1 (v1.1):** flat descriptor IR + interpreter (B1)
- **Layer 2 (v2.0+):** code generation atop the same IR (AOT for hot
  cells, optional JIT-LLVM for cold cells)

The IR is the load-bearing artifact. Build it once in v1.1; both layers
consume it.

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

## Architecture — B1 and JIT as layers, not alternatives

Earlier framings of this work treated **B1 (descriptor flatten)** and
**A1 (JIT compile)** as two competing solutions. They're actually
**composable layers** with a shared intermediate representation:

```
  Plan (factorization + per-stage variant codes)
                 │
                 ▼
  Walk all (stage, group) pairs ─→ stride_flat_desc_t[]
                 │                       (the IR)
        ┌────────┴────────┐
        ▼                 ▼
   Interpreter         Code generator
   (B1)                (JIT or AOT)
        │                 │
        ▼                 ▼
   exec_fwd_flat     exec_fwd_jitted
   (always available) (opt-in fast path)
```

The **flat descriptor array is the load-bearing artifact.** Both
consumers — the B1 interpreter and any future code generator — work
from the same array. This is a classic compiler-IR design: the
plan-time analysis (descriptor build) is separate from the
code-generation strategy (interpret vs JIT vs AOT).

**Implications:**

- **B1 ships first as the foundation.** Every subsequent codegen
  path consumes its IR. No work is wasted.
- **JIT and AOT are the same codegen pipeline at different
  invocation points.** JIT runs at plan time on the user's machine;
  AOT runs at library-build time on the developer's machine. Same
  generator, different timing.
- **The B1 interpreter is the reference implementation.** Any
  codegen path must produce results equivalent to the interpreter.
  This catches bugs and provides a fallback when JIT toolchain is
  unavailable.
- **The descriptor format is a stable internal interface.** Fields
  can be added; existing fields cannot be removed without breaking
  both the interpreter and the codegen. Worth treating as such from
  day one.

### Per-cell tiering decision

Once both layers exist, the planner picks per cell based on size and
expected use frequency:

| Cell shape | Path | Why |
|---|---|---|
| Tiny (N ≤ 256) | JIT/AOT | Unrolled code fits I-cache, runs many times in any workload |
| Mid (N = 512..8192) | JIT if plan persists, else B1 | Compile cost amortizes across many calls |
| Large (N ≥ 16384) | B1 (interpreter) | JIT'd unrolled code too big for I-cache; descriptor walk wins |
| Bluestein inner FFTs | B1 | Built per-Bluestein-execute; JIT compile cost can't amortize |

### Cache amortization math

JIT compile costs T_compile (~5 ms with asmjit, ~50 ms with LLVM ORC).
Per-execution speedup over B1 ≈ 5-10% (on top of B1's gain over baseline).
JIT pays off when:

```
N_executions × (T_b1 - T_jit) > T_compile
```

For a plan executing in T_baseline ≈ 2 ms (N=131072 K=4):
- 5% per-call gain × N executions > T_compile
- N > T_compile / (0.05 × T_baseline) = 50 ms / 0.1 ms = **500 calls**

Real-time DSP workloads ship millions of calls. Plan-once-execute-many
benchmarks ship thousands. The compile cost is paid in <1 second of
wall time. **JIT is profitable for almost any non-trivial use case
once the IR is in place.**

---

## Layer 1 — B1 — Pre-flattened iteration descriptors

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

## Layer 2 — Code generation atop B1's IR (JIT or AOT)

**Idea:** consume the flat descriptor array from B1 and emit a
fully-unrolled function with each descriptor's values baked as
immediates. The interpreter loop is unrolled away; the descriptor
array's data becomes the function's instruction stream.

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

**Implementation paths** (all four consume B1's IR; differ only in
when/how codegen runs):

| Path | When codegen runs | Lib dependency | Maintenance | Notes |
|---|---|---|---|---|
| **JIT-system**: `system("cl ...") + LoadLibrary` | plan time, ~seconds/cell | needs `cl`/`icx` at runtime | high — text-based codegen | impractical |
| **JIT-LLVM**: LLVM ORC | plan time, ~50 ms/cell | LLVM (~30 MB) | medium — IR generation | strong toolchain |
| **JIT-asmjit**: header-only x86 emitter | plan time, ~5 ms/cell | header-only | high — write x86 yourself | wild-card |
| **AOT**: pre-generate `.c` per cell, compile at lib build time | build time, 0 at runtime | none | low — codegen is build-time | **FFTW's actual approach** |

**Key insight:** since all four consume the same IR, the codegen logic
itself is shared. JIT and AOT differ in *when* the generator runs, not
*how*. We can pick a path per deployment:

- **Library author / contributor host:** AOT for hot cells, IR + B1 for
  cold cells.
- **End-user runtime with toolchain:** JIT-LLVM for cells not in the
  AOT-shipped set.
- **End-user runtime without toolchain:** B1 interpreter for everything.

**Expected gain over B1:**

- No descriptor-array memory cost (function in I-cache instead)
- No descriptor-walk overhead (inlined instructions)
- Compiler optimizes the whole function as a unit (cross-stage
  scheduling, peephole, register allocation)

Realistic estimate: B1's ~10-13% PLUS another 5-10% from compiler-level
optimization → **~15-25% total wall-time win** vs current executor.
Pushes toward 1.40× MKL.

**Trade-off vs B1 alone:**

| Aspect | B1 alone | B1 + JIT layer |
|---|---|---|
| Plan-time cost | ~µs | ~ms (codegen) — amortized across executions |
| Runtime deps | none | varies by path; AOT keeps zero runtime deps |
| Memory | +~4 MB plan size | +~50 KB I-cache per plan; ~0 plan-side |
| Engineering | 2-3 days | +1-3 weeks (path-dependent) |
| Reversibility | easy (skip `plan->use_flat`) | easy (per-cell tier-down to B1) |
| White paper claim | "data-layout optimization" | "runtime code generation atop common IR" |

---

## Recommendation

**Ship in layers:**

| Version | Layer added | Why this slot |
|---|---|---|
| **v1.1** | B1 (IR + interpreter) | Stable ~10-13% gain. No toolchain. Sets up everything that follows. |
| **v2.0** | AOT codegen for hot cells | +5-10% on those cells. Same IR, run codegen at build time. Library still has zero runtime deps. |
| **v2.x** | JIT-LLVM for arbitrary cells | +5-10% on cold cells the AOT set doesn't cover. Opt-in via build flag (LLVM dependency). |

**Why B1 first, even though JIT alone could work without B1's
interpreter:** the descriptor IR is the foundation. Building it forces
the right data discipline (uniform signatures, cf0-folded twiddles,
stable field layout). Once it exists, codegen is "compile this IR";
without it, codegen is "rewrite the executor from scratch every time
we add a new variant." The interpreter is a side benefit.

**Why AOT before JIT-LLVM:** AOT keeps the runtime dependency-free
property of the library. JIT-LLVM is for users who want zero-friction
calibration on novel hardware (recalibrate → JIT → cache to disk
→ done). Most users don't need that.

**Skip JIT-asmjit unless someone strongly wants it.** Hand-emitted x86
is a maintenance burden the wins don't justify when LLVM-ORC is
available.

---

## v1.1 implementation plan (Layer 1 / B1)

### Phase 0 — spike (1 day)

Goal: verify the 10-13% projected gain actually lands. Build the
minimum to bench, before committing to the full implementation.
Also: establish the descriptor format that Layer 2 will consume —
we want to get this right the first time since it'll be a stable
internal interface.

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

## What we're NOT doing in v1.1

- **Layer 2 (codegen).** B1 alone is the v1.1 deliverable. Codegen
  layered on top is v2.0+. The IR is built so the layering can land
  later without re-doing v1.1's work.
- **Full codelet fusion** (the v2.0+ "3× MKL ceiling" idea). Even
  Layer 2 codegen above B1's IR doesn't close the 3× gap by itself —
  closing it needs the codelets themselves to fuse stage transitions
  internally, which is a separate codelet-generator workstream.
- **Register-tiled data-flow fusion** — works only on AVX-512 hosts
  where the register file fits 17+ ymm/zmm of intermediate state.
  Not applicable to Raptor Lake AVX2.
- **Any of the wild brainstorm ideas** (bytecode VM, helper threads,
  GPU-assisted executor, self-modifying code) — interesting on paper,
  no clear engineering ROI for v1.x.
- **Touching the blocked executor.** That codepath has its own
  structure; layering descriptor IR onto it is a separate project.

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
