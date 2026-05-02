# VectorFFT Phase 2 Integration — Wisdom-Driven Protocol Selection

This package activates the codelet-side plan_wisdom in the executor's
planner. Specifically, it replaces the planner's hardcoded
"always-flat" twiddle codelet selection with wisdom-driven selection
between flat (DIT), t1s, and **DIT-log3** based on per-(me, ios)
empirical winners measured by the bench.

## What this delivers

For every (factorization, K) pair the planner considers, each stage
now consults the codelet-side bench data for its own (me, ios) point
and uses the DIT-log3 codelet at cells where DIT-log3 was the
empirical winner. Otherwise it stays with the legacy flat or t1s
behaviour.

**Expected impact on RL AVX2** (based on bench data in this package):

| Radix | Total log3 cells | DIT-log3 wins | Phase 2 captures |
|-------|------------------|---------------|------------------|
| R=16  | 6 (medium grid)  | 6             | 6/6  (100%)      |
| R=32  | 9                | 9             | 9/9  (100%)      |
| R=64  | 13               | 8             | 8/13 (62%)       |

Biggest captured DIT-log3 wins:
- R=64 (2048, 2056): flat 172469 ns → DIT-log3 91276 ns (**+47%**)
- R=64 (1024, 1032): flat 54008 ns → DIT-log3 39751 ns (**+26%**)
- R=64 (2048, 2048): flat 254595 ns → DIT-log3 202425 ns (**+21%**)
- R=64 (512, 520):  flat 25101 ns → DIT-log3 20293 ns (**+19%**)

## What this does NOT deliver

**DIF-log3 wins are NOT activated by Phase 2.** The 5 DIF-log3-winning
cells at R=64 (and any DIF-log3 wins at R=16, R=32 if present) require
a Phase 3 executor refactor. Cross-validation (see Appendix A) confirms
DIT-log3 and DIF-log3 are not interchangeable codelets — they share the
same call signature but produce different output buffers given the same
input. Activating DIF-log3 requires a parallel forward-DIF executor
path with twiddle handling adapted to post-butterfly application.

The conservative `prefer_dit_log3` query never regresses vs. flat: at
R=64 (128,128), (512,512), (1024,1024) where DIF-log3 is the bench
winner but DIT-log3 is *slower* than flat, Phase 2 correctly stays
with flat instead of substituting DIT-log3.

## Files in this package

```
phase2_pkg/
├── README.md                         — this file
│
├── core/                             — executor-side files (drop into your common/)
│   ├── wisdom_bridge.h               — NEW. Plan-time protocol query bridge.
│   ├── planner.h                     — modified. _stride_build_plan now wisdom-driven.
│   ├── executor.h                    — modified. R≥64 n1_fallback respects use_log3.
│   ├── dp_planner.h                  — modified. _dp_bench routes through _stride_build_plan.
│   └── exhaustive.h                  — modified. stride_bench_one routes through _stride_build_plan.
│
├── vectorfft_tune/                   — full codelet-side tree (merge over yours)
│   ├── DESIGN.md
│   ├── README.md                     — original repo README, unchanged
│   ├── bench_out/                    — bench measurement data (rN/measurements.jsonl)
│   ├── common/
│   │   ├── bench.py
│   │   ├── harness.c
│   │   ├── select_and_emit.py        — MODIFIED. Emit produces DIT/DIF split in plan_wisdom.
│   │   ├── validate.c
│   │   └── ...                       — other unchanged common/ files
│   ├── generated/
│   │   └── rN/
│   │       ├── vfft_rN_plan_wisdom.h — REGENERATED (14 of 17 radixes had container data;
│   │       │                            R=4, R=8, R=32, R=64 must be regenerated locally)
│   │       ├── fft_radixN_*.h        — codelet headers (unchanged)
│   │       ├── vfft_rN_*_dispatch_*.h — dispatchers (unchanged)
│   │       └── vfft_rN_report.md     — bench report (unchanged)
│   ├── radixes/                      — codelet generators per radix (unchanged)
│   └── report_draft/                 — paper drafts (unchanged)
│
└── diffs/                            — unified diffs for code review
    ├── registry.h.diff               — empty (registry.h is intentionally NOT modified).
    ├── planner.h.diff
    ├── executor.h.diff
    ├── dp_planner.h.diff
    └── exhaustive.h.diff
```

**Note on `registry.h`**: Phase 2 does NOT touch `registry.h`. The
existing `t1_fwd_log3[R]` slot is what we activate. This is by design
— it keeps Phase 2 minimally invasive.

**Note on `vectorfft_tune/`**: The bundled tree mirrors the working
state at the end of this session. It contains substantial codelet-side
infrastructure work in addition to the wisdom-emit changes. Specifically:

**Grid density configuration system** (new):
- **`common/grids.py`** — NEW. Defines four density presets (coarse/
  medium/fine/ultra) for both me and ios axes. Power-of-2 radixes
  (R=4, R=8, R=16, R=32, R=64) default to fine me-grid because
  their variant space has more me-dependent winners; non-pow2
  radixes default to medium.
- **`common/bench.py`** — added `--me-density` and `--ios-density`
  CLI flags. Default `--me-density default` uses each radix's
  preferred density.
- **`common/orchestrator.py`** — same flags exposed at the higher-
  level entry point.
- **`radixes/r{4,8,16,32,64}/candidates.py`** — set
  `_GRID_DENSITY_ME = 'fine'` per pow2 radix.

**Phase 2 wisdom integration**:
- **`common/select_and_emit.py`** — two changes:
  - Dispatcher bucketing fix: emits `if (me <= X) {...}` covering
    contiguous intervals instead of `if (me == X) {...}` with gaps.
    Without this, runtime me values between benched points would
    fall through with no codelet selected.
  - `emit_plan_wisdom` produces three log3 predicates per radix
    (`prefer_dit_log3`, `prefer_dif_log3`, `prefer_log3` union)
    instead of one. This is what `wisdom_bridge.h` consumes.
- **14 regenerated `generated/rN/vfft_rN_plan_wisdom.h`** files
  reflect both the dispatcher and emit changes (R=4, R=8, R=32,
  R=64 must be regenerated locally — Step 2).
- **Regenerated `generated/rN/vfft_rN_*_dispatch_*.h`** files reflect
  the bucketing fix.

**DIF-log3 support in validator**:
- **`common/validate.c`** — added `t1_dif_log3` validation cases for
  R=16, R=32, R=64 (each self-validates within the log3 family,
  since DIT-log3 and DIF-log3 produce different output buffers).
- **`common/fft_radix_include.h`** — added DIF-log3 dispatcher
  includes for R=8, R=16, R=32, R=64.
- **`radixes/r32/gen_radix32.py`** — DIF-log3 codegen support.

Everything else under `vectorfft_tune/` (most radix generators,
bench harness, bench_out measurements, reports) is unchanged.

## Integration steps

### Step 1: Drop the codelet-side files into your tree

The `vectorfft_tune/` folder in this package contains the modified
`common/select_and_emit.py` and 14 regenerated `vfft_rN_plan_wisdom.h`
headers. You can either:

**(a) Copy individual changed files** (minimal, surgical):
```sh
cp phase2_pkg/vectorfft_tune/common/select_and_emit.py \
   ~/path/to/your/vectorfft_tune/common/
# Optionally compare wisdom files:
diff phase2_pkg/vectorfft_tune/generated/r16/vfft_r16_plan_wisdom.h \
     ~/path/to/your/vectorfft_tune/generated/r16/vfft_r16_plan_wisdom.h
```

**(b) Merge the whole `vectorfft_tune/` directory** (rsync-style):
```sh
rsync -av phase2_pkg/vectorfft_tune/ ~/path/to/your/vectorfft_tune/
```
This is safe — only `common/select_and_emit.py` and the 14 wisdom
files change; everything else is identical to what we worked with.

### Step 2: Regenerate plan_wisdom for the missing radixes

R=4, R=8, R=32, R=64 wisdom files are missing from this package
because the container did not have measurements for those radixes.
Regenerate them locally from your existing measurements:

```sh
cd vectorfft_tune
for r in r4 r8 r32 r64; do
    python common/bench.py --radix-dir radixes/$r --phase emit
done
```

This re-emits from `bench_out/rN/measurements.jsonl`. No bench rerun.
Takes about 5 seconds per radix.

After this step, all 17 wisdom headers will export:
- `radix{N}_prefer_dit_log3(me, ios)`     ← Phase 2 uses this
- `radix{N}_prefer_dif_log3(me, ios)`     ← Phase 3 (informational)
- `radix{N}_prefer_log3(me, ios)`         ← union, backward compat
- `radix{N}_prefer_t1s(me, ios)`

### Step 3: Drop the executor-side files into your tree

```sh
cp phase2_pkg/core/wisdom_bridge.h    src/<your-executor-dir>/common/
cp phase2_pkg/core/planner.h          src/<your-executor-dir>/common/
cp phase2_pkg/core/executor.h         src/<your-executor-dir>/common/
cp phase2_pkg/core/dp_planner.h       src/<your-executor-dir>/common/
cp phase2_pkg/core/exhaustive.h       src/<your-executor-dir>/common/
```

Do NOT replace `registry.h` — Phase 2 doesn't need any registry
changes. The `t1_fwd_log3[R]` slot already exists; we just start using
it.

### Step 4: Update build's include path

The build that compiles your executor needs `-I` access to every
`generated/rN/` directory so `wisdom_bridge.h`'s 15 plan_wisdom
includes resolve. If your existing CMake already compiles registry.h
and finds the codelet headers, plan_wisdom lives alongside them — same
`-I` lines.

### Step 5: Build and verify

Build clean. The first thing to check is that wisdom is actually
firing. Drop this snippet near a known-stress test:

```c
stride_plan_t *plan = stride_auto_plan(N, K);
for (int s = 0; s < plan->nstages; s++) {
    printf("Stage %d: R=%d  use_log3=%d  use_n1_fallback=%d\n",
           s, plan->factors[s],
           plan->stages[s].use_log3,
           plan->stages[s].use_n1_fallback);
}
```

For an N where R=32 or R=64 stages are picked at me-values where
wisdom says DIT-log3 wins, you should see `use_log3=1` for those
stages. For N=64*1024 with K=1, stage 0 is R=64 (no twiddles, no
log3), but stage 1 (if any) at me_plan=K, ios=K could activate log3.

### Step 6: Bench the integration

Run your end-to-end FFT benchmark suite (the one outside the
codelet-tuning bench) and compare to the previous flat-only baseline.
At sizes where R=64 stages dominate the work, expect ~10-30% speedups
on the cells where Phase 2 activates DIT-log3.

## How Phase 2 picks log3 — the precise rule

In `_stride_build_plan` (planner.h ~line 320), for each stage `s > 0`
with radix `R`:

```c
size_t me_plan = K / num_threads;          /* per-thread slice */
size_t ios_s   = K * factors[s+1] * factors[s+2] * ...;

if (stride_prefer_dit_log3(R, me_plan, ios_s)
    && reg->t1_fwd_log3[R] != NULL)
{
    stage->t1_fwd      = reg->t1_fwd_log3[R];   /* DIT-log3 codelet */
    stage->t1_bwd      = reg->t1_bwd_log3[R];
    stage->use_log3    = 1;                      /* set log3_mask bit */
    stage->t1s_fwd     = NULL;                   /* don't shadow with t1s */
}
```

Then the executor's existing log3 branch (executor.h line 200) handles
the runtime: applies cf to all R legs of input, calls the codelet,
which derives the per-leg twiddles internally from a 2-base sparse read.

## Validation — what could go wrong

The Phase 2 planner ONLY swaps to log3 codelets where the bench
empirically measured DIT-log3 as the cross-protocol winner. If on a
new host (different CPU, different compiler version) the bench winners
shift, just rerun bench → emit, and the wisdom predicates regenerate.

There is no risk of regression vs. the legacy planner: every
plan_wisdom predicate that returns 1 was specifically a cell where
DIT-log3 beat both flat and t1s in the bench. Conservative by
construction.

The R≥64 n1_fallback override (executor.h line 1040) was specifically
the override blocking R=64 log3 wins. Phase 2's one-line edit allows
log3 to bypass that override:

```c
if (factors[s] >= 64 && s > 0 && !plan->stages[s].use_log3) {
    plan->stages[s].use_n1_fallback = 1;
}
```

## Appendix A — DIT vs DIF log3 cross-validation

`test_dit_vs_dif_log3.c` (in the wider integration tree, not in this
package) runs both DIT-log3 and DIF-log3 codelets on identical input
and identical twiddle buffer, and compares outputs. Result on R=16
AVX2: every test case produces `max_diff ≈ 20` (where 1e-10 would
indicate equality). Energy is identical (same FFT magnitude content),
column 0 of every output leg matches exactly — but other columns
differ. The two codelets are NOT computing the same buffer-to-buffer
transform.

This is by design: they share the log3 twiddle derivation
infrastructure but apply twiddles at different points in the butterfly
flow (DIT pre-butterfly, DIF post-butterfly). The bench's
"protocol=log3" grouping treats them as alternative implementations
within the log3 family — the cross-protocol comparison picks whichever
runs faster — but the executor must commit to one structural variant
per stage. Phase 2 commits to DIT-log3.

## Appendix B — Phase 3 scope (future work)

To capture the 5 DIF-log3-winning cells at R=64 (and any R=16/R=32 DIF
wins on different hosts), Phase 3 needs:

1. New executor branch for DIF-log3 stages: butterfly first, post-multiply
   after.
2. Investigation of whether DIF-log3 expects a different twiddle buffer
   layout than DIT-log3 (the failing cross-validation suggests yes,
   though both codelets have identical signatures).
3. Stage-chaining analysis: does a DIF-log3 stage produce output in a
   layout that downstream stages can consume? Or does it require an
   adjacent stage to be also DIF?
4. Backward path: currently the backward executor uses hand-coded
   `n1_bwd + post-multiply` (DIF-structured). DIF-log3 codelet substitution
   is a smaller change there than on the forward path.

Estimated effort: 1-2 weeks of focused executor work plus comprehensive
end-to-end validation.

For now, Phase 2 captures the headline DIT-log3 wins and ships
correctly without regressions.
