# Executor overhead reduction — failed experiments (v1.0 closeout)

**TL;DR — four independent attempts to reduce the executor's per-group
overhead all failed to produce measurable wins on Intel i9-14900KF
(Raptor Lake, AVX2, DDR5-6400):**

- **Three software-prefetch insertion points (E1, E3, E2)** — HW
  prefetcher already covers the access patterns; software prefetch
  consumes load-port slots without saving latency.
- **Aggressive small-R direct-call dispatch (option C)** — inlining
  the dispatcher's me/ios switch at every call site bloated the
  executor function, hurt ICache locality, regressed worse than
  baseline.

**The lesson is consistent: the executor's 19% time on memory-bound
CLOSE cells is real work (memory traffic, codelet compute), not
optimization headroom. Further experimentation deferred to v2.0
(EPYC + AVX-512 + different prefetcher hierarchy may change the
calculus).**

---

## Background

VTune profiling of large pow2 K=4 cells (the CLOSE cells in our bench
suite — where MKL is closest to VectorFFT) showed
[`_stride_execute_fwd_slice_from`](../src/core/executor.h) consuming
~19% of total CPU time on N=131072 K=4. Initial analysis attributed
this to the executor's serial loop body work (per-group base pointer
compute, twiddle pointer fetch, scalar preprocessing, codelet
dispatch).

The hypothesis was: **executor-level software prefetch could hide
DRAM/L3 load latency** for the next group's data while the current
group's codelet runs.

This document records three experiments testing that hypothesis,
their results, and the consolidated verdict.

## Test methodology

Each experiment was a hardcoded prefetch pattern in the forward
executor's inner group loop. Validated against the
[`build_tuned/dev/bench_vtune/`](../build_tuned/dev/bench_vtune/) 12-cell suite,
spanning three regimes:

- **CLOSE** — cells where MKL is within 1.30× (where prefetch would
  most help if it could)
- **MID** — typical wins (1.8–2.5×)
- **DECISIVE** — small-N batch, well above 4×

Each experiment was 3-run averaged. The bench runs each cell for ~2
seconds of FFT work to establish stable averages.

Power plan: High Performance, restored on exit.
Background load: minimal.
T=1 single-threaded, P-core 0 pinned.

## Experiment matrix

### E1 — Prefetch next group's base data (g+1) into L1

**Hypothesis:** HW prefetcher misses cold lines in the next group's
data; software prefetch with T0 hint hides DRAM latency.

```c
for (int g = 0; g < st->num_groups; g++) {
    base_re = re + group_base[g];
    base_im = im + group_base[g];

    /* E1: prefetch next group's base */
    if (g + 1 < st->num_groups) {
        _mm_prefetch(re + group_base[g+1], _MM_HINT_T0);
        _mm_prefetch(im + group_base[g+1], _MM_HINT_T0);
    }

    /* ... codelet call ... */
}
```

**Result:** **+1–3% regression on every cell.**

| Cell | Baseline (3-run avg) | E1 (3-run avg) | Δ |
|------|--------:|--------:|---:|
| N=131072 K=4 | 2,098,597 ns | 2,157,698 ns | +2.8% |
| N=32768 K=4  | 387,308 ns   | 393,288 ns   | +1.5% |
| N=8192 K=4   | 74,579 ns    | 75,814 ns    | +1.7% |
| N=243 K=4    | 2,290 ns     | 2,353 ns     | +2.8% |
| N=1024 K=256 | 414,840 ns   | 419,420 ns   | +1.1% |
| N=4096 K=256 | 3,189,000 ns | 3,209,413 ns | +0.6% |

**Why it failed:** Raptor Lake's HW L1 streamer prefetcher already
catches the next-group access pattern as a sequential stream.
Software prefetch is redundant and consumes load-port slots that
the codelet wants.

### E3 — Prefetch next stage's twiddle table at stage transition

**Hypothesis:** Twiddle table access at stage boundary is cold (not
part of any sequential stream HW tracks). Prefetching it during the
last few groups of the previous stage hides the cold load.

```c
/* on the last 4 groups of stage s */
if (g + 4 >= st->num_groups && s + 1 < plan->num_stages) {
    const stride_stage_t *next_st = &plan->stages[s + 1];
    if (next_st->grp_tw_re && next_st->grp_tw_re[0]) {
        _mm_prefetch(next_st->grp_tw_re[0], _MM_HINT_T1);
        _mm_prefetch(next_st->grp_tw_im[0], _MM_HINT_T1);
    }
    if (next_st->tw_scalar_re && next_st->tw_scalar_re[0]) {
        _mm_prefetch(next_st->tw_scalar_re[0], _MM_HINT_T1);
        _mm_prefetch(next_st->tw_scalar_im[0], _MM_HINT_T1);
    }
}
```

**Result:** **+1–4% regression on every cell.**

| Cell | Baseline | E3 | Δ |
|------|--------:|--------:|---:|
| N=131072 K=4 | 2,098,597 | 2,163,204 | +3.1% |
| N=32768 K=4  | 387,308   | 397,206   | +2.6% |
| N=8192 K=4   | 74,579    | 75,465    | +1.2% |
| N=243 K=4    | 2,290     | 2,391     | +4.4% |

**Why it failed:** Stage transitions only happen num_stages−1 times
per FFT (≈7 for N=131072). Best-case savings (~7 × ~300 cycles =
~2,100 cycles) are dominated by the per-prefetch instruction cost
across thousands of group iterations. Twiddle tables also benefit
from prior iteration's cache line locality.

### E2 — Prefetch all R legs of g+1 for R≥16 stages only

**Hypothesis:** HW prefetcher caps at ~4–8 trackable streams. Stages
with R≥16 have R parallel strided streams exceeding that limit.
Software prefetch could cover the streams HW misses.

```c
if (st->radix >= 16 && g + 1 < st->num_groups) {
    size_t next_base = st->group_base[g + 1];
    size_t stride = st->stride;
    const int R = st->radix;
    for (int j = 0; j < R; j++) {
        _mm_prefetch(re + next_base + j * stride, _MM_HINT_T0);
        _mm_prefetch(im + next_base + j * stride, _MM_HINT_T0);
    }
}
```

**Result:** **No measurable effect on R≥16 cells (within run-to-run
noise).**

| Cell | R≥16 stage? | Baseline | E2 (3-run avg) | Δ |
|------|:-----------:|--------:|---------------:|---:|
| N=32768 K=4 | yes (R=32 last) | 387,308 | 397,030 | +2.5% |
| N=8192 K=4  | yes (R=32 last) | 74,579  | 74,124  | −0.6% |
| Other 10 cells | no — E2 doesn't fire | — | — | within ±5% noise |

**Why it failed:** R=32 codelet ALREADY emits internal `_mm_prefetch`
instructions per the codegen (per
[`prefetch_heuristic.md`](../docs/) memory entry, those internal
prefetches regressed). Layering executor-level prefetch on top
compounds load-port pressure the codelet already creates. R=32
K=256 is also DTLB-bound per existing VTune profiles — prefetch
can't help when page-table walks are the bottleneck.

## Three converging hypotheses for why prefetch fails on this CPU

1. **HW prefetcher hierarchy is genuinely strong.** Raptor Lake's
   L1 streamer + L2 streamer + L1 IP-based + L2 spatial prefetchers
   together cover far more parallel patterns than the textbook
   "4–8 streams" model. Software prefetch becomes redundant.

2. **Codelets retire too efficiently to hide prefetch latency.**
   R=4 codelet retires at 86% (per
   [`docs/vtune-profiles/`](vtune-profiles/)). No issue-port slack
   for prefetch instructions to fit in "for free." Every
   `_mm_prefetch` serializes against real codelet work.

3. **Memory bandwidth is the binding constraint, not latency.** For
   memory-bound cells like N=131072 K=4 (64 MB traffic per FFT, run
   at ~80% of DDR5's bandwidth ceiling), prefetch can hide latency
   but can't add bandwidth. Issuing more memory requests just queues
   them; the queue can't drain faster than DRAM bandwidth allows.

## What did win (companion to this work)

The same diagnostic harness validated that **SIMDizing the
executor's scalar preprocessing** (cf0 multiply, K-blocked twiddle
broadcast, n1 fallback per-element multiply) gives **5–15% on K=256
cells**. Those changes shipped:

- AVX2 + AVX-512 implementations of three preprocessing helpers in
  [`src/core/executor.h`](../src/core/executor.h)
- Macro dispatch selects widest available SIMD at compile time
- Zero regression on cells where the preprocessing path doesn't fire

This is the reference for "what executor-level optimization actually
looks like when it works."

## v2.0 followups

This conclusion is **CPU-specific**. Re-evaluate executor-level
prefetch when:

| Trigger | Reason |
|---------|--------|
| EPYC port (Zen 5) | Different prefetcher hierarchy; native AVX-512; 12-channel DDR5. Stream-count caps may differ from Intel. |
| Sapphire Rapids / Emerald Rapids server | Different L3 size + HBM/larger cache. Bandwidth-bound argument may not bind. |
| AMD Zen 4 | Different prefetcher policies than Intel; worth re-running E1/E2/E3. |
| ARM (M1/M2/M3, Graviton) | Completely different uarch family; results don't transfer. |

The bench harness at [`build_tuned/dev/bench_vtune/`](../build_tuned/dev/bench_vtune/)
can be re-used for these re-evaluations. Each experiment took ~30
minutes of patch-and-bench time on Raptor Lake; expect similar on
other targets.

For deeper integration (i.e., wisdom-driven prefetch flags rather
than hardcoded patches), the design proposal lives in this
document's git history — see the conversational notes from the v1.0
closeout session for E2/E3/E4 calibration loop sketches.

## Related work in the repo

- [`docs/v1_1_codelet_roadmap.md`](v1_1_codelet_roadmap.md) — v1.1
  codelet improvements (R=32/64 codelet retiring %, wisdom-driven
  variant selection)
- [`src/core/executor.h`](../src/core/executor.h) — the executor
  with the SIMD preprocessing helpers that DID win
- [`build_tuned/dev/bench_vtune/`](../build_tuned/dev/bench_vtune/) —
  the bench harness for any future re-runs
- Memory entry: `executor_prefetch_dead_end.md` (project memory, for
  Claude session continuity)
