# The execute door: bind every plan's executor at create — roadmap item

**Date:** 2026-09-21 · **Status:** OPEN (owner: "add to roadmap") · the first piece
is SHIPPED (the bound K=1 interleaved fast path in `src/core/vfft_execute.h`).

## 1. What the door costs, measured

`benches/k1_fwd_ref_probe.exe --time [--k K]` times `vfft_execute` on replayed plans
(minimum of three processes) beside the planner's own time of the same winning
plan, which is the engine without the door:

```
                     engine   door before   door after (2026-09-21)
 N = 2                ~1          5.6            3.5
 N = 32 (pair 4.8)    14.6       18.6           16.3       MKL 16 (its dispatch included)
 N = 64 (pair 4.16)   30.0       33.4           30.9       MKL 31
 N = 128 (pair 4.32)  61.4       68.1           65.2       MKL 69
 N = 256 (pair 16.16) 134.6      141.4          135.4      MKL 137
```

Before: a NULL and direction check, `_vfft_sig_bad` (a transform-name lookup, then
the real, C2R and C2C branches), then two indirect calls: 4-5 ns per call, 25% of
the cell at 32. After: the bound K=1 fast path checks the four things that can be
wrong for such a plan inline and jumps to the bound engine; 2-3 ns per call
returned; 32 moved from 0.82x to 0.95x of MKL in the gauntlet, 64 from 0.91x to
0.98x. Gated by the forward reference (both placements), il_solo_gate,
ilprime_inner_gate and k1_pow2_gate.

Inside a transform-contiguous batch (the default K>1 interleaved geometry) the
loop re-enters the public door once per transform, now through the fast path:

```
 N = 32   K = 1 / 8 / 16      door per transform 16.3 / 17.1 / 18.2   engine 14.6
 N = 64   K = 1 / 8           31.1 / 32.8                              engine 30.0
 N = 128  K = 1 / 4           65.1 / 69.1                              engine 61.4
```

About one nanosecond per transform is the loop's re-entry; the rest of the growth
with K is the working set. Batches that leave L1 are memory-bound (N=256 at K=16:
174 ns per transform against 135 single; N=512 at K=16: 553 against 298) and no
door change touches that regime.

## 2. What remains

1. **The batch loop binds to the inner engine.** `vfft_execute` for a plan with
   `tcb` calls `vfft_execute(h->tcb, ...)` per transform, serial and in the threaded
   workers (`_tc_mt_arg`). The inner plan's contract is fixed at create and the batch
   pointers are checked once at the outer call, so the loop can call the inner
   plan's bound `k1_exec` directly. Saves at most 1-1.5 ns per transform: ~6% at
   N=32, 3% at 64, under 2% at 128, nothing above. Small, measurable with `--k`.
2. **Every plan binds its executor at create** (the execution-purity law: bind at
   plan time). The odd-real bridge and the real routes call their c2c child through
   the public door; split-layout K=1 and the trig plans walk the general branch
   chain. Each tier binds an `exec` with its own minimal contract check, and the
   public door becomes one cheap check plus one indirect call; today's dispatch
   becomes the binder. About a day across eight tiers; 2-3 ns per call each, visible
   only at sub-microsecond transforms.
3. **Not the stage calls inside an engine.** The pair's two kernel calls, chain3's
   three and the flat DIT's per-stage records are plain function-pointer calls, about
   a nanosecond each, with nothing to gate; that cost is the call boundary and only
   fusion removes it (the fused-codelet form ZTURN-T uses at pow2,
   `docs/research/sub2048_mkl_method/README.md` lever 8).

## 3. The measurement that decides item 1's priority

The K>1 interleaved tier against MKL's batched descriptor at the same K
(`bench_1d_vs_mkl.c` kzb mode) at small N: if the batch case is memory-bound at
the product's K, one nanosecond per transform is invisible and item 1 waits for
item 2.
