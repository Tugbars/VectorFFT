# Padding vs the SSE2 tail — and the planner pivot (2026-06-29)

> **Conclusion:** for odd K, **padding (round K → Kp = roundup(K,VW), run full-SIMD,
> ignore the pad columns) beats the SSE2/scalar tail in most rem=3 cells — but only
> when Kp is already in memory.** It is *not* a transparent replacement for the tail
> (a copy to pad erases the win). So the right design is **not** "tail vs pad" as a
> global decision — it's a **per-cell choice the PLANNER measures**, exactly like it
> already measures factorization and twiddle variant. Padding = *planning for Kp
> instead of K*, which the planner already knows how to do.

## The question

The arbitrary-K tail (per [[arbitrary_k_tail_strategy]]) finishes the `K mod VW`
leftover batch lanes with an SSE2 width-2 loop + a scalar straggler. The alternative
a caller could use instead: allocate `Kp = roundup(K,VW)` batch columns, zero the
pad, run pure full-width SIMD, ignore the pad outputs. Which is faster?

Key framing that made this cheap to measure: **padding needs no code change** — the
pad cost for K is literally our own full-width time at Kp. So we compare OUR `T(K)`
(tail) vs OUR `T(Kp)` (pad).

## Benchmark + methodology

`build_tuned/benches/bench_pad_vs_tail.c` (in-place c2c, no MKL — pure internal
compare):
- **Tight interleaving in one process**: alternate a K-burst and a Kp-burst every
  round, order-flipped. The *ratio of summed/per-round times cancels thermal drift*
  — cross-process absolute comparison is the noisy thing the measurement-variance
  lesson warns against.
- **Per-round ratio → MEDIAN** (not sum). First cut summed the bursts and got 2×
  swings on identical plans (thermal/denormal outliers inflate a sum). Median of
  per-round `pad/tail` ratios rejects outlier rounds while keeping the
  interleave's thermal fairness. (This mirrors the project's best-of-min rule.)
- **SAME reps for K and Kp within a cell** → the ratio is genuinely per-call.
- **Calibrated plans**: factorization from `spike_wisdom.txt` (nearest-K), built via
  `plan_create_ex` (factors + variants + use_dif). rem=3 only, N=256…4096, K=7…31.

## Results (median pad/tail ratio; <1 ⇒ PAD wins)

| K | N=256 | N=512 | N=1024 | N=2048 | N=4096 | verdict |
|---|---|---|---|---|---|---|
| **7**  | 0.665 | 0.986 | 0.780 | 0.756 | 0.722 | **PAD 5/5** (−1…−34%) |
| **11** | 0.855 | 0.910 | 0.661 | 0.674 | 0.752 | **PAD 5/5** (−9…−34%) |
| 15 | 1.030 | 0.966 | 0.955 | 0.819 | 1.057 | PAD 3/5 |
| 19 | 0.955 | 1.092 | 0.778 | 1.207 | 0.762 | mixed |
| 23 | 1.117 | 1.159 | 0.916 | 0.982 | 0.747 | mixed |
| 27 | 1.018 | 0.762 | 0.804 | 1.246 | 1.072 | mixed |
| 31 | 0.985 | 0.858 | 1.257 | 1.272 | 0.895 | mixed |

- **Padding wins 24/35 rem=3 cells (~69%)**, and wins *bigger* (0.66–0.92) than the
  tail wins (1.05–1.27, mostly ~1.08). Even where tail wins, it barely wins.
- **Small K (7, 11): padding wins everywhere, decisively (−15…−34%).** Mechanics:
  small K ⇒ the narrow tail is a *huge* fraction of the work (K=7 = one full group +
  a 3-lane tail = ~43% narrow), and padding only wastes one transform (1/K).
- **Larger K (15–31): mixed, leans padding, noisier** — the nearest-K factorization
  boundaries inject discontinuities (e.g. N=2048 flips `4.64.8`→`8.4.8.8` at K=19).
- **rem=1 (from the earlier rem-spread run): tail wins** — padding to Kp=K+3 wastes
  3/K (K=9→12 = 33%, K=5→8 = 60%); the single cheap scalar lane wins there.
- **Crossover rule:** padding wins when the wasted-compute fraction `(Kp−K)/K` is
  small *and* the tail is a big fraction — i.e. **small K, large rem (rem=3)**.

## The hard constraint — padding needs Kp IN MEMORY

To get full-width speed you need the Kp-th column to physically exist. Only three
ways, two are dead:

1. **Pre-padded caller buffer** (caller allocates `N·Kp`, zeros the pad) — ✅ this is
   what the table measures; it wins.
2. **Copy K→Kp scratch and back** — ~4 memory passes vs the FFT's ~2·nf passes; on a
   memory-bound kernel that's **+50…100%** → obliterates a 5–34% padding win.
3. **In-register padding** (masked-load rem lanes, zero lane VW, full butterfly,
   masked-store rem) — that is **exactly the masked tail we removed** because
   `vmaskmovpd` is slow on Raptor Lake; strictly worse than SSE2.

⇒ **Padding's win is real but only realizable with a pre-padded buffer.** There is no
free transparent version.

## The pivot — the PLANNER decides tail strategy (measured, per cell)

tail-vs-pad is plan/N/K-dependent and noisy — precisely what the planner already
resolves by *measuring* (factorization via `vfft_proto_dp_plan`, twiddle variant via
the separate MEASURE pass). So make it another measured axis. The clean reframe:

> **Padding = planning for Kp instead of K.** The DP planner already plans any K
> (`_vfft_proto_dp_bench` builds at K_eff and **executes at the real K_eff — so it is
> already tail-aware at odd K**). So "tail vs pad" is just: DP-plan at **K** (tail)
> *and* at **Kp** (pad, an ordinary even no-tail cell), compare the two winners'
> per-call costs, pick cheaper. The pad plan gets its **own** optimal
> factorization+variants at Kp (jointly optimal — not forced to reuse K's plan).

Wiring (small):
1. Calibration driver: for each cell, also DP-plan at Kp; record `pad` + `exec_K` in
   the wisdom entry (the format already carries `use_blocked`/`use_dif` flags — add
   one). It's a **per-PLAN** flag, not per-stage (the batch K is global across stages).
2. Front door: lookup `(N,K)` → if `pad`, run the stored Kp factorization at `me=Kp`.
3. The planner's best-of-min timing is **more rigorous** than the standalone harness
   (per-trial buffer reset + adaptive iter + thermal pacing) — the noisy verdicts
   above firm up.

References: planner build = [planner.h:151](../../../src/core/engine/planner.h#L151)
(`plan_create_ex`, stores factors/variants/use_dif); search = [dp_planner.h:732](../../../src/core/planning/dp_planner.h#L732)
(`vfft_proto_dp_plan`) + [dp_planner.h:408](../../../src/core/planning/dp_planner.h#L408)
(`_vfft_proto_dp_bench`).

## The admissibility gate (do NOT skip)

The planner can *decide* pad, but *executing* pad needs Kp memory. So **pad is only a
legal candidate when Kp is free**:
- caller passes a VW-padded batch (a `VFFT_BATCH_PADDED` contract), **or**
- there's already a repack/copy in the pipeline it can ride (the split-layout
  repack, OOP's allocated output) — pad the destination during a copy that happens
  anyway, zero extra passes.

If neither holds (tight in-place buffer), pad is **inadmissible** → keep the tail. So
the final shape: **the planner benches tail-vs-pad per cell, but only admits pad when
the buffer contract makes Kp free.** It then picks pad exactly for the rem=3-small-K
cells where it both *wins* and is *available*, and falls back to the SSE2 tail
everywhere else (rem=1, tight in-place buffers). The tail stays as the universal
default; padding is a measured fast-path, not a replacement.

## Status

Design pivot recorded; **not implemented**. Open decisions before wiring:
- which buffer mode(s) to expose (padded-contract vs ride-an-existing-repack);
- the wisdom field + the two-K bench in the calibrator + the front-door gate.

## See also
- [[arbitrary_k_tail_strategy]] — scrambled→SSE2 tail / natural→transpose; this doc
  is the "how to choose tail vs pad" companion.
- [[arbitrary_k_vectorization]] — the productionized SSE2 tail (§ 2026-06-29).
- [[arbitrary_k_scalartail_experiment]] — the SSE2-vs-masked bake-off + robust-bench
  methodology that this reuses.
- [[memory_bound_thesis]] — why the copy-to-pad path loses.
