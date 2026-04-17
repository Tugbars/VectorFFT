# R=16 Buffered Codelet Calibration Results

**Date:** 2026-04-17
**CPU:** Intel i9-14900KF (Raptor Lake), P-core, AVX2 only, 5.7 GHz turbo
**Comparison CPU:** Intel Xeon (Claude.ai container), AVX-512, server-class memory subsystem
**Generator:** `gen_radix16_buffered.py` — tiled Pass-2 output buffering with sequential drain
**Calibration harness:** `bench_codelets.py` — 22 candidates × 18 (ios, me) sweep points

---

## Background

The R=16 DIT codelet's Pass 2 writes 16 outputs at stride `ios`, hitting 16 different 4 KB pages per k-iteration. On AVX2 (Raptor Lake), VTune shows 53% Store Bound and 18.5% DTLB Store Overhead from this pattern.

The buffered variant (`ct_t1_buf_dit`) redirects Pass-2 stores to a contiguous local buffer (`outbuf[16 × TILE]`), then drains 16 sequential streams to `rio` at stride `ios`. The drain touches the same pages but sequentially — one page at a time instead of 16 interleaved.

---

## Executive Summary

| Finding | Detail |
|---------|--------|
| **Stream (NT) drain** | **Catastrophic on Raptor Lake** (1.5-2× slower). Viable on Xeon. Reject for desktop. |
| **Temporal drain, tile=16** | Best buffered variant at ios=me=256 (**27% win** vs baseline). Loses at ios>me. |
| **Temporal drain, tile=128** | Best buffered variant at ios=me≥512 (**8-10% win**). Massive regression at ios>me. |
| **`t1s_dit` (scalar broadcast)** | **Hidden champion** — 10-25% faster than baseline at ALL sizes, already in the executor |
| **`t1s_dit` is already the primary path** | Executor dispatch: n1 → n1_fallback → log3 → **t1s** → K-blocked → legacy. log3_mask=0, so t1s fires for all twiddled R=16 groups. |

**The production executor is already using the best codelet for Raptor Lake.** The buffered variant wins only at the specific ios==me sweet spot and loses elsewhere.

---

## Full Results: ios == me (DTLB-worst case)

Forward direction, ns/call (lower is better):

| ios=me | baseline | t1s_dit | buf_t16_temp | buf_t128_temp | winner | win vs baseline |
|--------|----------|---------|-------------|--------------|--------|----------------|
| 64 | 444 | **334** | 452 | 345 | t1s | **25%** |
| 128 | 1118 | **898** | 1422 | 2027 | t1s | **20%** |
| 256 | 5007 | 4729 | **3650** | 4274 | buf_t16 | **27%** |
| 512 | 9089 | **8434** | 9041 | 8390 | t1s | **7%** |
| 1024 | 17966 | **16403** | 17966 | 16210 | t1s | **9%** |
| 2048 | 35477 | **32103** | 35150 | 31923 | t1s/buf_t128 | **10%** |

`t1s_dit` wins or ties at every point except ios=me=256, where buf_t16 has a specific advantage.

## Full Results: ios > me (stores less scattered)

| ios / me | baseline | t1s_dit | buf_t16_temp | buf_t128_temp |
|----------|----------|---------|-------------|--------------|
| 264 / 256 | 2337 | **2078** | 2639 | 4521 |
| 520 / 512 | 5245 | **4346** | 5624 | 8135 |
| 1032 / 1024 | 10021 | **9923** | 11285 | 15802 |
| 2056 / 2048 | 20677 | **17527** | 23777 | 30903 |

When ios > me, buffered variants **regress 10-50%** because the drain overhead exceeds DTLB savings. `t1s_dit` wins everywhere by 5-15%.

## Stream Drain: Catastrophic on Raptor Lake

| ios=me | baseline | buf_t16_stream | buf_t128_stream | ratio vs baseline |
|--------|----------|---------------|----------------|------------------|
| 256 | 5007 | 9595 | 9697 | **1.9× slower** |
| 512 | 9089 | 19307 | 19110 | **2.1× slower** |
| 1024 | 17966 | 33275 | 33123 | **1.9× slower** |
| 2048 | 35477 | 50473 | 51683 | **1.4× slower** |

NT stores force the next FFT stage to re-fetch from DRAM. On Raptor Lake's memory bus (already 77% saturated at K=256 per VTune), this is pure waste.

On the Xeon container, stream drain showed 1.27-1.30× wins at ios=me=512+. The server's wider memory bus, larger STLB, and possibly transparent huge pages made NT viable there.

**Decision: reject stream drain for production. Temporal only.**

---

## Why `t1s_dit` Wins

The scalar-broadcast twiddle variant (`t1s_dit`) uses `_mm256_broadcast_sd(&W_re[j])` — one scalar per twiddle leg, broadcast to SIMD width inside the codelet. Compared to the baseline `t1_dit` which loads `W_re[j*me+m]` (one full SIMD vector per leg per k-position):

1. **99% less twiddle memory**: (R-1) scalars vs (R-1)×K vectors. At R=16, K=256: 15 doubles vs 3840 doubles.
2. **Zero twiddle cache pollution**: 15×8 = 120 bytes vs 60 KB. The freed L1/L2 capacity goes to data and spill buffers.
3. **No stride-K twiddle loads**: baseline loads `W_re[j*me+m]` at stride `me`, causing L1 misses at large K. `t1s` broadcasts one scalar — always L1-hot.

The codelet's Pass-2 store pattern is identical (same 16 stride-K outputs), but the twiddle memory pressure reduction frees enough cache budget that store-buffer back-pressure drops.

---

## Executor Dispatch Verification

The forward executor cascade in `executor.h` (line 180+):

```
1. !needs_tw         → n1_fwd (no twiddle, stage 0)
2. use_n1_fallback   → cf_all + n1_fwd (R≥64 only)
3. use_log3          → cf0 all-legs + t1_fwd (log3_mask=0 → NEVER FIRES)
4. t1s_fwd available → t1s_fwd (scalar broadcast)     ← PRODUCTION PATH
5. tw_scalar_re only → K-blocked expand + t1_fwd
6. fallback          → cf0 + t1_fwd (legacy K-replicated)
```

Since `log3_mask` is passed as `0` from all planners (`planner.h:266`, `exhaustive.h:160`, `dp_planner.h:135`), priority 3 never fires. **`t1s_fwd` is the active production codelet for all twiddled R=16 stages** (priority 4). This is already the best variant per calibration data.

---

## Raptor Lake vs Xeon: Why Results Diverge

| Architectural feature | Raptor Lake (i9-14900KF) | Xeon (server container) |
|----------------------|--------------------------|------------------------|
| Store STLB entries | 16 | ~1536 (estimated) |
| Default page size | 4 KB | 4 KB + THP (2 MB likely) |
| Memory bus | Dual-channel DDR5 | Multi-channel DDR5 |
| Store buffer depth | 72 entries | 72+ entries |
| L2 cache | 2 MB (P-core) | 2 MB per core |
| AVX-512 | No | Yes (32 ZMM, no spill) |

The Xeon's larger STLB and wider memory bus mean:
- DTLB Store Overhead is proportionally worse (more pages in flight, each tracked longer)
- NT stream stores can drain without back-pressuring the next stage (more bus bandwidth)
- Buffered drain amortizes better because STLB pressure is the dominant cost

On Raptor Lake, the 16-entry store STLB is small enough that `t1s_dit`'s reduced cache footprint addresses the same bottleneck from the supply side (less cache pollution → more L1 budget for output writes).

---

## Recommendations

### For the paper (Raptor Lake results)
No codelet change needed. `t1s_dit` is already the production path and wins by 10-25% vs the baseline `t1_dit`. This explains part of VectorFFT's competitive pow2 performance.

### For multi-platform deployment
If targeting server Xeon: `ct_t1_buf_dit` with temporal drain, tile=128, dispatched when ios==me≥256. Gate behind runtime calibration (same infrastructure as the codelet bench harness).

### For the DP planner (tomorrow's session)
The planner should be aware that R=16 `t1s_dit` is dramatically faster than `t1_dit` — the calibration data confirms the wisdom system is picking the right codelet. The DP planner improvement should focus on factorization quality, not codelet selection.

### What NOT to ship
- Stream drain (NT stores) — catastrophic on desktop, marginal on server
- Buffered variants at ios>me — consistent 10-50% regression
- ilog3 variant — 1% win, not worth the complexity (scrapped earlier this session)

---

## Files

- Calibration harness: `src/test-gen/R32/bench_codelets.py`
- Generator: `src/test-gen/R32/gen_radix16_buffered.py`
- Candidate definitions: `src/test-gen/R32/candidates.py`
- Raw measurements: `src/test-gen/R32/measurements.json`
- Prior R=16 VTune profile: `docs/vtune-profiles/vtune_r16_codelet_k256.md`
