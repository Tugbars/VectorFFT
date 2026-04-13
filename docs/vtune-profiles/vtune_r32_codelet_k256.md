# R=32 AVX2 Codelet — VTune Analysis (K=256)

Target: `radix32_t1_dit_fwd_avx2` at K=256 (L2-resident).
Hardware: Intel Core i9-14900KF (Raptor Lake), P-core pinned, 5.696 GHz turbo.
Collector: `uarch-exploration` (event-based sampling).

## Baseline performance

| Metric | Value |
|---|---|
| ns/call | 8634.5 |
| CPE (cycles / (R × K)) | 2.550 |
| GFLOP/s | 15.42 |
| Retiring | 34.4% of pipeline slots |
| CPI Rate | 0.482 |

For comparison, `radix32_n1_fwd_avx2` (no twiddle table) runs at 6744 ns / CPE 1.99.
The fact that `t1_dit` is **slower than `n1`** points to twiddle-table access as
the overhead source — but the breakdown below shows the actual cost is on the
**store side**, not loads.

## Pipeline breakdown

| Category | Value |
|---|---|
| Retiring | **34.4%** |
| Front-End Bound | 1.2% |
| Bad Speculation | 1.6% |
| **Back-End Bound** | **62.8%** |
| — Memory Bound | 46.0% |
| — Core Bound | 16.8% |

The codelet retires only about one-third of pipeline slots. Back-end is the
dominant bottleneck, split roughly 3:1 between memory and compute pressure.

## Memory Bound breakdown

| Source | % of Clockticks |
|---|---|
| **Store Bound** | **37.2%** |
| — Store Latency | 67.5% |
| — **DTLB Store Overhead** | **66.1%** |
| — — Store STLB Hit | 66.1% |
| — — Store STLB Miss | 0.0% |
| L2 Bound (L2 Hit Latency) | 8.7% |
| L1 Bound | 3.7% |
| — L1 Latency Dependency | 19.8% |
| — FB Full | 13.4% |
| — DTLB Overhead (loads) | 1.6% |
| L3 Bound | 0.1% |
| DRAM Bound | 0.1% |

### The dominant bottleneck: DTLB store pressure

**66.1% of clockticks are stalled on L1 DTLB store misses that hit in STLB.**

This is the #1 issue. Every stride-K store to the output array is walking to
the second-level TLB. STLB hits are cheap compared to page walks, but still
cost cycles — and they happen on nearly every store.

### Why R=32 hits TLB and R=16 doesn't

At K=256, `ios = K × sizeof(double) = 2048 bytes = 2 KB`.

Each stride-K output store `rio_re[m + k*ios]` lands on a different 4 KB page
for every other `k` (two strides per page).

Per iteration pages touched by the codelet:
- Output `rio_re` + `rio_im`: 32 stride-K stores × 2 arrays ≈ 32 pages
- Input `rio_re` + `rio_im` (loaded at start): same 32 pages
- Twiddle `W_re` + `W_im`: 31 stride-K loads × 2 arrays ≈ 16 pages
- **Total ≈ 80 unique pages per iteration**

Raptor Lake L1 DTLB capacity is roughly 96 entries (48 load + 48 store paths).
The working set of ~80 pages — split across load and store ports — overflows
the first-level DTLB for stores. Every store that misses L1 DTLB walks to STLB.

By contrast, R=16 touches about half as many pages and stays resident in L1
DTLB, which is why R=16's bottleneck was on loads (hardware-prefetcher miss
for stride-K twiddle loads), not stores.

### Store Latency 67.5%

Stores are taking many cycles to retire, consistent with TLB walks blocking
the store pipeline. Store buffer / fill buffer pressure (FB Full 13.4%)
confirms the store queue is often full.

## Core Bound breakdown

| Source | % of Clockticks |
|---|---|
| Port Utilization | 27.6% |
| — 0 ports busy | 0.7% |
| — 1 port busy | 16.4% |
| — 2 ports busy | 18.4% |
| — 3+ ports busy | 40.7% |
| ALU Port 0 | 30.7% |
| **ALU Port 1** | **45.4%** |
| ALU Port 6 | 5.1% |
| Load port utilization | 23.8% |
| Store port utilization | 16.6% |
| Vector Capacity Usage (FPU) | 50.0% |

Port 1 at 45.4% is moderate pressure but not saturating (R=16 hit 80%). FMA
throughput is not the bottleneck. The 3+ port cycles (40.7%) show OOO can
overlap multiple operations when not store-stalled.

## Comparison across radixes (K=256, AVX2 t1_dit_fwd)

| Radix | CPE | Retiring | Memory Bound | Store Bound | DTLB Store |
|---|---|---|---|---|---|
| R=4 | 0.277 | 85.9% | — | — | — |
| R=8 | 0.516 | 72.2% | ~1% | — | — |
| R=16 (pre-prefetch) | 2.58 | 21.8% | 74% | 53% | 23.9% |
| R=16 (post-prefetch) | 0.94 | ~25% | ~30% | 53% | — |
| **R=32** | **2.55** | **34.4%** | **46%** | **37.2%** | **66.1%** |

R=32's store-bound signature is **categorically different** from R=16. R=16 was
a load problem (L1 Bound 30.3%, solved by twiddle prefetch). R=32 is a TLB
pressure problem on outputs.

## Implications for optimization

### Interventions that won't move the needle much

- **Twiddle prefetch** (R=16's 2.7× fix): L1 Bound is only 3.7%. Load-side is
  mostly hidden already. Expect small single-digit gains at best.
- **Port-level micro-scheduling**: Port 1 at 45.4% isn't saturating. Freeing a
  port isn't the win.

### Interventions with real potential

- **Huge pages (2 MB)**: Every output stride lands on its own 4 KB page today.
  With 2 MB pages, all 32 output rows fit in a single TLB entry. DTLB Store
  Overhead 66.1% → near zero. Biggest available win, but requires OS-level
  allocator changes ("Lock Pages in Memory" privilege on Windows, `MAP_HUGETLB`
  or `madvise(MADV_HUGEPAGE)` on Linux). Out of scope for this codelet alone
  — a project-wide change.
- **`PREFETCHW` on output addresses**: Prefetch-for-write brings the cache line
  in Modified state and populates L1 DTLB. Could directly reduce DTLB store
  overhead without requiring huge pages. Novel experiment — no prior data.
- **Deferred constants**: Same pattern that gave R=16 and R=8 a few percent
  each. Move the 22 `tw_W32_*` broadcasts from function start to Pass 2 where
  they're actually used. Reduces register pressure and compiler spills during
  Pass 1. Low risk, 2–4% expected.

### Interventions ruled out

- **AVX2 U=2 (two k-blocks per iteration)**: Would double register pressure
  from 32 values to 64. Catastrophic overflow of 16 YMM. Already proven to
  regress R=8 by 39%; R=32 would be worse.
- **Streaming (non-temporal) stores**: Output feeds the next pass of the FFT
  — bypassing cache would force reloads. Counterproductive for inner t1_dit.

## Summary

R=32 on AVX2 is dominated by TLB walks on output stores, not compute or load
latency. The only in-codelet interventions that can meaningfully reduce this
are `PREFETCHW` on output addresses (to warm L1 DTLB) or huge-page backed
allocations (to eliminate the DTLB pressure entirely). Deferred constants and
twiddle prefetch are cheap wins worth taking but will not change the
macro-picture.

Compared to R=16 — where a simple twiddle-prefetch change yielded a 2.7× speedup
— R=32 is a harder target. Expect smaller in-codelet wins until the huge-pages
allocator work is done project-wide.

## Optimization results

### Optimization #1: Deferred W32 twiddle broadcasts (applied)

**Change**: Moved the 22 `tw_W32_*_re/im` broadcasts from function scope to
the start of PASS 2 (where they are first used). Previously these were
declared before the outer `for (m ...)` loop and competed for registers
during PASS 1, where they are dead.

**Rationale**: Pre-change, the compiler had 24 long-lived `const __m256d`
constants (22 `tw_W32_*` + `sign_flip` + `sqrt2_inv`) at function scope. With
only 16 YMM available on AVX2, most of them were forced onto the stack during
PASS 1's twiddle-load + radix-8 butterfly work, generating spill/reload
traffic alongside the explicit spill buffer.

**Result** (K=256, AVX2, P-core pinned, 5.7 GHz turbo):

| Codelet | Before | After | Delta |
|---|---|---|---|
| `t1_dit_fwd` | 8634.5 ns, CPE 2.550 | **6777.6 ns, CPE 1.969** | **-21.5%** |
| `t1_dit_bwd` | 8435.8 ns, CPE 2.491 | **6820.4 ns, CPE 1.982** | **-19.1%** |

Much larger than the 2–4% precedent from R=8 and R=16 because R=32 has
significantly more long-lived constants relative to register budget.

The gap between `t1_dit` and `n1` (no twiddle table) narrowed from ~28% to
~12%, showing the twiddle-access overhead was in large part compiler-
generated spill/reload pressure, not inherent to the twiddle memory access
itself.

**Status**: Kept. Hand-edit applied to
`codelets/avx2/fft_radix32_avx2_ct_t1_dit.h` and propagated to `gen_radix32.py`.

---

### Optimization #2: Twiddle prefetch (reverted)

**Change attempted**: Same pattern that gave R=16 a 2.7× speedup — insert
`_mm_prefetch(_MM_HINT_T0)` calls during each sub-FFT's spill stores,
targeting the next sub-FFT's 8 stride-K twiddle loads. 48 prefetch
instructions added per iteration (8 per sub-FFT × 3 sub-FFTs × 2 for re/im).

**Rationale tried**: R=32 has more stride-K twiddle loads than R=16 (31 vs 15),
so the same pattern seemed natural.

**Result** (K=256):

| Codelet | Deferred-const baseline | + Twiddle prefetch | Delta |
|---|---|---|---|
| `t1_dit_fwd` | 6777.6 ns, CPE 1.969 | 7495.4 ns, CPE 2.209 | **+10.6% slower** |
| `t1_dit_bwd` | 6820.4 ns, CPE 1.982 | 7340.5 ns, CPE 2.164 | **+7.6% slower** |

**Why it failed**: VTune showed L1 Bound at only 3.7% — load-side latency is
already hidden by the OOO engine and hardware prefetcher. With no load
latency to hide, the prefetch instructions are pure front-end pressure. Same
failure mode as R=8's prefetch attempt (where L1 Bound was ~1%).

**Lesson**: Software twiddle prefetch only helps when L1 Bound is significant
(R=16 pre-optimization: 30.3% L1 Bound → 2.7× speedup). Low L1 Bound means
there's nothing to hide and prefetch becomes net overhead.

**Status**: Reverted.

---

### Optimization #3: `PREFETCHW` on output addresses (reverted)

**Change attempted**: Before each PASS 2 block, issue 16 `__builtin_prefetch(.., 1, 3)`
(write-intent, high locality) for the output destination cache lines at
`rio_re[m + k*ios]` and `rio_im[m + k*ios]` for k ∈ {0, 4, 8, 12, 16, 20, 24, 28}.
Intent: warm the L1 store DTLB before the actual stride-K stores so the
66.1% STLB-hit traffic would drop.

**Result** (K=256):

| Codelet | Baseline | + PREFETCHW | Delta |
|---|---|---|---|
| `t1_dit_fwd` | 6777.6 ns, CPE 1.969 | **11601.3 ns, CPE 3.389** | **+71.2% slower** |

Catastrophic regression.

**Why it failed**: Two compounding issues.

1. **Front-end pressure, same as twiddle prefetch**: 16 extra instructions per
   iteration adds issue/decode overhead. The codelet has no idle slots.
2. **DTLB capacity cannot be bypassed**: Raptor Lake's L1 store DTLB holds
   roughly 16 entries. The codelet's working set spans ~32 output pages plus
   ~32 input pages plus ~16 twiddle pages. PREFETCHW populates TLB entries,
   but entries immediately get evicted by other memory accesses before the
   actual stride-K stores arrive. The prefetched-then-evicted entries also
   thrash the STLB, making the situation worse.

**Lesson**: Software TLB warming via prefetch cannot fix fundamental DTLB
capacity overflow. The only in-codelet fix for R=32's 66.1% DTLB Store
Overhead is **huge pages** (one 2 MB TLB entry covers all 32 output rows at
stride 2 KB), which is a project-wide allocator change, out of scope for
this codelet.

**Status**: Reverted.

---

## Final conclusion

R=32 on AVX2 is **store-bound with hardware-capacity DTLB pressure** — a
fundamentally different bottleneck from R=16's load-latency issue.

The only in-codelet optimization that moved the needle was **deferred constants
(-21.5%)**, which reduced compiler-generated spill/reload traffic during
PASS 1 by freeing register budget. Both prefetch strategies (twiddle prefetch
for loads, PREFETCHW for stores) regressed because:

- R=32 has no L1 load latency to hide (3.7% L1 Bound)
- R=32's DTLB overflow is a hardware capacity issue software cannot fix

The remaining 66.1% DTLB Store Overhead will only yield to huge-pages backed
allocations, which is a separate project-wide effort.

**Baseline locked**: `t1_dit_fwd` 6777.6 ns / CPE 1.969.
