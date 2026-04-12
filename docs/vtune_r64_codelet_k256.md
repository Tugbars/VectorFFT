# R=64 AVX2 Codelet — VTune Analysis (K=256)

Target: `radix64_t1_dit_fwd_avx2` at K=256 (L2-resident).
Hardware: Intel Core i9-14900KF (Raptor Lake), P-core pinned, 5.705 GHz turbo.
Collector: `uarch-exploration` (event-based sampling).

## Baseline performance

| Metric | Value |
|---|---|
| ns/call | 21185.9 |
| CPE (cycles / (R × K)) | 3.072 |
| GFLOP/s | 15.23 |
| Retiring | 27.2% of pipeline slots |
| CPI Rate | 0.609 |

For comparison, `radix64_n1_fwd_avx2` (no twiddle table) runs at 18036 ns / CPE 2.615.
The `t1_dit` → `n1` gap (17%) is smaller than R=32 saw (28%), because R=64's
bottleneck is no longer purely twiddle-access.

## Pipeline breakdown

| Category | Value |
|---|---|
| Retiring | **27.2%** |
| Front-End Bound | 1.1% |
| Bad Speculation | 0.5% |
| **Back-End Bound** | **71.2%** |
| — Memory Bound | 57.5% |
| — Core Bound | 13.7% |

R=64 retires even less than R=32 (34.4%). Back-end saturation dominates.

## Memory Bound breakdown — and what's new vs R=32

| Source | R=32 | R=64 |
|---|---|---|
| Memory Bound | 46.0% | **57.5%** |
| — **Load DTLB Overhead** | 1.6% | **32.9%** ← NEW |
| — Store DTLB Overhead | 66.1% | 43.7% |
| — Store Latency | 67.5% | 74.7% |
| — L1 Bound | 3.7% | 7.2% |
| — L1 Latency Dependency | 19.8% | 14.6% |
| — L2 Bound | 8.7% | **16.3%** |
| — FB Full | 13.4% | 2.4% |
| — **Split Loads** | — | **5.1%** ← NEW |
| — **Split Stores** | — | **7.0%** ← NEW |

### The three stacked bottlenecks

1. **Load DTLB overflow (32.9%)**: R=64's working set touches enough stride-K
   twiddle and input pages that the load-side L1 DTLB overflows too (not just
   the store DTLB). This is new vs R=32 where load DTLB was essentially zero
   (1.6%).
2. **Store DTLB (43.7%)**: still high, though lower percent-of-clockticks than
   R=32 because more clockticks are now absorbed by load DTLB walks — the
   total DTLB cost has grown, just spread across load and store paths.
3. **L2 Bound (16.3%)**: spill buffer plus input/twiddle working set exceeds
   L1d capacity comfortably. Some PASS 2 spill reloads hit L2.

### Page budget

At K=256, `ios = 2 KB`. R=64's pages touched per iteration:
- Output `rio_re` + `rio_im`: 64 stride-K stores × 2 ≈ 64 pages
- Input `rio_re` + `rio_im`: same ~64 pages (PASS 1 loads)
- Twiddle `W_re` + `W_im`: 63 stride-K loads × 2 ≈ 32 pages
- **Total ≈ 160 unique pages**

Against Raptor Lake's ~96-entry L1 DTLB (split roughly 48 load / 48 store),
the working set is ~3.3× capacity. Both load and store DTLBs overflow.

## Core Bound breakdown

| Source | R=32 | R=64 |
|---|---|---|
| Port Utilization | 27.6% | 24.4% |
| 3+ Ports Utilized | 40.7% | 33.0% |
| Port 0 | 30.7% | 22.2% |
| Port 1 | 45.4% | 34.9% |
| Load Op Utilization | 23.8% | 18.0% |
| Store Op Utilization | 16.6% | 15.4% |
| Vector Capacity (FPU) | 50.0% | 50.0% |

Port pressure actually **lower** than R=32 because so many cycles are stalled
on memory. Compute is not the bottleneck.

## Critical difference from R=32: front-end has budget

| Metric | R=32 | R=64 |
|---|---|---|
| Front-End Bound | 1.2% | **1.1%** |
| DSB Coverage | 94.2% | 32.3% |
| DSB Misses | 8.1% | 13.8% |

R=64's DSB coverage dropped to 32% (vs R=32's 94%), so more instructions
stream through MITE instead of the uop cache. Despite this, front-end is
still not the bottleneck — memory is. Importantly, adding instructions does
not immediately hit a front-end wall the way it does for R=32.

## Code structure — why R=32's wins don't port

Unlike R=32, R=64's codelet does **not** hoist internal twiddle broadcasts at
function scope. Internal twiddles are inlined as
`_mm256_set1_pd(iw_re[N])` at each use site, letting the compiler fold them
as broadcast-from-memory operands to FMA. This means the "deferred constants"
trick that gave R=32 a 21.5% win has no target here — there is nothing to
defer.

## Optimization results

### Optimization #1: Twiddle prefetch (reverted)

**Change**: Interleaved 4 `_mm_prefetch(_MM_HINT_T0)` pairs (re+im) per PASS 1
sub-FFT, targeting the next sub-FFT's 7–8 stride-K twiddle loads. Placed
before the first 4 spill stores of each sub-FFT. 7 sub-FFTs × 4 pairs × 2 =
56 prefetch instructions per iteration.

**Rationale**: R=64 has L1 Bound 7.2% and Load DTLB 32.9% — real load latency
to hide. R=32's prefetch failed because L1 Bound was only 3.7%, but R=64's
signature looked like R=16's pre-optimization state (which got 2.7× from
prefetch).

**Result** (K=256):

| Codelet | Baseline | + Twiddle prefetch | Delta |
|---|---|---|---|
| `t1_dit_fwd` | 21185.9 ns, CPE 3.072 | 22101.7 ns, CPE 3.281 | **+4.3% slower** |
| `t1_dit_bwd` | 20554.3 ns, CPE 2.981 | 21920.3 ns, CPE 3.254 | **+6.6% slower** |

**Why it failed**: Not the front-end (R=64 had budget). Likely:

- Memory port contention. Load/store ports were at 18%/15.4%. Adding 56
  prefetches per iteration drives memory ports higher. Prefetches share ports
  with real loads/stores.
- **L2-latency distance**. Many twiddle loads hit L2, not L1. T0 prefetch
  brings to L1 but the prefetch itself carries L2 latency. Our prefetch →
  store → use distance (~10–15 cycles) may not cover the ~13-cycle L2 hit
  latency for the fetch to complete.
- **Spill buffer eviction**. Aggressive prefetch may evict the hot spill
  buffer from L1, causing PASS 2 loads to hit L2 — the L2 Bound 16.3% could
  have grown.

**Status**: Reverted.

---

### Optimization #2: Conservative twiddle prefetch (reverted)

**Change**: Same structure as #1 but halved — 2 prefetch pairs per sub-FFT
instead of 4. 7 sub-FFTs × 2 pairs × 2 = 28 prefetches total per iteration
(vs 56 in attempt #1).

**Rationale**: #1 regressed partly from port contention. Halving the count
should halve port pressure. Test whether there's any count at which twiddle
prefetch breaks even.

**Result** (K=256):

| Codelet | Count 56 | Count 28 | Baseline |
|---|---|---|---|
| `t1_dit_fwd` | +4.3% slower | **+2.5% slower** | 21185.9 ns |
| `t1_dit_bwd` | +6.6% slower | **+6.1% slower** | 20554.3 ns |

**Observations**:
- fwd regression **scales with count** (56→28 halved cost from +4.3% to +2.5%)
  → real front-end/port contention exists.
- bwd regression **stuck at ~6% regardless of count**
  → different dominant cost in backward direction (candidate: L1 eviction of
  hot spill buffer during PASS 2 reloads).

**Status**: Reverted. No count tried yielded a net positive.

---

## Remaining candidates (not yet tried)

To investigate in a future pass, cheapest-first:

### A. T1 hint prefetch (L2, not L1)

Replace `_MM_HINT_T0` with `_MM_HINT_T1` in the prefetch instructions. T1 still
populates the L1 DTLB (warming TLB for the actual load), but only brings data
to L2 — avoiding eviction of the hot spill buffer from L1d. Could isolate
whether the #1/#2 regressions came from L1 thrashing or true port contention.

Setup cost: one-line change to existing prefetch code.

### B. NFUSE=2 (s-reg fusion in last sub-FFT)

R=64 currently uses NFUSE=0; every sub-FFT spills all 8 values. R=32 uses
NFUSE=2 where the last sub-FFT keeps 2 values in `s0`, `s1` registers and
PASS 2's k1=0,1 columns read from s-regs instead of spill. Saves 4 store/load
pairs (8 bytes × 4 re + 8 bytes × 4 im = 64 bytes) per iteration.

Small savings, but isolates cleanly from prefetch. Requires generator-side
change.

### C. Ultra-conservative prefetch (1 pair per sub-FFT = 14 total)

Scaling from #1 (56: +4.3%) and #2 (28: +2.5%) extrapolates to 14 ≈ +0%. May
hit net-zero for `fwd` (bwd still stuck at ~6%). Lowest effort experiment of
this set — just remove half the prefetches from attempt #2.

### D. Manual hoisting of repeated inline broadcasts

PASS 2 inlines `_mm256_set1_pd(iw_re[N])` at every twiddle use. Some indices
(e.g. `iw_re[2]`, `iw_re[4]`) appear in multiple k1 columns. The compiler
should CSE these but ICX may not always. Manually factoring `__m256d wr =
_mm256_set1_pd(iw_re[N]), wi = ...;` before each column's radix-4 would force
it. Tiny potential win. Requires generator-side change.

### E. Reordering for page locality (Split Stores 7.0%)

At K=256, `ios = 2 KB` → every other output `k` index straddles a 4 KB page
boundary. 7% of stores are page-splits. If PASS 2 output order could be
reorganized to write same-page stores consecutively, some split stores would
become normal aligned stores. Significant complexity — output index order is
dictated by FFT structure — probably requires a post-PASS-2 shuffle buffer
to decouple write order from FFT order. Gains unclear.

### F. `PREFETCHW` on outputs (explicitly ruled out)

R=32 tried this and saw **-71%**. R=64 has 2× the output pages (64 vs 32).
Same L1 DTLB capacity overflow, worse. Not attempted, not likely worth trying
without first fixing the capacity issue via huge pages.

---

## Final conclusion

R=64 on AVX2 has **three stacked memory bottlenecks** that together consume
57.5% of pipeline slots:

1. Load DTLB overflow (new vs R=32)
2. Store DTLB overflow (ongoing from R=32)
3. L2 load pressure (new vs R=32)

No single in-codelet change can address all three. The R=32 playbook of
deferred constants doesn't apply (no hoisted broadcasts to move). Twiddle
prefetch regresses because memory ports are contested and L2 latency exceeds
the prefetch-to-use distance.

Remaining candidates, all speculative:

- **Huge pages (2 MB)**: would collapse 160-page working set to ~4 pages.
  Addresses all three DTLB issues at once. Project-wide allocator change,
  out of scope for this codelet.
- **`PREFETCHW` on outputs**: R=32 tried this and regressed 71%. R=64 has
  the same capacity overflow, worse (64 pages vs 32). Likely also regresses.
  Not attempted.
- **Split Stores 7.0% investigation**: if any of these splits are
  eliminable via store reordering, modest gain. Not yet attempted.
- **Conservative prefetch (fewer instructions)**: the 56-prefetch attempt
  failed; maybe 14 or 28 would be different. Not tried yet.

**Baseline locked (unchanged)**: `t1_dit_fwd` 21185.9 ns / CPE 3.072.

## Comparison across radixes (K=256, AVX2 t1_dit_fwd)

| Radix | CPE | Retiring | Dominant bottleneck |
|---|---|---|---|
| R=4 | 0.277 | 85.9% | — (peak efficiency) |
| R=8 | 0.516 | 72.2% | Dependency chains |
| R=16 (post-prefetch) | 0.94 | ~25% | Store Bound residual |
| R=32 (post-deferred) | 1.97 | 34.4% | Store DTLB 66% |
| **R=64** | **3.07** | **27.2%** | Load DTLB + Store DTLB + L2 |

Each radix doubling roughly doubles CPE on AVX2, and the dominant bottleneck
shifts: compute-chain-bound for small R, memory-bound for R=16, store-DTLB
for R=32, full memory hierarchy stress for R=64. Huge pages is the only
mechanism with potential to change this trend without architectural change
(AVX-512 register file).
