# VTune Profile: R=11 Isolated Codelet (t1_dit_fwd, K=256)

**Date:** 2026-04-13  
**CPU:** Intel i9-14900KF (Raptor Lake), P-core only, 5.70 GHz turbo  
**Binary:** `vfft_bench_codelet.exe 256 5000 11 t1_dit_fwd`  
**Codelet:** `radix11_t1_dit_fwd_avx2` — prime-radix DFT-11 (genfft Winograd-style), single stage, AVX2  
**Data size:** R=11, K=256, total=2816 elements = 44 KB (re+im), fits L1  
**Analysis:** Microarchitecture Exploration (uarch-exploration)

---

## Baseline performance

| Metric | Value |
|---|---|
| ns/call | 1333 |
| CPE (cycles / (R × K)) | **2.70** |
| GFLOP/s | 27 |
| Retiring | **58.5%** of pipeline slots |
| CPI Rate | 0.283 |

For comparison, `radix11_n1_fwd_avx2` (no twiddle table) runs at 865 ns /
CPE 1.75. The `t1_dit` / `n1` ratio is ~1.54x, similar to R=10.

R=11 is a **prime-radix codelet** — the DFT-11 butterfly cannot decompose
into smaller sub-butterflies. genfft generates a Winograd-style single-stage
implementation. This is a different regime from the small composites
(R=10, R=12, R=20, R=25) which decompose into parallel sub-butterflies.

---

## Pipeline breakdown

| Category | Value |
|---|---|
| **Retiring** | **58.5%** |
| Front-End Bound | 1.7% |
| **Bad Speculation** | **4.6%** ← unusual |
| Back-End Bound | 35.2% |
| — Memory Bound | 6.4% |
| — Core Bound | 28.8% |

R=11 has the lowest Memory Bound (6.4%) and the highest Bad Speculation
(4.6%) of any codelet we have profiled. The combination is a signature of
prime-radix Winograd-style code: minimal memory pressure (single stage,
small working set) but unusual speculation behavior from the generated
butterfly's branch/memory-disambiguation patterns.

---

## Memory Bound breakdown

| Source | % of Clockticks |
|---|---|
| L1 Bound | 2.1% |
| — **L1 Latency Dependency** | **43.0%** ← highest in family |
| — FB Full | 9.0% |
| — DTLB Overhead (loads) | 0.1% |
| — Split Loads | 0.3% |
| L2 Bound | 5.9% |
| L3 Bound | 0.2% |
| DRAM Bound | 0.1% |
| **Store Bound** | **0.0%** |
| — Store Latency | 0.1% |
| — Split Stores | 16.4% of store-cycles |
| — DTLB Store Overhead | 0.2% |

Key observations:

- **L1 Latency Dependency 43.0%** is the highest in the small-composite +
  prime-radix family. This is the radix-11 FMA dependency chain
  serialization. The Winograd DFT-11 butterfly has fewer independent
  sub-products than the radix-5 or radix-3 butterflies, producing a longer
  serial chain.
- **Memory Bound only 6.4%** — the lowest of all t1_dit codelets profiled.
  Single-stage execution + small working set means almost no cache pressure.
- **Store Bound 0.0%** — like R=12, the compute work fully absorbs store
  latency. Split stores 16.4% of zero-store-cycles is meaningless.
- **L2 Bound 5.9%** — slightly elevated. Some twiddle reloads going to L2.
  Not significant overall.

---

## Bad Speculation — unusual signature

| Source | % of Pipeline Slots |
|---|---|
| Branch Mispredict | 0.2% |
| **Machine Clears** | **4.4%** |
| — Other Nukes | 0.0% |

**Machine Clears at 4.4%** is genuinely unusual — no other codelet we
profiled shows more than ~0.1%. Possible causes for prime-radix
Winograd-style code:

- **Memory disambiguation failures**: the genfft-generated butterfly may
  have load/store address patterns that the disambiguator predicts wrong,
  triggering re-execution.
- **FP subnormal handling**: if intermediate radix-11 calculations produce
  subnormal values for some inputs, FP assists (which trigger machine
  clears) could fire.
- **Self-modifying code detection**: unlikely but possible if the
  generated code layout has unusual properties.

The 4.4% machine clear cost amounts to ~6 billion clockticks lost out of
the 138 billion total. Worth investigating further but not catastrophic —
R=11 is still 58.5% retiring.

Not pursued in this session; flagged as future work.

---

## Core Bound breakdown

| Source | % of Clockticks |
|---|---|
| Port Utilization | 21.1% |
| — 0 Ports Utilized | 0.0% |
| — 1 Port Utilized | 8.1% |
| — 2 Ports Utilized | 9.8% |
| — **3+ Ports Utilized** | **71.6%** |
| ALU Operation Utilization | 36.6% |
| — **Port 0** | **62.6%** |
| — **Port 1** | **64.4%** |
| — Port 6 | 9.0% |
| Load Operation Utilization | 44.4% |
| Store Operation Utilization | 31.2% |
| Vector Capacity Usage (FPU) | 50.0% |

Both FMA ports balanced at ~63%, indicating evenly distributed butterfly
work. Load op utilization 44.4% is high — the radix-11 butterfly does
many twiddle multiplications, each requiring a twiddle-pair load.

---

## Cross-radix comparison (K=256, AVX2 t1_dit_fwd)

| Radix | CPE | Retiring | Memory Bound | Bad Spec | Dominant bottleneck |
|---|---|---|---|---|---|
| R=4 | 0.277 | 85.9% | — | 0.1% | peak |
| R=8 | 0.516 | 72.2% | 1% | 0.1% | R=8 FMA chain |
| R=10 | 1.98 | 63.4% | 10.7% | 0.1% | R=5 + R=2 FMA chains |
| **R=11** | **2.70** | **58.5%** | **6.4%** | **4.6%** | **Prime DFT-11 FMA chain + machine clears** |
| R=12 | ~2.36 | 57.1% | 8.4% | 0.2% | R=3 + R=4 FMA chains |
| R=16 post | 0.94 | ~25% | ~30% | — | was store bound, fixed |
| R=20 | 1.17 | 53.8% | 10.3% | — | R=5 FMA chain |
| R=25 | 1.93 | 50.0% | 20.1% | — | R=5 + emerging store DTLB |
| R=32 post | 1.97 | 34.4% | 46.0% | — | DTLB store 66% |
| R=64 | 3.07 | 27.2% | 57.5% | — | stacked memory |

R=11 is unique in this family:

- Highest L1 Latency Dependency (FMA chain) — the prime DFT-11 has the
  longest serial dependency chain
- Lowest Memory Bound — minimal cache pressure, single-stage execution
- Only codelet with significant Bad Speculation — generated code pattern
  triggers machine clears

---

## Optimization analysis (no changes applied)

### Deferred constants — not applicable

R=11 is single-stage. There is no PASS 2 to defer constants into.
The W11 internal twiddles are inlined into the butterfly directly by
genfft. Cannot apply the deferred-constants pattern that worked for
R=32 (which has separate PASS 1 and PASS 2).

### Twiddle prefetch — predicted to regress

L1 Bound 2.1% — no load latency to hide. Same regime as R=8/R=10/R=12/
R=20/R=25, all of which regressed or were null.

### `PREFETCHW` on outputs — not applicable

Store Bound is 0.0%. Nothing to optimize.

### Investigation of 4.4% Machine Clears — flagged as future work

This is genuinely unusual and worth investigating. To diagnose:

1. **Disassemble** `radix11_t1_dit_fwd_avx2` and look for branchy code or
   unusual load/store address patterns
2. **VTune Memory Access analysis** with disambiguation focus — would
   show whether this is memory ordering / disambiguation failures
3. **VTune `MACHINE_CLEARS.*` event groups** — would categorize the
   clears (memory ordering vs FP assist vs other)
4. **Test with different input data ranges** — if subnormals are the
   cause, providing only normalized inputs would eliminate the clears

Not investigated in this session. R=11 is still 58.5% retiring; the
4.4% machine clear cost is meaningful but secondary to the FMA-chain
dependency that dominates Core Bound.

---

## Final conclusion

R=11 on AVX2 is **near its achievable ceiling** for this microarchitecture,
bounded by the prime-radix DFT-11 Winograd FMA dependency chain. The
4.4% machine clears represents a small recoverable fraction (worth ~3% of
total runtime if entirely eliminated) but is not the dominant factor.

The codelet's profile is distinct from the composite radixes:
- **Lowest memory pressure** (6.4% Memory Bound, 0% Store Bound)
- **Longest FMA chain** (43% L1 Latency Dependency)
- **Unique speculation issue** (4.4% Machine Clears)

**Baseline locked (unchanged)**: `t1_dit_fwd` 1333 ns / CPE 2.70.

---

## Heuristic reinforced

Prefetch heuristic still holds: L1 Bound 2.1% means no load latency to
hide, predicted regression matches the pattern. R=11 doesn't fit either
the "small composite at ceiling" cluster (it's prime, single-stage) or
the "memory-bound large pow2" cluster (it's compute-bound). It's its own
category: **prime-radix Winograd-style codelets** — characterized by
maximum FMA chain length and unique speculation behavior from the
generated code patterns.

Future work: investigate the 4.4% machine clear cost. May be applicable
to other genfft prime codelets (R=13, R=17, R=19, R=23).
