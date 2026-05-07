# N=131072 K=4 — VTune deep-dive vs MKL, EXHAUSTIVE validation

**Cell:** N=131072 (= 2^17), K=4. Single-thread, P-core 2 pinned, AVX2.
This is the closest-margin cell in the v1.0 production bench (1.17×
MKL — see `docs/performance/vfft_perf_tuned_1d_mkl.txt`).

**TL;DR — the "ILP weakness" framing was wrong, but there's a real
ceiling to find:**

- VectorFFT achieves **70% pipeline retiring** with **IPC 3.94** at this
  cell. MKL achieves **24.8% retiring** with **IPC 1.47**.
- VectorFFT is **8.8% memory-bound** (mostly bandwidth). MKL is
  **56.3% memory-bound** — moves ~6× more data because of split-complex
  ↔ interleaved gather/scatter overhead.
- VectorFFT executes **2.43× more instructions** than MKL per FFT
  (41.24 B vs 16.95 B). Their compute kernel is denser. We compensate
  with **2.68× higher IPC** — the two effects nearly cancel, leaving
  the 1.17× wall-time margin.
- **EXHAUSTIVE search of 887 candidates returns the same plan as the
  MEASURE top-K=5** that's already in wisdom. The 8-stage
  `4×4×4×4×8×4×4×4` factorization is empirically the optimum on this
  CPU — no planning gap to close.
- **The actual ceiling is ~3× MKL, not 1.17×.** Closing it requires
  codelet fusion (one monolithic 8-stage kernel instead of 8 separate
  codelet calls). Gated to v2.0+ work; currently structural.

---

## Data — uarch metrics (vtune -collect uarch-exploration, ITT-tagged)

Both libraries measured under identical conditions: same buffer, same
ITT-tagged region, same elevated VTune session, P-core 2 pinned.

| Metric | **VectorFFT** | **MKL** | Ratio (VFFT/MKL) |
|--------|---:|---:|---:|
| Wall time per FFT | 2.36 ms | 2.55 ms | **0.92×** (8% faster) |
| **CPI Rate** | **0.254** | **0.680** | **0.37×** |
| **IPC** | **~3.94** | **~1.47** | **2.68×** |
| **Retiring** | **70.0%** | **24.8%** | **2.82×** |
| Front-End Bound | 8.4% | 3.8% | 2.21× |
| Bad Speculation | 1.1% | 0.3% | 3.67× |
| **Back-End Bound** | **20.5%** | **71.0%** | **0.29×** |
| ↳ Memory Bound | 8.8% | **56.3%** | 0.16× |
| ↳↳ Memory Bandwidth share of MB | 74.3% | 72.5% | similar |
| ↳ Core Bound | 11.6% | 14.7% | 0.79× |

## Function-level CPU time within the ITT region

VFFT side (factorization `4×4×4×4×8×4×4×4`, 8 stages, mostly r4):

| Function | CPU Time | Share |
|---|---:|---:|
| `radix4_t1s_dit_fwd_avx2` | 0.630s | 33.6% |
| `_stride_execute_fwd_slice_from` | 0.395s | 21.0% |
| `radix4_n1_fwd_avx2` | 0.325s | 17.4% |
| `stride_cmul_scalar_avx2` | 0.134s | 7.1% |
| `radix8_t1s_dit_fwd_avx2` | 0.130s | 6.9% |
| (others) | ~0.06s | ~3.7% |

MKL side:

| Function | CPU Time | Share |
|---|---:|---:|
| `[MKL FFT]@owns_crRadix4FwdNorm_64f` | 1.185s | **58.6%** (compute) |
| `[MKL FFT]@gather_dd_dd` | 0.379s | 18.7% (split→interleaved) |
| `[MKL FFT]@scatter_dd_dd` | 0.266s | 13.2% (interleaved→split) |
| `func@0x1403f5655` (ntoskrnl) | 0.078s | 3.9% |

MKL's **gather + scatter overhead = 31.9% of their total CPU time**.
That's the cost of converting between split-complex (the bench's
input/output layout) and MKL's internal interleaved representation
on every call. VectorFFT's split-complex-native layout pays zero of
this cost.

## What the metrics actually say

### VectorFFT — clean execution

70% pipeline retiring + IPC 3.94 is essentially a well-behaved
codelet at the K-axis ceiling. The remaining 30% breaks down as:

- **8.4% front-end bound** — DSB (decode stream buffer) bandwidth
  takes 6.9% of slots. Codelet body is large enough to occasionally
  stress the µop cache. Not actionable without restructuring code.
- **1.1% bad speculation** — branch predictor is happy. Codelet has
  predictable structure.
- **20.5% back-end bound:**
  - **8.8% memory-bound, 74% of which is bandwidth.** N=131072 ×
    K=4 × 16 bytes = 8 MB working set. Larger than L2 (~1 MB), fits
    L3 (~36 MB). Inner stages stay in L1/L2; large-stride passes
    touch L3 and DRAM. The 8.8% reflects DDR5 ceiling pressure on
    those passes, not catastrophic stalls.
  - **11.6% core-bound.** Execution port pressure / dependency chain
    stalls. At K=4, only 4 SIMD lanes per group means the OoO engine
    can overlap fewer independent butterflies. Compare to the same
    codelet at K=256 (per `MEMORY.md` R=4 profile): 86% retiring
    there. The 16-percentage-point drop from K=256 to K=4 is the
    K-axis tax.

### MKL — bandwidth-strangled with denser compute

24.8% retiring is low. The breakdown:

- **3.8% front-end bound** — code is compact, fits µop cache well.
- **71.0% back-end bound:**
  - **56.3% memory-bound, 73% of which is bandwidth.** Massive DRAM
    pressure. Their gather/scatter passes touch the data twice
    extra (split↔interleaved conversion), and their inner kernel's
    interleaved access pattern is less cache-friendly than ours.
  - **14.7% core-bound.** Similar magnitude to ours.

MKL's compute kernel `owns_crRadix4FwdNorm_64f` is a denser, more
specialized codelet — probably hand-tuned in assembly with carefully
scheduled FMAs. It does fewer total instructions per FFT than our
many-stage plan. That's why the wall-time gap is only 1.17× despite
the 3× retiring difference: we do more instructions but execute them
3× more efficiently per cycle.

### Why the 1.17× margin — the instruction-count breakdown

Direct from the VTune summary report (per ITT-tagged task):

| Metric | **VectorFFT** | **MKL** | Ratio |
|---|---:|---:|---:|
| **Instructions retired** | **41.24 B** | **16.95 B** | **VFFT does 2.43× MORE** |
| Clockticks | 10.46 B | 11.53 B | VFFT 9% fewer cycles |
| IPC | 3.94 | 1.47 | VFFT 2.68× higher |
| Frequency | 5.58 GHz | 5.71 GHz | similar |

**The two effects almost exactly cancel.** VFFT does 2.43× more
instructions but executes them 2.68× more efficiently per cycle. Wall
time = instructions ÷ IPC × frequency, so:

```
VFFT cycles = 41.24 B insts / 3.94 IPC = 10.47 B cycles → 1.88 ms compute
MKL cycles  = 16.95 B insts / 1.47 IPC = 11.53 B cycles → 2.02 ms compute
Wall-time ratio: VFFT / MKL = 0.91× (VFFT 9% faster from the IPC margin).
```

The remaining gap to the 1.17× production-bench margin comes from
MKL's gather/scatter overhead — already absorbed into their reported
wall time.

### Where the instructions actually go

Theoretical work: `5 × N × log2(N) × K = 5 × 131072 × 17 × 4 ≈ 44.6 M
FLOPs/FFT`. Times ~1500 reps in the ITT region → **~67 B FLOPs total**.
AVX2 packed FMA = 8 FLOPs/instruction → **~8.4 B FMAs minimum**.

| Library | Total instructions | FMA density | Implication |
|---|---:|---:|---|
| **MKL** | 16.95 B | **~50%** (8.4 / 16.95) | Highly compact — minimal overhead per FMA. Decades of hand-tuned scheduling. |
| **VFFT** | 41.24 B | **~20%** (8.4 / 41.24) | **80% non-FMA overhead** per FMA. See breakdown below. |

VFFT's 80% non-FMA instructions break down across:

1. **8 codelet stage calls.** Each codelet is a separate function:
   prologue, twiddle pointer fetch, K-loop body, return. MKL fuses
   many internal radix steps into one monolithic kernel.
2. **Per-stage executor overhead** (`_stride_execute_fwd_slice_from`
   at 21% of CPU time). Per-group base pointer compute, twiddle table
   walk, scalar preprocessing — paid 8 times.
3. **Scalar twiddle preprocessing** (`stride_cmul_scalar_avx2` at
   7.1%). Per-group cf0 multiply + K-blocked twiddle broadcast. SIMD'd
   in a prior session, but still a separate pass before each codelet.
4. **Split-complex twiddle loads.** Real and imag twiddle tables at
   different addresses → more load instructions per FMA than MKL's
   interleaved twiddle layout.

### Where the actual ceiling is

The interesting counterfactual: **if VFFT achieved MKL's instruction
density**, we'd run at:

```
~17 B insts ÷ 3.94 IPC ÷ 5.58 GHz = ~0.77 ms wall time
vs MKL 2.55 ms = 3.0× win (vs the current 1.17×)
```

**The actual ceiling for this cell is ~3× MKL, not 1.17×.** Closing
that gap would require:

- **Codelet fusion** — generate one monolithic 8-stage codelet that
  fuses all radix-4 steps + the single radix-8 step into one tight
  function. Eliminates the 8× executor-overhead pay and the 8× codelet
  function-call overhead. Probably 5-10× engineering effort beyond
  what's in v1.0.
- **Twiddle layout** — interleaved twiddle (real and imag adjacent)
  halves the load count per butterfly, at the cost of a separate
  twiddle prep pass. Trade-off depends on whether the prep pass
  amortizes.

Both are deferred post-v1.0. The current 1.17× is **structural** in
the sense that closing it requires architecture changes, not just
calibration tuning.

### Net story

| Fact | Implication |
|---|---|
| VFFT at 70% retiring, IPC 3.94 | Codelet+executor are well-implemented; small tuning headroom |
| MKL at 24.8% retiring, IPC 1.47 | MKL is bandwidth-strangled at this cell |
| MKL pays 32% gather/scatter | Layout conversion overhead they can't avoid |
| VFFT does 2.43× more instructions | Architectural cost of multi-stage planner-driven decomposition |
| VFFT IPC 2.68× higher | Architectural advantage of fewer dependencies per call |
| Net wall-time margin | 1.17× (the two effects nearly cancel) |
| **Architectural ceiling for this cell** | **~3× MKL, gated on codelet fusion (v2.0+ work)** |

## EXHAUSTIVE vs MEASURE — what the search found

The current wisdom for `(N=131072, K=4)` was produced by MEASURE top-K=5
calibration. To check whether top-K=5 left meaningful headroom on the
table, we ran a forced EXHAUSTIVE re-calibration on this single cell
(`build_tuned/dev/exhaustive_one/`).

```
[exh] CURRENT pick: 4x4x4x4x8x4x4x4  best_ns=1662345  variants=[FLAT,T1S,T1S,T1S,T1S,T1S,T1S,T1S]
[exh] EXHAUSTIVE finished in 412.0s
  N=131072 K=4: 30 unique decompositions
  Best of this search: 16x4x4x32x16 = 1782730 ns (887 total candidates)
[exh] EXHAUSTIVE pick: 4x4x4x4x8x4x4x4  best_ns=1662345  (preserved from prior MEASURE run)
[exh] improvement: 1.000x
```

**887 candidates evaluated. Zero improvement.** The 8-stage current
plan is the empirical optimum.

The interesting result is the runner-up: the search's best newly-measured
plan was a 5-stage `16×4×4×32×16` at 1,782,730 ns — **7.2% slower**
than the 8-stage incumbent. Naive intuition (fewer stages = less memory
traffic = faster) would have favored the 5-stage plan. Reality:

- r4 retires at ~70% of pipeline slots at this K (per VTune above)
- r16, r32 retire at ~25-34% per existing per-codelet profiles
- **Per-codelet retiring efficiency dominates over memory-pass count**
  at K=4 on this CPU. The 8-stage plan does more passes but each
  pass runs at much higher IPC, beating the 5-stage plan's better
  memory profile.

The cost model would have predicted the opposite ranking. Measurement
overrides the model. This is the architectural bet of the v5 wisdom
format paying off.

## Conclusions

1. **The 1.17× MKL margin at this cell is structural, not a planning
   gap.** VectorFFT executes at IPC 3.94 vs MKL's 1.47 — we are
   2.7× more efficient per cycle. We close 1.17× ahead because MKL
   pays 32% gather/scatter overhead. No factorization change can
   affect either of these. The **actual ceiling is ~3× MKL** if we
   matched their instruction density via codelet fusion (v2.0+).

2. **EXHAUSTIVE confirms MEASURE top-K=5 is the optimum** for this
   cell. The cost model's pre-bench ranking was wrong (favored fewer
   stages); measurement caught the correct answer (8 stages of mostly
   r4).

3. **The "ILP weakness" framing in the original perf doc was
   misleading.** It implied VFFT had a planning gap or codelet
   weakness. The actual story is the opposite: at this cell VFFT runs
   *unusually well* (70% retiring is high for a K=4 batched pow2 FFT).
   The remaining 30% is the natural K=4 ceiling on Raptor Lake — same
   ceiling MKL faces, except they pay it with both port pressure AND
   massive memory bandwidth pressure on top.

4. **Architecture validation:** the v5 wisdom format (factorization +
   per-stage variant codes from plan-level measurement) found the
   right plan that the cost model alone would have missed. This
   reinforces the architectural rule documented in
   `docs/dev/wisdom_bridge_predicates.md`: **plan-level measurement is
   the only accurate variant chooser; cost models and per-radix
   predicates are siblings, both isolated-codelet extrapolations.**

## Reproducing

```
# Whole-bench uarch profile (needs admin for hardware events)
build_tuned/dev/bench_vtune/run.bat --collect uarch-exploration
# (run from elevated cmd or `Start-Process -Verb RunAs`)

# Filter summary to one task
"%VTUNE%/bin64/vtune.exe" -report summary \
  -result-dir build_tuned/dev/bench_vtune/vt_uarch-exploration \
  -filter task=VFFT_N131072_K4_CLOSE -format text
"%VTUNE%/bin64/vtune.exe" -report summary \
  -result-dir build_tuned/dev/bench_vtune/vt_uarch-exploration \
  -filter task=MKL_N131072_K4_CLOSE -format text

# EXHAUSTIVE re-calibration of one cell
build_tuned/dev/exhaustive_one/run.bat 131072 4
```

## See also

- [docs/dev/wisdom_bridge_predicates.md](wisdom_bridge_predicates.md) —
  why plan-level wisdom beats cost-model extrapolation in general
- [docs/performance/v1_0_results.md](../performance/v1_0_results.md) —
  the production bench numbers this dive zoomed into
- `MEMORY.md` per-radix VTune profiles (R=4 at 86% retiring at K=256
  is the K=256 reference point that the K=4 70% retiring should be
  contrasted against)
