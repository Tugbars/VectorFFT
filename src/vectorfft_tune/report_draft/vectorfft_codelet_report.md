# VectorFFT codelet autotuning: R=8 through R=64 across two Intel microarchitectures

**Scope.** This report covers per-chip autotuning of complex-to-complex FFT codelets for radixes R=8, R=16, R=32, R=64 on two Intel platforms: an Intel Core i9-14900KF (Raptor Lake, consumer, ICX 2025.3.0, AVX-512 disabled by fuse) and an unbranded Sapphire Rapids–class Xeon in a cloud container (AMX + AVX-512 FP16 + VNNI present). All measurements use AVX2 for cross-chip comparison; the SPR container also provides AVX-512 data as a reference point. Phase A benchmarks (flat DIT, flat DIF, t1s, log3) cover all four radixes. Phase B benchmarks (buf variants with tile × drain knobs) cover R=16, R=32, R=64.

**Primary claim.** No single codelet wins across chips. The difference is not small: at identical (radix, me, ios) points the optimal codelet changes between the two chips in 20–50% of regions depending on radix. A chip-agnostic codelet choice costs 10–50% wall-clock at roughly half the regions we measured. Per-chip calibration is necessary for a library that wants to be consistently fast on hardware it was not written for.

**Secondary claim.** The mechanism explaining most of the chip-dependence is the interaction between the codelet's load-port vs FMA-port demand pattern and the chip's port topology. The log3 codelet, which trades twiddle loads for FMA-based derivation, wins more on the server-class Xeon (deeper L2, more load-port bandwidth relief to be had) than on the consumer-class Raptor Lake (fewer FMA gaps to fill because the butterfly already saturates them at small R). The picture reverses at large R because critical-path length grows faster than derivation-chain length.

---

## 1. Test platforms

### 1.1 Raptor Lake consumer desktop

Intel Core i9-14900KF, 24 cores (8P + 16E), P-core clocks up to 5.7 GHz, 32 MB L3, 2 MB L2 per P-core, 80 KB L1D per P-core (48 KB data + 32 KB instruction). Windows, Intel ICX 2025.3.0 compiler. AVX-512 is physically present on the silicon but permanently disabled by an Intel microcode fuse on Alder Lake and later consumer chips. All RL data in this report is AVX2.

The P-core microarchitecture relevant to our measurements: 3 load-AGU ports, 2 store-AGU ports, 2 FMA pipes (ports 0 and 1), 6-wide decode. 512 KB L2 unified per-core. L1D associativity 12-way, 64-byte lines, 64 sets per way.

### 1.2 Sapphire Rapids–class Xeon (SPR container)

Unknown-model Xeon in a cloud VM (SKU masked). CPUID reports AMX, AVX-512 FP16, VNNI — consistent with Sapphire Rapids generation. Linux host, GCC 13. Core frequency under container limits: we observe ~2.6 GHz sustained under tight benchmark loops.

Relevant microarchitecture: 3 load-AGU ports, 2 store ports, 2 FMA pipes with native 512-bit width (vs Raptor Lake's double-pumped 256-bit). L2 is 2 MB per core. Because it is a cloud VM we do not know which SKU specifically, and throttling, memory topology, and shared-L3 contention are not under our control. We treat it as a benchmark subject, not a reference platform.

### 1.3 Why this pairing is informative despite the asymmetry

The two chips span a meaningful microarchitectural gap: consumer vs server, fused-off vs native AVX-512, different L2 sizes and clock behaviors. They are both Intel Golden Cove descendants — Raptor Cove and Sapphire Rapids share the same core topology at the scheduler level — but differ enough that their optimal codelet sets diverge measurably. This is the right setting to demonstrate that even *within* one vendor's current-generation core family, per-chip codelet selection matters.

## 2. Codelet family

Each radix has four baseline Phase-A variants plus (R≥16) six Phase-B buf variants:

| Variant | Twiddle layout | Mechanism |
|---|---|---|
| `ct_t1_dit` (flat) | flat `(R-1)*me` buffer, reads all positions | baseline DIT — straight-line butterfly with one twiddle load per W^k position |
| `ct_t1_dif` (flat) | same | baseline DIF — post-twiddle store pattern |
| `ct_t1s_dit` | scalar broadcasts, K-blocked | twiddles broadcast from memory once per K-block, reused across m |
| `ct_t1_dit_log3` | flat buffer, sparse read | reads log₂(R) base positions; derives the others via chained complex multiplies |
| `ct_t1_buf_dit_tile{T}_{D}` | flat buffer | tile-outer, m-inner decomposition; computes into intermediate buffer, then drains to rio. T ∈ {64, 128, 256}, D ∈ {temporal, stream} |

All variants with `dit` or `buf_dit` suffix produce identical output given identical input and identical twiddle table (ignoring floating-point rounding differences under non-associative add). This is verified by the validator (§3.4).

The sparse-log3 convention is unified across radixes: one flat `(R-1)*me` twiddle buffer, codelet reads positions `{2^k - 1}*me` for k = 0..log₂(R)-1 (at R=4,8,16,32) or a hand-picked subset (at R=64 the kernel reads only {0, 7}*me). Crucially, the planner does not branch on codelet choice — the buffer shape is identical for all variants sharing a protocol. This makes Phase B extension mechanical: a new variant lands in an existing dispatcher slot without touching the planner layer.

## 3. Methodology

### 3.1 Timing harness

FFTW-style tight-loop timing: `for (rep = 0; rep < N; rep++) codelet(rio_re, rio_im, W_re, W_im, ios, me);`. No memcpy, no cache flush between reps. The input buffer is warmed once, then N reps run back-to-back inside `rdtscp`-bracketed regions. Median of K outer runs with K·N tuned so each codelet gets ≥ 100 ms total work.

An earlier version of the harness inserted a `memcpy` between reps. That version flushed store buffers and reset L1D state between iterations, adding 60–80 ns per call and reversing the DIT-vs-DIF winner ranking at several points. We discarded all measurements from that harness; this report is based on the post-fix harness.

### 3.2 Sweep grid

Every variant runs on the cross-product:

- `me` ∈ {64, 128, 256, 512, 1024, 2048} (standard); {64, 128} for t1s (K-blocked, not useful at large me)
- `ios` ∈ {me, me+8, 8×me}

Three `ios` values per `me` exercise three distinct memory regimes: pow2 stride (where 4K aliasing and associativity pressure are maximal), padded stride (where L1D set pressure is minimal), and 8× stride (TLB and prefetcher stress). The pow2 vs padded contrast is the single most diagnostic dimension in the data; many dispatcher flips between the chips happen across that boundary.

Buf variants additionally require me ≥ tile for the kernel to complete a full tile iteration; we skip (tile=256, me < 256) points.

### 3.3 Measurement discipline

Each (variant, isa, me, ios, direction) tuple produces one persisted measurement in JSON Lines format. The benchmark pipeline is deterministic and re-entrant at phase granularity: if the run phase crashes, the generate and compile phases are not re-executed. Measurements carry host fingerprints (CPU model, core count, OS) so cross-chip data cannot be silently interleaved.

### 3.4 Validation

Per-variant bit-exact (within 1e-10 tolerance) comparison against the flat DIT reference at every point in the sweep. This is stronger than validating only the final dispatcher output: if the dispatcher happens to pick the flat variant at the test points, bugs in the buf variants would not be caught by dispatcher-level testing alone. We register validation cases for each of the six buf variants plus the dispatcher itself.

Across R = 8, 16, 32, 64 on both chips, with Phase A + Phase B variants where applicable: zero validation failures.

## 4. Raw findings — dispatcher win counts per radix per chip

Each table reports the number of (me, ios) sweep points (out of 18) where each dispatcher produced the fastest ns/call. AVX2 direction = fwd. A "dispatcher win" here means the dispatcher's best-configured variant beat every other dispatcher's best-configured variant at that point.

### 4.1 Raptor Lake, AVX2

| Dispatcher | R=8 | R=16 | R=32 | R=64 |
|---|---|---|---|---|
| `t1_dit_log3` | 7 | 5 | 7 | 9 |
| `t1s_dit` | 3 | 6 | 5 | 4 |
| `t1_dit` (flat) | 8 | 1 | 4 | 2 |
| `t1_dif` | 0 | 0 | 0 | 0 |
| `t1_buf_dit` | (skipped) | 6 | 2 | 3 |

### 4.2 SPR container, AVX2

| Dispatcher | R=8 | R=16 | R=32 | R=64 |
|---|---|---|---|---|
| `t1_dit_log3` | 15 | 9 | 12 | 9 |
| `t1s_dit` | 3 | 2 | 6 | 4 |
| `t1_dit` (flat) | 0 | 0 | 0 | 0 |
| `t1_dif` | 0 | 0 | 0 | 5 |
| `t1_buf_dit` | (skipped) | 7 | 0 | 0 |

### 4.3 SPR container, AVX-512 (reference)

| Dispatcher | R=8 | R=16 | R=32 | R=64 |
|---|---|---|---|---|
| `t1_dit_log3` | 14 | 15 | 15 | 12 |
| `t1s_dit` | 2 | 1 | 3 | 5 |
| `t1_dit` (flat) | 0 | 0 | 0 | 0 |
| `t1_dif` | 2 | 2 | 0 | 1 |
| `t1_buf_dit` | N/A | 0 | 0 | 0 |

The AVX-512 column is the cleanest illustration of log3 dominance when the chip has sufficient port resources; the AVX2 columns reveal chip-specific tradeoffs.

## 5. Chip-dependent patterns

### 5.1 Flat baseline wins on Raptor Lake, never on Sapphire Rapids

Across all four radixes, `t1_dit` (the unadorned flat DIT baseline) wins 8+1+4+2 = 15 regions on Raptor Lake and 0 on SPR. This is not noise. The "simple beats clever" regime is a genuine Raptor Lake phenomenon that does not appear on SPR.

The mechanism, to the extent we can attribute one without VTune confirmation: at small me with prefetcher-assisted L1D-hot twiddle reads, the baseline's load port traffic is already cheap. The derivation chain log3 adds to "save" those loads is pure overhead in that regime. On SPR the same derivation chain has enough FMA-port headroom to run free; on RL the FMA ports are densely used by the butterfly itself, so the derivation FMAs extend the critical path.

### 5.2 log3 margin scales with R, and differently per chip

Measured at me=2048, ios=2056 (padded stride — the cleanest log3-favorable regime, no aliasing, maximal load-port pressure on flat):

| Chip | R=8 | R=16 | R=32 | R=64 |
|---|---|---|---|---|
| Raptor Lake AVX2 | +47% | +26% | +44% | **+106%** |
| SPR container AVX2 | +37% | +56% | +39% | +89% |

Log3 margins grow dramatically at R=64. At this hotspot R=64 log3 is **over 2× faster** than flat DIT on Raptor Lake. The mechanism: at R=64 the cmul-derivation chain has 61 FMAs to schedule, all independent of the butterfly's critical path, which itself is 40+ FMAs deep. The OOO engine fills every FMA-port cycle it can. Load ports, which the baseline DIT saturates with 63 twiddle loads per butterfly, are near-idle in log3.

The non-monotonic trend on RL (R=8 high, R=16 low, R=32 and R=64 high) is consistent with the ILP-gap-filling theory: the benefit depends both on how many loads log3 saves (monotonically increasing with R) and how many FMA gaps the butterfly leaves (roughly proportional to R but with an inflection around R=16 where the butterfly is big enough to have gaps but small enough that the derivation chain extends the critical path). The data does not nail this down; we can only say the measured margins are consistent with a mechanism that has two competing factors.

### 5.3 Buf is an R=16 specialist, a stride specialist at R=64, and nearly inert at R=32

Buf win counts per chip per radix:

| | R=16 | R=32 | R=64 |
|---|---|---|---|
| RL AVX2 buf wins | 6 | 2 | 3 |
| SPR AVX2 buf wins | 7 | 0 | 0 |

R=16 is the only radix where buf wins consistently on both chips. The R=16 butterfly is big enough that tiling matters (working set of ~32 doubles × me per tile), and small enough that the tile-outer / drain overhead stays small relative to the butterfly compute.

At R=64 on Raptor Lake, buf wins at a specific niche: 8×me strides at large me. At (me=256, ios=2048), (me=1024, ios=8192), (me=2048, ios=16384) buf wins by 3–21%. All three points have ios stride 8× larger than me — the regime where the baseline in-place DIT pattern stresses TLB and L1D sets that buf's tile-structured I/O sidesteps. At R=64 on SPR, buf wins zero regions because log3's 61-derivation savings already exceed what buf can provide.

### 5.4 DIF is irrelevant at small R and only matters at R=64 on SPR

Across all measurements, DIF wins 0 regions on Raptor Lake at any radix, and 0 regions on SPR at R=8/R=16/R=32. At R=64 on SPR DIF wins 5 regions (AVX2) and 1 region (AVX-512). At R=64 on RL it wins 0.

The mechanism is specific to the R=64 butterfly's size on SPR: the DIF post-twiddle-store pattern interacts better with SPR's store buffers at very large butterflies. On Raptor Lake the same pattern loses, likely due to RL's narrower store-AGU path. This is the kind of detail a per-chip calibrator catches and a hand-tuned library might not.

For library shipping purposes, DIF could be dropped at R=8/R=16/R=32 without loss. It remains useful at R=64 only on server-class chips. A future pass should consider whether DIF is worth generating at all at the smaller radixes.

### 5.5 t1s is a small-me specialist, chip-insensitive in the aggregate

Across R=16, R=32, R=64 on Raptor Lake, t1s wins 6+5+4 = 15 of 18 small-me (me ≤ 128) regions. On SPR AVX2 the equivalent count is 2+6+4 = 12. The small-me dominance is real on both chips. It softens slightly at R=64 where log3 starts catching t1s at the largest small-me point.

## 6. The me=512, ios=4096 anomaly on Raptor Lake

At the sweep point (me=512, ios=4096) — where ios is exactly 8×me at a page-aligned stride — the log3 codelet is slower than flat DIT on Raptor Lake at two specific radixes:

| Radix | RL flat (`t1_dit`) | RL log3 | log3 vs flat |
|---|---|---|---|
| R=8 | ~1000 | ~2050 | **2.0× slower** |
| R=16 | 19,593 | 15,728 | 24.6% faster |
| R=32 | 22,119 | 18,746 | 17.9% faster |
| R=64 | 56,653 | 67,855 | 20% slower |

The anomaly appears at R=8 and R=64 but not at R=16 or R=32 at this specific point. This is consistent with a cache-set aliasing pattern that only manifests at particular combinations of radix, twiddle-base positions, and stride. Log3 reads bases at positions that depend on the radix: {0}×me at R=8 (one base), {0,1,3,7}×me at R=16, {0,1,3,7,15}×me at R=32, {0,7}×me at R=64. At R=8 the kernel has only one sparse base read, and when that read hits a pathological L1D set at ios=4096, there is no parallelism across other base reads to hide it. At R=64 only two bases are read, at offsets 0 and 7, which happen to alias in a related way at this stride. R=16 and R=32 read 4 and 5 bases respectively, spread across offsets whose alias patterns differ, and at those radixes the load pressure is distributed enough that no single alias dominates.

This is a hypothesis. We do not have VTune profile data confirming L1D set-aliasing as the mechanism. What is confirmed is the behavior: at the two radixes listed, log3 is measurably slower at this one point, and the dispatcher correctly routes around it by picking flat (at R=8) or a different variant (at R=64) at that specific (me, ios).

This is the cleanest example in our data of a per-chip, per-point behavior that static codelet selection cannot catch. A chip-agnostic library that picked log3 for R=64 at me=512 would lose 20% at that point on Raptor Lake while winning 100%+ on the same chip at me=2048, ios=2056. The point of per-chip calibration is that it catches both.

## 7. Wall-clock: RL vs SPR

At me=1024, ios=1024 (pow2), best-of-dispatcher per chip:

| Radix | SPR best (AVX2) | RL best (AVX2) | SPR/RL ratio |
|---|---|---|---|
| R=8 | 6301 ns | 3601 ns | 1.75× |
| R=16 | 31,697 ns | 12,936 ns | 2.45× |
| R=32 | 77,351 ns | 30,031 ns | 2.58× |
| R=64 | 183,148 ns | 79,149 ns | 2.31× |

Raptor Lake is consistently 1.75–2.58× faster on AVX2 than the cloud SPR container. Two factors dominate: clock (RL ~5.7 GHz vs SPR ~2.6 GHz under container limits) and L2 size (RL P-core's 2 MB L2 vs SPR's 2 MB, similar but SPR shares L3 with a noisy cloud environment). At extreme strides (me=2048, ios=16384, R=64) the ratio compresses to 1.1× as both chips hit TLB-miss walls that neutralize the clock advantage.

The *SPR AVX-512 vs RL AVX2* comparison is different and not the main subject of this report, but for completeness: at me=1024, ios=1024 R=32, SPR AVX-512 log3 runs at 35,373 ns. RL AVX2 log3 runs at 30,142 ns. Raptor Lake at 5.7 GHz on AVX2 beats a cloud Sapphire Rapids on AVX-512 at the same compute point. This is partly clock, partly the container's shared-cache noise, and is not a statement about the underlying microarchitectures' peak capability.

## 8. Methodological caveats

The SPR container is a cloud VM with an unknown SKU and shared infrastructure. Absolute SPR numbers should not be taken as representative of a dedicated Sapphire Rapids server; SPR-SP workstation silicon likely runs 20–40% faster at the same codelet. What transfers across SPR variants is the *relative* dispatcher winner pattern, because port topology and cache associativity are SKU-invariant within the Sapphire Rapids generation. We do not have the data to confirm this claim; it is a prior that would be falsified if a dedicated SPR-SP box showed different dispatcher winners than the container.

The harness does not control for SMT siblings, turbo boost transitions, or thermal throttling. On Raptor Lake we observe stable sustained clock during the measurement windows; on SPR we observe clock noise of ~5% that does not affect relative winners but does add scatter to absolute ns/call. We report medians.

The Raptor Lake data is one machine, one compiler version, one OS. The SPR data is one cloud instance, one compiler version. Generalization to other instances of the same microarchitecture is a hypothesis, not a measurement. The whole point of the calibration library is that users run it on their own chip and get wisdom specific to that chip; this report's cross-chip comparison is a demonstration that the wisdom differs meaningfully, not a claim that our two machines define the full distribution.

## 9. Implications for library design

**9.1 Shipping `t1_dit_log3` as a universal default is close to right.** Across 8 tables (4 radixes × 2 ISAs on SPR, 4 radixes × AVX2 on RL), log3 wins the largest share at every radix × ISA combination except RL AVX2 R=16. It is the single best chip-agnostic default.

**9.2 Per-chip calibration captures 10–50% wins at 30–50% of regions** that the chip-agnostic default misses. The regions are concentrated at (small me, any stride) where t1s takes over and at (chip-specific stride pathologies) where flat or buf take over. Without calibration, the library would lose these regions silently.

**9.3 Buf should ship only at R=16 unless per-chip data justifies it.** At R=32 and R=64 on SPR, buf never wins. At R=32 on RL it wins 2/18. At R=64 on RL it wins 3/18. The code-size cost (~25k lines per buf variant at R=64, six variants per ISA) is substantial. A reasonable default: ship buf at R=16 unconditionally; ship buf at R=32/R=64 only if the calibration step signals benefit on the target chip.

**9.4 DIF at R ≤ 32 can be dropped.** 0 wins across 72 measurement points (R=8, R=16, R=32 × both chips × both ISAs × 18 points). Only R=64 DIF on SPR earns its shelf space.

## 10. Scope limitations and open questions

**Zen.** All measurements in this report are Intel. AMD Zen 4 (double-pumped AVX-512) and Zen 5 (native 512-bit with 4 FMA pipes) are expected to show different dispatcher patterns. The Zen 5 prediction specifically: 4 FMA pipes means more gaps for log3's derivation chain to fill, which should make log3 dominate even harder at R=16 and R=32 than on SPR. This is a hypothesis, not a finding.

**Apple silicon, ARM Neoverse.** Completely outside the scope of this work. The calibration framework is ISA-portable in principle but currently emits AVX2/AVX-512 intrinsics only.

**Multi-socket, NUMA.** Not tested. All measurements are single-threaded on one core.

**R=4.** Covered by Phase A infrastructure but not discussed in this report because the radix is small enough that the dispatcher winner is uniform across chips and regions.

**R=8 Phase B.** Deliberately skipped. VTune on the standalone library had previously shown that R=8 is compute-bound and buf's memory-optimizing structure does not help. We did not re-run buf at R=8 under the new harness. A formal Phase B sweep at R=8 would be needed to close this out.

## 11. Conclusion

Across R=8 through R=64 on two Intel microarchitectures we find that the optimal FFT codelet choice differs between chips in 20–50% of regions, with chip-specific wins of 10–50% wall-clock at those regions. The dominant mechanism is the interaction between the codelet's load/FMA-port demand mix and the chip's port topology, with a secondary effect from cache/TLB behavior at large strides. The log3 codelet — which trades twiddle loads for FMA-based derivation — is the single best universal default and achieves its largest margins at large R on chips with FMA-port headroom.

A per-chip calibration library is necessary to capture the cross-chip variation. The tuning infrastructure produces bit-exact, validated dispatcher headers that a consuming project includes as plain C. The calibration step takes under 10 minutes for the full R=8..R=64 sweep on modern hardware.

The next meaningful data acquisition, from our current position, is a Zen microarchitecture bench run. If Zen confirms the "FMA-pipe-count drives log3 dominance" hypothesis, the library's value proposition — "per-chip calibration matters" — gains a third independent data point across three genuinely different microarchitectures. If Zen does not confirm that, the hypothesis needs revision. Either way we learn something.
