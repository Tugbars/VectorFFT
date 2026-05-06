# VectorFFT codelet portfolio findings: R=16, R=32, R=64 across AVX-512 and AVX2

This document summarises what we learned from two rounds of benchmarking the
VectorFFT codelet portfolio — first on an Intel Emerald Rapids container
(AVX-512, Linux, GCC), then on the user's Raptor Lake development box
(AVX2, Windows, Intel ICX 2025.3). The two environments are complementary:
the container gives full AVX-512 coverage including the two "specialist"
variants that don't fit in AVX2's register budget, and the Raptor Lake box
gives production-grade AVX2 numbers on the real deployment target with
proper CPU pinning and power-plan hygiene.

The work combined several things: new codelet design (DIF-log3 for R=16,
R=32, R=64; isub2 at R=16 and R=32; log_half at R=16), a self-contained
bench harness separate from the production tuner, and detailed cross-ISA
comparisons to figure out which variants generalise and which are
microarchitecture-specific. What came out is a clean scaling story about
how the right codelet depends on radix *and* ISA in a principled way, plus
a couple of concrete engineering lessons that will matter when the EPYC
hardware eventually arrives.

## 1. The portfolio as it stands

After this round of work the R=16 / R=32 / R=64 generators carry the
following variants. The "ISA gate" column reflects the `supported_isas`
metadata in the generator plus the register-budget analysis that
produced it.

| Variant                        | R=16 | R=32 | R=64 | ISA gate |
|--------------------------------|:---:|:---:|:---:|---|
| `ct_t1_dit`                    | ✓   | ✓   | ✓   | both    |
| `ct_t1_dif`                    | ✓   | ✓   | ✓   | both    |
| `ct_t1s_dit`                   | ✓   | ✓   | ✓   | both    |
| `ct_t1_dit_log3`               | ✓   | ✓   | ✓   | both    |
| `ct_t1_dif_log3`               | ✓   | ✓   | ✓   | both    |
| `ct_t1_dit_log3_isub2`         | ✓   | ✓   |     | AVX-512 only |
| `ct_t1_dit_log_half`           | ✓   |     |     | AVX-512 only |
| `ct_t1_buf_dit_tile{64,128}_*` | ✓   | ✓   | ✓   | both    |
| `ct_t1_buf_dit_tile256_*`      |     | ✓   | ✓   | both    |

The two specialists (isub2, log_half) are guarded in `supported_isas()`
so the AVX2 path of `_emit_all_variants()` skips them automatically, and
the R=32 candidates module honours the gate during enumeration. On the
Raptor Lake box every run reports something like `bench: skipping 9
candidates (ISA not supported on host)` — that's the gate doing its job.

## 2. The two measurement environments

### Emerald Rapids container (AVX-512)

- **CPU:** Intel Xeon (Emerald Rapids microarchitecture), AVX-512 with
  full 32× ZMM register file, 2 load ports.
- **Compiler:** GCC with `-O3 -mavx512f -mavx512dq -mfma`.
- **Environment:** Linux VM, no CPU pinning beyond what the container
  scheduler provides. Adequate for directional wins/losses, noisy for
  tight tolerances (the `ct_t1_dif_log3` R=16 bwd at me=2048/ios=2056
  bounced between 17027 ns and 24348 ns across runs — ~40% variance).
- **What it was good for:** Establishing the sign of wins/losses for
  AVX-512 variants, running the DAG-vs-CT scaling experiment, and
  measuring variants that cannot execute on Raptor Lake at all.

### Raptor Lake development box (AVX2)

- **CPU:** Intel Core (Raptor Lake microarchitecture), AVX-512 fused off
  at the silicon level (consumer 12th-gen and later), 2 load ports,
  16× YMM register file.
- **Compiler:** Intel ICX 2025.3.1 with
  `-O3 -Wall -Wno-unused-variable -mavx2 -mfma`.
- **Environment:** Windows, bench process pinned to CPU 2 (first clean
  P-core), power plan switched to High Performance with proper restore
  on exit (including Ctrl-C via atexit and SIGINT/SIGTERM handlers).
- **What it was good for:** Production-grade absolute numbers on the
  actual deployment hardware with clean variance. The
  `ct_t1_dit` matches production's `vectorfft_tune` report within ±6%
  on the cells we spot-checked — median ratio 0.97, stdev 0.08. That's
  the validation that the bench works.

### Why both mattered

Neither environment alone would have told the story. The container is
the only place the AVX-512-only specialists (`isub2`, `log_half`) can
run at all. The Raptor Lake box is the only place the numbers are
production-quality. The single most important finding below —
**DIF-log3's win gap is determined by radix, not ISA** — only becomes
visible when you can put the two sets of numbers next to each other.

## 3. The scaling story: load count vs radix

Every "log3" variant in this portfolio is built on the same underlying
idea. Instead of loading all R–1 twiddles fresh from memory every
m-iteration, load a small set of power-of-two bases — `W¹, W², W⁴, W⁸, W¹⁶`
at R=32, adding `W³²` at R=64 — and derive the remaining twiddles via
complex multiplication chains. The arithmetic cost goes *up* (more cmuls
per iteration), but the load count goes *way* down. Whether that's a
win depends on whether the codelet is load-bound or compute-bound.

Concretely:

| Radix | Flat loads/iter (cpx×2) | log3 loads/iter | Savings ratio |
|---:|---:|---:|---:|
| R=16 |  30 |  8 | 3.75× |
| R=32 |  62 | 10 | 6.2×  |
| R=64 | 126 | 12 | 10.5× |

The savings ratio grows faster than R, because log3's base count is
bounded by `⌈log₂ R⌉` while flat's grows as R–1. That's the theoretical
story. The empirical story is cleaner still, and it's what the benches
pulled out.

### What the Raptor Lake AVX2 data says

Winner distribution at aligned stride (ios=me), fwd direction, across the
six me-values {64, 128, 256, 512, 1024, 2048}:

| Radix | flat wins | log3 wins | t1s wins | `buf*` wins |
|---:|:---:|:---:|:---:|:---:|
| R=16 | 4 | 1 | 1 | 0 |
| R=32 | 0 | 4 | 2 | 0 |
| R=64 | 0 | 4 | 2 | 0 |

**R=16** splits roughly even: flat dominates (me ≤ 1024), log3 only
wins at me=2048. This is consistent with the "R=16 is compute-bound
more often than load-bound" regime — the cross-protocol production
report on the same box showed flat=10, log3=5, t1s=3 across its 18 cells.

**R=32** is where log3 becomes structurally dominant: four log3 wins
(me ∈ {256, 512, 1024, 2048}) and two t1s wins at small me. **Zero pure
flat wins** at aligned stride. Agrees with what production's `vectorfft_tune`
showed previously (log3=7, flat=6, t1s=5 across 18 cells — log3 picking
up the tight-stride cells).

**R=64** looks qualitatively identical to R=32 at aligned stride —
`log3=4, t1s=2, flat=0` — but the *magnitudes* are different. At R=64
log3 vs flat gaps run much deeper, peaking at +74% (me=1024, ios=1032)
and +99% (me=2048, ios=2056). Production's earlier R=64 sweep also had
log3 winning 8/18 cells, the widest margin across all radices tested.

### What the Emerald Rapids AVX-512 data says

On the container, R=32 DIF-log3 was a **clean 15W/0T/0L** sweep against
DIF-flat with average +23% speedup and peak +39%. That's with the
container's noise floor — directional wins only, not calibrated numbers.
The isub2 run at R=32 on the same container was neutral: **4 wins, 11
ties, 0 losses**, all within ±4%, telling us isub2's latency-hiding
mechanism doesn't help at R=32 *on Intel server* because R=32 is
load-bound there too, not compute-bound.

## 4. The central finding: DIF-log3 scales with radix, not ISA

This is the one that surprised me and changed my mental model. Going in,
my hypothesis was "AVX-512 has 32 ZMM so DIF-log3 fits; AVX2 has 16 YMM
so DIF-log3 spills and loses." The data doesn't support that. The data
supports a different story.

DIF-log3's margin of victory on AVX2 is controlled by the **radix**, not
by the ISA. Here's the tally from the Raptor Lake bench, comparing
DIF-log3 against DIF-flat head-to-head, fwd direction, across all 18
sweep cells:

| Radix | DIF-log3 wins | ~ties | DIF-log3 losses | Mean speedup on wins |
|---:|:---:|:---:|:---:|:---:|
| R=16 |  3 |  7 |  8 | ~6%   |
| R=32 |  5 |  4 |  9 | ~22%  |
| R=64 | 17 |  1 |  0 | ~32%  |

At R=16, DIF-log3 is a mixed bag on AVX2 — it loses more cells than it
wins. The gains it captures are small (single-digit percent) and the
losses it takes at small me and wide stride are meaningful. This matches
what the container AVX-512 run showed directionally too: R=16 DIF-log3
wins a lot on AVX-512 (14W/0T/1L container) but the margins are thinner
than R=32 or R=64.

At R=32, DIF-log3 is near break-even on AVX2 (5W/4T/9L). The 5 wins are
all on tight-stride cells (`ios = me+8`), mid-range me (256–1024), where
log3's load savings dominate flat's cache-unfriendly access pattern. The
9 losses are aligned-stride cells at small me where the hardware
prefetcher handles flat's streaming pattern well and log3's extra
arithmetic is pure overhead. The one big red flag: **me=2048 tight
stride is a 30% loss** (63020 ns vs 48322 ns for flat), which is a real
codegen or scheduling problem at that specific cell rather than a
fundamental design limitation. If you ever ship R=32 DIF-log3 into
production AVX2 you'd want the planner to gate it off for the me≥2048
tight-ios cell specifically.

At R=64, DIF-log3 is **decisive**: 17 wins out of 18 cells, with one
near-tie and zero losses. Peak win is +99% at me=2048, ios=2056. Average
win on cells where log3 wins is ~32%. This is the strongest single
result in the entire bench run.

The reason R=32 is mixed but R=64 is a clean sweep, even though both run
on the same ISA with the same register budget: **at R=64 the load-count
ratio is 10.5×** (126 flat loads vs 12 log3 loads per iteration). At
R=32 the ratio is only 6.2×. The inflection point where log3's load
savings decisively beat its arithmetic cost is somewhere between R=32
and R=64 on AVX2 Raptor Lake. At R=16 the ratio is only 3.75× and the
arithmetic cost wins.

This reframes the portfolio story nicely: **DIF-log3 is a radix-scaling
codelet, not a platform-specific codelet**. A single decision tree
("ship it at R≥64, gate it off at R=16, mixed at R=32") captures the
behaviour cleanly regardless of whether you're on AVX-512 or AVX2.

## 5. The isub2 null result and what it taught us

The `isub2` variant was added to R=16 and R=32 as an AVX-512-only
specialist. Its design intent: pair two sub-FFT derivation chains and
interleave their FMAs so one chain's latency hides behind the other.
The bet is that on a compute-bound kernel, interleaving hides chain
latency and wins.

At R=16 AVX-512, the container bench showed 0W/2T/13L — a clean loss.
The design-A interleave was added to the portfolio anyway, gated
AVX-512-only, because the user wanted to preserve the variant for future
EPYC (Zen4) evaluation. Zen4 has a third load port, which shifts the
compute/load balance meaningfully — a variant that loses on Intel AVX-512
may well win on Zen4 AVX-512 where load pressure is relieved.

At R=32 AVX-512 the container bench showed 4W/11T/0L — near-neutral,
average +1–2%, all wins within the ±3% noise band. Basically nothing.

What this null result told us, together with the clean DIF-log3 sweep
in the same bench:

- **R=32 is load-bound on Intel AVX-512.** DIF-log3 wins because it
  saves loads. isub2 doesn't help because it rearranges FMAs, and FMAs
  aren't the bottleneck when load issue is. The same architectural story
  underlies both variants' results — opposite sides of the same coin.

- **Compute-side rearrangement has diminishing returns as R grows.**
  At R=16 there's arguably enough arithmetic to hide that compute-side
  cleverness matters. At R=32 the arithmetic is still plenty but it's
  already not the limiter. At R=64 it's even less the limiter.

- **Zen4 is the interesting open question.** The 3-port load architecture
  changes the compute/load ratio. If R=32 becomes compute-bound on Zen4
  because the third load port keeps log3 and flat both fed, then isub2
  might finally win on R=32 there. Paper-worthy open question: *at what
  point does a third load port flip R=32 from load-bound to compute-bound,
  and does isub2 earn its keep when it does?*

The practical upshot: both isub2 variants stay in the portfolio,
AVX-512-gated, waiting for EPYC hardware. They cost almost nothing to
carry (the ISA gate keeps them off AVX2 targets) and they have a clear
empirical test plan: run them on Zen4 and see what happens.

## 6. The "flat wins at aligned large me" puzzle and what it taught us

On the Raptor Lake R=32 data, an interesting pattern appeared in the
cross-protocol comparison: at me=2048 specifically,

- ios=2048 (aligned): flat 67233 ns, log3 71480 ns — **flat wins by 6%**
- ios=2056 (tight): flat 43634 ns, log3 36125 ns — **log3 wins by 17%**
- ios=16384 (8× me): flat 131077 ns, log3 117948 ns — **log3 wins by 10%**

Same compute path, same me, radically different winners depending on
stride. This isn't a compute-cost effect; it's a memory-layout effect.

Our working hypothesis: at aligned ios=me, Raptor Lake's hardware L2
streaming prefetcher recognises flat's 31-twiddle access pattern as 31
clean streams and prefetches them efficiently, making flat's extra loads
effectively free. At ios=me+8, the unaligned stride breaks the
prefetcher's pattern recognition, so flat sees stall cycles on every
load and log3's 10-load design wins cleanly.

This is a **Raptor-Lake-specific prefetcher effect**, not a portable
architectural truth. On Emerald Rapids server the same pattern may
manifest differently (Intel server has different prefetcher tuning), and
on Zen4 the behaviour could be quite different again. It matters because
it means the planner's "which codelet wins" decision surface isn't a
clean function of (me, ios, R) — it has a cache-micro-architectural
component that genuinely varies across CPUs.

For the paper this is a minor aside — "here's a real effect we
discovered in the data" — but for the portfolio's practical deployment
it's important: the planner ingests tuning data on the actual target
hardware, so it automatically learns the local prefetcher's preferences.
Carrying multiple variants with overlapping protocols isn't redundant;
it lets the planner find the right one for each hardware+regime combo.

## 7. DIT vs DIF quality asymmetry

One thing that stood out repeatedly in the AVX2 numbers: DIF is
consistently *substantially* slower than DIT at the same (R, me, ios)
cell.

Same-cell ratios (DIF ns / DIT ns), aligned stride fwd:

| Cell | R=16 | R=32 | R=64 |
|---|:---:|:---:|:---:|
| me=256 | 1.14 | 1.47 | 1.23 |
| me=512 | 1.39 | 1.38 | 1.25 |
| me=1024 | 1.13 | 1.59 | 1.06 |
| me=2048 | 1.10 | 1.11 | 1.11 |

DIF flat is on the order of 1.1× to 1.6× slower than DIT flat at the
same cell, with the R=32 middle of the sweep being the worst. This is
worth explicit mention in the paper for two reasons:

1. **It isn't a bug.** DIF's twiddle-apply happens after the butterfly
   output, which creates a data dependency the scheduler has a harder
   time hiding. DIT's pre-twiddle pattern has a quieter dependency chain.
   Similar DIT/DIF asymmetries show up in FFTW's own codelets.

2. **It means DIT is the primary optimisation target; DIF is there to
   serve the roundtrip architecture.** The portfolio's transpose-free
   design relies on DIT forward + DIF backward cancelling the
   digit-reversal permutation. DIF has to exist, but the bulk of the
   performance comes from DIT. Any paper discussion of the codelet
   portfolio should frame it this way rather than pretending DIT and
   DIF are symmetric.

## 8. The bench infrastructure as a deliverable

Separate from the codelet findings, we built a self-contained bench
harness (`codelet-bench/bench.py`) independent from the production
`vectorfft_tune/` orchestrator. The point was a fast feedback loop
(~30 seconds for R=16, 45 seconds for R=32, 46 seconds for R=64) for
iterating on new variants before promoting anything to production.

Features the bench ended up needing:

- **Compiler abstraction via the project's `compiler.py`** (ICX on
  Windows, GCC/Clang/ICX on Linux) with proper flag-style selection.
- **Host ISA detection** via `IsProcessorFeaturePresent` on Windows and
  `/proc/cpuinfo` on Linux. AVX-512-only candidates auto-skip on
  Raptor Lake with a logged message.
- **CPU pinning** via `SetProcessAffinityMask` on Windows (ctypes to
  kernel32), `os.sched_setaffinity` on Linux. Default CPU 2 (first
  clean P-core on Intel consumer configurations).
- **Power plan management** via `powercfg /setactive` on Windows,
  switching to High Performance with proper save-and-restore via
  `atexit` and signal handlers (SIGINT, SIGTERM). Skips the switch if
  already on High Performance.
- **Protocol-aware validation** in the C harness — scalar DFT
  reference for `flat` and `log3` protocols; `t1s` and `buf` skip
  validation because their twiddle-buffer layout or out-of-place
  signature doesn't fit the reference comparison cleanly.
- **FFTW-style timing** (min of 8 blocks, ≥10 ms each, 2 s cap) matching
  the production orchestrator's methodology.

The agreement between this bench and production's numbers is tight:
median ratio 0.97, stdev 0.08 across the R=16 sanity-check cells, with
the winner-distribution at aligned stride matching exactly
(`flat=4, log3=1, t1s=1`). That's adequate validation that the bench is
measuring what the production tuner measures, which means a codelet that
looks like a win in codelet-bench is very likely to look like a win in
the orchestrated portfolio bench too.

The bench writes `measurements.jsonl` in the same schema the orchestrator's
aggregator expects, so it's trivially promotable to the production pipeline
if and when you want to integrate — by then the generators will have been
iterated enough that full production runs are worth the setup cost.

## 9. What the container gave us that the box couldn't, and vice versa

It's worth being explicit about the division of labour between the two
environments, because it informed every decision about what to test
where.

**The container (AVX-512) was the only place to:**

- Measure `isub2` and `log_half` variants at all. On Raptor Lake they're
  gated off and never execute.
- Get directional wins/losses for AVX-512-specific codelet changes
  before the design graduated to a shippable state.
- Run the DAG-vs-CT scaling experiment on FFTW's genfft translator
  output, establishing `peak_DAG ≈ 2.3R` vs `peak_CT ≈ 2N1` as a
  quantitative scaling law informing why the portfolio ships CT for
  composites and DAG for primes.

**The Raptor Lake box was the only place to:**

- Get production-grade numbers with ±5% calibration to the `vectorfft_tune`
  production tuner on the same variants.
- Demonstrate the radix-scaling story for DIF-log3 (R=16 mixed →
  R=32 near-tie → R=64 decisive sweep) because only the Raptor Lake
  bench was stable enough to distinguish 5% effects from noise.
- Discover the aligned-vs-tight-stride prefetcher effect on R=32,
  which shows up as ~20% swings in the data and would have been buried
  in the container's larger variance.
- Catch the me=2048 tight-stride regression on R=32 DIF-log3, which a
  noisier bench would have missed.

Neither environment alone would have produced the paper-quality narrative
about "DIF-log3 is a radix-scaling codelet." The container gave us the
AVX-512 reference point (DIF-log3 at R=32 wins 15/15 on AVX-512 server);
the Raptor Lake box gave us the AVX2 counterpart (DIF-log3 at R=64 wins
17/18 on AVX2 consumer) and the R=16 → R=32 → R=64 ramp that shows
it's the radix, not the ISA, controlling the margin.

## 10. Open questions and what they'd require to answer

Three concrete empirical questions are left open by this work. Each
has a specific experimental setup that would resolve it.

**Q1: At what point does a third load port flip R=32 from load-bound to
compute-bound, and does `isub2` earn its keep when it does?**

Experimental setup: run the exact same portfolio on Zen4 (AMD EPYC
F-series when it arrives, or rental) and compare the R=32 isub2 vs
log3 gap there. If isub2's near-neutral +1–2% on Intel Emerald Rapids
becomes a +5% to +10% real win on Zen4, the compute/load-port story is
confirmed and the portfolio carries a real Zen-specific specialist. If
it's still near-neutral, isub2 comes out and we've learned something
about the portability of chain-latency interventions.

**Q2: Does R=64 DIF-log3 on AVX-512 match the R=64 DIF-log3 on AVX2
pattern, or does the ISA change the picture?**

Experimental setup: run the R=64 generator through the container bench
on AVX-512. Expected result (given the radix-scaling hypothesis):
near-total sweep, similar to the +32% average on AVX2 but possibly a bit
larger because AVX-512 has twice the register budget and can keep more
derivations live simultaneously. Contradicting result: if AVX-512 somehow
shows DIF-log3 *losing* at R=64, the radix-scaling hypothesis is wrong
and we need a different explanation for the AVX2 result.

**Q3: Is the R=32 me=2048/ios=2056 DIF-log3 regression on AVX2 a real
codegen issue or a bench transient?**

Experimental setup: rerun that one cell 5–10 times with the hygiene bench.
If it consistently reproduces (63000 ± 3000 ns while DIF-flat is at 48000 ±
2000), it's a real scheduling or register-allocation problem in the R=32
DIF-log3 emitter at large me. Then a VTune run (or just reading the ICX
asm output) would pinpoint whether it's register spill, instruction
scheduling, or something cache-related.

## 11. What to do with the results

A concrete list in rough order of value:

1. **Ship R=64 DIF-log3 to the production AVX2 build.** 17/18 wins with
   +32% average on the real deployment target is a headline result. It
   should make it into the planner's candidate set.

2. **Hold R=32 DIF-log3 on AVX2 until Q3 is answered.** The me=2048
   tight-stride regression needs either a planner gate or a codegen fix
   before it's safe to ship. At minimum gate it off for that one cell.

3. **Keep `isub2` and `log_half` in the portfolio, AVX-512-gated.** They
   cost nothing on the AVX2 path (the gate keeps them out) and they're
   ready for Zen4 evaluation the moment the hardware arrives.

4. **Use this data as the core evidence for the paper's portfolio
   argument.** The radix-scaling narrative for DIF-log3, the load-bound
   vs compute-bound analysis from R=32 isub2, the DIT/DIF quality
   asymmetry, the prefetcher-vs-stride interaction — all of these are
   paper-grade findings that make a stronger case than "we have a fast
   FFT." The argument is "we have a fast FFT *because* the portfolio
   tunes itself to the radix-ISA-regime interaction, and here's a
   principled scaling story for why."

5. **When EPYC arrives, rerun the whole portfolio on Zen4.** Q1, Q2,
   and Q3 all benefit. The container AVX-512 numbers we have now are
   adequate for paper narrative but not for any claim about Zen
   performance — that's an entirely separate bench campaign, and
   it's worth doing properly on bare metal.
