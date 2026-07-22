# VectorFFT session report — §6a25 → §6a30
**The z-contract completion, the jit reckoning, and the 2D campaign opening**
Container session, July 16 2026 · 1 vCPU, AVX2+AVX512 · MKL sequential AVX2 baseline
All comparisons same-process interleaved medians unless noted. Container weather this session was exceptionally violent (±25% cross-process swings observed) — cross-run ratio deltas are cited nowhere as evidence.

---

## Executive summary

| § | item | verdict | headline number |
|---|---|---|---|
| 6a25 | Model-B fused last-stage terminator (#8) | **CLOSED, measured negative** | avx512 edition: parity-at-best (median +1.7% in noise band), +2.3…+8.2% elsewhere |
| 6a26 | rfft native z terminator (fwd) | **SHIPPED, BIT-exact** | z/MKL restored ≥1.0 on rfft lead cells; −13…−32% at (100000,4) |
| 6a27 | rfft PIC codelet set + jit verdicts | rsp **fixed**; natural jit **CLOSED negative**; natural-z jit built, bit-exact, **CLOSED slow** | jit −8.9% vs generic at K=4; jit_z +63% vs jit-split |
| 6a27+ | Post-ship regression audit | **no regressions**; c2r jit newly active, measured **parity** | c2r jit −0.4…+4.0% band = inside ±3% |
| 6a28 | c2r native z-in initiator | **SHIPPED, first-attempt 12/12 BIT** | bwd z/MKL: 0.965× → parity at (2000,4); −10.9% at 100K |
| 6a29 | 2D r2c/c2r z contract | **SHIPPED v1**; pre-existing **SEGFAULT fixed** | gate 18/18 |
| 6a29+ | v1.0 reconciliation (claim challenged) | "first 2D numbers" **withdrawn**; **no regression**; finding restated precisely | c2c control 1.03× today on v1.0 config; MKL real/complex = 2.24× |
| 6a30 | 2D wrapper elimination + fused z | **SHIPPED** | split −15/−16% fwd; **2D z tax eliminated** (parity) |

---

## §6a25 — Model-B (#8): closed as measured negative

The attribution overturned §6a20's diagnosis: a stub-ls arm showed the scalar scaffold + specials floor at **11.8 µs** — the deficit was the avx2 codelet (~260 µs vs ~179 µs for model-A's last-stage + postprocess). The avx512 edition (one generator invocation; the mode had anticipated 32-zmm pressure) closed the width gap but not the structural one:

| cell | inner shape | A (µs) | B-avx512 (µs) | delta |
|---|---|---|---|---|
| (512,256) ×5 instances | r2 r4 r4 r8 | 344–359 | 350–363 | −1.8, +3.7, +2.0, +0.3, +1.7% → median **+1.7%** (lone win = noise outlier) |
| (256,256) | r4 r4 r8 | 132.9 | 141.6 | **+6.6%** |
| (1024,256) | r4 r4 r4 r8 | 815.8 | 834.4 | +2.3% |
| (4096,32) | r8 r4 r8 r8 | 491.9 | 532.1 | **+8.2%** |

Deleting the scratch round-trip genuinely buys −49 µs of inner; the fused codelet's choreography (strided leg gathers per group-pair under full register pressure) hands it back. Setter NOT wired; both codelet editions + decision bench stay in tree. Bycatch: the **±3% cross-process noise band became an explicit doctrine number**.

---

## §6a26 — rfft native z terminator (fwd): shipped, BIT-exact

**Motivation measured first:** the §6a24 convert-around cost +19–23% on rfft cells and surrendered the MKL lead at (2000,4): split/MKL 1.145× → z/MKL **0.960×**.

**Design:** `zo` threaded through both natural executors + mt; stage-0 branches to `_rfft_stage0_z` — k0 specials interleave from `nat_k0`, the terminator codelet lands in a plan-owned L1 scratch (`zscr`, plan-time-sized chunk width `zch = 768/(r·K)` capped at kmax), rows interleaved to z while hot; mid via a new `zo` mode.

**The debugging saga (doctrine-grade):** first landing heap-corrupted (D2 one-sided slot partition — each slot written exactly once, upper slots conjugated at row r·m−f). Then a 1–2 ULP divergence in hcn-family rows only: the codelet was exonerated **twice** by isolation (stride-agnostic, bit-stable at every k); entry tracers finally showed the split arm skipping the per-k loop entirely — split runs the **ranged hcnr**, enabled by `#define VFFT_RFFT_RANGED 1` **inside vfft.c** at line 37. The "refutation" that had ruled hcnr out was doubly invalid: `-U` cannot undo an in-source define, and the define grep covered flags and two headers, not the tree. **Fix better than a gate relaxation: chunked hcnr in z-mode** — the same ranged codelet as split's single sweep, in zch-column chunks; splitting the codelet's column loop across calls preserves per-column arithmetic → **gate 12/12 BIT, fwd and bwd**.

**Tax curve (generic cascade both arms — the true configuration, see §6a27):**

| cell | z vs split | z/MKL-CCE | split/MKL-CCE |
|---|---|---|---|
| (200,4) ×3 | +34.8…+40.8% | 1.18–1.25× | 1.59–1.76× |
| (2000,4) | +21.5% | **1.15×** | 1.40× |
| (1000,8) | +16.4% | 1.26× | 1.46× |
| (20000,4) | +2.5…+6.3% | 0.63× | 0.66× |
| (50000,4) | −1.0% | 0.93× | 0.92× |
| (100000,4) | **−12.8…−31.7%** | 0.67× | 0.53–0.59× |

Tiny cells are L1-resident — there is no memory round-trip to fuse away, so the interleave's ALU cost is an inherent floor (convert paid the same). The structural payoff is at DRAM scale: one interleaved stream has half the distinct rows of two split planes (TLB/prefetch density). Bycatch: the rfft.h:523 `-Waggressive-loop-optimizations` warning resolved by the mid restructure.

---

## §6a27 — PIC codelet set, jit verdicts, and the regression audit

**The rsp fix (real, shipped):** `codelets_linux.rsp` contained **zero rfft-family PIC objects** — the rfft natural/packed jit had *never bound* in this container (silent dlopen failure → generic, forever). 115 rfft/c2r avx2 codelets compiled `-fPIC` (rsp 94→209 lines; `tools/build_rfft_pic_rsp.sh`). The rfft natural jit bound for the first time.

**Then the measurements killed it.** Same-process four-arm:

| cell | path | split generic vs jit | z native vs z jit_z |
|---|---|---|---|
| (2000,4) | RFFT | generic **−8.9%** (12.94 vs 14.21 µs) | native 15.69 vs jit_z 24.10 |
| (2000,16) | RFFT | generic −1.0% | native 55.5 vs jit_z 66.0 |
| (2000,64) / (128,67) | STRIDE | rfft jit fields nil (path doesn't apply) | — |

The rfft path serves only small K, and across that entire domain the emitted per-k terminator's call overhead at small vl exceeds the cascade gain over generic's single ranged hcnr sweep. **jit_natural and jit_natural_z unbound with rationale at the bind site.**

**The natural-z jit (built and closed):** `emit_rfft_jit.py --mode natural-z` — same cascade and per-k log3 terminator with stores redirected through zscr. **Bit-exact against the natural jit** (gate 12/12 on the jit↔jit_z pairing — store-redirect-only emission works) but +63% slow. Variant surgery on the emitted .c attributed it: codelet→scratch is 0.7 µs *faster* than direct scatter; **the interleave loop alone costs ~11 µs in the jit TU vs 2.2 µs for the identical static-inline helper in the main build** (~5×, -O3/-march=haswell TU, not root-caused — mode closed, observation recorded).

**Doctrine reinforcement (the weather ghost):** the initial "jit −19%/−29% wins" at 20K/100K were cross-process comparisons on a ±25% day; the same-process four-arm showed generic ahead. Cross-run deltas are not evidence at any magnitude — same-process interleaved arms only.

**Post-ship regression audit (user-prompted):**

| surface | verdict |
|---|---|
| shipped fwd paths | no change — jits never bound in shipped config |
| c2c jit after VERSION 5→6 | steady-state unregressed; one-time recompile per shape |
| §6a24 zo plumbing on split | sub-noise; historical bands matched all session |
| **c2r jit** | **newly ACTIVE** (pre-existing bind woken by the rsp): −0.4, +0.6, +0.3, +0.4…+4.0% band = **parity within ±3%**; left bound (bit-exact, exercises the emitter) |
| packed fwd jit | **unreachable** — all create sites pass SPLIT; dead code |

Lesson: an infrastructure fix that makes previously-failing resolves succeed is a behavioral change everywhere those resolves are called — audit the callers, not just the fix.

---

## §6a28 — c2r native z-in initiator: the clean one

The bwd mirror of §6a26, landed in **one patch with a first-attempt 12/12 BIT gate** — the D2 partition, chunked-scratch layout, stride-agnosticism proof, and verbatim-anchor discipline were all pre-paid. `c2r_plan_t` wraps `rfft_plan_t`, so `zscr`/`zch` were already there. The chunk's z rows are deinterleaved into P/M family planes filling **exactly the cells the fwd terminator wrote** (same one-sided predicate); the same `nat_init` codelet runs on scratch pointers. Bonus fix: the PACKED-input layout's z entry fed split planes before (bwd twin of the fwd latent bug) — now a proper CCE→packed pack.

| cell | z-in tax (was, convert) | z/MKL-CCE-bwd (was) |
|---|---|---|
| (200,4) | ~+43% (was +35% — micro-cell floor, same trade fwd accepted) | 1.13–1.18× (was 1.26×) |
| (2000,4) | +17.9…+20.9% (was +24.1%) | **0.99–1.01× parity** (was 0.965×) |
| (1000,8) | +16.7% (was +31.7%) | **1.227×** (was 1.100×) |
| (100000,4) | **−10.9%** | 0.756× |

**1D z contract: COMPLETE both directions** — fwd z-out (§6a26), bwd z-in (§6a28), STRIDE native both ways (§6a24), PACKED correct both ways.

---

## §6a29 — 2D z contract + the correction

**Correctness first: the 2D z sentinel SEGFAULTED** (NULL plane straight into the tiled row pass). §6a29 defined the contract — `z[2*(i*H2+f)]`, MKL's 2D CCE shape — and shipped v1 convert-around. **Gate extended to 18/18 ALL PASS.**

**The claim that got challenged, and the reconciliation (credit: the challenge was correct twice over).** The original §6a29 text claimed "first 2D-vs-MKL numbers ever recorded" and framed the gap as "2D trails MKL 1.6–1.7×." Both wrong: `v1_0_results.md` §2 records 2D C2C beating MKL 1.26–1.41× (i9-14900KF, PATIENT, cooled), and the 0.60× compared against MKL's **real-CCE** baseline — a very different animal from the complex-split baseline v1.0's win was earned against. Control run, this container, (256×256):

| arm | µs | ratio |
|---|---|---|
| vfft c2c-2D (in-place scrambled, PATIENT, copy-corrected) | 452.0 | — |
| MKL complex SPLIT (v1.0 config) | 467.0 | **dag/MKL 1.033×** — no regression |
| MKL complex interleaved | 419.4 | 0.928× |
| MKL real CCE | 208.4 | **MKL real/complex-split = 2.24×** |

*Real-CCE* = MKL's real-domain transform (half the flops of complex, Hermitian half-spectrum) stored as interleaved complex pairs — MKL at full native strength. The corrected finding: **our r2c-2D harvested only 1.31× from the real-transform advantage where MKL harvests 2.24×.** Phase attribution (pre-existing `VFFT_2D_PROFILE` counters, 98.8% accounted at 256² fwd): wrapper memcpys **14.7%**, transpose-in 8.4%, inner-r2c **37.3%**, transpose-out+pad 12.4%, col-c2c 17.0%, perm-pack **9.0%**.

---

## §6a30 — wrapper elimination + fused z: the 2D z tax dies

The OOP entries paid a full-plane memcpy in + half out purely because they reused the in-place ABI — the core never mutates its input. Copy-free OOP-native executors added; the z variants fold the (de)interleave into the existing phase-3 pack / phase-1 unpack perm loops. **The §6a29 z2tmp convert machinery, one day old, cleanly retired.** Gate 18/18 (bit-identical: same phases, same pad bytes).

| cell/dir | split before → after | z tax before → after | split/MKL-real | z/MKL-real |
|---|---|---|---|---|
| 256² fwd | 332.4 → **283.1 (−14.8%)** | +16.2% → **−0.7%** | 0.615 → **0.741×** | 0.529 → **0.747×** |
| 256² bwd | 312.0 → 282.7 (−9.4%) | +18.0% → +2.3% | 0.768 → 0.859× | 0.651 → 0.840× |
| 512² fwd | 1652.3 → **1382.2 (−16.3%)** | +12.7% → **−0.5%** | 0.601 → **0.715×** | 0.533 → **0.719×** |
| 512² bwd | 1482.3 → 1322.5 (−10.8%) | +16.6% → −0.1% | 0.768 → 0.870× | 0.658 → 0.871× |

**2D z runs at split parity everywhere.** Harvest ratio: 1.31× → **1.60×** vs MKL's 2.24×. The "parked" native-2D-z shipped itself for free once the ABI was fixed — the interleave had a pack loop to fuse into all along.

---

## Doctrine crystallized this session

1. **Cross-process comparisons are banned as evidence at any magnitude** — the ±3% band is a floor; on a ±25% weather day, even 20% "improvements" were phantoms. Same-process interleaved arms only.
2. **Compile-time-feature questions are answered by grepping the tree for the define** — never by flag archaeology; `-U` cannot undo an in-source `#define`.
3. **Variant surgery on emitted artifacts** (delete one loop, recompile, dlopen, time) attributes in one run what pointer-level reasoning couldn't in ten.
4. **Claims of primacy get checked against the tree's own record before writing** — the v1.0 miss; and different MKL baselines (real-CCE vs complex-split) answer different questions and must never be conflated.
5. **Infrastructure fixes that un-fail resolves are behavioral changes at every caller** — audit the callers.
6. **Ship correct → attribute → replace** beats speculative optimization: the z2tmp convert lived one day, by design.

## Final state

| surface | status |
|---|---|
| 1D z contract | native both directions, BIT-gated, MKL parity-or-lead on rfft cells, structural win at DRAM scale |
| 2D z contract | native (fused pack/unpack), **zero tax**, 18/18 gated; segfault dead |
| rfft/c2r jit | unbound with evidence (rfft path); c2r jit active at parity; PIC rsp complete + rebuild script in tree |
| Model-B | closed negative, reproducible |
| Archive | `/mnt/user-data/outputs/VectorFFT-main-fftnd-accfix.tar.gz`, current through §6a30 |

## Queue (post-session)

1. **2D campaign, next slice: inner-r2c (43.3% of post-§6a30 total)** — the row pass through the stride engine at K=8 tiles; f2dprof needs p2/p3 probes on the OOP path first.
2. Transpose pair (~24%).
3. 3-stage DIF bwd jit residual (~8.7%, broadcast hoisting suspect).
4. Gap-A post-tw OOP generator mode (spec ready).
5. r2c.h:973 c2r tail guard; n1 codelet profiling R=8/16/32; TLB/hugetlbfs at large K.
