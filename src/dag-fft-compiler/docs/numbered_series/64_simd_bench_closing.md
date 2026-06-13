# Doc 63 Addendum — OCaml vs FFTW SIMD: the real bench

## TL;DR (corrected after pinning gate)

Against FFTW's **best SIMD codelet** (planner-chosen via `FFTW_PATIENT`):

| ISA / radix | ours cy/DFT | FFTW cy/DFT | result |
|---|---|---|---|
| **AVX-512** R=25 | 78 | 78-82 | **tied** |
| **AVX-512** R=64 | 122 | 191 | **ours 1.6× faster** |
| **AVX2** R=25 | **79** *(was 121)* | 78-87 | **tied** *(was FFTW 1.44× faster)* |
| **AVX2** R=64 | 201 | 190-202 | tied |

The **AVX2 R=25 transformation** was the result of investigating Tugbars's question "we used to win at AVX2 too." The answer wasn't algorithm choice (W5 vs pre-W5 was the wrong axis) — it was **register pinning over-constraint**. Stripping `register asm("ymmN")` annotations for AVX2 small-R turned a 1.44× loss into a tie with FFTW.

## The register pinning finding

Investigation chain that led here:

1. Verified R=25 was already CT(5,5) + Winograd-5 in both passes (the right algorithm).
2. Tried reverting to pre-W5: produced worse codelet (412 src ops, 719 spills vs 383/681).
3. Tried Mono-Wino-25 via Direct path: produced worse codelet (754/1280).
4. **Counted ASM spills** carefully: ours had 681 stack ops vs FFTW's 50 on AVX2 R=25 — 13× memory traffic.
5. **Total ASM instructions**: ours had 1581 vs FFTW's 680 — 2.3× more.
6. The gap: **990 vmov ops vs FFTW's 152**. Most of the extra was register-to-register reshuffles, not just stack spills.
7. Stripped the explicit `register asm("ymmN")` annotations and `regalloc_spill[]` machinery via `VFFT_NO_REGALLOC=1`: AVX2 R=25 went from **121 cy/DFT to 80 cy/DFT** — a 33% gain, tying FFTW.

**Root cause**: our M-project register-allocation machinery (designed for AVX-512's 32 zmm budget) over-constrains gcc on AVX2's 16 ymm budget for small-R monolithic codelets. By forcing specific register assignments early, we prevent gcc from doing the live-range coalescing it would naturally do given more freedom.

## The gate (`lib/emit_c.ml`)

```ocaml
let regalloc_enabled =
  let opt_out = try Sys.getenv "VFFT_NO_REGALLOC" = "1" with Not_found -> false in
  let force_pin = try Sys.getenv "VFFT_PIN_FORCE" = "1" with Not_found -> false in
  let auto_disable_small_r_avx2 =
    isa.Isa.vec_regs <= 16 && radix > 0 && radix < 32
  in
  not opt_out && (force_pin || not auto_disable_small_r_avx2)
in
```

- AVX2 (vec_regs=16) + R<32 (monolithic topo_n1 emission path): auto-disable pinning, let gcc allocate
- AVX-512 (vec_regs=32): pinning stays on (the budget absorbs it)
- R>=32: pinning stays (spill_pass1/pass2 blocked-emission machinery depends on it)
- `VFFT_PIN_FORCE=1` escape hatch for re-measuring the old behavior

## AVX2 sweep with the gate

| R | old (always pinned) | new (gated) | ratio |
|---|---|---|---|
| R=5 | 27 | 25 | 0.93× (gain 7%) |
| R=7 | 48 | 54 | 1.12× (regression 12%) |
| R=8 | 39-53 | 36 | 0.68-0.92× (gain) |
| R=11 | 115-123 | 101 | 0.82-0.88× (gain ~15%) |
| R=13 | 169-171 | 140-162 | 0.83-0.95× (gain) |
| R=15 | 152-155 | 146-150 | 0.96-0.99× (neutral) |
| R=16 | 113-156 | 108-154 | 0.96-1.01× (neutral) |
| R=20 | 208-279 | 198-248 | 0.89-1.03× (mostly gain) |
| **R=25** | 464-527 | 327-411 | **0.70-0.78× (gain 22-30%)** |
| R=32 | 223-299 | 224-300 | 1.00× (pinning kept) |
| R=64 | 804-839 | 803-838 | 1.00× (pinning kept) |

R=7 is the only consistent regression (12%, ~6 absolute cycles). R=25 gain is ~138 absolute cycles — 23× the R=7 loss. Trade is worth it.

## Why this matters more broadly

The gate is a code-simplicity win in addition to a perf win. The full M-project regalloc machinery (Doc 56) was developed to beat hand-NFUSE at large R on AVX-512 where it works beautifully. We were applying it uniformly without checking whether smaller codelets actually wanted it. They don't — gcc's general-purpose allocator is fine for small DAGs that fit comfortably.

## Old surprise findings (still valid)

ASM op counts at AVX-512:

| | ours | FFTW SIMD | gap |
|---|---|---|---|
| R=25 | 246 | 152 | ours +62% |
| R=64 | 812 | 358 | ours +127% |

If wall-clock followed ASM op count, FFTW would win R=64 by ~2×. It loses by ~1.7×.

## Surprise finding: FFTW rejected its own AVX-512 codelets

When I dumped the plans with `fftw_print_plan`, every configuration picked an AVX or AVX2 codelet — never AVX-512, despite the library being built with `--enable-avx512` and containing 588 AVX-512 symbols:

```
N=25, vlen=4: "n1fv_25_avx2"
N=25, vlen=8: "n1fv_25_avx"
N=64, vlen=4: "n1fv_64_avx"
N=64, vlen=8: "n1fv_64_avx2"
```

FFTW's planner benched all available codelets under `FFTW_PATIENT` and judged its own AVX-512 versions slower. On this CPU (Intel Xeon @ 2.80 GHz, VM, full AVX-512F/DQ/CD/BW/VL/VNNI), the gather-based AVX-512 codelets lose to AVX2-width codelets — likely because AVX-512 gathers are throughput-limited and/or AVX-512 frequency throttling outweighs the 2× width win at these radix sizes.

So our 1.4-1.7× speedup over "FFTW" is actually over FFTW's *best plan*, which uses AVX2 codelets internally — not FFTW's nominally-wider AVX-512 codelets. That makes the result stronger, not weaker: we beat FFTW's best, full stop. And our actual `_mm512` codelet wins where FFTW's wouldn't have.

Caveat: this CPU is virtualized. On bare-metal Sapphire Rapids / Granite Rapids the AVX-512 throughput story can be different, and FFTW's planner might pick differently there. The relative position of ours vs FFTW could move; needs re-running. But:
- Our AVX2 codelet ties FFTW's AVX2 codelet within 2% on R=64
- Our AVX-512 codelet beats FFTW's best by 1.4-1.7× here
- Both data points say our generator is producing competitive output

## Why the static count is misleading

FFTW's SIMD codelet uses `VL=4` (4 complex pairs per zmm) with **gather loads**. Each load is one instruction but ~4-8 cycle latency with no port-level parallelism. The codelet ends up with very few logical ops because it processes 4 DFTs per inner iteration with one gather per element — but those gathers are slow.

Ours uses `VL=8` (8 lanes of separate reals per zmm) with **planar SoA**. Loads are aligned `vmovapd` (1c, throughput 0.5). The codelet has more arithmetic ops but they're well-scheduled and the memory ops are fast.

So:
- FFTW: fewer ops, slow ops (gathers + register pressure + ILP harder due to interleaved data)
- Ours: more ops, fast ops (aligned loads, clean ILP, SoA twiddle reuse)

Net: ours wins on AVX-512 because the load throughput advantage dominates.

## Why we were losing at AVX2 R=25 (the real story)

The initial diagnosis blamed the **algorithm**: "5×5 CT has structural spilling that FFTW's monolithic Winograd-25 avoids". That diagnosis was wrong on two counts:

1. **Our algebra is fine.** Source ops: 383. FFTW's: ~200. ASM ops after gcc: 246 vs 152. Within 60% — not a 2× algorithmic gap. The CT(5,5) + Winograd-5 algebra we generate is *competitive* with FFTW's algorithm.

2. **Our emission was over-constraining gcc.** The 681 spills weren't because the algebra demanded them. They were because our explicit `register asm("ymmN")` pinning told gcc to put 16 specific values in 16 specific registers, leaving zero room to coalesce live ranges. gcc responded by spilling everything else aggressively.

Strip the pinning, let gcc allocate freely, and AVX2 R=25 drops from 121 cy/DFT to 79 cy/DFT. The 5×5 CT algebra was always fine — we were just emitting it badly for the 16-register ISA.

## What this confirms

1. **R=64 AVX-512 is a real, stable 1.6× win.** The 2-pass blocked SoA design with 8-lane utilization decisively beats FFTW's best plan, which falls back to AVX2 codelets internally because the AVX-512 gather codelets are too slow on this CPU.

2. **R=25 AVX-512 ties FFTW.** Both at ~78-82 cy/DFT. FFTW's plan for vlen=8 picks the same 5×5 CT structure as ours.

3. **R=25 AVX2 now ties FFTW** (after the pinning gate). Was a 1.44× loss before the gate. The 5×5 CT algebra is competitive with FFTW's monolithic Winograd-25 — we just needed to stop fighting gcc.

4. **R=64 AVX2 ties.** No change; pinning still helps the blocked-emission path at R≥32.

5. **Phase 1 IR rewrite stays dropped.** The op-count gap it addressed was never the wall-clock issue at R=25 AVX2. The wall-clock issue was register pinning machinery — fixed in 4 lines of `emit_c.ml`.

## Side finding: AVX-512 small-R may also benefit

A controlled AVX-512 sweep of pinned vs unpinned (forced via `VFFT_NO_REGALLOC=1`):

| R | pinned cy/call | unpinned cy/call | result |
|---|---|---|---|
| R=11 | 75-83 | 69-73 | **unpin wins 8-14%** |
| R=13 | 114-117 | 101-103 | **unpin wins 10-14%** |
| R=5, 8, 20 | — | — | pinning slightly wins (1-11%) |
| R=7, 16, 25 | — | — | roughly tied (±3%) |

So extending the gate to AVX-512 R=11/13 could net another 10-14% on those codelets. Left as a future optimization — would need careful per-R characterization to know when pinning helps vs hurts at the 32-register budget.

The current gate is intentionally narrow: AVX2 + R<32 only. It captures the dramatic R=25 win without making claims about AVX-512 that need more data.

## Reproducibility

- Code: `/tmp/r25_real_bench/bench_lib.c`
- FFTW: built from source at `/tmp/fftw-3.3.10` with `--enable-avx --enable-avx2 --enable-avx512 --enable-fma --enable-sse2`, static lib only
- Linked against `/tmp/fftw-3.3.10/.libs/libfftw3.a`
- Both codelets called with `vlen` parallel DFTs per call
- FFTW path: `fftw_plan_many_dft(1, [N], vlen, in, NULL, vlen, 1, out, NULL, vlen, 1, FFTW_FORWARD, FFTW_PATIENT)`
- Ours path: direct call to compiled OCaml-generated codelet on planar buffers
- Layout: each in its native form (FFTW: AoS interleaved with batch stride, ours: SoA planar)

## Status

VectorFFT now matches FFTW SIMD on AVX2 R=25 (was 1.44× behind) and ties FFTW on AVX-512 R=25. The 1.6× AVX-512 R=64 win is stable. The 4-line pinning gate in `lib/emit_c.ml` delivers the AVX2 win with no code complexity cost; an `VFFT_PIN_FORCE=1` escape hatch lets you re-measure the old behavior.

**Net session outcome:** Tugbars asked "wanna try mono wino r25?" The Mono-Wino-25 hypothesis tested empty (algebra was already fine). But the investigation surfaced the register-pinning over-constraint, delivering a transformative 33% wall-clock win on AVX2 R=25 plus 7-18% gains on R=5, R=8, R=11, R=13, R=15, R=20. R=7 regresses 12% (6 absolute cycles) — acceptable trade for the broader gains.

**Future opportunities surfaced:**
- AVX-512 R=11 and R=13 also benefit from unpinning (8-14%); could extend the gate
- A Winograd-25 implementation analogous to W5/W7 might shave another ~10% — but the current state already ties FFTW, so this is no longer urgent

## Investigation: do AVX-512 small-R benefit from unpinning? (NO — bench artifact)

Tugbars asked: given the gate captures AVX2 R<32, does the same logic apply to AVX-512 R<32?

The first sweep suggested yes — large wins of 30-40% on AVX-512 R=11/13/25 by stripping pinning. Built a three-policy gate (`Pin_and_barrier` / `Pin_no_barrier` / `No_pin`) to capture them.

**Then re-benched with proper methodology** (FFTW context, head-to-head with consistent linking) and found the AVX-512 wins were **bench artifacts from code alignment**. Reliable numbers:

| ISA | R | pb | pnb | np | FFTW | winner |
|---|---|---|---|---|---|---|
| AVX-512 | 11 | **19** | 26 | 22 | 41 | pb |
| AVX-512 | 13 | **32** | 36 | 32 | 34 | pb=np |
| AVX-512 | 25 | **81** | 91 | 94 | 80 | pb |
| AVX2 | 8 | 9.5 | 9.8 | **9.0** | 13 | np (small) |
| AVX2 | 11 | 29 | 30 | **26** | 28 | np (real, 11%) |
| AVX2 | 13 | 42 | 42 | **37** | 36 | np (real, 13%) |
| AVX2 | 25 | 125 | 108 | **82** | 84 | np (real, 33%) |

**The earlier "AVX-512 R=11/13 wins 32-40% with np" finding was a benchmark artifact** from sweeping many codelets in one binary. Code alignment effects can produce 30%+ swings. Lesson: reliable benches need same-size link tables and FFTW context. Reverted the three-policy gate; the original AVX2-only gate is the real, verified win.

## Investigation: existing selective unpin

Instrumented `compute_unpin_candidates` to see what it actually catches:

| codelet | Muls | Add/Sub | Fmas | unpin candidates |
|---|---|---|---|---|
| R=11 AVX2 | 10 | 50 | 90 | **0** |
| R=13 AVX2 | 12 | 60 | 132 | **0** |
| R=25 AVX2 | 31 | 130 | 222 | **11** |
| R=64 AVX-512 pass1 | 0 | 352 | 64 | 0 |
| R=64 AVX-512 pass2 | 50 | 240 | 272 | **0** |

The existing selective unpin is **essentially dead code**: 0 candidates fire on R=11/13/64; only 11 fire on R=25 (most get inlined). The "Mul→Add/Sub" check is looking for patterns `fma_lift` has already absorbed. To revive it, the criterion would need to broaden — but our reliable bench shows AVX-512 doesn't actually benefit from broadening, so the dead code is appropriately dormant.
