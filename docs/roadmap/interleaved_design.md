# Interleaved layout support — pitfalls, performance strategy, architecture

*Roadmap design doc, 2026-07-14. The v1.1 headline adoption feature: the world's data is
interleaved complex (`std::complex`, numpy, MKL default, FFTW native API); VectorFFT's
structural speed advantage comes precisely from NOT computing on it. This doc is about
having both.*

> **TL;DR** — "Interleaved support" is two orthogonal problems wearing one name:
> **(1) component interleave** (re/im pairs vs split planes) and **(2) batch geometry**
> (transform-major `z[k·2N + 2i]` vs our lane-major `re[i·K+k]`). Problem 1 is nearly
> free for us: every datum already passes through cache-resident scratch (tiles) or
> cache-resident fused blocks, and layout conversion inside those windows adds ~zero DRAM
> traffic — the tiled/fused architecture is *uniquely* positioned here, in a way FFTW had
> to buy with buffered plans. Problem 2 (batched 1D) is a genuine corner-turn and the
> real engineering. Compute stays split-complex internally, always — going
> interleaved-native in the butterflies would surrender the shuffle-free FMA structure
> that beats FFTW/MKL in the first place.

---

## 1. Why the engine is split, and why that must not change internally

Split-complex lane-batching gives butterflies with **zero shuffles**: complex mul is 4
FMA-port ops on plain vectors, twiddles broadcast, and the SIMD width is pure batch.
Interleaved-native arithmetic (FFTW's paired-vl style) pays a shuffle per complex mul
(`vpermilpd`/`vaddsubpd` dance), contends for the shuffle port, and halves effective
arithmetic density in the worst stages. The measured 1.2–1.8× over FFTW is substantially
*this choice*. Conclusion: interleave lives at the **boundary**; the interior is split
forever.

## 2. The two problems

**P1 — component interleave, no batch dimension** (all rank ≥ 2 transforms, and 1D K=1):
user buffer `z[2·flat + {0,1}]`. Element i of engine-lane k sits at `z[2(i·K+k)]` — a
pure stride-2 component split, no reordering across elements. This fuses into existing
data movement (§4).

**P2 — transform-major batched 1D**: K transforms, each contiguous: `z[k·2N + 2i]`. The
engine wants `re[i·K + k]`. That is a **(K×N) corner turn** *plus* the component split —
a real transpose, not a load-time shuffle. Naive strided gathers across `2N`-strided
rows are a TLB/cache disaster at large N. This is the same class of problem as the 2D
row pass, and the same machinery solves it — but it costs sweeps unless fused (§5).

## 3. Pitfalls (the honest list)

| # | pitfall | severity | mitigation |
|---|---|---|---|
| 1 | Treating P1+P2 as one feature; shipping P1 and calling batched-1D "supported" | design-fatal | separate flags, separate phases, separate benches |
| 2 | Shuffle-port pressure in fused IL loads/stores (unpck/perm on port 5 vs FMA 0/1) | measurable, per-codelet | stage-0/last-stage only; measure; wisdom chooses IL-fused vs scratch-bounce per cell |
| 3 | Codelet-count explosion ({split,IL-in,IL-out,IL-both} × ISA × everything) | build/registry blow-up | NOT a codelet family — an **emitter IR pass** (`Load(Input i)` → paired-load+unpack; symmetric store). Flag on existing codelets; new symbols only where the planner actually places them |
| 4 | In-place interleaved: internal split intermediates must live somewhere; in-place deinterleave is a cycle-following permutation (serial, cache-hostile) | correctness/perf trap | never permute in place; convert **through the tile scratch / fused block** the data already passes through. Same trick that killed the c2r reverse-order hazard |
| 5 | AVX2 register pressure in IL mono codelets (zero-spill R=16/R=20 sit at 12–24 regs; +2–4 shuffle temps may tip spills) | AVX2-specific | IL-mono default-on for AVX-512 (32 regs); AVX2 measured per codelet, scratch-bounce fallback |
| 6 | K=1 interleaved — the single most common call in the wild — is our engine's worst geometry (lane-batching wants K ≥ VW) | adoption-critical | 1D K=1 as degenerate 2D: N = N1×N2, tiled row pass supplies the lanes (N2-point at K=B) + native N1 pass; the IL gather rides the tile transpose. Needs its own bench story |
| 7 | Alignment: complex pairs are 16B-atoms; 256/512-bit pair loads split cache lines on unaligned user buffers | minor, real | document 32/64B-preferred; masked/split-load tails; NT stores for reinterleave-out at DRAM sizes |
| 8 | Wisdom dimensionality (layout joins T in the key) | bookkeeping | `lay=` field; {split, IL} realistically one flag per side |
| 9 | Benchmark methodology drift: the canonical v1.0 results were always vs MKL **CCE-interleaved** (its best path) — but `bench_fft2d_vs_mkl.c` carried a `DFTI_REAL_REAL` split config that the 3D/4D session benches inherited, silently comparing against MKL's 2.3–5.2×-slower split path (measured). Corrected 2026-07-14: multi-dim addendum tables retracted/re-benched vs CCE; all three bench files now CCE-default with a warning comment | happened; fixed | one canonical MKL config, asserted in every bench header; on AVX-512 hosts also pin `MKL_ENABLE_INSTRUCTIONS` to the vfft build's ISA for matched comparisons |
| 10 | API overreach toward FFTW-guru iodim generality | scope creep | `VFFT_LAYOUT_SPLIT/INTERLEAVED` per side, transform-major batch convention for IL 1D, nothing more in v1 |

## 4. Maximizing performance — the fusion thesis

The performance program is one sentence: **every conversion rides a memory movement
that already exists.** Concretely, the boundary sites:

- **Tiled last-axis pass** (1D-as-2D rows, 2D/3D/nd rows, r2c rows): the gather is
  already a transpose through per-thread scratch. An IL-gather variant
  (`stride_transpose_il2sp[_pair]`) reads pairs and writes split scratch — the
  deinterleave of 2 complex per 256-bit reg is 1 `vunpcklpd` + 1 `vunpckhpd`, *cheaper
  per element than the 8×4 line-fill it joins*. Scatter symmetric. Expected cost vs
  split: ~0 sweeps, low-% kernel delta.
- **Axis-0 / native first pass** (fwd) and its bwd mirror (the last pass to touch
  memory): an emitter flag fuses deinterleave into stage-0 loads / reinterleave into
  last-stage stores. One IR pass, no new math, +1–2 shuffle temps.
- **Fused fftnd blocks**: inside a cache-resident block, conversion is L2-bandwidth,
  not DRAM. The fusion work of this week is precisely what makes P1 ≈ free at rank ≥ 2.
- **Batched-1D corner turn (P2)**: v1 = explicit blocked IL-transpose sweeps through
  tile scratch (in: fused with nothing; out: fusable into the final pass's store). At
  DRAM-resident sizes the 1D chain is already multi-sweep, so +1–2 sweeps is a bounded,
  *measured* tax; the fully fused form (batched-1D restructured as 2D-with-lanes so the
  tiled pass IS the boundary) is the v1.2 refinement. At cache sizes the turn is
  cache-hit shuffling — small.

Acceptance bars to calibrate against, per family: **rank ≥ 2 IL within ~1–3% of split;
1D mono IL within ~3–5% (AVX-512), scratch-bounce parity on AVX2; batched-1D IL ≤ one
sweep-equivalent over split at DRAM sizes.** All wisdom-cell verdicts, per the house
rule.

## 5. Supporting architecture (build plan)

1. **Layout descriptor**: `vfft_layout_t {SPLIT, INTERLEAVED}` for input and output
   independently (IL→split import and split→IL export are free byproducts). Batched-1D
   IL fixes the transform-major convention.
2. **Transpose layer** (`transpose.h`): `il2sp`/`sp2il` scalar+pair variants, unit-tested
   as permutations, microbenched against the split kernels. *No generator work — this is
   Phase 1 and unlocks all of rank ≥ 2.*
3. **Emitter IR pass** (OCaml): `--il-load` / `--il-store` emit-time flags rewriting
   stage-0 loads / final stores; provenance-stamped; registered as `_il*` symbol
   suffixes in a side table with graceful fallback (missing IL flavor → split +
   scratch-bounce). Register-pressure guard per ISA.
4. **Executor plumbing**: plan carries layout flags; orchestrators select IL flavors at
   the two boundary positions only (fwd: first pass load / last pass store; bwd
   mirrored). fftnd fwd first-touch is always axis 0 (or the fused group's axis-s at
   s=… axes<s run first, so axis 0 regardless); bwd last-touch is axis 0's bwd.
5. **Calibrator/wisdom**: `lay=` joins the key next to `T=`; per-cell verdicts include
   the AVX2 IL-mono vs bounce choice.
6. **Gates**: layout equivalence is *exact* — conversions move bits, no arithmetic — so
   the primary test is memcmp: IL-path output, converted, equals split-path output
   bit-for-bit, every family, every T. Plus conv-IL end-to-end and the MKL-default
   (interleaved CCE) bench column.
**STATUS 2026-07-14 — P1a SHIPPED**: converters + universal wrapper + r2c IL-out,
every family bit-gated (test_il_layout.c). Measured two-sweep ceiling at 64³:
**1.933×** — the strided-rows work made the split transform memory-floor-lean, so
unfused conversion now costs a transform's worth of traffic. P1b/P2 fusion is
therefore not an optimization but the feature.

7. **Phasing**: P1 transpose variants + rank ≥ 2 IL (largest win/effort, zero generator
   risk) → P2 emitter flags + 1D mono IL (AVX-512 first) → P3 batched-1D corner turn +
   the K=1 story → P4 calibrator integration + the MKL-default bench tables.

## 6. Open questions (decide before P3)

- **Batch geometry commitment**: is transform-major (`idist = N`, FFTW/MKL default) the
  only supported IL batch layout in v1, or do we also take lane-major-interleaved
  (cheap for us, rare in the wild)?
- **K=1 priority**: does the single-transform interleaved story ship with P1 (as
  degenerate-2D) or wait for P3? It's the adoption-visible case.
- **Mixed in/out layouts** (IL-in → split-out import mode): free to support; worth API
  surface? Relevant if the trading stack wants zero-copy ingest into split-domain
  pipelines.
