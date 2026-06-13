# 3-Stage Bailey Experiment — Final Summary

## Math derivation

For N = R³, input n = a·R² + b·R + c (a slow, c fast), output m = mₐ·R² + m_b·R + m_c:

    mn mod R³ = (mₐ·c + m_b·b + m_c·a)·R²
              + (m_b·c + m_c·b)·R
              + m_c·c

    W_N^{mn} = W_R^{mₐc + m_b·b + m_c·a}
             · W_{R²}^{m_b·c + m_c·b}
             · W_N^{m_c·c}

Grouping by input variable shows FFT-over-a absorbs only W_R^{m_c·a}
(slow-input axis pairs with fast-OUTPUT digit — standard digit reversal).

## Algorithm

  1. Stage 1: FFT_R along a-axis (slow, stride R²) → A[m_c, b, c]
  2. Twiddle 1: *= W_N^{m_c · (b·R + c)}
  3. Stage 2: FFT_R along b-axis (middle, stride R) → B[m_c, m_b, c]
  4. Twiddle 2: *= W_{R²}^{m_b·c}
  5. Stage 3: FFT_R along c-axis (fast, stride 1) → C[m_c, m_b, mₐ]
  6. Final permutation:
       out[mₐ·R² + m_b·R + m_c] = C[m_c·R² + m_b·R + mₐ]

## Codelet mapping

  Stage 1: UG_UG, in_leg_stride=R², in_group_stride=1, me=R²
  Stage 2: R UG_UG calls, in_leg_stride=R, in_group_stride=1, me=R each
  Stage 3: UL_UL, in_leg_stride=1, in_group_stride=R, me=R², in-place
  Permute: SIMD 16×16 transpose (R 16×16 transposes, one per m_b)

UL_UL fits stage 3 because its load AND store do in-register 8×8 transposes,
matching the (leg-contiguous, batch-strided) memory pattern natively.

## Optimizations applied

  AVX-512 vectorized twiddles: cut each twiddle pass from ~10,000 ns
    (GCC didn't autovectorize scalar form) to ~1,370 ns.

  SIMD 16×16 permutation: scalar 3251 ns → SIMD 2570 ns (1.26×).
    Uses same unpacklo/unpackhi/permutex2var/shuffle_f64x2 primitives the
    codelet uses for its load/store transposes. Bit-exact.

## Correctness

  R=4  N=64    scalar ref vs naive DFT:   max_rel = 5.79e-13  PASS
  R=8  N=512   scalar ref vs naive DFT:   max_rel = 1.77e-11  PASS
  R=8  N=512   all-codelet vs naive DFT:  max_rel = 1.77e-11  PASS
  R=16 N=4096  all-codelet vs v3 ref:     max_rel = 6.56e-11  PASS

## Performance (stable across 14 successive runs)

At N=4096 (R=16):

  v3 (2-stage Bailey):       ~13,100 ns
  3-stage + SIMD permute:    ~19,900 ns
  Delta:                      ~6,700 ns  (3-stage 51% slower)

At N=512 (R=8):
  FFTW (PATIENT):                ~730 ns
  3-stage:                     ~2,500 ns  (≈3.4× FFTW)

Per-stage breakdown at N=4096 (with vectorized twiddles + SIMD permute):

  Stage 1 UG_UG (me=256):     3254 ns
  Twiddle 1 (vectorized):     1368 ns
  Stage 2 (R UG_UG calls):    3539 ns
  Twiddle 2 (vectorized):     1376 ns
  Stage 3 UL_UL (me=256):     3277 ns
  SIMD permutation:           2570 ns
  ─────────────────────────────────
  Sum of parts:              15384 ns
  Integrated:               ~19,900 ns
  Wrapper/cache overhead:    ~4500 ns

## Where the gap comes from

  v3:        2 codelets + 1 fused twid+xpose
             = ~7700 + 2400 = ~10,000 ns sum-of-parts
  3-stage:   3 codelets + 2 twiddles + 1 permutation
             = ~10,070 + 2,750 + 2,570 = ~15,400 ns sum-of-parts

3-stage trades v3's single fused-twid+xpose for one extra FFT stage + one
extra twiddle pass + one permutation. Net cost: ~5,400 ns more sum-of-parts.

## Best-case ceiling

Saved if we (a) generate real t1 OOP codelets that absorb twiddle during
load and (b) fold permutation into stage-3 stores via a codelet variant
with non-uniform output stride:

  Both twiddles fused:        −2,750 ns
  Permutation folded in:      −2,570 ns
                              ─────────
                              −5,320 ns

  Best-case 3-stage:         ~14,600 ns
  v3 today:                  ~13,100 ns

Best-case 3-stage still ~10% slower than v3. Three codelet calls cost
more than two even when everything else is fused.

## Where 3-stage actually matters

Not useful for:
  N ≤ 4096 with our current codelet set. v3's 2-stage handles these well
  because radix-32 and radix-64 cleanly cover the sqrt(N) factorizations.

Useful for:
  - N > 4096 where single-radix codelets don't cover sqrt(N).
    Example: N=16384 has no X·Y with X,Y ∈ {8,16,32,64}; three-way
    16×32×32 = 16384 is the natural decomposition.
  - N with working set significantly exceeding L1, where the third level
    of recursion helps tile. At N=4096 we're at ~32KB per buffer,
    comfortably L1-resident.
  - In combination with real t1 codelets (which would also benefit v3).

## Verdict

3-stage Bailey is correctly implementable with our existing codelets
(UG_UG for stages 1+2, UL_UL for stage 3). With AVX-512 vectorized
twiddles + SIMD 16×16 permutation, it runs at ~19,900 ns at N=4096.

It is ~51% slower than v3 at this size, and even maximum further
optimization only narrows the gap to ~10%. The implementation is sound
and ready to extend to N > 4096 when needed. For N ≤ 4096, v3 remains
the production design.

## Files in bundle

  test_3stage_scalar.c       Scalar reference, R=4 and R=8
  test_3stage_codelet.c      Codelet stages 1+2 + scalar stage 3 POC
  bench_3stage.c             Full codelet 3-stage w/ SIMD permute vs v3
  bench_3stage_breakdown.c   Per-stage timing at N=4096
  bench_3stage_vs_fftw.c     Head-to-head vs FFTW
  permute_simd.c             Standalone SIMD permutation perf test
  permute_simd.h             SIMD 16×16 transpose header
  oop_r{8,16}_{UG_UG,UL_UL}.c    Codelets used
  stride_fft1d_v3.h          v3 reference for correctness at N=4096
  build_all.sh               One-shot build
