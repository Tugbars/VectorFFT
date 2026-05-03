# VectorFFT core (`src/core/`)

This folder is VectorFFT's planner + executor + transform implementations, header-only. It replaces the older `src/stride-fft/` tree long-term. v1.0 ships from this folder.

The core is built around one idea: modern high-end CPUs are memory-bound on FFT, not compute-bound. We win by optimizing the cache hierarchy, batch dimension, and stage-level scheduling — not by minimizing arithmetic count, which is the axis FFTW spent 25 years optimizing.

The result, on Intel i9-14900KF (AVX2):
- **207/207 wins vs MKL on 1D C2C** (median 2.35×, range 1.02-9.81×)
- **Wins on every measured cell vs FFTW for DCT-II/III/IV, DST-II/III, DHT** (1.85-3.84×)
- **Wins 1.08-1.63× over MKL on 2D C2C** at all tested sizes
- 1D R2C, 2D R2C work but are gated on v1.1 codelet-fusion; see Performance section.

---

## Quick start

```c
#include "src/core/planner.h"
#include "src/core/env.h"

int main(void) {
    stride_env_init();
    stride_set_num_threads(8);

    stride_registry_t reg;
    stride_registry_init(&reg);

    /* 1D complex FFT, length 4096, batched 256 transforms */
    int N = 4096;
    size_t K = 256;
    double *re = aligned_malloc(N * K * sizeof(double), 64);
    double *im = aligned_malloc(N * K * sizeof(double), 64);
    /* ... fill (re, im) with input ... */

    stride_plan_t *plan = stride_auto_plan(N, K, &reg);
    stride_execute_fwd(plan, re, im);   /* in-place, natural order */
    stride_execute_bwd(plan, re, im);   /* unnormalized inverse: bwd(fwd(x)) = N*x */
    stride_plan_destroy(plan);
}
```

For real-input FFTs, DCT/DST family, and 2D, see the [API table](#api--convention-support) and the per-transform sections in `planner.h`.

---

## Transforms supported

| Transform | Status | Notes |
|-----------|--------|-------|
| 1D C2C | ✅ shipped | Forward + backward, split-complex, batched, wisdom-tuned |
| 1D R2C / C2R | ✅ shipped | Pair-packing algorithm |
| 2D C2C | ✅ shipped | Tiled (B=8) + Bailey methods |
| 2D R2C / C2R | ✅ shipped | Tiled row R2C + col C2C with K-padding |
| 3D anything | ❌ v1.1 | Recurses on 2D once 2D R2C is performant |
| ND anything | ❌ v1.1 | |
| DCT-I (REDFT00) | ❌ deferred | Lowest demand |
| DCT-II (REDFT10) | ✅ shipped | Makhoul + N=8 specialized straight-line codelet |
| DCT-III (REDFT01) | ✅ shipped | Same plan as DCT-II |
| DCT-IV (REDFT11) | ✅ shipped | Lee 1984 — single N/2-point complex FFT |
| DST-I (RODFT00) | ❌ deferred | |
| DST-II (RODFT10) | ✅ shipped | Wraps DCT-II via Wang's identity |
| DST-III (RODFT01) | ✅ shipped | Wraps DCT-III |
| DST-IV (RODFT11) | ❌ deferred | |
| DHT (Hartley) | ✅ shipped | Reuses R2C, self-inverse |

Why some r2r are "wrap" while others get specialized algorithms: the wrappers (DST-II/III) cost only one O(N×K) sign-flip + reverse pass on top of the inner DCT-II/III, which is a few percent at typical N/K. Lee 1984 for DCT-IV is needed because the textbook DCT-IV-via-DCT-III+DST-III approach pays 2× R2C cost; Lee gets it back to 1× by going directly to a half-size complex FFT.

---

## Performance

All numbers are single-threaded on Intel i9-14900KF (AVX2) unless noted.

| Transform | vs FFTW | vs MKL |
|-----------|---------|--------|
| 1D C2C | wins broadly | **207/207 wins** (median 2.35×, range 1.02-9.81×) |
| 1D R2C | ~1.5× | loses (codelet-fusion gap, v1.1) |
| 2D C2C | wins | 1.08-1.63× |
| 2D R2C | **0.33-0.77×** (loses) | not benched (no batched MKL R2C 2D) |
| DCT-II | 1.48× JPEG | predicted 4-13× (not benched yet) |
| DCT-III | similar | similar |
| DCT-IV | 1.85-3.84× | **4-13×** |
| DST-II | 1.91-2.77× | 2.5-8× (caveat: MKL TT solves a different transform) |
| DST-III | similar | similar |
| DHT | 1-2× | no MKL DHT API |

The R2C and 2D R2C losses are the same root cause: our R2C is structurally a 3-pass design (pack → C2C → butterfly), while FFTW and MKL fuse the pack into the first DIT stage and the butterfly into the last DIT stage. Closing this gap requires new codelet variants per radix (fused-first-stage, fused-last-stage); that's the headline v1.1 item.

The DST vs MKL comparison has a caveat — MKL's `MKL_STAGGERED_SINE_TRANSFORM` is part of their Trigonometric Transforms library, designed for PDE boundary-value problems, with a different mathematical definition than FFTW's RODFT10. The timing comparison still says something useful ("how fast can MKL do an N-length sine-like transform K times"), but it's not same-math.

---

## API / convention support

| Feature | Status | Notes |
|---------|--------|-------|
| Split-complex (`re`, `im` separate) | ✅ shipped | Native layout, no interleaved support |
| Interleaved complex (`fftw_complex`) | ❌ never | Would need API rework, not planned |
| In-place execution | ✅ shipped | All transforms |
| Out-of-place execution | ❌ v1.1 | r2r and R2C have 3-pointer convenience wrappers that allocate scratch internally; no native OOP |
| Double precision (FP64) | ✅ shipped | |
| Single precision (FP32) | ❌ never | Codelet generators are FP64-only |
| Half precision (FP16) | ❌ never | |

The split-complex layout is intentional, not an oversight. Split-complex enables:
- Zero-copy I/O for I/Q data sources where re and im come from separate ADCs/channels
- Free conjugation (negate `im`)
- Free Hartley transform (`H[k] = Re - Im` of R2C output)
- Cache-friendly cross-correlation: `conj(A) * B` becomes four pointwise loops on uniform arrays
- Predictable memory footprint for streaming (no allocator hits in the hot loop)

The trade-off: users coming from FFTW's interleaved `fftw_complex` API need a one-time conversion at the boundary.

In-place is the only mode for v1.0. Convenience wrappers for r2r and R2C (`stride_execute_2d_r2c`, `stride_execute_dct2`, etc.) accept a 3-pointer signature (`in`, `out_re`, `out_im`), but they allocate temp scratch internally because the in-place override needs the buffer to be sized for the larger of input/output.

---

## Numeric / size constraints

| Constraint | Where it applies |
|-----------|------------------|
| K must be ≥ 2 | R2C / DCT / DST / DHT / DCT-IV (K=1 hard-rejected, returns NULL) |
| K must be multiple of 4 | All C2C codelet paths — codelets have no scalar tail at vl<4. 2D R2C pads internally; 1D users hit this if they pass non-mult-of-4 K |
| N must be even | R2C / DCT / DST family / DHT (inherits R2C) |
| N must be ≥ 2 | All transforms |
| Arbitrary N (primes) | ✅ shipped via Bluestein (any prime) and Rader (smooth-N-1 primes) |

The K-multiple-of-4 constraint is the most surprising one for users. It surfaces for any caller passing batched K not divisible by 4. The 2D R2C path pads internally (K_pad = ceil((N2/2+1)/4)*4) to dodge this, paying ~6% extra col-FFT work for typical N2. Removing the constraint requires v1.1 codelet rewrites with proper scalar tails.

---

## Threading

| Feature | Status |
|---------|--------|
| 1D C2C K-split MT | ✅ shipped |
| 1D C2C group-parallel MT | ✅ shipped |
| Bluestein / Rader block-walk MT | ✅ shipped (T-aware block sizing) |
| 1D R2C block-walk MT | ✅ shipped |
| 2D C2C tile-parallel + K-split MT | ✅ shipped |
| 2D R2C forward tile-parallel | ✅ shipped |
| 2D R2C backward MT | ❌ v1.1 (single-threaded; reverse-iteration constraint blocks naive tile parallelism) |
| Thread pinning | ✅ shipped |

The threading model is heterogeneous on purpose — different transforms benefit from different parallel decompositions:
- Pure 1D C2C uses K-split (each thread owns a contiguous batch slice) plus group-parallel within stages.
- Bluestein and Rader use block-walk: each thread owns a batch block and runs the full pipeline serially within it.
- 2D row passes parallelize over independent tiles; col passes use K-split.
- 2D R2C backward currently runs single-threaded because the in-place backward must process tiles in reverse order to avoid overwriting future tiles' input. Tile-parallel reverse iteration is solvable but not in scope for v1.0.

---

## Wisdom

| Feature | Status |
|---------|--------|
| 1D C2C wisdom (MEASURE) | shipped |
| 1D R2C wisdom (DIT-only) | shipped — DIF rejected as safety |
| 1D r2r wisdom (DCT/DST/DHT) | shipped (inherits R2C wisdom) |
| 2D C2C wisdom | partial — col FFT non-wisdom (K-split corruption safety) |
| 2D R2C wisdom | partial — both inner plans non-wisdom (paranoia) |

Wisdom v5 stores per-stage variant codes (FLAT / LOG3 / T1S / BUF) plus orientation (DIT / DIF) and blocked-executor parameters. The MEASURE pass times each (factorization × executor × split-point) cell and writes the winner.

Two known wisdom limitations:
1. **R2C requires DIT** because the fused first/last-stage paths assume stage 0 is twiddle-free and executes first. DIF inner plans get silently mis-handled — `_r2c_force_dit_inner` rejects them and falls back to a non-wisdom DIT plan.
2. **K-split + variant-coded plan corruption** at certain (T, K, N) combinations (notably 2D 1024² T=2/T=4) produces garbage output. The mechanism is not yet root-caused — investigation deferred to v1.1. Workaround: 2D plans skip wisdom for the column FFT.

---

## File map

| File | Role |
|------|------|
| `planner.h` | Public planning entry points: `stride_auto_plan`, `stride_wise_plan`, `stride_plan_2d`, `stride_plan_2d_r2c`, `stride_dct2_*`, `stride_dst2_*`, `stride_dct4_*`, `stride_dht_*`, `stride_r2c_*` |
| `executor.h` | Stage execution: `stride_execute_fwd`, `stride_execute_bwd`, slice variants, threaded dispatch |
| `executor_blocked.h` | Wide + blocked two-phase executor (when wisdom selects it) |
| `registry.h` | Codelet function-pointer registry (radix → n1/t1/n1_oop function pointers) |
| `factorizer.h` | Greedy largest-first factorization, SIMD-aware reorder |
| `dp_planner.h` | DP-based factorization search with memoization |
| `exhaustive.h` | Exhaustive (factorization × variant × orientation) search for wisdom |
| `wisdom_bridge.h` | Per-codelet variant-selection predicates (load-bearing as MEASURE coarse-pass prior; deprecation plan in roadmap_wisdom_bridge_retirement.md) |
| `bluestein.h` | Arbitrary-prime FFT via Bluestein convolution; T-aware block sizing |
| `rader.h` | Smooth-N-1 prime FFT via Rader; shares block-walk infrastructure with Bluestein |
| `r2c.h` | 1D R2C / C2R (pair-packing) |
| `fft2d.h` | 2D C2C (tiled + Bailey) |
| `fft2d_r2c.h` | 2D R2C / C2R |
| `dct.h` | DCT-II / DCT-III via Makhoul (+ N=8 specialized codelets in `vectorfft_tune/generated/dct8/`) |
| `dct4.h` | DCT-IV via Lee 1984 |
| `dst.h` | DST-II / DST-III via DCT wrappers |
| `dht.h` | Discrete Hartley Transform |
| `transpose.h` | 8×4 (AVX2) and 8×8 (AVX-512) line-filling SIMD transpose, used by 2D and Bailey |
| `threads.h` | Thread pool primitives, pinning, num-threads management |
| `env.h` | One-time runtime initialization (`stride_env_init`) |
| `workspace.h` | Per-thread scratch buffer management |

---

## Known v1.0 limitations / v1.1 backlog

| Item | Type | Why deferred |
|------|------|--------------|
| Out-of-place execution | API | Needs callsite + plan-struct refactor |
| 3D / ND R2C and C2C | feature | Recurses on 2D; no point until 2D R2C is performant |
| R2C codelet-fusion | codelet | Fused-first-stage and fused-last-stage codelet variants per radix — closes the FFTW/MKL gap on 1D R2C and 2D R2C |
| Specialized r2hc_8/16/32 codelets | codelet | Per-N straight-line codelets (analogous to our DCT N=8) |
| Codelet vl<4 scalar tail | codelet | Removes the K-multiple-of-4 caller constraint |
| K-split + variant-coded plan corruption | bug | Investigation; possibly codelet thread-safety audit |
| 256² MT scaling regression | bug | Investigation |
| DCT-I / DST-I / DST-IV | feature | Completionist, low demand |
| Single precision (FP32) | codelet | Would double codelet generator output |
| Convolution helpers | feature | FFTW and VkFFT have these |
| Native zero-padded transforms | feature | FFTW and VkFFT have these |
| EPYC / Zen 5 / AOCC validation | porting | Different ISA path; codelet generators need cross-uarch tuning |

The v1.1 priority is R2C codelet-fusion. Two transforms benefit (1D R2C, 2D R2C) and the fix is well-scoped: extend the radix codelet generators to emit fused first-stage (twiddle + scaled output) and fused last-stage (input scale + butterfly) variants, then replace R2C's three-pass design with a one-pass walk through fused stages.

---

## Building

The new core is header-only, included via `planner.h`. The build harness is `build_tuned/build.py` (Python, calls Intel ICX with the right include paths). For an integrated CMake build, see `build_tuned/CMakeLists.txt` once it's added (CMake integration is on the v1.0 polish list).

To compile a test or bench against this core:
```
cd build_tuned
python build.py --src test_tuned_core.c
python build.py --src bench_1d_vs_mkl.c --mkl
python build.py --src test_fft2d_r2c.c --fftw
```
