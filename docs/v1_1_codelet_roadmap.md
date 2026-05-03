# v1.1 codelet roadmap

This is the planned codelet work for v1.1 of VectorFFT, in order of impact. v1.0 ships from `src/core/` with the limitations documented in [`src/core/README.md`](../src/core/README.md); the items below are what's needed to close those gaps.

---

## Three new codelet categories

### 1. R2C-fusion codelets (per radix) — closes the 1D R2C / 2D R2C gap

The current 1D R2C is structurally a 3-pass design: pack → C2C → butterfly. FFTW and MKL fuse the pack into the first DIT stage and the butterfly into the last DIT stage, which is why we lose to MKL on 1D R2C and to FFTW on 2D R2C. Closing this requires two new variants of every existing radix codelet, one for each "edge" of the inner FFT:

| Variant | What it does | Replaces today's |
|---------|--------------|------------------|
| `t1_r2c_first_R` | First-stage codelet that reads 2K reals (pair-packed `x[2n] + i·x[2n+1]`) directly from input layout. Combines pack + first DIT butterfly. | Explicit pack-pass + plain `n1_R` |
| `t1_r2c_last_R` | Last-stage codelet that applies the X[k] post-process butterfly inline with the final DIT write. | Plain `n1_R` + explicit butterfly pass |
| `t1_c2r_first_R` | Symmetric for C2R: pre-process butterfly fused into first DIT stage. | Pre-process pass + `n1_R` |
| `t1_c2r_last_R` | Symmetric for C2R: ×2 unpack scaling fused into last stage. | `n1_R` + explicit unpack |

This is **4 new variants × every radix we ship** (R = 2/3/4/5/6/7/8/10/11/12/13/16/17/19/20/25/32/64). Roughly 70–80 new codelets. The generator framework exists (we already have `t1_oop` and `n1_scaled` partially wired for the R2C path), but the comprehensive matrix needs filling in. This single workstream closes both the 1D R2C MKL gap and the 2D R2C FFTW gap.

### 2. Specialized straight-line codelets at small N — closes JPEG / audio-codec gap

Like our `dct2_n8_avx2` codelet — entire transform as one straight-line kernel for hot small-N cells:

| Codelet | Use case |
|---------|----------|
| `r2hc_8` | JPEG block size, 8-tap audio frames |
| `r2hc_16` | MDCT short-window (MP3/AAC) |
| `r2hc_32` | MDCT long-window |
| `e10_16`, `e10_32` | DCT-II short/long windows |
| `e11_16`, `e11_32` | DCT-IV (MDCT proper) — replaces Lee at small N where Lee's overhead dominates |
| `dst_*` analogs | DST equivalents if usage justifies |

Each is ~150–200 LOC straight-line, generated once. Maybe 10–15 codelets total. Wins at small-K cells where current Lee/Makhoul wrapper overhead dominates the actual compute.

### 3. Codelet `vl<4` scalar tail — removes the K-multiple-of-4 caller constraint

Every existing radix codelet today has the structure:

```c
for (size_t k = 0; k < vl; k += 4) { /* AVX2 */ }
```

with no scalar tail. We need:

```c
for (; k + 4 <= vl; k += 4) { /* AVX2 SIMD */ }
for (; k < vl; k++)          { /* scalar */ }
```

to every `n1`, `t1`, `t1s`, `t1_oop`, `n1_scaled`, `t1_r2c_*`, `t1_c2r_*` codelet. **Not a new category — a per-codelet modification.** Touches all ~50+ existing codelets across the registry.

This is what unblocks K=1, K=2, K=3, K=5, K=6, K=7, K=9, etc. caller-side. Currently the planner rejects K=1 (R2C only) and 2D R2C pads internally — ugly workarounds. Proper fix here.

---

## Optional / nice-to-have categories

### FP32 versions — single-source via SIMD type macros

Standard approach: typedef-driven single-source codelets, compile twice for FP64 / FP32 with a precision macro:

```c
/* vfft_simd.h */
#ifdef VFFT_FP32
  typedef float       vfft_real_t;
  typedef __m256      vfft_simd_t;     /* 8-wide */
  #define VFFT_LANES  8
  #define VFFT_LOAD   _mm256_loadu_ps
  #define VFFT_STORE  _mm256_storeu_ps
  #define VFFT_FMA    _mm256_fmadd_ps
  #define VFFT_SET1   _mm256_set1_ps
#else  /* FP64 default */
  typedef double      vfft_real_t;
  typedef __m256d     vfft_simd_t;     /* 4-wide */
  #define VFFT_LANES  4
  #define VFFT_LOAD   _mm256_loadu_pd
  #define VFFT_STORE  _mm256_storeu_pd
  #define VFFT_FMA    _mm256_fmadd_pd
  #define VFFT_SET1   _mm256_set1_pd
#endif
```

Codelets use `vfft_simd_t`, `VFFT_LOAD`, `VFFT_LANES` everywhere. Generator emits one source file per radix that compiles cleanly under either precision. FFTW does this exact pattern.

**What this buys cheaply:**
- Single source of truth — bug fix in FP64 propagates to FP32 automatically.
- Generator output halves vs "two separate sets."
- Per-codelet symbol naming via `_PASTE(radix8_n1_fwd, _avx2)` macro that picks `_pd` or `_ps` suffix; both versions can coexist in the binary.
- Twiddle tables already use `cos`/`sin` constants — emit them as `static const vfft_real_t[]`, works for both.

**Caveats — what stays manual:**
- **Loop bounds shift from `k+=4` to `k+=VFFT_LANES`**, which means K-multiple-of-4 becomes K-multiple-of-8 at FP32. Acceptable as long as the scalar tail (category 3 above) is in place.
- **Optimal radix ordering changes.** At FP32 + AVX2, an R=8 butterfly fits in 8 SIMD registers and runs faster relative to R=4 than at FP64. The factorizer's "greedy largest-first" + wisdom-driven joint search would re-pick winners on FP32 — same calibration pass, different wisdom file.
- **Specific tricks (log3, buf, t1s) may have different break-even points.** Memory pressure scales differently with width. So `wisdom_bridge` predicates would need re-tuning per precision.
- **AVX-512 widens further** (8-wide FP64 / 16-wide FP32), so the lane-count macro pattern keeps working but the optimal scheduling might want different unrolling. The codelet body unrolls would carry over (correctness), perf-tuned versions would benefit from a separate AVX-512-FP32 generator pass.

**Concrete v1.1 path with this:**
1. Add `vfft_simd.h` with the typedef/macro layer. Backwards-compatible — current FP64 path is the default.
2. Modify `gen_radix*.py` to emit codelets in macro form (mostly find-and-replace `__m256d` → `vfft_simd_t`, `_mm256_loadu_pd` → `VFFT_LOAD`, etc.).
3. Build twice with different `-DVFFT_FP32` and a symbol suffix, ship both in the binary.
4. Run the wisdom calibrator on FP32 cells separately; ship a `vfft_wisdom_fp32.txt` alongside the FP64 wisdom.

Total cost: ~1 session for the macro layer + generator changes, plus a calibration run (overnight job, no human time). Compared to "generate a separate FP32 set from scratch" which would be 2-3× the codelet body work, this is dramatically cheaper. Same approach scales to FP128 (`long double` or double-double) if EPYC scientific users ever ask.

### 2D R2C stride-aware first-stage

Subset of category 1 with strided-load codelets — reads input with stride N2 between samples, no row-transpose needed. Eliminates the gather/scatter passes around the inner FFT in 2D R2C. Independent improvement on top of category 1.

### EPYC / Zen 5 retuned codelets

Same generator, different prefetch / log3 / buf calibration per uarch. Codelet **bodies** carry over; per-codelet `wisdom_bridge` predicates and the wisdom file are the only things that change. Calibration runtime, not codegen time.

### DCT-I, DST-I, DST-IV completionist transforms

Each is one new wrapper transform plus optional small-N specialized codelets. Only worth doing if there's user demand — the audio/JPEG/MDCT use cases are already covered by the DCT-II/III/IV and DST-II/III set we shipped in v1.0.

---

## Suggested v1.1 codelet sequencing

1. **`vl<4` scalar tail first** — it's a pure modification, no new generator logic, fixes a real correctness sharp edge, and is groundwork for everything else. The new R2C variants will be cleaner if they don't need to pad.
2. **R2C-fusion codelets second** — biggest perf win, closes both 1D and 2D R2C.
3. **Straight-line `r2hc`/`e10`/`e11` small-N third** — completes the small-N story for r2r and R2C, mirrors what we did for DCT N=8.
4. **FP32, EPYC, completionist transforms** in any order after that.

---

## What this v1.1 work delivers

| Today (v1.0) | After v1.1 codelet work |
|--------------|-------------------------|
| 1D R2C: ~1.5× over FFTW, loses to MKL | Wins ~2× over MKL like 1D C2C does |
| 2D R2C: 0.33–0.77× of FFTW (loses) | Wins over FFTW like 2D C2C does (1.08–1.63× over MKL) |
| K=1 rejected, K not multiple of 4 silently broken in 1D C2C | Any K accepted, scalar tail handles it |
| FP64 only | FP64 + FP32, same source, separate wisdom |
| Intel Raptor Lake tuned | Intel + Zen 4/5 tuned via per-uarch calibration |
| Small-N r2r/R2C wrapper-overhead dominated | Small-N has straight-line specialized codelets |
