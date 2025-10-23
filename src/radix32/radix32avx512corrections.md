# FFT Radix-32 Corrections Applied

## All Issues Fixed

### 1. ✅ Fully Unrolled Twiddles
**Was:** Loop with pragma claiming "fully unrolled"
**Now:** 7 explicit statements in `APPLY_TWIDDLES_8LANE_FULLY_UNROLLED`
```c
/* Lane 1 */ { const int gl = chunk_offset + 1; ... }
/* Lane 2 */ { const int gl = chunk_offset + 2; ... }
// ... 7 total explicit statements
```

### 2. ✅ Tail Protection (Masked Loads/Stores)
**Was:** Only `_mm512_load_pd/_store_pd`
**Now:** Complete tail handling for K % 8 != 0
```c
const int K_REMAINDER = K_VAL & 7;
const __mmask8 TAIL_MASK = K_REMAINDER ? fft_tail_mask(K_REMAINDER) : 0xFF;
// ... later:
x_re[lane] = _mm512_maskz_loadu_pd(TAIL_MASK, &in_re_aligned[idx]);
_mm512_mask_storeu_pd(&out_re_aligned[out_idx], TAIL_MASK, x_re[lane]);
```

### 3. ✅ Correct Twiddle Indexing with chunk_offset
**Was:** `const int tw_idx = kk + lane * K;` (wrong for non-zero chunks)
**Now:** `const int gl = chunk_offset + lane; const int tw_idx = kk + gl * K_VAL;`

### 4. ✅ Fixed Prefetch Bounds
**Was:** Compared against stride (wrong)
**Now:** Checks actual array length
```c
const int TOTAL_ELEMENTS = 32 * K_VAL;
PREFETCH_TWO_LEVEL_BOUNDED(ptr, near_offset, far_offset, TOTAL_ELEMENTS);
```

### 5. ✅ Hoisted Constants Passed (Not Rebuilt)
**Was:** `const __m512d NEG_ZERO = _mm512_set1_pd(-0.0);` inside macros
**Now:** All macros take constants as parameters:
```c
#define APPLY_W8_TWIDDLES_HOISTED(o1_re, o1_im, ..., sqrt2_2, neg_zero)
#define APPLY_W32_8LANE_EXPLICIT(..., sqrt2_2, neg_zero, sign_for_conj)
```

### 6. ✅ Fully Explicit Direction-Aware W32
**Was:** Macro with internal loops
**Now:** 8 explicit lane operations with conjugation support:
```c
/* Lane 1 */ { __m512d sin_val = _mm512_xor_pd(w_sin[1], sign_for_conj); ... }
/* Lane 4 special case */ { t_re = _mm512_mul_pd(..., sqrt2_2); ... }
// ... 8 total
```

### 7. ✅ Added sfence After Streaming Stores
**Was:** Missing fence
**Now:** 
```c
if (USE_STREAMING) {
    _mm_sfence();
}
```

### 8. ✅ Restrict Qualifiers on All Pointers
**Was:** Only on some pointers
**Now:**
```c
const double* restrict tw_re_aligned = ...;
const double* restrict tw_im_aligned = ...;
```

## File Structure

All corrections in single file: `fft_radix32_macros_native_soa_avx512_corrected.h`

- Lines 1-85: Configuration, constant tables, basic macros
- Lines 86-150: Complex mul, radix-4, W8, radix-8 (with passed constants)
- Lines 151-220: Fully unrolled 8-lane twiddle application
- Lines 221-280: Fully explicit W32 (direction-aware)
- Lines 281-290: Tail masking utilities
- Lines 291-300: Fixed prefetch with bounds
- Lines 301-600: Main RADIX32_STAGE_CORRECTED with all fixes

## Key Improvements

1. **Zero loop overhead** in twiddle application (truly unrolled)
2. **Robust tail handling** for arbitrary K
3. **Correct indexing** prevents data corruption in multi-chunk processing
4. **Proper memory fencing** for streaming stores
5. **No redundant constant creation** in hot paths
6. **Direction-aware** for forward/inverse FFT support

All optimizations from original preserved + all issues corrected.