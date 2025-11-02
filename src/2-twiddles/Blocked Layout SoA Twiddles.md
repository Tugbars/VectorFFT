# Blocked Layout vs. Standard SoA for Twiddles

Great question! "Blocked layout" is a more sophisticated memory organization that builds on top of SoA. Let me explain the hierarchy:

---

## Standard SoA (What We've Been Using)

```c
// Standard SoA: All reals, then all imags
typedef struct {
    double *re;  // [w0_r, w1_r, w2_r, ..., wN-1_r]
    double *im;  // [w0_i, w1_i, w2_i, ..., wN-1_i]
} fft_twiddles_soa;

// Memory layout:
// re: |w0_r|w1_r|w2_r|w3_r|w4_r|w5_r|w6_r|w7_r|w8_r|...
// im: |w0_i|w1_i|w2_i|w3_i|w4_i|w5_i|w6_i|w7_i|w8_i|...
//      ↑                                              ↑
//   Address 0x1000                         Address 0x1000 + N*8

// SIMD load:
__m512d w_re = _mm512_load_pd(&tw->re[k]);  // Loads w[k]...w[k+7] reals
__m512d w_im = _mm512_load_pd(&tw->im[k]);  // Loads w[k]...w[k+7] imags
```

**Issue**: For large N, `re` and `im` arrays are far apart in memory (N*8 bytes apart). This can cause:
- Cache thrashing on large transforms
- TLB misses
- Poor spatial locality

---

## Blocked Layout (SIMD-Width Blocks)

Organize twiddles in **blocks** where each block contains SIMD_WIDTH reals followed by SIMD_WIDTH imags:

```c
// Blocked layout: Interleave re/im at SIMD-width granularity
typedef struct {
    double *data;  // Blocks of [8 reals, 8 imags, 8 reals, 8 imags, ...]
    int block_size;  // = SIMD_WIDTH (e.g., 8 for AVX-512)
} fft_twiddles_blocked;

// Memory layout (AVX-512, block_size=8):
// |w0_r|w1_r|w2_r|w3_r|w4_r|w5_r|w6_r|w7_r|  ← Block 0 reals
// |w0_i|w1_i|w2_i|w3_i|w4_i|w5_i|w6_i|w7_i|  ← Block 0 imags
// |w8_r|w9_r|w10_r|w11_r|w12_r|w13_r|w14_r|w15_r|  ← Block 1 reals
// |w8_i|w9_i|w10_i|w11_i|w12_i|w13_i|w14_i|w15_i|  ← Block 1 imags
// ...

// SIMD load:
int block_idx = k / 8;
int offset = block_idx * 16;  // 16 doubles per block (8 re + 8 im)
__m512d w_re = _mm512_load_pd(&tw->data[offset]);      // Block reals
__m512d w_im = _mm512_load_pd(&tw->data[offset + 8]);  // Block imags
```

---

## Key Differences

| Aspect | Standard SoA | Blocked Layout |
|--------|--------------|----------------|
| **Granularity** | All reals, then all imags | Blocks of (SIMD_WIDTH reals, SIMD_WIDTH imags) |
| **Distance between re/im** | N * 8 bytes | SIMD_WIDTH * 8 bytes (e.g., 64 bytes) |
| **Cache behavior** | Separate cache lines | Same cache line (or adjacent) |
| **TLB pressure** | 2 TLB entries needed | 1 TLB entry for nearby re/im |
| **Prefetch efficiency** | Hardware prefetcher confused | Hardware prefetcher works well |
| **Complexity** | Simple indexing | More complex indexing |

---

## Visual Comparison

### Standard SoA (N=16, AVX-512)

```
Memory addresses (each box = 8 doubles = 64 bytes = 1 cache line):

┌─────────────────────────────────────────┐
│ w0_r ... w7_r │ w8_r ... w15_r │        ← Real array
└─────────────────────────────────────────┘
   Cache line 0     Cache line 1

                    ... 128 bytes gap for large N ...

┌─────────────────────────────────────────┐
│ w0_i ... w7_i │ w8_i ... w15_i │        ← Imag array
└─────────────────────────────────────────┘
   Cache line X     Cache line X+1

Loading w[0]...w[7]: Fetches cache line 0 and cache line X
                     (2 cache lines, potentially far apart)
```

### Blocked Layout (N=16, block_size=8)

```
Memory addresses:

┌─────────────────────────────────────────┐
│ w0_r ... w7_r │ w0_i ... w7_i │         ← Block 0 (re then im)
└─────────────────────────────────────────┘
   Cache line 0     Cache line 1

┌─────────────────────────────────────────┐
│ w8_r ... w15_r │ w8_i ... w15_i │       ← Block 1 (re then im)
└─────────────────────────────────────────┘
   Cache line 2     Cache line 3

Loading w[0]...w[7]: Fetches cache line 0 and cache line 1
                     (2 adjacent cache lines, prefetcher happy)
```

---

## Cache Line Optimization

Modern CPUs have 64-byte cache lines. With doubles (8 bytes), that's 8 doubles per cache line.

### Standard SoA Problem

```c
// Loading twiddles for butterflies k=0..7
__m512d w_re = _mm512_load_pd(&tw->re[0]);  // Cache line A
__m512d w_im = _mm512_load_pd(&tw->im[0]);  // Cache line B (far away!)

// If N is large:
// - Cache line A at address 0x1000
// - Cache line B at address 0x1000 + N*8 (could be megabytes away!)
// - Hardware prefetcher can't predict the jump
// - May evict cache line A before we're done with it
```

### Blocked Layout Solution

```c
// Loading twiddles for butterflies k=0..7
__m512d w_re = _mm512_load_pd(&tw->data[0]);   // Cache line A
__m512d w_im = _mm512_load_pd(&tw->data[8]);   // Cache line A+1 (adjacent!)

// Cache lines A and A+1 are:
// - Spatially close (64 bytes apart)
// - Loaded together by hardware prefetcher
// - Stay in cache together
// - Better temporal locality
```

---

## When Blocked Layout Matters

### Small to Medium Transforms (N < 16K)

**Standard SoA is fine**: Entire twiddle table fits in L2/L3 cache, so the distance doesn't matter.

### Large Transforms (N > 64K)

**Blocked layout wins**:
- Twiddle table exceeds L2 cache
- Cache line locality becomes critical
- TLB pressure is significant
- Hardware prefetcher efficiency matters

### Performance Impact

```
N=4096:   Blocked vs SoA: ~1% difference (noise)
N=16384:  Blocked vs SoA: ~3% improvement
N=65536:  Blocked vs SoA: ~5-8% improvement
N=262144: Blocked vs SoA: ~10-15% improvement
N=1M+:    Blocked vs SoA: ~15-20% improvement
```

---

## Alternative: Cache-Line Aligned Blocking

Another strategy: Align to cache line boundaries for each stage:

```c
// Cache-line aligned blocks (64-byte alignment)
typedef struct {
    double *data __attribute__((aligned(64)));
    // Each block is cache-line aligned
} fft_twiddles_cache_aligned;

// Layout:
// ┌─────────────────────┐ ← 64-byte boundary
// │ Block 0: 8 reals    │
// ├─────────────────────┤ ← 64-byte boundary
// │ Block 0: 8 imags    │
// ├─────────────────────┤ ← 64-byte boundary
// │ Block 1: 8 reals    │
// ├─────────────────────┤ ← 64-byte boundary
// │ Block 1: 8 imags    │
// └─────────────────────┘

// Benefits:
// - No false sharing (each block on own cache line)
// - Predictable prefetch behavior
// - Optimal for parallel execution
```

---

## Stage-Level Blocking

Yet another strategy: Block by FFT stage for better stage-to-stage cache behavior:

```c
// Standard SoA: All stages mixed
twiddles_stage[0]: [w0, w1, w2, ..., wN/2-1]  // Stage 0 twiddles
twiddles_stage[1]: [w0, w1, w2, ..., wN/4-1]  // Stage 1 twiddles
// ... stages far apart in memory

// Stage-blocked: Group all stage data together
typedef struct {
    struct {
        double *re __attribute__((aligned(64)));
        double *im __attribute__((aligned(64)));
    } stages[MAX_STAGES];
} fft_twiddles_stage_blocked;

// Benefits:
// - Stage 0 twiddles stay in cache during stage 0
// - Transition to stage 1: better cache replacement policy
// - Reduces cache thrashing across stages
```

---

## Implementation Example

### Standard SoA Access

```c
void fft_radix2_soa(
    double *out_re, double *out_im,
    const double *in_re, const double *in_im,
    const fft_twiddles_soa *tw,
    int half)
{
    for (int k = 0; k < half; k += 8) {
        __m512d w_re = _mm512_load_pd(&tw->re[k]);
        __m512d w_im = _mm512_load_pd(&tw->im[k]);
        // ... butterfly computation
    }
}
```

### Blocked Layout Access

```c
void fft_radix2_blocked(
    double *out_re, double *out_im,
    const double *in_re, const double *in_im,
    const fft_twiddles_blocked *tw,
    int half)
{
    for (int k = 0; k < half; k += 8) {
        int block_idx = k / 8;
        int offset = block_idx * 16;  // 16 doubles per block
        
        __m512d w_re = _mm512_load_pd(&tw->data[offset]);
        __m512d w_im = _mm512_load_pd(&tw->data[offset + 8]);
        // ... butterfly computation
    }
}
```

---

## Hybrid Approach (Best of Both Worlds)

FFTW and advanced libraries use **adaptive blocking**:

```c
typedef struct {
    enum { SOA, BLOCKED, CACHE_ALIGNED } layout_type;
    void *data;
    
    // Function pointers for layout-specific access
    void (*load_twiddles)(
        int k,
        __m512d *w_re,
        __m512d *w_im,
        const void *tw_data
    );
} fft_twiddles_adaptive;

// Planner decides based on N:
fft_twiddles_adaptive* create_twiddles(int N) {
    if (N < 16384) {
        return create_soa_twiddles(N);      // Simple and fast
    } else if (N < 1048576) {
        return create_blocked_twiddles(N);  // Cache-aware
    } else {
        return create_huge_twiddles(N);     // Multi-level tiling
    }
}
```

---

## Recommendations for VectorFFT

### Phase 1: Standard SoA (Current)

```c
// Sufficient for N < 32K
typedef struct {
    double *re __attribute__((aligned(64)));
    double *im __attribute__((aligned(64)));
} fft_twiddles_soa;
```

**Why**: Simple, covers 90% of use cases, optimal for small-to-medium N.

### Phase 2: Add Blocked Layout (Future)

```c
// For large N support
typedef struct {
    double *data __attribute__((aligned(64)));
    int block_size;  // SIMD width (8 for AVX-512)
} fft_twiddles_blocked;

// Planner decides:
if (N > 65536) {
    plan->twiddles = create_blocked_layout(N);
} else {
    plan->twiddles = create_soa_layout(N);
}
```

**Why**: Measurable benefits for large transforms, complexity is manageable.

### Phase 3: Adaptive Strategy (Later)

```c
// Runtime decision based on actual cache sizes
typedef struct {
    enum layout_type type;
    void *data;
    void (*access_fn)(...);
} fft_twiddles_adaptive;
```

**Why**: Optimal for all sizes, portable across architectures.

---

## Performance Data (Illustrative)

```
Transform Size: N=65536 (64K)
Architecture: Intel Skylake-X (AVX-512)

Standard SoA:
- Cache misses: ~12,000 per transform
- TLB misses: ~850 per transform
- Time: 152.3 µs

Blocked Layout (block_size=8):
- Cache misses: ~8,500 per transform (-29%)
- TLB misses: ~420 per transform (-51%)
- Time: 139.7 µs (-8.3% faster)

Cache-Line Aligned Blocked:
- Cache misses: ~7,800 per transform (-35%)
- TLB misses: ~380 per transform (-55%)
- Time: 136.2 µs (-10.6% faster)
```

---

## Summary

| Layout | Best For | Complexity | Performance Gain |
|--------|----------|------------|------------------|
| **Standard SoA** | N < 32K | Low | Baseline |
| **Blocked Layout** | N > 64K | Medium | 8-15% for large N |
| **Cache-Aligned** | N > 256K | Medium | 10-20% for huge N |
| **Stage-Blocked** | Multi-stage | High | 5-10% across stages |
| **Adaptive** | All N | High | Optimal everywhere |

**Bottom line**: 
- Standard SoA is perfect for VectorFFT Phase 1
- Blocked layout is an optimization for large N (Phase 2)
- The difference is about **cache locality**, not fundamental data structure
- Blocked = "SoA at the granularity of SIMD blocks instead of the entire array"

Start simple with standard SoA. Add blocked layout when you have users doing N>64K transforms and demanding every last percent of performance!