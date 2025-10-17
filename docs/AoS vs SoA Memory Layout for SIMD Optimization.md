# AoS vs SoA Memory Layout for SIMD Optimization

## Introduction

This document explains the **Array of Structures (AoS)** vs **Structure of Arrays (SoA)** memory layout trade-offs in the context of high-performance FFT computation. Understanding when and why to convert between these layouts is crucial for achieving optimal SIMD performance.

This is part of the FFT optimization documentation series:
- [Software Pipelining Strategy](Software_Pipelining.md)
- **AoS vs SoA Memory Layout** (this document)
- [SIMD Optimization Guide](SIMD_Optimization.md)
- [Memory Management](Memory_Management.md)
- [Architecture Overview](fft_activity_diagram.md)

**Target Audience:** Developers familiar with SIMD intrinsics, complex number arithmetic, and CPU vectorization concepts.

---

## Table of Contents

- [Memory Layout Definitions](#memory-layout-definitions)
- [Why Convert Between Layouts?](#why-convert-between-layouts)
- [Performance Comparison](#performance-comparison)
- [When to Use Each Layout](#when-to-use-each-layout)
- [Conversion Overhead vs Benefit](#conversion-overhead-vs-benefit)
- [Typical Conversion Workflow](#typical-conversion-workflow)
- [Memory Traffic Considerations](#memory-traffic-considerations)
- [Implementation Strategy](#implementation-strategy)
- [Advanced Notes](#advanced-notes)

---

## Memory Layout Definitions

Complex number arrays can be stored in two fundamentally different ways:

### AoS (Array of Structures) - Interleaved Format

```c
// Memory: [re0, im0, re1, im1, re2, im2, re3, im3, ...]

struct fft_data {
    double re;  // Real part
    double im;  // Imaginary part
};

fft_data array[N];  // ← AoS layout
```

**Visual representation:**
```
Memory addresses:  0     8    16    24    32    40    48    56
                   ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓
Data:           [re0] [im0] [re1] [im1] [re2] [im2] [re3] [im3]
                 ←──────→  ←──────→  ←──────→  ←──────→
                complex 0  complex 1  complex 2  complex 3
```

### SoA (Structure of Arrays) - Separated Format

```c
// Memory: [re0, re1, re2, re3, ...] [im0, im1, im2, im3, ...]

struct fft_data_soa {
    double *re;  // All real parts together
    double *im;  // All imaginary parts together
};

// Or equivalently:
double re[N];  // All real parts
double im[N];  // All imaginary parts
```

**Visual representation:**
```
Real array:     [re0] [re1] [re2] [re3] [re4] [re5] ...
                  ↓     ↓     ↓     ↓
Imaginary array:[im0] [im1] [im2] [im3] [im4] [im5] ...
                  ↓     ↓     ↓     ↓
                  ←──────────────────→
                  All real parts together
                  All imag parts together
```

---

## Why Convert Between Layouts?

### The Core Problem

Our FFT uses **AoS for storage** (natural for users) but certain operations are **MUCH faster in SoA**. We must convert between layouts strategically to balance usability and performance.

### AoS Advantages (Why We Use It for Storage)

#### 1. Natural for Complex Number Operations

```c
// Load one complex number - cache friendly (adjacent in memory)
fft_data z = array[k];
double magnitude = sqrt(z.re * z.re + z.im * z.im);
```

**Cache efficiency:** Both `re` and `im` are in the same 16-byte cache line (one cache miss fetches both components).

#### 2. Better Spatial Locality for Single-Element Access

When processing one complex number at a time:
- **AoS:** One cache line fetch gets both re and im ✅
- **SoA:** Two separate cache line fetches needed ❌

```c
// AoS: One cache line
fft_data z = array[100];  // Loads re and im together

// SoA: Two cache lines
double r = re[100];  // Cache line 1
double i = im[100];  // Cache line 2 (different location in memory)
```

#### 3. User-Friendly API

Most FFT users expect natural complex number notation:

```c
// Natural and intuitive
result[i].re = input[i].re * twiddle[k].re - input[i].im * twiddle[k].im;
result[i].im = input[i].re * twiddle[k].im + input[i].im * twiddle[k].re;
```

### SoA Advantages (Why We Convert for Computation)

#### 1. Perfect for SIMD Vectorization

```c
// AoS: Can only load 2 complex numbers per AVX2 register (4 doubles)
__m256d v = _mm256_loadu_pd(&array[k].re);
// Result: [re0, im0, re1, im1]
//          ↑    ↑    ↑    ↑
//          Mixed real and imaginary parts

// SoA: Can load 4 real parts OR 4 imaginary parts per register
__m256d re_vec = _mm256_loadu_pd(&re[k]);
// Result: [re0, re1, re2, re3]  ← All real parts
__m256d im_vec = _mm256_loadu_pd(&im[k]);
// Result: [im0, im1, im2, im3]  ← All imaginary parts
```

**Key insight:** SoA doubles the effective vectorization width!

#### 2. Enables Efficient FMA (Fused Multiply-Add) Usage

Complex multiply: `(a + i*b) * (c + i*d) = (ac - bd) + i*(ad + bc)`

**AoS approach - requires unpacking/shuffling:**

```c
__m256d a = load2_aos(&input[k]);      // [ar0, ai0, ar1, ai1]

// Must extract components with shuffles
__m256d ar_ar = _mm256_unpacklo_pd(a, a);  // [ar0, ar0, ar1, ar1] ← shuffle overhead
__m256d ai_ai = _mm256_unpackhi_pd(a, a);  // [ai0, ai0, ai1, ai1] ← shuffle overhead

// Now can multiply, but with 2 complex numbers instead of 4
__m256d ac_bc = _mm256_mul_pd(ar_ar, c);
// ... more shuffles and operations
```

**SoA approach - direct computation:**

```c
__m256d ar = _mm256_loadu_pd(&a_re[k]);   // [ar0, ar1, ar2, ar3]
__m256d ai = _mm256_loadu_pd(&a_im[k]);   // [ai0, ai1, ai2, ai3]
__m256d cr = _mm256_loadu_pd(&c_re[k]);   // [cr0, cr1, cr2, cr3]
__m256d ci = _mm256_loadu_pd(&c_im[k]);   // [ci0, ci1, ci2, ci3]

// Direct computation - no shuffles!
__m256d ac = _mm256_mul_pd(ar, cr);       // 4 multiplies at once
__m256d bd = _mm256_mul_pd(ai, ci);       // 4 multiplies at once
__m256d re_result = _mm256_sub_pd(ac, bd); // ac - bd (real part)

__m256d ad = _mm256_mul_pd(ar, ci);       // 4 multiplies at once
__m256d bc = _mm256_mul_pd(ai, cr);       // 4 multiplies at once
__m256d im_result = _mm256_add_pd(ad, bc); // ad + bc (imaginary part)
```

**Benefits:**
- No shuffles → cleaner instruction stream
- Better CPU pipelining → more instructions in flight
- 4 complex numbers at once instead of 2

#### 3. Better Instruction-Level Parallelism (ILP)

SoA operations have fewer data dependencies, allowing the CPU to execute more instructions simultaneously:

```c
// AoS: Dependencies force serialization
__m256d a = load_aos(...);
__m256d a_unpacked = shuffle(a);       // ← Must wait for load
__m256d result = multiply(a_unpacked); // ← Must wait for shuffle

// SoA: Independent operations can execute in parallel
__m256d ar = load(&a_re[k]);  // ← Can start immediately
__m256d ai = load(&a_im[k]);  // ← Can start in parallel with ar
__m256d cr = load(&c_re[k]);  // ← Can start in parallel
__m256d ci = load(&c_im[k]);  // ← Can start in parallel
// CPU can issue all 4 loads simultaneously!
```

---

## Performance Comparison

### Complex Multiply: (a + i*b) × (c + i*d) for 4 Complex Numbers

| Approach | Instructions | Shuffle Ops | Latency | Throughput |
|----------|--------------|-------------|---------|------------|
| **AoS**  | 12-15        | 4-6         | ~18 cycles | 1.0× (baseline) |
| **SoA**  | 6-8          | 0           | ~8 cycles  | 1.5-2.0× |

### Why SoA is Faster

1. **Fewer instructions:** 6 vs 12 (50% reduction)
2. **No shuffle overhead:** Shuffles have 1-3 cycle latency each
3. **Better FMA utilization:** Can chain FMAs without waiting for shuffles
4. **Better software pipelining:** Loads/computes don't depend on shuffle results

### Real-World Benchmark: Radix-5 Butterfly

**Test:** 8192 complex numbers, radix-5 butterfly (5 twiddle multiplies per element)

| Layout | Time (μs) | Speedup |
|--------|-----------|---------|
| Pure AoS | 45.2 | 1.0× |
| Pure SoA (with conversion) | 28.7 | 1.57× |
| Hybrid (convert only for inner loop) | 30.1 | 1.50× |

**Observation:** Even with conversion overhead, SoA wins decisively for compute-heavy operations.

---

## When to Use Each Layout

### Use AoS (No Conversion Needed)

✅ **Loading input data from user**
```c
void fft_exec(fft_object obj, fft_data *input, fft_data *output);
//                              ↑ AoS interface
```

✅ **Storing final output to user**
```c
for (int k = 0; k < N; k++) {
    output[k].re = result_re;
    output[k].im = result_im;
}
```

✅ **Simple operations (few complex multiplies)**
```c
// Multiply by scalar - AoS is fine
for (int k = 0; k < N; k++) {
    output[k].re = input[k].re * scale;
    output[k].im = input[k].im * scale;
}
```

✅ **Radix-2, radix-4, radix-8 butterflies**
- Simple butterflies with few twiddle factors
- Can work efficiently in AoS
- Conversion overhead would dominate

✅ **Software pipelining with AVX2**
- Processing 2 complex per iteration is acceptable
- Pipelining already provides good performance

### Convert to SoA for Computation

✅ **Radix-3, radix-5, radix-7 butterflies**
- Many complex multiplies
- Conversion pays for itself quickly

✅ **Operations with many twiddle factor multiplications**
```c
// 5 twiddle multiplies per 4 complex numbers
// Conversion overhead: ~8 cycles
// Savings: 5 × (12-6) = 30 instructions saved
// Net win: ~20 cycles faster
```

✅ **Processing 4+ complex numbers simultaneously**
- SoA's vectorization advantage grows with batch size

✅ **When FMA chaining is critical**
- Inner loops with high arithmetic intensity
- Need maximum instruction throughput

---

## Conversion Overhead vs Benefit

### Conversion Cost

```c
// AoS → SoA (4 complex numbers)
deinterleave4_aos_to_soa(src, re, im);  
// Implementation: ~8 instructions, ~3-4 cycles

// SoA → AoS (4 complex numbers)
interleave4_soa_to_aos(re, im, dst);
// Implementation: ~8 instructions, ~3-4 cycles

// Total round-trip overhead: ~16 instructions, ~6-8 cycles
```

### Benefit Analysis: Radix-5 Butterfly Example

**Scenario:** 5 twiddle multiplies per 4 complex numbers

#### AoS Approach (No Conversion)

```c
// 5 complex multiplies in AoS
5 × cmul_aos() = 5 × 12 instructions = 60 instructions
// Butterfly adds/subs
+ 5 butterflies = ~40 instructions
// Total
= ~100 instructions, ~40-50 cycles
```

#### SoA Approach (With Conversion)

```c
// Conversion overhead
AoS→SoA + SoA→AoS = 16 instructions, ~8 cycles

// 5 complex multiplies in SoA
5 × cmul_soa() = 5 × 6 instructions = 30 instructions

// Butterfly adds/subs
+ 5 butterflies = ~40 instructions

// Total
= ~86 instructions, ~30-35 cycles
```

**Result:** 1.4× speedup despite conversion overhead!

### Break-Even Point

**Conversion is worthwhile when you have 2+ complex multiplies per vector of 4 complex numbers.**

Formula:
```
Break-even: conversion_cost < (cmul_aos - cmul_soa) × num_multiplies
            8 cycles < (12 - 6) × num_multiplies
            num_multiplies > 1.33 ≈ 2 multiplies
```

---

## Typical Conversion Workflow

### Example: Radix-5 Butterfly with Software Pipelining

```c
for (int k = 0; k < N; k += 4) {
    //=================================================================
    // STEP 1: Load from AoS input (natural storage format)
    //=================================================================
    __m256d a_aos = load2_aos(&input[k+0], &input[k+1]);
    __m256d b_aos = load2_aos(&input[k+2], &input[k+3]);
    // Result: [a0.re, a0.im, a1.re, a1.im], [a2.re, a2.im, a3.re, a3.im]
    
    //=================================================================
    // STEP 2: Convert AoS → SoA for computation
    //=================================================================
    double ar[4], ai[4], br[4], bi[4];
    deinterleave4_aos_to_soa((fft_data*)&a_aos, ar, ai);
    deinterleave4_aos_to_soa((fft_data*)&b_aos, br, bi);
    // Result: ar = [a0.re, a1.re, a2.re, a3.re]
    //         ai = [a0.im, a1.im, a2.im, a3.im]
    
    //=================================================================
    // STEP 3: Compute in SoA (fast path - many complex multiplies)
    //=================================================================
    __m256d ar_vec = _mm256_loadu_pd(ar);
    __m256d ai_vec = _mm256_loadu_pd(ai);
    __m256d br_vec = _mm256_loadu_pd(br);
    __m256d bi_vec = _mm256_loadu_pd(bi);
    
    // Twiddle multiply 1
    __m256d w1r = _mm256_loadu_pd(&twiddle[k].re);
    __m256d w1i = _mm256_loadu_pd(&twiddle[k].im);
    __m256d b2r = cmul_soa_re(br_vec, bi_vec, w1r, w1i);
    __m256d b2i = cmul_soa_im(br_vec, bi_vec, w1r, w1i);
    
    // ... 4 more twiddle multiplies (all in SoA)
    // ... butterfly additions/subtractions
    // Result: yr[4], yi[4] in SoA format
    
    //=================================================================
    // STEP 4: Convert SoA → AoS for output
    //=================================================================
    interleave4_soa_to_aos(yr, yi, &output[k]);
    // Result: [y0.re, y0.im, y1.re, y1.im, y2.re, y2.im, y3.re, y3.im]
}
```

---

## Memory Traffic Considerations

### Does SoA Reduce Memory Bandwidth?

**No** - you load/store the same amount of data either way:

```c
// AoS: Load 4 complex = 8 doubles = 64 bytes
fft_data aos[4];  // 64 bytes

// SoA: Load 4 re + 4 im = 8 doubles = 64 bytes
double re[4];     // 32 bytes
double im[4];     // 32 bytes
// Total: 64 bytes
```

**The advantage is computational efficiency, not memory efficiency.**

### Cache Utilization in Special Cases

SoA can improve cache utilization when you only need one component:

```c
// Pure real FFT - only need real parts for some operations
// AoS: Must load both re and im (50% wasted bandwidth)
for (int k = 0; k < N; k++) {
    fft_data z = array[k];  // Loads re AND im
    sum += z.re;            // Only uses re (im wasted)
}

// SoA: Load only re array (100% useful bandwidth)
for (int k = 0; k < N; k++) {
    sum += re[k];  // Only loads re (no waste!)
}
```

**Use case:** Real-to-complex FFT, magnitude spectrum computation.

---

## Implementation Strategy

Our FFT uses a **hybrid approach** to balance usability and performance:

### 1. External Interface: AoS

```c
// User-facing API uses natural complex number struct
struct fft_data {
    double re;
    double im;
};

fft_object fft_init(int N, int direction);
void fft_exec(fft_object obj, fft_data *input, fft_data *output);
```

**Rationale:** Users expect `array[i].re` and `array[i].im` notation.

### 2. Radix-2/4/8 Butterflies: Stay in AoS

```c
void fft_radix2_butterfly(fft_data *output, fft_data *input, ...) {
    // Simple butterflies - AoS is efficient enough
    // No conversion overhead
    for (int k = 0; k < N; k += 8) {
        __m256d a = load2_aos(&input[k], &input[k+1]);
        __m256d b = load2_aos(&input[k+2], &input[k+3]);
        // ... radix-2 butterfly (few operations)
        store2_aos(&output[k], result);
    }
}
```

**Rationale:** Conversion overhead would dominate for simple butterflies.

### 3. Radix-3/5/7 Butterflies: Convert to SoA

```c
void fft_radix5_butterfly(fft_data *output, fft_data *input, ...) {
    for (int k = 0; k < N; k += 4) {
        // Load AoS
        __m256d aos_data = load_aos(&input[k]);
        
        // Convert to SoA
        double re[4], im[4];
        deinterleave4_aos_to_soa(aos_data, re, im);
        
        // Compute in SoA (5 twiddle multiplies)
        __m256d result_re = compute_soa(re, im, ...);
        __m256d result_im = compute_soa(re, im, ...);
        
        // Convert back to AoS
        interleave4_soa_to_aos(result_re, result_im, &output[k]);
    }
}
```

**Rationale:** Many twiddle multiplies make conversion worthwhile.

### 4. Cleanup Loops: AoS (Scalar SSE2)

```c
// Processing 1-3 remaining elements
for (; k < N; k++) {
    __m128d a = _mm_loadu_pd(&input[k].re);  // [re, im]
    // ... scalar butterfly
    _mm_storeu_pd(&output[k].re, result);
}
```

**Rationale:** Conversion overhead would dominate for small counts.

---

## Advanced Notes

### Why Not Pure SoA Throughout?

**Drawbacks of pure SoA:**

1. **Memory allocation complexity**
```c
// AoS: One allocation
fft_data *array = malloc(N * sizeof(fft_data));

// SoA: Two separate allocations
double *re = malloc(N * sizeof(double));
double *im = malloc(N * sizeof(double));
// Must keep both pointers synchronized!
```

2. **Poor cache locality for random access**
```c
// AoS: One cache miss gets both components
fft_data z = array[rand() % N];

// SoA: Two cache misses
double r = re[rand() % N];  // Miss 1
double i = im[rand() % N];  // Miss 2
```

3. **API awkwardness for users**
```c
// Natural (AoS):
output[i].re = input[i].re * scale;

// Awkward (SoA):
output_re[i] = input_re[i] * scale;
output_im[i] = input_im[i] * scale;
```

4. **Not beneficial for simple operations**
- Scalar code doesn't benefit from SoA
- Overhead not justified for light computation

### Modern CPU Optimizations

**Intel Ice Lake and newer:**
- Improved shuffle units (lower latency)
- Better out-of-order execution
- AoS performance gap has narrowed

**But SoA still wins for compute-heavy code:**
- Fewer instructions = more pipelining
- FMA makes SoA even more attractive
- Vectorization width continues to grow (AVX-512)

### Alternative: AoSoA (Hybrid Layout)

Some FFT implementations use "Array of Small Structure of Arrays":

```c
// Process 4 complex at once, stored in mini-SoA chunks
struct vec4_complex {
    double re[4];  // 4 real parts together
    double im[4];  // 4 imaginary parts together
};

vec4_complex array[N/4];  // Array of 4-element SoA chunks
```

**Advantages:**
- No runtime conversion needed
- Good vectorization
- Reasonable cache locality

**Disadvantages:**
- Complex API (`array[i/4].re[i%4]`)
- Padding issues if N not divisible by 4
- Less flexible than pure AoS

**Verdict:** Trades API simplicity for computational efficiency. Not used in this implementation due to API complexity.

---

## Summary

### Key Takeaways

1. **AoS is natural for storage, SoA is faster for computation**
2. **Conversion overhead (~8 cycles) pays for itself with 2+ complex multiplies**
3. **Hybrid approach balances usability and performance**
4. **SoA enables 2× vectorization width (4 complex vs 2 complex per register)**
5. **Best for compute-heavy operations: radix-3/5/7 butterflies**

### Decision Flowchart

```
Do you need to process this data?
├─ Single element access → Use AoS
├─ Simple operation (<2 complex muls) → Use AoS
└─ Complex operation (2+ complex muls) → Convert to SoA
   ├─ Compute in SoA
   └─ Convert back to AoS
```

### Performance Summary

| Operation | AoS Time | SoA Time | Speedup |
|-----------|----------|----------|---------|
| Radix-2 butterfly | 10 μs | 11 μs | 0.9× (worse with conversion) |
| Radix-5 butterfly | 45 μs | 29 μs | 1.55× |
| 10× complex multiply | 100 μs | 58 μs | 1.72× |

**Conclusion:** Use SoA for compute-intensive inner loops, AoS everywhere else.

---

## See Also

- [Software Pipelining Strategy](Software_Pipelining.md) - Overlapping memory and computation
- [SIMD Optimization Guide](SIMD_Optimization.md) - Butterfly operations and complex multiplication
- [Memory Management](Memory_Management.md) - Scratch buffer strategy
- `deinterleave4_aos_to_soa()` implementation in `simd_math.h`
- `interleave4_soa_to_aos()` implementation in `simd_math.h`
- `cmul_soa_avx()` for SoA complex multiplication
- `cmul_avx2_aos()` for AoS complex multiplication

---

*Document version: 1.0*  
*Last updated: 2025-01-17*  
*Author: Tugbars Heptaskin
