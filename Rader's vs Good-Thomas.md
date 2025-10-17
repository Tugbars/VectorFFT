# Understanding Rader's Algorithm in Radix-7 and Good-Thomas in Radix-8

## Table of Contents

1. [Introduction](#introduction)
2. [Prime-Length FFT Challenge](#prime-length-fft-challenge)
3. [Rader's Algorithm for Radix-7](#raders-algorithm-for-radix-7)
4. [Good-Thomas Algorithm for Radix-8](#good-thomas-algorithm-for-radix-8)
5. [Algorithm Selection Rationale](#algorithm-selection-rationale)
6. [The N=56 Case Study](#the-n56-case-study)
7. [Implementation Details](#implementation-details)
8. [Performance Characteristics](#performance-characteristics)

---

## Introduction

Mixed-radix FFT implementations must handle different radix types with fundamentally different decomposition strategies. This library implements two distinct approaches:

- **Rader's Algorithm**: Applied to prime radices (Radix-7)
- **Split-Radix/Good-Thomas**: Applied to composite radices with power-of-2 structure (Radix-8)

This document explains the mathematical foundations of each approach, the reasoning behind the algorithm selection, and the technical considerations when combining them in multi-stage transforms.

---

## Prime-Length FFT Challenge

### The Cooley-Tukey Limitation

The Discrete Fourier Transform is defined as:

```
X[k] = Σ(n=0 to N-1) x[n] · exp(-2πi·kn/N)
```

Traditional Cooley-Tukey FFT factorization decomposes N-point DFTs into smaller DFTs based on the factors of N. For composite N (e.g., N=8=2³), recursive factorization is straightforward. However, for prime N (e.g., N=7), no smaller factorization exists using standard index mapping techniques.

### The Prime Number Problem

For prime N, the DFT matrix exhibits no exploitable block structure under standard Cooley-Tukey decomposition. The index set {0, 1, 2, ..., N-1} cannot be partitioned into disjoint subgroups using modular arithmetic alone. Alternative mathematical structures must be employed.

---

## Rader's Algorithm for Radix-7

### Mathematical Foundation

Rader's algorithm transforms a prime-length DFT into a cyclic convolution by exploiting the multiplicative group structure of integers modulo N.

#### Separating the DC Component

The k=0 output is computed independently:

```
X[0] = Σ(n=0 to 6) x[n]
```

This requires only additions—no complex multiplications.

#### Primitive Root Theory

For N=7, the primitive root g=3 generates all non-zero elements modulo 7:

```
Powers of 3 mod 7:
3^0 = 1
3^1 = 3
3^2 = 9 mod 7 = 2
3^3 = 27 mod 7 = 6
3^4 = 81 mod 7 = 4
3^5 = 243 mod 7 = 5
3^6 = 729 mod 7 = 1

Generated sequence: {1, 3, 2, 6, 4, 5}
```

This sequence determines the permutation structure of the algorithm.

#### The Rader Transform Identity

Using the substitutions:

```
n = g^l mod N  (l = 0, 1, ..., N-2)
k = g^(-q) mod N (q = 0, 1, ..., N-2)
```

where g^(-1) = 5 is the multiplicative inverse of 3 modulo 7, the DFT becomes:

```
X[g^(-q)] = x[0] + Σ(l=0 to 5) x[g^l] · W_7^(g^(l-q))
```

This is a cyclic convolution of permuted inputs with twiddle factors.

### Permutation Structure

**Input permutation** (reordering by powers of g=3):
```
Original indices:    [1, 2, 3, 4, 5, 6]
Permuted indices:    [1, 3, 2, 6, 4, 5]
Powers of g:         [3^0, 3^1, 3^2, 3^3, 3^4, 3^5]
```

**Output permutation** (reordering by powers of g^(-1)=5):
```
Convolution index q → Output index g^(-q) mod 7
q=0 → 5^0 mod 7 = 1 → Y[1]
q=1 → 5^1 mod 7 = 5 → Y[5]
q=2 → 5^2 mod 7 = 4 → Y[4]
q=3 → 5^3 mod 7 = 6 → Y[6]
q=4 → 5^4 mod 7 = 2 → Y[2]
q=5 → 5^5 mod 7 = 3 → Y[3]

Output permutation: [1, 5, 4, 6, 2, 3]
```

### Cyclic Convolution Computation

Each output q = 0..5 is computed as:

```
conv[q] = Σ(l=0 to 5) permuted_input[l] · twiddle[(q-l) mod 6]
```

where:

```c
twiddle[q] = exp(-2πi · output_perm[q] / 7)
```

### Computational Complexity

A single Radix-7 Rader butterfly requires:
- **36 complex multiplications** (6 convolution outputs × 6 terms each)
- **30 complex additions**
- **14 memory accesses** (7 loads + 7 stores)

This compares to 1 multiplication and 2 additions for a Radix-2 butterfly.

### Implementation Structure

The code in `fft_radix7.c` follows this structure:

```c
// 1. DC component computation
__m256d y0 = _mm256_add_pd(
    _mm256_add_pd(_mm256_add_pd(x0, x1), _mm256_add_pd(x2, x3)),
    _mm256_add_pd(_mm256_add_pd(x4, x5), x6)
);

// 2. Input permutation (g=3 powers)
__m256d tx0 = x1;  // g^0 = 1
__m256d tx1 = x3;  // g^1 = 3
__m256d tx2 = x2;  // g^2 = 2
__m256d tx3 = x6;  // g^3 = 6
__m256d tx4 = x4;  // g^4 = 4
__m256d tx5 = x5;  // g^5 = 5

// 3. Cyclic convolution (6 terms per output)
__m256d v0 = cmul_avx2_aos(tx0, tw_brd[0]);
v0 = _mm256_add_pd(v0, cmul_avx2_aos(tx1, tw_brd[5]));
v0 = _mm256_add_pd(v0, cmul_avx2_aos(tx2, tw_brd[4]));
v0 = _mm256_add_pd(v0, cmul_avx2_aos(tx3, tw_brd[3]));
v0 = _mm256_add_pd(v0, cmul_avx2_aos(tx4, tw_brd[2]));
v0 = _mm256_add_pd(v0, cmul_avx2_aos(tx5, tw_brd[1]));
// ... (repeat for v1..v5)

// 4. Output permutation application
y1 = _mm256_add_pd(x0, v0);  // q=0 → output[1]
y5 = _mm256_add_pd(x0, v1);  // q=1 → output[5]
y4 = _mm256_add_pd(x0, v2);  // q=2 → output[4]
y6 = _mm256_add_pd(x0, v3);  // q=3 → output[6]
y2 = _mm256_add_pd(x0, v4);  // q=4 → output[2]
y3 = _mm256_add_pd(x0, v5);  // q=5 → output[3]
```

---

## Good-Thomas Algorithm for Radix-8

### Factorization-Based Approach

Radix-8 = 2³ allows factorization-based decomposition. While the Good-Thomas algorithm traditionally applies to coprime factors, the power-of-2 structure of Radix-8 enables an optimized variant often called **split-radix decomposition**.

### Split-Radix Structure

The implementation decomposes Radix-8 as nested Radix-4 butterflies followed by Radix-2 combination:

1. **Two parallel Radix-4 butterflies** (even and odd index groups)
2. **W₈ twiddle application** to odd outputs
3. **Final Radix-2 combination** to produce 8 outputs

This is a Gentleman-Sande variant of Cooley-Tukey optimized for powers of 2.

### Butterfly Structure

#### Even Butterfly (indices 0, 2, 4, 6)

```c
// Radix-4 on even inputs
__m256d a_e = x[0];
__m256d c_e = x[2];
__m256d e_e = x[4];
__m256d g_e = x[6];

__m256d sumCG = _mm256_add_pd(c_e, g_e);
__m256d difCG = _mm256_sub_pd(c_e, g_e);
__m256d sumAE = _mm256_add_pd(a_e, e_e);
__m256d difAE = _mm256_sub_pd(a_e, e_e);

e[0] = _mm256_add_pd(sumAE, sumCG);
e[2] = _mm256_sub_pd(sumAE, sumCG);
e[1] = _mm256_sub_pd(difAE, rotate_90(difCG));
e[3] = _mm256_add_pd(difAE, rotate_90(difCG));
```

#### Odd Butterfly (indices 1, 3, 5, 7)

The odd butterfly follows an identical structure, producing `o[0..3]`.

### W₈ Twiddle Factors

The key to Radix-8 efficiency is the special structure of W₈ twiddles:

```
W₈^0 = 1
W₈^1 = (1/√2)(1 - i)  = exp(-πi/4)
W₈^2 = -i              = exp(-πi/2)
W₈^3 = (-1/√2)(1 + i) = exp(-3πi/4)
```

These are applied to odd outputs `o[1..3]`:

```c
const double c8 = 0.7071067811865476;  // 1/√2

// W₈^1 multiplication on o[1]
if (transform_sign == 1) {  // forward FFT
    double r = o[1].re, i = o[1].im;
    o[1].re = (r + i) * c8;
    o[1].im = (i - r) * c8;
} else {  // inverse FFT
    double r = o[1].re, i = o[1].im;
    o[1].re = (r - i) * c8;
    o[1].im = (i + r) * c8;
}

// W₈^2 = -i multiplication (swap + negate)
double temp = o[2].re;
o[2].re = (transform_sign == 1) ? o[2].im : -o[2].im;
o[2].im = (transform_sign == 1) ? -temp : temp;

// W₈^3 multiplication on o[3]
if (transform_sign == 1) {
    double sum = (o[3].re + o[3].im) * c8;
    o[3].re = -sum;
    o[3].im = -sum;
} else {
    double dif = (o[3].re - o[3].im) * c8;
    o[3].re = -dif;
    o[3].im = dif;
}
```

### Final Radix-2 Combination

```c
// Combine even and odd butterflies
for (int m = 0; m < 4; ++m) {
    output[m]     = e[m] + o[m];  // First half
    output[m + 4] = e[m] - o[m];  // Second half
}
```

### Computational Complexity

A single Radix-8 split-radix butterfly requires:
- **8 complex multiplications** (from per-stage twiddles)
- **24 complex additions** (two radix-4 + final radix-2)
- **16 memory accesses** (8 loads + 8 stores)
- **W₈ twiddles are compile-time constants** (no additional loads)

---

## Algorithm Selection Rationale

### Comparison Matrix

| Property | Radix-7 | Radix-8 | Implication |
|----------|---------|---------|-------------|
| **Mathematical nature** | Prime | Composite (2³) | Determines available algorithms |
| **Factorization** | None | Multiple (4×2, 2×2×2) | Enables recursive decomposition |
| **Algorithm choice** | Rader (required) | Split-radix (optimal) | No alternative vs. best of several |
| **Complex multiplications** | 36 per butterfly | 8 per butterfly | 4.5× cost difference |
| **FFT-specific symmetry** | None | Extensive | Power-of-2 optimizations unavailable for primes |
| **Twiddle structure** | Convolution kernel | Standard exponentials | Different mathematical properties |

### Why Rader for Radix-7

Rader's algorithm is the **only known efficient method** for prime-length FFTs. The alternatives are:

1. **Direct DFT computation**: O(N²) complexity—completely impractical
2. **Bluestein's chirp-z algorithm**: Converts prime DFT to convolution but requires zero-padding to next power-of-2, introducing overhead
3. **Rader's algorithm**: O(N log N) complexity by converting to cyclic convolution

For N=7, Rader is selected because it provides O(N log N) complexity without padding overhead. Bluestein would require padding to N=16, wasting bandwidth.

### Why Split-Radix for Radix-8

For powers of 2, multiple decomposition strategies exist:

1. **Pure Radix-2**: 3 stages of radix-2 butterflies
2. **Radix-4**: Combines pairs of radix-2 butterflies
3. **Split-radix**: Optimally combines radix-4 (even) and radix-4 (odd) with special twiddles
4. **Rader's algorithm**: Could technically be applied but would be grossly inefficient

The split-radix approach is selected because it:
- Minimizes total multiplications (theoretical minimum for power-of-2 FFTs)
- Exploits W₈ twiddle symmetries (many are real or imaginary)
- Fits naturally into SIMD architectures (4-element AVX2 vectors align with radix-4)

### The Core Principle

**Rader is used when mathematically necessary** (prime radices). **Factorization-based methods are used when mathematically possible** (composite radices). This minimizes computational cost while maintaining correctness for all supported lengths.

---

## The N=56 Case Study

### Multi-Radix Decomposition

For N=56 = 7×8, the FFT decomposes into two stages. The library's `get_fft_execution_radices()` function determines the factorization order:

```c
// N=56 factorization
int execution_radices[] = {7, 8};  // or {8, 7}
```

Both orderings are mathematically equivalent but have different memory access patterns.

### Stage Execution Flow

#### Decomposition 1: Radix-7 → Radix-8

```
Input: x[0..55]
  ↓
Stage 1 (Radix-7): 8 butterflies
  Butterfly 0: x[0, 8, 16, 24, 32, 40, 48] → temp[0..6]
  Butterfly 1: x[1, 9, 17, 25, 33, 41, 49] → temp[7..13]
  Butterfly 2: x[2, 10, 18, 26, 34, 42, 50] → temp[14..20]
  ... (8 butterflies, stride=8)
  ↓
Stage 2 (Radix-8): 7 butterflies
  Butterfly 0: temp[0, 7, 14, 21, 28, 35, 42, 49] → X[0..7]
  Butterfly 1: temp[1, 8, 15, 22, 29, 36, 43, 50] → X[8..15]
  ... (7 butterflies, stride=7)
  ↓
Output: X[0..55]
```

#### Decomposition 2: Radix-8 → Radix-7

The data flow is identical but with stages reversed.

### Twiddle Factor Organization

The library maintains separate twiddle storage for each stage due to differing requirements.

#### Stage Twiddle Counts

For N=56 with {Radix-7, Radix-8} decomposition:

**Stage 1 (Radix-7, 8 butterflies of size 8)**
- Each butterfly needs: 6 twiddles (for inputs 1..6, input 0 has no twiddle)
- Total: 8 × 6 = **48 complex twiddles**
- Storage: `stage_tw[6*k + j]` for k=0..7, j=0..5

**Stage 2 (Radix-8, 7 butterflies of size 7)**
- Each butterfly needs: 7 twiddles (for inputs 1..7)
- Total: 7 × 7 = **49 complex twiddles**
- Storage: `stage_tw[7*k + j]` for k=0..6, j=0..6

The offset computation in `fft_init()` accumulates these:

```c
int twiddle_factors_size = 0;

// Stage 1: Radix-7
int sub_fft_size = 8;
twiddle_factors_size += (7 - 1) * sub_fft_size;  // 6 × 8 = 48

// Stage 2: Radix-8  
sub_fft_size = 7;
twiddle_factors_size += (8 - 1) * sub_fft_size;  // 7 × 7 = 49

// Total: 97 complex twiddles needed
```

#### Storage Layout

```c
// In fft_set structure:
fft_data *twiddle_factors;          // Contiguous array[97]
int stage_twiddle_offset[MAX_STAGES]; // [0, 48] for 2 stages

// Stage 1 twiddles: twiddle_factors[0..47]
//   Butterfly k, input j: twiddle_factors[0 + 6*k + j]

// Stage 2 twiddles: twiddle_factors[48..96]
//   Butterfly k, input j: twiddle_factors[48 + 7*k + j]
```

### Stride Evolution

The recursive function `mixed_radix_dit_rec()` tracks input stride:

```c
// Initial call for N=56
mixed_radix_dit_rec(output, input, fft_obj, sgn,
                    56,    // data_length
                    1,     // stride (initial)
                    0,     // factor_index
                    0);    // scratch_offset

// After Stage 1 (Radix-7):
//   next_stride = 1 × 7 = 7
//   Radix-8 butterflies read inputs separated by 7

// After Stage 2 (Radix-8):
//   next_stride = 7 × 8 = 56
//   Outputs are naturally ordered
```

This stride mechanism generalizes to arbitrary radix combinations.

### Rader Convolution Twiddle Handling

The Rader convolution twiddles for Radix-7 are **computed at runtime** within `fft_radix7_butterfly()` rather than precomputed in `fft_init()`. This is because:

1. **Transform direction dependency**: The convolution kernel changes sign for inverse transforms
2. **Minimal overhead**: Only 6 complex exponentials per function call
3. **Cache efficiency**: Broadcasting into AVX2 registers (`tw_brd[]`) is done once per execution

```c
// Computed once per fft_radix7_butterfly() call
const double base_angle = (transform_sign == 1 ? -2.0 : +2.0) * M_PI / 7.0;

__m256d tw_brd[6];  // Broadcast for SIMD
for (int q = 0; q < 6; ++q) {
    const int output_index = output_perm[q];  // [1, 5, 4, 6, 2, 3]
    double angle = output_index * base_angle;
    double wr = cos(angle);
    double wi = sin(angle);
    tw_brd[q] = _mm256_set_pd(wi, wr, wi, wr);
}
```

The per-stage DIT twiddles are separate and stored in `twiddle_factors[]`.

---

## Implementation Details

### Twiddle Indexing Conventions

#### Radix-7 (Rader) Indexing

```c
// Per-stage DIT twiddles (applied before Rader convolution)
for (int lane = 1; lane < 7; ++lane) {
    __m256d tw = load2_aos(&stage_tw[6*k + (lane-1)], ...);
    x[lane] = cmul_avx2_aos(x[lane], tw);
}

// Note: Only 6 twiddles per butterfly (x[0] has no twiddle)
```

#### Radix-8 (Split-Radix) Indexing

```c
// Per-stage DIT twiddles
for (int lane = 1; lane < 8; ++lane) {
    __m256d tw = load2_aos(&stage_tw[7*k + (lane-1)], ...);
    x[lane] = cmul_avx2_aos(x[lane], tw);
}

// Note: 7 twiddles per butterfly (x[0] has no twiddle)
```

The difference in stride (6 vs. 7) is critical for correct indexing when accessing `stage_tw[]`.

### Scratch Buffer Management

The recursive function allocates scratch space per-stage:

```c
// Stage requirements
const int stage_outputs_size = radix * sub_len;
const int stage_tw_size = (radix - 1) * sub_len;

int need_this_stage = stage_outputs_size;
if (!using_precomputed_twiddles) {
    need_this_stage += stage_tw_size;
}

// Child stages use offset space
const int child_scratch_base = scratch_offset + need_this_stage;
```

For N=56:
- **Stage 1 (Radix-7)**: Needs 7×8 = 56 complex scratch
- **Stage 2 (Radix-8)**: Needs 8×7 = 56 complex scratch
- **Total**: 112 complex values minimum (children execute serially)

The library over-allocates `4 × max_padded_length` to handle worst-case Bluestein scenarios.

### Output Permutation Implementation

Radix-7 requires explicit output reordering based on the generator inverse (g⁻¹ = 5):

```c
// Convolution results in natural order q=0..5
__m256d v0 = ...; // q=0
__m256d v1 = ...; // q=1
__m256d v2 = ...; // q=2
__m256d v3 = ...; // q=3
__m256d v4 = ...; // q=4
__m256d v5 = ...; // q=5

// Map to output bins using permutation [1,5,4,6,2,3]
__m256d y1 = _mm256_add_pd(x0, v0);  // q=0 → bin 1
__m256d y5 = _mm256_add_pd(x0, v1);  // q=1 → bin 5
__m256d y4 = _mm256_add_pd(x0, v2);  // q=2 → bin 4
__m256d y6 = _mm256_add_pd(x0, v3);  // q=3 → bin 6
__m256d y2 = _mm256_add_pd(x0, v4);  // q=4 → bin 2
__m256d y3 = _mm256_add_pd(x0, v5);  // q=5 → bin 3

// Store in correct order
STOREU_PD(&output_buffer[k + 1*seventh].re, y1);
STOREU_PD(&output_buffer[k + 2*seventh].re, y2);
STOREU_PD(&output_buffer[k + 3*seventh].re, y3);
STOREU_PD(&output_buffer[k + 4*seventh].re, y4);
STOREU_PD(&output_buffer[k + 5*seventh].re, y5);
STOREU_PD(&output_buffer[k + 6*seventh].re, y6);
```

Radix-8 has no output permutation—results are stored in natural order.

### Transform Direction Handling

#### Global Twiddle Sign

The main twiddle table `fft_obj->twiddles[]` is sign-adjusted during `fft_init()`:

```c
// In fft_init()
if (transform_direction == -1) {  // Inverse FFT
    for (int i = 0; i < twiddle_count; i++) {
        fft_config->twiddles[i].im = -fft_config->twiddles[i].im;
    }
}
```

This handles most radices (2, 3, 4, 5, 8, etc.) uniformly.

#### Rader-Specific Direction Handling

Rader's convolution twiddles must be recomputed:

```c
// In fft_radix7_butterfly()
const double base_angle = (transform_sign == 1 ? -2.0 : +2.0) * M_PI / 7.0;
```

This cannot use the global twiddle table because the convolution kernel structure differs from standard FFT twiddles.

#### W₈ Direction Handling

The W₈ twiddles in Radix-8 are applied with direction-dependent operations:

```c
// W₈^1 multiplication
if (transform_sign == 1) {  // Forward: (r+i)/√2, (i-r)/√2
    o[1].re = (r + i) * c8;
    o[1].im = (i - r) * c8;
} else {  // Inverse: (r-i)/√2, (i+r)/√2
    o[1].re = (r - i) * c8;
    o[1].im = (i + r) * c8;
}
```

### SIMD Optimization Strategy

#### Radix-7 (8× Unrolling)

The AVX2 path processes 8 butterflies simultaneously:

```c
for (; k + 7 < seventh; k += 8) {
    // Load 8 copies of x[0..6] (56 complex values)
    // Apply 8 sets of DIT twiddles
    // Compute 8 Rader convolutions in parallel
    // Store 56 outputs
}
```

This amortizes the cost of loading convolution twiddles and improves instruction-level parallelism despite the long dependency chains in cyclic convolution.

#### Radix-8 (8× Unrolling)

Similarly, 8 radix-8 butterflies are processed together:

```c
for (; k + 7 < eighth; k += 8) {
    // Load 8 copies of x[0..7] (64 complex values)
    // Compute 8 even butterflies (radix-4)
    // Compute 8 odd butterflies (radix-4)
    // Apply W₈ twiddles to odd outputs
    // Radix-2 combination and store
}
```

The split-radix structure has shorter dependency chains than Rader, achieving higher throughput.

---

## Performance Characteristics

### Computational Cost Analysis

For a single N=56 = 7×8 FFT:

#### Stage 1: 8 Radix-7 Butterflies
- Complex multiplications: 8 × 36 = **288**
- Complex additions: 8 × 30 = **240**

#### Stage 2: 7 Radix-8 Butterflies
- Complex multiplications: 7 × 8 = **56**
- Complex additions: 7 × 24 = **168**

#### Total Operation Count
- Complex multiplications: **344**
- Complex additions: **408**

### Comparison to Reference Implementations

#### Pure Radix-2 (N=64, closest power of 2)
- Complex multiplications: ~192 (3N log₂N / 2)
- Complex additions: ~384 (3N log₂N)

The Rader-based N=56 FFT requires **~80% more multiplications** than a slightly larger radix-2 FFT, demonstrating the cost of prime factorization.

#### FFTW-Style N=56
FFTW would likely choose a similar {7, 8} decomposition, though the exact butterfly implementation differs. The operation count would be comparable.

### Memory Bandwidth Characteristics

#### Radix-7 Memory Access Pattern
- **Strided loads**: 7 reads with stride = `sub_len`
- **Strided stores**: 7 writes with stride = `sub_len`
- **Twiddle loads**: 6 per butterfly (if not precomputed)
- **Convolution twiddle broadcasts**: 6 (amortized over 8× unroll)

For large `sub_len`, the strided access pattern can cause cache thrashing. Prefetching partially mitigates this:

```c
if (k + 16 < seventh) {
    for (int j = 0; j < 7; ++j) {
        _mm_prefetch((const char *)&sub_outputs[j * seventh + k + 16].re, 
                     _MM_HINT_T0);
    }
}
```

#### Radix-8 Memory Access Pattern
- **Strided loads**: 8 reads with stride = `sub_len`
- **Strided stores**: 8 writes with stride = `sub_len`
- **Twiddle loads**: 7 per butterfly
- **W₈ twiddles**: 0 (compile-time constants)

The power-of-2 structure aligns better with cache line boundaries, and the absence of runtime-computed convolution twiddles reduces memory pressure.

### SIMD Efficiency Metrics

#### Radix-7 ILP (Instruction-Level Parallelism)
The cyclic convolution has inherent dependencies:

```
conv[q] depends on all 6 permuted inputs
→ Cannot fully parallelize within a single butterfly
→ 8× unrolling exploits parallelism across butterflies
```

Peak AVX2 efficiency: ~60-70% (limited by dependency chains and non-power-of-2 structure)

#### Radix-8 ILP
The split-radix structure has high ILP:

```
Even butterfly: 4 independent outputs (e[0..3])
Odd butterfly: 4 independent outputs (o[0..3])
→ Both can be computed in parallel
→ Only W₈ multiplication creates dependencies
```

Peak AVX2 efficiency: ~85-95% (limited primarily by memory bandwidth)

### Cache Behavior

For N=56:
- **L1 cache**: Entire working set (56 complex × 16 bytes = 896 bytes) fits easily
- **No thrashing expected** for reasonable sub-FFT sizes

For larger multi-stage FFTs involving Radix-7:
- Rader's non-contiguous access pattern has higher cache miss rates than Radix-8
- The library's prefetch strategy (`_MM_HINT_T0` at k+16 ahead) targets L1/L2 caches

### Algorithmic Scaling

| N | Factorization | Rader Count | Computational Cost |
|---|---------------|-------------|-------------------|
| 7 | 7 | 1 | 36 muls |
| 56 | 7×8 | 8 | 288 muls (Stage 1) |
| 448 | 7×64 | 64 | 2,304 muls (Stage 1) |
| 3136 | 7²×64 | 512 | 18,432 muls (Stage 1) |

The cost scales linearly with the number of Radix-7 butterflies. For FFTs heavily involving prime factors, Rader becomes the dominant computational bottleneck.

### Optimization Trade-offs

The library makes several design choices:

1. **Runtime Rader twiddle computation**: Small overhead (~6 `sincos()` calls) vs. reduced memory footprint and direction-handling complexity

2. **8× SIMD unrolling**: Amortizes twiddle load costs and improves ILP, at the cost of increased register pressure

3. **Separate storage for each radix**: Wastes some memory (different strides) but simplifies indexing logic

4. **Prefetch at k+16**: Empirically tuned distance balancing latency hiding vs. cache pollution

These choices prioritize **correctness and maintainability** while achieving competitive performance for mixed-radix scenarios.

---

## Summary

The library's implementation of Radix-7 (Rader) and Radix-8 (split-radix) represents fundamentally different mathematical approaches necessitated by the prime versus composite nature of the radices. Rader's algorithm is employed for Radix-7 because it is the only efficient algorithm for prime-length FFTs, converting the problem into a cyclic convolution through number-theoretic permutations. Split-radix decomposition is used for Radix-8 because it optimally exploits the power-of-2 factorization structure, minimizing arithmetic complexity and maximizing SIMD efficiency.

When combined in multi-stage transforms such as N=56, careful attention is required for twiddle storage layout (6 vs. 7 per butterfly), convolution twiddle computation (runtime vs. precomputed), and output permutation handling (explicit for Radix-7, implicit for Radix-8). The resulting implementation achieves correct transforms across all supported lengths while maintaining reasonable performance despite the inherent computational cost of prime factorization.
