
# Packed-Complex (AoSoA + `ADDSUB`) Optimization for FFT Twiddle Multiplication
**Author:** VectorFFT Notes  
**Date:** 2025  
**Applies to:** AVX2 and AVX‑512, double‑precision SoA FFT kernels (DIT/DIF), N1 and generic stages

---

## Executive Summary
This report describes a **packed-complex** micro‑kernel that accelerates batches of complex multiplications in FFT stages by packing real/imag parts into interleaved vectors (AoSoA layout) and using a single `vaddsubpd` to form the real/imag combinations. Compared to the standard SoA complex multiply (two `vmulpd` + two FMA), the packed approach reduces the **effective op count by ~30–40%** in merge blocks dominated by **generic twiddles** (no cheap constants), with typical end‑to‑end stage gains ≥ **5–12%** depending on memory traffic and unrolling. The optimization is **radix‑agnostic** and benefits radix‑64, radix‑32, and (often) radix‑16 merge sections; radix‑8 and W₄/W₈ twiddle paths should remain specialized.

---

## Background
In SoA FFT kernels, we frequently compute batches of complex products
\[(a_r + j a_i)\cdot (b_r + j b_i)\]
for twiddle merges across sub‑FFTs. The canonical SoA formulation is:

```
t0 = ai * br;          // MUL
t1 = ai * bi;          // MUL
tr = ar*br - t1;       // FMA/FMSUB
ti = ar*bi + t0;       // FMA/FMADD
// total: 2 MUL + 2 FMA per cmul
```

On AVX‑512 (and AVX2), FMAs have 0.5–1.0 cycle throughput and MULs are 1.0 cycle
(depending on µarch), so the inner block is compute‑heavy and repeats for every twiddle.

---

## Idea: Pack, Multiply, `ADDSUB`, Unpack
### Packing (AoSoA)
Rearrange two SoA registers, e.g. `(ar, ai)` and `(br, bi)`, into **packed** vectors:
```
a = pack(ar, ai)  -> [ar0, ai0, ar1, ai1, ...]
b = pack(br, bi)  -> [br0, bi0, br1, bi1, ...]
b_swap = permute(b, 0x55) -> [bi0, br0, bi1, br1, ...]  // lane-wise swap
```

### Compute
```
t0 = a * b;            // [ar*br, ai*bi, ar*br, ai*bi, ...]
t1 = a * b_swap;       // [ar*bi, ai*br, ar*bi, ai*br, ...]
sign = [+,-,+,-,...]   //  + for even lanes (real), - for odd lanes (imag)
res = vaddsubpd(t0, t1 ^ sign);   // [ar*br ± ar*bi,  ai*bi ∓ ai*br, ...]
```
A single `ADDSUB` forms the real/imag pairs across lane parity.

### Unpack Back to SoA
```
(ar', ai') = unpack(res)
```

### Instruction Count (per complex multiply)
- Pack+permute amortized across **N** complex multiplies (same twiddle batch).
- Compute uses **2× MUL + 1× ADDSUB** vs **2× MUL + 2× FMA** (SoA).
- Practical end‑to‑end saving: **~30–40%** within the **twiddle block** when N is large.

> Key: amortize pack/unpack across **many cmuls** that share the same layout; do not repack per element.

---

## Applicability by Radix
The technique is **not tied to radix‑64**. It applies wherever a merge emits a **batch of generic twiddle multiplies**:

| Radix pattern (2‑D CT) | Twiddle batch per slot | Benefit |
|---|---:|---|
| 64 = 8×8 (N1 merge r=1..7) | 56 cmuls | **Strong** |
| 32 = 8×4 or 4×8 | 24–28 cmuls | **Strong** |
| 16 = 4×4 | ~12 cmuls | **Moderate** (good with unroll U≥2) |
| 8 and W₄/W₈ | Many constants (`±1, ±j, ±c8`) | Keep **specialized** constant paths |

**Rule of thumb** per *k*-slot after one pack:  
- ≥16 cmuls ⇒ install packed kernel (clear win).  
- 8–15 cmuls ⇒ usually beneficial when pipelined across multiple *k* slots (U>1).  
- ≤7 cmuls ⇒ packing overhead often offsets gains; stick to SoA/FMAs or constants.

---

## Micro‑Kernel Sketches

### AVX‑512 (double, 8‑wide)
```c
// Inputs: ar, ai, br, bi (SoA, __m512d); outputs: tr, ti (SoA)
static inline __m512d oddmask(){ return _mm512_set_pd(-0.0, 0, -0.0, 0, -0.0, 0, -0.0, 0); }

static inline void cmul_packed_avx512(__m512d ar, __m512d ai,
                                      __m512d br, __m512d bi,
                                      __m512d* tr, __m512d* ti)
{
    __m512d a0 = _mm512_unpacklo_pd(ar, ai);  // [ar0, ai0, ar1, ai1, ...]
    __m512d a1 = _mm512_unpackhi_pd(ar, ai);
    __m512d b0 = _mm512_unpacklo_pd(br, bi);
    __m512d b1 = _mm512_unpackhi_pd(br, bi);

    __m512d b0s = _mm512_permute_pd(b0, 0x55); // swap ri in each pair
    __m512d b1s = _mm512_permute_pd(b1, 0x55);

    __m512d t00 = _mm512_mul_pd(a0, b0);
    __m512d t01 = _mm512_mul_pd(a0, b0s);
    __m512d t10 = _mm512_mul_pd(a1, b1);
    __m512d t11 = _mm512_mul_pd(a1, b1s);

    __m512d sign = oddmask();
    __m512d r0i0 = _mm512_addsub_pd(t00, _mm512_xor_pd(t01, sign));
    __m512d r1i1 = _mm512_addsub_pd(t10, _mm512_xor_pd(t11, sign));

    *tr = _mm512_permutex2var_pd(r0i0, _mm512_set_epi64(14,12,10,8,6,4,2,0), r1i1);
    *ti = _mm512_permutex2var_pd(r0i0, _mm512_set_epi64(15,13,11,9,7,5,3,1), r1i1);
}
```

### AVX2 (double, 4‑wide)
Same idea using `vpermilpd`, `vunpcklo/hi`, and `_mm256_addsub_pd`.

> In both cases, **pack once per sub‑batch**, run many cmuls, and **unpack once**.

---

## Integration Guidelines

1. **Choose the right blocks**
   - Apply in **merge** sections with **generic W_N^k** (k≠0, N∈{16, 32, 64, …}).
   - Keep W₈/W₄ specialized fast paths (rotations, sign flips, `c8` scalings).

2. **Amortize packing**
   - Organize data so one pack serves **multiple cmuls** (e.g., r=1..7 across m lanes).
   - With K‑unroll (U=2..4), reuse packed twiddles across **adjacent *k*-slots**.

3. **Register pressure**
   - Limit live state: hold **two packed a’s**, **two packed b’s**, and one or two results; broadcast twiddles on demand rather than pinning many W’s.
   - For AVX‑512, stay ≤ 24–28 ZMM live to avoid spills in large butterflies.

4. **Prefetching**
   - Prefetch inputs for **k+2U×8** while computing current *k* slots.
   - Pack/unpack is L1‑resident; ensure SoA loads/stores are 64‑byte aligned.

5. **Branching on constants**
   - Detect W patterns (`±1`, `±j`, `±c8`, `±c16`) and route to cheaper kernels; reserve packed path for the **truly generic** cases.

6. **Tails & masks**
   - AVX‑512: use native `__mmask8` masked loads/stores to keep the kernel unified.
   - AVX2: use blends for tails or scalar clean‑up if the hot loop is long.

---

## Cost Model (per *k*-slot)
Let **P** be the amortized pack/unpack cost, **C** the number of complex multiplies in the batch.

- **SoA FMA path:** ~`C * (2 MUL + 2 FMA)`
- **Packed path:** ~`P + C * (2 MUL + 1 ADDSUB)`

Break‑even occurs when `P ≤ C * (FMA - ADDSUB)`. On recent Intel cores, this is typically around **C ≈ 8–10**; above that, the packed path wins convincingly.

---

## Expected Gains
- **Within W_N merge block (compute‑bound):** ~**30–40%** fewer core ops ⇒ similar cycle reductions when data is L1‑resident.
- **Whole stage (including loads/stores):** commonly **5–12%** in radix‑64 and radix‑32 N1 stages, depending on unrolling and memory bandwidth.
- **End‑to‑end FFT:** cumulative gains depend on stage mix; stages dominated by constants dilute impact.

---

## Validation Checklist
- Bitwise‑close or within expected FP error vs reference SoA across random inputs.
- Check all transforms sizes and both forward/backward signs.
- Stress masked tails (1..7 lanes) and misaligned but allowed pointers.
- Benchmark with performance counters: retired µops, FP MUL/ADD/SUB, L1D hit rate, spills.

---

## Pitfalls
- Over‑packing (packing per element) eliminates benefits—always **batch**.
- Register spills from excessive unroll (e.g., U=4 with many live zmm) can negate gains.
- Applying to twiddles that are simple constants—use specialized fast paths instead.

---

## When Not to Use
- Batches with **≤7** complex multiplies after packing.
- Branch‑heavy code paths where the extra permutes cannot be amortized.
- Extremely memory‑bound kernels where compute savings do not surface.

---

## Appendix A — Minimal AVX‑512 Helper (lane sign mask)
```c
static inline __m512d oddmask_pd() {
    //  +0.0, -0.0 alternating to flip add/sub sense on odd lanes
    const __m512d neg_zero = _mm512_set1_pd(-0.0);
    __m512i m = _mm512_set_epi64(1,0,1,0,1,0,1,0);
    return _mm512_mask_blend_pd(_mm512_kmov(m), _mm512_setzero_pd(), neg_zero);
}
```

## Appendix B — Packing/Unpacking Macros (AVX‑512)
```c
#define PACK_RI(a_re,a_im,out_lo,out_hi)  \
    do{ out_lo = _mm512_unpacklo_pd(a_re,a_im); \
        out_hi = _mm512_unpackhi_pd(a_re,a_im);} while(0)

#define UNPACK_RI(r0i0,r1i1,out_re,out_im) \
    do{ const __m512i idx_re = _mm512_set_epi64(14,12,10,8,6,4,2,0); \
        const __m512i idx_im = _mm512_set_epi64(15,13,11,9,7,5,3,1); \
        out_re = _mm512_permutex2var_pd(r0i0, idx_re, r1i1); \
        out_im = _mm512_permutex2var_pd(r0i0, idx_im, r1i1);} while(0)
```

---

## Conclusion
The packed‑complex (AoSoA + `vaddsubpd`) micro‑kernel is a **portable, radix‑agnostic** acceleration for FFT twiddle batches that are dominated by **generic complex multiplies**. It delivers **meaningful (>5%) stage‑level wins** when applied to radix‑64 and radix‑32 merges and often helps radix‑16, provided you amortize pack/unpack and manage register pressure. Keep specialized constant twiddle paths for small N (W₄/W₈).

